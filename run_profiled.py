"""Profiled ICR Probe pipeline for RAGTruth dataset.

Phases:
  1. model_loading   — Load LLM (single GPU)
  2. data_loading    — Load RAGTruth
  3. icr_extraction  — Per-sample forward → ICR Score → layer-mean feature
  4. probe_training  — Train ICRProbe MLP (sentence-level)
  5. evaluation      — Sentence-level AUROC/F1 on test set
  6. token_level     — Token-level detection on subset of test samples

Does NOT modify any src/ files.

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_profiled.py \
        --model_name /gz-fs/models/Meta-Llama-3.1-8B-Instruct \
        --ragtruth_data_dir ../../data/RAGTruth/dataset \
        --ragtruth_model_filter llama-2-7b-chat \
        --ragtruth_task_types QA \
        --token_level_n_samples 20 \
        --profile_output_dir ./profiling_results/icr_llama_QA/
"""

import os, argparse, logging, gc, json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.icr_score import ICRScore
from src.utils import ICRProbe
from profiler import ICRProfiler, clear_gpu_memory
from ragtruth_loader import RAGTruthICRLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Teacher-forcing → ICRScore format adapter
# ---------------------------------------------------------------------------

def teacher_forcing_to_icr_format(hf_hidden_states, hf_attentions, prompt_len):
    """Convert single-pass HF output to ICRScore's autoregressive-style nested list."""
    num_hs_layers = len(hf_hidden_states)
    num_attn_layers = len(hf_attentions)
    total_len = hf_hidden_states[0].shape[1]
    response_len = total_len - prompt_len

    icr_hidden = [[hf_hidden_states[l][:, :prompt_len, :].detach() for l in range(num_hs_layers)]]
    for t in range(response_len):
        icr_hidden.append([hf_hidden_states[l][:, prompt_len+t:prompt_len+t+1, :].detach()
                           for l in range(num_hs_layers)])

    icr_attn = [[hf_attentions[l][:, :, :prompt_len, :prompt_len].detach()
                 for l in range(num_attn_layers)]]
    for t in range(response_len):
        icr_attn.append([hf_attentions[l][:, :, prompt_len+t:prompt_len+t+1, :prompt_len+t+1].detach()
                         for l in range(num_attn_layers)])

    return icr_hidden, icr_attn


def compute_core_positions(tokenizer, prompt_text, response_text):
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).input_ids
    prompt_len = prompt_ids.shape[1]
    full_text = prompt_text + "\n" + response_text
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=True).input_ids
    total_len = full_ids.shape[1]
    core_positions = {"user_prompt_start": 0, "user_prompt_end": prompt_len, "response_start": prompt_len}
    return core_positions, prompt_len, total_len, full_ids


def extract_icr_features_for_sample(
    model, tokenizer, sample, max_length, icr_device,
    top_k=None, top_p=0.1, pooling="mean", use_induction_head=True,
    return_token_scores=False,
):
    """Extract ICR feature for one sample.

    Returns:
        feature: np.array (num_layers,) — sentence-level feature
        token_scores: list[list] [layer][token] — if return_token_scores
        success: bool
    """
    prompt_text = sample["prompt_text"]
    response_text = sample["response_text"]
    core_positions, prompt_len, total_len, full_ids = compute_core_positions(
        tokenizer, prompt_text, response_text)

    if max_length and total_len > max_length:
        full_ids = full_ids[:, :max_length]
        total_len = max_length
        if prompt_len >= total_len:
            return (None, None, False) if return_token_scores else (None, False)

    if total_len - prompt_len < 2:
        return (None, None, False) if return_token_scores else (None, False)

    device = next(model.parameters()).device
    with torch.no_grad():
        output = model(full_ids.to(device), output_hidden_states=True, output_attentions=True)

    icr_hidden, icr_attn = teacher_forcing_to_icr_format(
        output.hidden_states, output.attentions, prompt_len)
    del output; clear_gpu_memory()

    try:
        scorer = ICRScore(hidden_states=icr_hidden, attentions=icr_attn,
                          core_positions=core_positions, icr_device=icr_device)
        icr_scores, _ = scorer.compute_icr(
            top_k=top_k, top_p=top_p, pooling=pooling,
            attention_uniform=False, hidden_uniform=False,
            use_induction_head=use_induction_head)
        n_layers = len(icr_scores)
        feature = np.array([float(np.mean(icr_scores[l])) if icr_scores[l] else 0.0
                            for l in range(n_layers)], dtype=np.float32)
    except Exception as e:
        logger.warning(f"ICR failed: {e}")
        del icr_hidden, icr_attn; clear_gpu_memory()
        return (None, None, False) if return_token_scores else (None, False)

    raw = icr_scores if return_token_scores else None
    del scorer, icr_hidden, icr_attn
    if not return_token_scores: del icr_scores
    clear_gpu_memory()
    return (feature, raw, True) if return_token_scores else (feature, True)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--torch_dtype", type=str, default="float16", choices=["float16","bfloat16","float32"])
    p.add_argument("--ragtruth_data_dir", type=str, required=True)
    p.add_argument("--ragtruth_task_types", type=str, nargs="+", default=None)
    p.add_argument("--ragtruth_max_length", type=int, default=1024)
    p.add_argument("--ragtruth_model_filter", type=str, nargs="+", default=None)
    p.add_argument("--icr_top_k", type=int, default=None)
    p.add_argument("--icr_top_p", type=float, default=0.1)
    p.add_argument("--icr_pooling", type=str, default="mean")
    p.add_argument("--icr_use_induction_head", action="store_true", default=True)
    p.add_argument("--no_induction_head", dest="icr_use_induction_head", action="store_false")
    p.add_argument("--probe_num_epochs", type=int, default=100)
    p.add_argument("--probe_batch_size", type=int, default=32)
    p.add_argument("--probe_lr", type=float, default=5e-4)
    p.add_argument("--probe_weight_decay", type=float, default=1e-5)
    p.add_argument("--probe_lr_factor", type=float, default=0.5)
    p.add_argument("--probe_lr_patience", type=int, default=5)
    p.add_argument("--halu_threshold", type=float, default=0.5)
    p.add_argument("--token_level_n_samples", type=int, default=20)
    p.add_argument("--profile_output_dir", type=str, default="./profiling_results/")
    p.add_argument("--gpu_util_interval", type=float, default=2.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def run_probe_training(train_X, train_y, val_X, val_y, args, pp):
    dev = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = ICRProbe(input_dim=train_X.shape[1]).to(dev)
    criterion = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.probe_lr, weight_decay=args.probe_weight_decay)
    sched = ReduceLROnPlateau(opt, mode="min", factor=args.probe_lr_factor, patience=args.probe_lr_patience)

    tl = DataLoader(TensorDataset(torch.tensor(train_X,dtype=torch.float32),
                                   torch.tensor(train_y,dtype=torch.float32)),
                    batch_size=args.probe_batch_size, shuffle=True)
    vl = DataLoader(TensorDataset(torch.tensor(val_X,dtype=torch.float32),
                                   torch.tensor(val_y,dtype=torch.float32)),
                    batch_size=args.probe_batch_size, shuffle=False)

    best_loss, best_metrics, best_sd = float("inf"), {}, None
    for epoch in range(args.probe_num_epochs):
        with pp.epoch_scope():
            model.train(); tot_loss = 0; nb = 0
            for xb, yb in tl:
                xb, yb = xb.to(dev), yb.to(dev)
                with pp.operation("forward"):
                    out = model(xb); loss = criterion(out, yb.unsqueeze(1))
                with pp.operation("backward"):
                    opt.zero_grad(); loss.backward(); opt.step()
                tot_loss += loss.item(); nb += 1
            train_loss = tot_loss / max(nb, 1)

            with pp.operation("validation"):
                vm = _val_probe(model, vl, criterion, dev, args.halu_threshold)
            sched.step(vm["val_loss"])

            pp.log_epoch_metric("train_loss", train_loss)
            pp.log_epoch_metric("val_loss", vm["val_loss"])
            pp.log_epoch_metric("auroc", vm["ROC-AUC"])
            pp.log_epoch_metric("f1", vm["F1"])

            if vm["val_loss"] < best_loss:
                best_loss = vm["val_loss"]; best_metrics = dict(vm)
                best_sd = {k: v.clone() for k, v in model.state_dict().items()}

            if (epoch+1) % 10 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{args.probe_num_epochs} — "
                            f"train={train_loss:.4f} val={vm['val_loss']:.4f} "
                            f"AUROC={vm['ROC-AUC']:.4f} F1={vm['F1']:.4f}")
    if best_sd: model.load_state_dict(best_sd)
    return model, best_metrics


def _val_probe(model, vl, criterion, dev, thr=0.5):
    model.eval()
    losses, preds_c, labels = [], [], []
    with torch.no_grad():
        for xb, yb in vl:
            xb, yb = xb.to(dev), yb.to(dev)
            out = model(xb)
            losses.append(criterion(out, yb.unsqueeze(1)).item())
            preds_c.extend(out.squeeze().cpu().numpy().tolist())
            labels.extend(yb.cpu().numpy().tolist())

    preds_bin = [1 if p >= thr else 0 for p in preds_c]
    TP = sum(1 for i in range(len(preds_c)) if preds_c[i] >= thr and labels[i] == 1)
    FP = sum(1 for i in range(len(preds_c)) if preds_c[i] >= thr and labels[i] == 0)
    FN = sum(1 for i in range(len(preds_c)) if preds_c[i] < thr and labels[i] == 1)
    TN = sum(1 for i in range(len(preds_c)) if preds_c[i] < thr and labels[i] == 0)
    prec = TP/(TP+FP) if TP+FP>0 else 0
    rec = TP/(TP+FN) if TP+FN>0 else 0
    f1 = f1_score(labels, preds_bin, zero_division=0)
    try:
        auroc = float(roc_auc_score(labels, preds_c))
    except: auroc = 0.0
    try: pcc = float(np.corrcoef(labels, preds_c)[0,1])
    except: pcc = 0.0

    return {"val_loss": float(np.mean(losses)), "Precision": prec, "Recall": rec,
            "F1": f1, "ROC-AUC": auroc, "PCC": pcc if not np.isnan(pcc) else 0.0}


# ---------------------------------------------------------------------------
# Token-level detection (paper Section 5.5)
# ---------------------------------------------------------------------------

def run_token_level_detection(probe_model, llm, tokenizer, test_samples, test_labels,
                              args, icr_device, output_dir):
    """Each response token's L-dim ICR vector → probe → truthfulness probability."""
    n = min(args.token_level_n_samples, len(test_samples))
    if n <= 0: return []

    dev = next(probe_model.parameters()).device
    probe_model.eval()
    results = []

    for idx in range(n):
        sample = test_samples[idx]
        label = int(test_labels[idx])

        feat, tscores, ok = extract_icr_features_for_sample(
            llm, tokenizer, sample, args.ragtruth_max_length,
            icr_device, args.icr_top_k, args.icr_top_p,
            args.icr_pooling, args.icr_use_induction_head,
            return_token_scores=True)
        if not ok or tscores is None: continue

        n_layers = len(tscores)
        n_tok = len(tscores[0]) if tscores else 0
        if n_tok == 0: continue

        tf = np.zeros((n_tok, n_layers), dtype=np.float32)
        for l in range(n_layers):
            for t in range(min(len(tscores[l]), n_tok)):
                tf[t, l] = float(tscores[l][t])

        with torch.no_grad():
            tp = probe_model(torch.tensor(tf, dtype=torch.float32).to(dev)).squeeze().cpu().numpy()
        with torch.no_grad():
            sp = float(probe_model(torch.tensor(feat,dtype=torch.float32).unsqueeze(0).to(dev)).squeeze().cpu())

        full_text = sample["prompt_text"] + "\n" + sample["response_text"]
        full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=True).input_ids[0]
        plen = tokenizer(sample["prompt_text"], return_tensors="pt", add_special_tokens=True).input_ids.shape[1]
        resp_ids = full_ids[plen:]

        tdet = []
        for t in range(min(n_tok, len(resp_ids))):
            tok_txt = tokenizer.decode([resp_ids[t].item()])
            prob = float(tp[t]) if np.ndim(tp) > 0 else float(tp)
            tdet.append({"token": tok_txt, "prob_truthful": round(prob, 4)})

        results.append({
            "sample_idx": idx, "task_type": sample.get("task_type", ""),
            "ground_truth": "truthful" if label == 1 else "hallucinated",
            "sentence_pred": round(sp, 4),
            "response_preview": sample["response_text"][:300],
            "token_detection": tdet,
        })
        logger.info(f"Token [{idx}]: {results[-1]['ground_truth']}, sent={sp:.4f}, tokens={len(tdet)}")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "token_level_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Token-level saved to {path} ({len(results)} samples)")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    profiler = ICRProfiler(output_dir=args.profile_output_dir)

    # Phase 1: Model Loading
    with profiler.phase("model_loading", args.gpu_util_interval) as p:
        logger.info(f"Loading: {args.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=dtype_map[args.torch_dtype],
            device_map={"": args.device}, low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        profiler.set_model_info(model, args.model_name)
    logger.info(f"Loaded. Layers={model.config.num_hidden_layers}, Hidden={model.config.hidden_size}")

    # Phase 2: Data Loading
    with profiler.phase("data_loading", args.gpu_util_interval) as p:
        loader = RAGTruthICRLoader(data_dir=args.ragtruth_data_dir,
                                    task_types=args.ragtruth_task_types,
                                    model_filter=args.ragtruth_model_filter)
        (train_samples, train_labels), (test_samples, test_labels), stats = loader.load_splits()
        stats.update({"detection_model": args.model_name,
                      "ragtruth_model_filter": args.ragtruth_model_filter,
                      "ragtruth_task_types": args.ragtruth_task_types,
                      "max_token_length_setting": args.ragtruth_max_length})
        profiler.set_dataset_info(stats)
    logger.info(f"Data: train={len(train_samples)}, test={len(test_samples)}")

    # Phase 3: ICR Extraction
    icr_device = next(model.parameters()).device
    with profiler.phase("icr_extraction", args.gpu_util_interval) as p:
        train_feats, train_vlabels = [], []
        for i, s in enumerate(tqdm(train_samples, desc="ICR Train")):
            with p.operation("forward_pass"):
                feat, ok = extract_icr_features_for_sample(
                    model, tokenizer, s, args.ragtruth_max_length, icr_device,
                    args.icr_top_k, args.icr_top_p, args.icr_pooling, args.icr_use_induction_head)
            if ok and feat is not None:
                train_feats.append(feat); train_vlabels.append(train_labels[i])
            if (i+1) % 10 == 0: clear_gpu_memory()

        test_feats, test_vlabels = [], []
        for i, s in enumerate(tqdm(test_samples, desc="ICR Test")):
            with p.operation("forward_pass"):
                feat, ok = extract_icr_features_for_sample(
                    model, tokenizer, s, args.ragtruth_max_length, icr_device,
                    args.icr_top_k, args.icr_top_p, args.icr_pooling, args.icr_use_induction_head)
            if ok and feat is not None:
                test_feats.append(feat); test_vlabels.append(test_labels[i])
            if (i+1) % 10 == 0: clear_gpu_memory()

    if not train_feats or not test_feats:
        raise RuntimeError("No valid samples after ICR extraction.")

    train_X = np.stack(train_feats); train_y = np.array(train_vlabels, dtype=np.int32)
    test_X = np.stack(test_feats);   test_y = np.array(test_vlabels, dtype=np.int32)
    logger.info(f"Features: train={train_X.shape}, test={test_X.shape}")

    # Phase 4: Probe Training
    with profiler.phase("probe_training", args.gpu_util_interval) as p:
        probe, best_m = run_probe_training(train_X, train_y, test_X, test_y, args, p)
    logger.info(f"Probe best: AUROC={best_m.get('ROC-AUC',0):.4f} F1={best_m.get('F1',0):.4f}")

    # Phase 5: Sentence-level Eval
    with profiler.phase("evaluation", args.gpu_util_interval) as p:
        dev = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        probe.eval()
        with torch.no_grad():
            preds = probe(torch.tensor(test_X,dtype=torch.float32).to(dev)).squeeze().cpu().numpy()
        try: auroc = float(roc_auc_score(test_y, preds))
        except: auroc = 0.0
        f1v = float(f1_score(test_y, (preds>=args.halu_threshold).astype(int), zero_division=0))
    logger.info("=" * 60)
    logger.info(f"SENTENCE-LEVEL:  AUROC={auroc:.4f}  F1={f1v:.4f}")
    logger.info("=" * 60)

    # Phase 6: Token-level Detection
    if args.token_level_n_samples > 0:
        logger.info(f"Token-level on {args.token_level_n_samples} test samples...")
        with profiler.phase("token_level_detection", args.gpu_util_interval) as p:
            run_token_level_detection(probe, model, tokenizer,
                                     test_samples, test_labels,
                                     args, icr_device, args.profile_output_dir)

    # Save everything
    profiler.save()
    fd = os.path.join(args.profile_output_dir, "features")
    os.makedirs(fd, exist_ok=True)
    np.save(os.path.join(fd, "train_features.npy"), train_X)
    np.save(os.path.join(fd, "train_labels.npy"), train_y)
    np.save(os.path.join(fd, "test_features.npy"), test_X)
    np.save(os.path.join(fd, "test_labels.npy"), test_y)
    logger.info(f"All saved to {args.profile_output_dir}")


if __name__ == "__main__":
    main()