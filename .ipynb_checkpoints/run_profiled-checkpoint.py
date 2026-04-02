"""Optimized ICR Probe pipeline for RAGTruth — single 48GB GPU.

Strategy:
  - Short sequences (<=1200 tokens): Use original ICRScore class directly on GPU.
    Full speed, 100% faithful to original code.
  - Long sequences (>1200 tokens): Layer-by-layer GPU computation.
    Same math as ICRScore, but processes one layer at a time to avoid OOM.
    GPU only holds ~1 layer of [H,N,N] attention at a time.

Both paths produce identical ICR scores.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python run_profiled.py \
        --model_name /gz-fs/models/Meta-Llama-3.1-8B-Instruct \
        --ragtruth_data_dir ../../data/RAGTruth/dataset \
        --ragtruth_model_filter llama-2-7b-chat \
        --ragtruth_task_types Summary \
        --profile_output_dir ./profiling_results/icr_llama3.1-8B_llama-2-7b-chat_Summary/
"""

import os, sys, argparse, logging, gc, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(__file__))
from src.icr_score import ICRScore, js_divergence
from src.utils import ICRProbe
from profiler import ICRProfiler, clear_gpu_memory
from ragtruth_loader import RAGTruthICRLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Sequences longer than this use layer-by-layer mode.
# 1200 tokens: full [32,32,1200,1200]*2B = 2.8GB attention, fits 48GB with model+hs+intermediates.
MAX_FULL_TOKENS = 1200


# ---------------------------------------------------------------------------
# Teacher-forcing → ICRScore format (for fast path)
# ---------------------------------------------------------------------------

def teacher_forcing_to_icr_format(hf_hidden_states, hf_attentions, prompt_len):
    """Convert HF output to ICRScore's nested list format. Stay on GPU."""
    num_hs_layers = len(hf_hidden_states)
    num_attn_layers = len(hf_attentions)
    total_len = hf_hidden_states[0].shape[1]
    response_len = total_len - prompt_len

    icr_hidden = [[hf_hidden_states[l][:, :prompt_len, :].detach() for l in range(num_hs_layers)]]
    for t in range(response_len):
        icr_hidden.append([hf_hidden_states[l][:, prompt_len+t:prompt_len+t+1, :].detach()
                           for l in range(num_hs_layers)])

    icr_attn = [[hf_attentions[l][:, :, :prompt_len, :prompt_len].detach() for l in range(num_attn_layers)]]
    for t in range(response_len):
        icr_attn.append([hf_attentions[l][:, :, prompt_len+t:prompt_len+t+1, :prompt_len+t+1].detach()
                         for l in range(num_attn_layers)])
    return icr_hidden, icr_attn


# ---------------------------------------------------------------------------
# Layer-by-layer ICR for long sequences (faithful math, avoids OOM)
# ---------------------------------------------------------------------------

def _calc_skew_ent(attn_map):
    """ICRScore._calculate_skewness_entropy — exact."""
    N = attn_map.size(0)
    rs = attn_map.sum(1, keepdim=True)
    rn = attn_map / (rs + 1e-12)
    idx = torch.arange(1, N+1, device=attn_map.device, dtype=attn_map.dtype).view(1, -1)
    mu = (rn * idx).sum(1)
    var = ((idx - mu.unsqueeze(1))**2 * rn).sum(1)
    m3 = ((idx - mu.unsqueeze(1))**3 * rn).sum(1)
    skew = m3 / (var**1.5 + 1e-12)
    ent = -(rn * torch.log2(rn + 1e-12)).sum(1)
    v = rs.squeeze() > 0
    return (skew[v].mean().item() if v.any() else 0.0,
            ent[v].mean().item() if v.any() else 0.0)


def compute_icr_layerwise(hf_hs, hf_attn, prompt_len, dev,
                          top_k=20, top_p=None, pooling="mean",
                          use_induction_head=True, skew_thr=3, ent_thr=3):
    """Layer-by-layer ICR: same math as ICRScore, one layer's attention at a time.

    Args:
        hf_hs: list of (L+1) tensors, each [1, N, H] (hidden states from HF output)
        hf_attn: nested list [step][layer], where:
            hf_attn[0][l] = [1, H, prompt_len, prompt_len]  (prompt self-attention)
            hf_attn[t+1][l] = [1, H, 1, prompt_len+t+1]    (response token t's attention)
        prompt_len: int
        dev: torch.device
    """
    total_len = hf_hs[0].shape[1]
    resp_len = total_len - prompt_len
    n_layers = len(hf_attn[0])         # number of transformer layers
    n_heads = hf_attn[0][0].shape[1]   # hf_attn[0][0] is [1, H, P, P]
    if resp_len < 1:
        return [], 0.0

    # Hidden states: stack once (small, ~1.3GB max)
    hs = torch.stack([h.squeeze(0) for h in hf_hs]).to(dev)  # [L+1, N, H]
    a, b, c = 0, prompt_len, prompt_len

    scores_all, tp_list = [], []
    for li in range(n_layers):
        # Build this layer's [H, N, N] from per-token slices
        # hf_attn[0][li] is [1, H, P, P] — prompt's self-attention for layer li
        inp = hf_attn[0][li][0, :, :prompt_len, :prompt_len].to(dev)  # [H, P, P]
        inp_pad = F.pad(inp, (0, total_len - prompt_len))  # [H, P, N]
        del inp
        rows = []
        for t in range(resp_len):
            # hf_attn[t+1][li] is [1, H, 1, prompt_len+t+1]
            r = hf_attn[t+1][li][0, :, 0, :]  # [H, prompt_len+t+1]
            rows.append(F.pad(r, (0, total_len - prompt_len - t - 1)))  # [H, N]
        out = torch.stack(rows, dim=1).to(dev)  # [H, resp_len, N]
        del rows
        la = torch.cat([inp_pad, out], dim=1)  # [H, N, N]
        del inp_pad, out

        # Mask (same as set_other_attn_scores_to_zero)
        mask = torch.zeros(n_heads, total_len, total_len, dtype=torch.bool, device=dev)
        mask[:, a:b, a:b] = True; mask[:, c:, c:] = True
        la[~mask] = 0
        del mask

        # Induction head detection
        if use_induction_head:
            se = torch.zeros(n_heads, 2, device=dev)
            for h in range(n_heads):
                se[h, 0], se[h, 1] = _calc_skew_ent(la[h])
            ih = (se[:, 0] >= skew_thr) & (se[:, 1] <= ent_thr)
            if ih.sum() < n_heads // 8:
                ih[:] = False
                ih[se[:, 0].topk(n_heads // 8).indices] = True
        else:
            ih = torch.ones(n_heads, dtype=torch.bool, device=dev)

        # Pool over induction heads (response region)
        sel = la[:, prompt_len:, :][ih]  # [n_sel, resp, N]
        pooled = sel.mean(0) if sel.shape[0] > 0 else torch.zeros(resp_len, total_len, device=dev)
        del la, sel
        torch.cuda.empty_cache()

        # ICR per token
        layer_scores, tp_layer = [], []
        for tok in range(resp_len):
            ca = pooled[tok, :prompt_len+tok+1]
            k = min(top_k, len(ca)) if top_k is not None else len(ca)
            if top_p is not None: k = int(top_p * len(ca))
            k = max(1, k)
            tp_layer.append(k / max(len(ca), 1e-6))
            tv, ti = torch.topk(ca, k=k)
            hd = hs[li+1, prompt_len+tok] - hs[li, prompt_len+tok]
            att_hs = hs[li][ti]
            wi = torch.sum(hd * att_hs, dim=1) / (torch.norm(att_hs, dim=1) + 1e-8)
            try:
                sc = js_divergence(wi, tv)
                sc = sc if not (isinstance(sc, float) and (np.isnan(sc) or np.isinf(sc))) else 0.0
            except:
                sc = 0.0
            layer_scores.append(sc)
        del pooled
        scores_all.append(layer_scores)
        tp_list.append(tp_layer)

    del hs; torch.cuda.empty_cache()
    return scores_all, np.mean(tp_list) if tp_list else 0.0


# ---------------------------------------------------------------------------
# Unified extraction
# ---------------------------------------------------------------------------

def extract_icr(model, tokenizer, sample, dev, top_k=20, top_p=None,
                pooling="mean", use_ih=True, return_raw=False):
    """Extract ICR. Short→original ICRScore on GPU. Long→layer-by-layer."""
    pt = sample["prompt_text"]
    rt = sample["response_text"]
    pid = tokenizer(pt, return_tensors="pt", add_special_tokens=True).input_ids
    plen = pid.shape[1]
    full = tokenizer(pt + "\n" + rt, return_tensors="pt", add_special_tokens=True).input_ids
    tlen = full.shape[1]
    rlen = tlen - plen
    if rlen < 2:
        return (None, None, False) if return_raw else (None, False)

    with torch.no_grad():
        out = model(full.to(dev), output_hidden_states=True, output_attentions=True)

    try:
        if tlen <= MAX_FULL_TOKENS:
            # FAST: original ICRScore on GPU
            cp = {"user_prompt_start": 0, "user_prompt_end": plen, "response_start": plen}
            ih, ia = teacher_forcing_to_icr_format(out.hidden_states, out.attentions, plen)
            del out; gc.collect(); torch.cuda.empty_cache()
            scorer = ICRScore(hidden_states=ih, attentions=ia, core_positions=cp, icr_device=dev)
            scores, _ = scorer.compute_icr(top_k=top_k, top_p=top_p, pooling=pooling,
                                           attention_uniform=False, hidden_uniform=False,
                                           use_induction_head=use_ih)
            del scorer, ih, ia
        else:
            # SAFE: layer-by-layer
            hs_ref, at_ref = list(out.hidden_states), list(out.attentions)
            # Build per-token nested list (views, not copies)
            at_nested = [[at_ref[l][:, :, :plen, :plen] for l in range(len(at_ref))]]
            for t in range(rlen):
                at_nested.append([at_ref[l][:, :, plen+t:plen+t+1, :plen+t+1] for l in range(len(at_ref))])
            del out; gc.collect(); torch.cuda.empty_cache()
            scores, _ = compute_icr_layerwise(hs_ref, at_nested, plen, dev,
                                              top_k=top_k, top_p=top_p, pooling=pooling,
                                              use_induction_head=use_ih)
            del hs_ref, at_nested
    except Exception as e:
        logger.warning(f"ICR failed (seq={tlen}): {e}")
        try: del out
        except: pass
        gc.collect(); torch.cuda.empty_cache()
        return (None, None, False) if return_raw else (None, False)

    gc.collect(); torch.cuda.empty_cache()
    nl = len(scores)
    feat = np.array([float(np.mean(scores[l])) if scores[l] else 0.0 for l in range(nl)], dtype=np.float32)
    feat = np.nan_to_num(feat, nan=0.0)
    if return_raw:
        return feat, scores, True
    return feat, True


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
    p.add_argument("--ragtruth_model_filter", type=str, nargs="+", default=None)
    p.add_argument("--icr_top_k", type=int, default=20)
    p.add_argument("--icr_top_p", type=float, default=None)
    p.add_argument("--icr_pooling", type=str, default="mean")
    p.add_argument("--icr_use_induction_head", action="store_true", default=True)
    p.add_argument("--no_induction_head", dest="icr_use_induction_head", action="store_false")
    p.add_argument("--probe_num_epochs", type=int, default=50)
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
    m = ICRProbe(input_dim=train_X.shape[1]).to(dev)
    crit = nn.BCELoss()
    opt = torch.optim.Adam(m.parameters(), lr=args.probe_lr, weight_decay=args.probe_weight_decay)
    sch = ReduceLROnPlateau(opt, mode="min", factor=args.probe_lr_factor, patience=args.probe_lr_patience)
    tl = DataLoader(TensorDataset(torch.tensor(train_X, dtype=torch.float32),
                                  torch.tensor(train_y, dtype=torch.float32)),
                    batch_size=args.probe_batch_size, shuffle=True)
    vl = DataLoader(TensorDataset(torch.tensor(val_X, dtype=torch.float32),
                                  torch.tensor(val_y, dtype=torch.float32)),
                    batch_size=args.probe_batch_size, shuffle=False)
    best_loss, best_m_dict, best_sd = float("inf"), {}, None
    for ep in range(args.probe_num_epochs):
        with pp.epoch_scope():
            m.train(); tl_sum, nb = 0, 0
            for xb, yb in tl:
                xb, yb = xb.to(dev), yb.to(dev)
                with pp.operation("forward"):
                    o = m(xb); loss = crit(o, yb.unsqueeze(1))
                with pp.operation("backward"):
                    opt.zero_grad(); loss.backward(); opt.step()
                tl_sum += loss.item(); nb += 1
            trl = tl_sum / max(nb, 1)
            with pp.operation("validation"):
                vm = _val(m, vl, crit, dev, args.halu_threshold)
            sch.step(vm["val_loss"])
            pp.log_epoch_metric("train_loss", trl)
            pp.log_epoch_metric("val_loss", vm["val_loss"])
            pp.log_epoch_metric("auroc", vm["auroc"])
            pp.log_epoch_metric("f1", vm["f1"])
            if vm["val_loss"] < best_loss:
                best_loss = vm["val_loss"]; best_m_dict = dict(vm)
                best_sd = {k: v.clone() for k, v in m.state_dict().items()}
            if (ep+1) % 10 == 0 or ep == 0:
                logger.info(f"Ep {ep+1}/{args.probe_num_epochs} tl={trl:.4f} vl={vm['val_loss']:.4f} "
                            f"AUROC={vm['auroc']:.4f} F1={vm['f1']:.4f}")
    if best_sd: m.load_state_dict(best_sd)
    return m, best_m_dict

def _val(m, vl, crit, dev, thr):
    m.eval(); ls, pc, lb = [], [], []
    with torch.no_grad():
        for xb, yb in vl:
            xb, yb = xb.to(dev), yb.to(dev)
            o = m(xb); ls.append(crit(o, yb.unsqueeze(1)).item())
            pc.extend(o.squeeze().cpu().numpy().tolist()); lb.extend(yb.cpu().numpy().tolist())
    pb = [1 if p >= thr else 0 for p in pc]
    try: auc = float(roc_auc_score(lb, pc))
    except: auc = 0.0
    return {"val_loss": float(np.mean(ls)), "auroc": auc,
            "f1": f1_score(lb, pb, zero_division=0)}


# ---------------------------------------------------------------------------
# Token-level
# ---------------------------------------------------------------------------

def run_token_level(probe, llm, tok, test_s, test_l, args, dev, out_dir):
    n = min(args.token_level_n_samples, len(test_s))
    if n <= 0: return []
    pdev = next(probe.parameters()).device; probe.eval(); res = []
    for i in range(n):
        s, lb = test_s[i], int(test_l[i])
        f, ts, ok = extract_icr(llm, tok, s, dev, args.icr_top_k, args.icr_top_p,
                                args.icr_pooling, args.icr_use_induction_head, return_raw=True)
        if not ok or ts is None: continue
        nl, nt = len(ts), len(ts[0]) if ts else 0
        if nt == 0: continue
        tf = np.zeros((nt, nl), dtype=np.float32)
        for l in range(nl):
            for t in range(min(len(ts[l]), nt)):
                v = float(ts[l][t]); tf[t,l] = v if not (np.isnan(v) or np.isinf(v)) else 0.0
        with torch.no_grad():
            tp = probe(torch.tensor(tf).to(pdev)).squeeze().cpu().numpy()
            sp = float(probe(torch.tensor(f).unsqueeze(0).to(pdev)).squeeze().cpu())
        ftxt = s["prompt_text"] + "\n" + s["response_text"]
        fids = tok(ftxt, return_tensors="pt", add_special_tokens=True).input_ids[0]
        pl = tok(s["prompt_text"], return_tensors="pt", add_special_tokens=True).input_ids.shape[1]
        rids = fids[pl:]
        td = [{"token": tok.decode([rids[t].item()]),
               "prob_truthful": round(float(tp[t]) if np.ndim(tp)>0 else float(tp), 4)}
              for t in range(min(nt, len(rids)))]
        res.append({"sample_idx": i, "task_type": s.get("task_type",""),
                     "ground_truth": "truthful" if lb==1 else "hallucinated",
                     "sentence_pred": round(sp,4), "token_detection": td})
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "token_level_results.json"), "w") as fp:
        json.dump(res, fp, indent=2, ensure_ascii=False)
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    dtm = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    prof = ICRProfiler(output_dir=args.profile_output_dir)
    dev = torch.device(f"cuda:{args.device}")

    with prof.phase("model_loading", args.gpu_util_interval):
        logger.info(f"Loading: {args.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, dtype=dtm[args.torch_dtype],
            device_map={"": args.device}, low_cpu_mem_usage=True,
            attn_implementation="eager")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        model.eval(); prof.set_model_info(model, args.model_name)
    logger.info(f"Loaded. L={model.config.num_hidden_layers} H={model.config.hidden_size}")

    with prof.phase("data_loading", args.gpu_util_interval):
        ldr = RAGTruthICRLoader(data_dir=args.ragtruth_data_dir,
                                task_types=args.ragtruth_task_types,
                                model_filter=args.ragtruth_model_filter)
        (tr_s, tr_l), (te_s, te_l), stats = ldr.load_splits()
        stats.update({"detection_model": args.model_name, "icr_top_k": args.icr_top_k})
        prof.set_dataset_info(stats)
    logger.info(f"Data: train={len(tr_s)}, test={len(te_s)}")

    with prof.phase("icr_extraction", args.gpu_util_interval) as p:
        tr_f, tr_vl = [], []
        for i, s in enumerate(tqdm(tr_s, desc="ICR Train")):
            with p.operation("forward_pass"):
                f, ok = extract_icr(model, tokenizer, s, dev, args.icr_top_k, args.icr_top_p,
                                    args.icr_pooling, args.icr_use_induction_head)
            if ok and f is not None: tr_f.append(f); tr_vl.append(tr_l[i])
        te_f, te_vl = [], []
        for i, s in enumerate(tqdm(te_s, desc="ICR Test")):
            with p.operation("forward_pass"):
                f, ok = extract_icr(model, tokenizer, s, dev, args.icr_top_k, args.icr_top_p,
                                    args.icr_pooling, args.icr_use_induction_head)
            if ok and f is not None: te_f.append(f); te_vl.append(te_l[i])

    if not tr_f or not te_f: raise RuntimeError("No valid samples.")
    trX, trY = np.stack(tr_f), np.array(tr_vl, dtype=np.int32)
    teX, teY = np.stack(te_f), np.array(te_vl, dtype=np.int32)
    for nm, X, Y in [("train",trX,trY),("test",teX,teY)]:
        v = ~np.isnan(X).any(1)
        if not v.all():
            logger.warning(f"Removing {(~v).sum()} {nm} NaN samples")
            if nm=="train": trX,trY=X[v],Y[v]
            else: teX,teY=X[v],Y[v]
    logger.info(f"Features: train={trX.shape}, test={teX.shape}")

    with prof.phase("probe_training", args.gpu_util_interval) as p:
        probe, bm = run_probe_training(trX, trY, teX, teY, args, p)
    logger.info(f"Probe: AUROC={bm.get('auroc',0):.4f} F1={bm.get('f1',0):.4f}")

    with prof.phase("evaluation", args.gpu_util_interval):
        probe.eval()
        with torch.no_grad():
            preds = probe(torch.tensor(teX, dtype=torch.float32).to(dev)).squeeze().cpu().numpy()
        try: auroc = float(roc_auc_score(teY, preds))
        except: auroc = 0.0
        f1v = float(f1_score(teY, (preds >= args.halu_threshold).astype(int), zero_division=0))
    logger.info("=" * 60)
    logger.info(f"SENTENCE-LEVEL:  AUROC={auroc:.4f}  F1={f1v:.4f}")
    logger.info("=" * 60)

    if args.token_level_n_samples > 0:
        with prof.phase("token_level_detection", args.gpu_util_interval):
            run_token_level(probe, model, tokenizer, te_s, te_l, args, dev, args.profile_output_dir)

    prof.save()
    fd = os.path.join(args.profile_output_dir, "features"); os.makedirs(fd, exist_ok=True)
    np.save(os.path.join(fd, "train_features.npy"), trX)
    np.save(os.path.join(fd, "train_labels.npy"), trY)
    np.save(os.path.join(fd, "test_features.npy"), teX)
    np.save(os.path.join(fd, "test_labels.npy"), teY)
    logger.info(f"All saved to {args.profile_output_dir}")

if __name__ == "__main__":
    main()