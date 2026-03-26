"""Profiled ICR Probe pipeline entry point for RAGTruth dataset.

Runs the full ICR hallucination detection pipeline with profiling:
  Phase 1: model_loading   — Load LLM with hidden_states + attentions output
  Phase 2: data_loading    — Load RAGTruth via RAGTruthICRLoader
  Phase 3: icr_extraction  — Per-sample: tokenize → forward → ICR score → aggregate
  Phase 4: probe_training  — Train ICRProbe MLP on aggregated ICR features
  Phase 5: evaluation      — Final metrics on test set

Does NOT modify any existing src/ files. Uses ICRScore and ICRProbe as-is.

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_profiled.py \
        --model_name meta-llama/Meta-Llama-3.1-8B \
        --ragtruth_data_dir /path/to/RAGTruth/ \
        --ragtruth_max_length 1024 \
        --profile_output_dir ./profiling_results/ragtruth_llama/
"""

import os
import sys
import argparse
import logging
import gc

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ICR imports (from existing src/)
from src.icr_score import ICRScore
from src.utils import ICRProbe

# Profiling imports (at project root)
from profiler import ICRProfiler, clear_gpu_memory
from ragtruth_loader import RAGTruthICRLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Teacher-forcing → ICRScore format adapter
# ---------------------------------------------------------------------------

def teacher_forcing_to_icr_format(hf_hidden_states, hf_attentions, prompt_len):
    """Convert HuggingFace single-pass output to ICRScore's expected nested list format.

    ICRScore was designed for autoregressive generation where each token is generated
    one at a time. In teacher-forcing mode, the entire sequence is processed in one
    forward pass. This function slices the single-pass output to match ICRScore's
    expected input structure.

    Args:
        hf_hidden_states: Tuple of (num_layers+1) tensors, each (1, total_len, H).
            Includes embedding layer output + all transformer layer outputs.
        hf_attentions: Tuple of num_layers tensors, each (1, n_head, total_len, total_len).
        prompt_len: Number of tokens in the prompt (before response starts).

    Returns:
        icr_hidden: List[List[Tensor]] — [output_size+1][layer] where:
            icr_hidden[0][l] = (1, prompt_len, H)           — prompt hidden states at layer l
            icr_hidden[t][l] = (1, 1, H) for t=1..resp_len  — response token t hidden states
        icr_attn: List[List[Tensor]] — [output_size+1][layer] where:
            icr_attn[0][l] = (1, n_head, prompt_len, prompt_len)  — prompt self-attention
            icr_attn[t][l] = (1, n_head, 1, prompt_len+t)        — response token t attention
    """
    num_hs_layers = len(hf_hidden_states)   # num_hidden_layers + 1 (includes embedding)
    num_attn_layers = len(hf_attentions)    # num_hidden_layers
    total_len = hf_hidden_states[0].shape[1]
    response_len = total_len - prompt_len

    # --- Hidden states ---
    icr_hidden = []

    # Step 0: prompt tokens
    step0_hs = []
    for l in range(num_hs_layers):
        step0_hs.append(hf_hidden_states[l][:, :prompt_len, :].detach())
    icr_hidden.append(step0_hs)

    # Steps 1..response_len: one response token each
    for t in range(response_len):
        step_t_hs = []
        for l in range(num_hs_layers):
            step_t_hs.append(hf_hidden_states[l][:, prompt_len + t: prompt_len + t + 1, :].detach())
        icr_hidden.append(step_t_hs)

    # --- Attentions ---
    icr_attn = []

    # Step 0: prompt self-attention
    step0_attn = []
    for l in range(num_attn_layers):
        step0_attn.append(hf_attentions[l][:, :, :prompt_len, :prompt_len].detach())
    icr_attn.append(step0_attn)

    # Steps 1..response_len: attention of each response token over all preceding tokens
    for t in range(response_len):
        step_t_attn = []
        for l in range(num_attn_layers):
            # Row: prompt_len + t (the current response token)
            # Cols: 0 to prompt_len + t (all tokens up to and including current)
            attn_slice = hf_attentions[l][:, :, prompt_len + t: prompt_len + t + 1, :prompt_len + t + 1].detach()
            step_t_attn.append(attn_slice)
        icr_attn.append(step_t_attn)

    return icr_hidden, icr_attn


def compute_core_positions(tokenizer, prompt_text, response_text):
    """Compute core_positions dict for ICRScore.

    Tokenizes prompt and response separately to determine boundaries.

    Returns:
        core_positions: dict with user_prompt_start, user_prompt_end, response_start
        prompt_len: int, number of prompt tokens
        total_len: int, total token count
    """
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).input_ids
    prompt_len = prompt_ids.shape[1]

    # For the full sequence, tokenize prompt + response together
    full_text = prompt_text + "\n" + response_text
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=True).input_ids
    total_len = full_ids.shape[1]

    core_positions = {
        "user_prompt_start": 0,
        "user_prompt_end": prompt_len,
        "response_start": prompt_len,
    }

    return core_positions, prompt_len, total_len, full_ids


def extract_icr_features_for_sample(
    model,
    tokenizer,
    sample,
    max_length,
    icr_device,
    top_k=None,
    top_p=0.1,
    pooling="mean",
    use_induction_head=True,
):
    """Extract ICR feature vector for a single sample.

    Returns:
        feature: numpy array of shape (num_layers,) — mean ICR score per layer
        success: bool — whether extraction succeeded
    """
    prompt_text = sample["prompt_text"]
    response_text = sample["response_text"]

    core_positions, prompt_len, total_len, full_ids = compute_core_positions(
        tokenizer, prompt_text, response_text
    )

    # Apply max_length truncation
    if max_length and total_len > max_length:
        full_ids = full_ids[:, :max_length]
        total_len = max_length
        # Adjust if prompt_len exceeds max_length
        if prompt_len >= total_len:
            return None, False

    response_len = total_len - prompt_len
    if response_len < 2:
        return None, False

    device = next(model.parameters()).device
    input_ids = full_ids.to(device)

    # Forward pass
    with torch.no_grad():
        output = model(
            input_ids,
            output_hidden_states=True,
            output_attentions=True,
        )

    hf_hidden_states = output.hidden_states
    hf_attentions = output.attentions

    # Convert to ICRScore format
    icr_hidden, icr_attn = teacher_forcing_to_icr_format(
        hf_hidden_states, hf_attentions, prompt_len
    )

    # Free HF output immediately
    del output, hf_hidden_states, hf_attentions
    clear_gpu_memory()

    # Compute ICR scores
    try:
        icr_scorer = ICRScore(
            hidden_states=icr_hidden,
            attentions=icr_attn,
            core_positions=core_positions,
            icr_device=icr_device,
        )

        icr_scores, _ = icr_scorer.compute_icr(
            top_k=top_k,
            top_p=top_p,
            pooling=pooling,
            attention_uniform=False,
            hidden_uniform=False,
            use_induction_head=use_induction_head,
        )

        # Aggregate: mean ICR score per layer → feature vector
        num_layers = len(icr_scores)
        feature = np.array([
            float(np.mean(icr_scores[layer])) if icr_scores[layer] else 0.0
            for layer in range(num_layers)
        ], dtype=np.float32)

    except Exception as e:
        logger.warning(f"ICR computation failed for sample: {e}")
        del icr_hidden, icr_attn
        clear_gpu_memory()
        return None, False

    # Cleanup
    del icr_scorer, icr_hidden, icr_attn, icr_scores
    clear_gpu_memory()

    return feature, True


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Profiled ICR Probe pipeline on RAGTruth")

    # Model
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model name/path (e.g., meta-llama/Meta-Llama-3.1-8B)")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--torch_dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])

    # RAGTruth data
    parser.add_argument("--ragtruth_data_dir", type=str, required=True,
                        help="Path to RAGTruth dataset dir (response.jsonl + source_info.jsonl)")
    parser.add_argument("--ragtruth_task_types", type=str, nargs="+", default=None,
                        help="Task types to include: QA Summary Data2txt (default: all)")
    parser.add_argument("--ragtruth_max_length", type=int, default=1024,
                        help="Max token length for sequences (default: 1024)")
    parser.add_argument("--ragtruth_model_filter", type=str, nargs="+", default=None,
                        help="Only include samples from these source models (partial match)")

    # ICR extraction params
    parser.add_argument("--icr_top_k", type=int, default=None)
    parser.add_argument("--icr_top_p", type=float, default=0.1)
    parser.add_argument("--icr_pooling", type=str, default="mean", choices=["mean", "max", "min"])
    parser.add_argument("--icr_use_induction_head", action="store_true", default=True)
    parser.add_argument("--no_induction_head", dest="icr_use_induction_head", action="store_false")

    # Probe training params
    parser.add_argument("--probe_num_epochs", type=int, default=100)
    parser.add_argument("--probe_batch_size", type=int, default=16)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_weight_decay", type=float, default=1e-5)
    parser.add_argument("--probe_lr_factor", type=float, default=0.5)
    parser.add_argument("--probe_lr_patience", type=int, default=5)
    parser.add_argument("--halu_threshold", type=float, default=0.5,
                        help="Threshold for binary classification in validation metrics")

    # Profiling
    parser.add_argument("--profile_output_dir", type=str, default="./profiling_results/")
    parser.add_argument("--gpu_util_interval", type=float, default=2.0)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Phase 4: Probe training (replicated from ICRProbeTrainer for profiling)
# ---------------------------------------------------------------------------

def run_probe_training(
    train_features, train_labels,
    val_features, val_labels,
    args, phase_profiler,
):
    """Train ICRProbe with per-epoch profiling.

    Replicates the training loop from ICRProbeTrainer to enable per-epoch
    and per-operation profiling without modifying src/icr_probe.py.

    Returns:
        model: trained ICRProbe
        best_metrics: dict of best validation metrics
    """
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    input_dim = train_features.shape[1]
    model = ICRProbe(input_dim=input_dim).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min",
        factor=args.probe_lr_factor,
        patience=args.probe_lr_patience,
    )

    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(train_features, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(val_features, dtype=torch.float32),
        torch.tensor(val_labels, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.probe_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.probe_batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_metrics = {}
    best_state_dict = None

    for epoch in range(args.probe_num_epochs):
        with phase_profiler.epoch_scope():
            # --- Train ---
            model.train()
            total_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                with phase_profiler.operation("forward"):
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch.unsqueeze(1))

                with phase_profiler.operation("backward"):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            train_loss = total_loss / max(n_batches, 1)

            # --- Validate ---
            with phase_profiler.operation("validation"):
                val_metrics = _validate_probe(model, val_loader, criterion, device, args.halu_threshold)

            # Scheduler step
            scheduler.step(val_metrics["val_loss"])

            # Log epoch metrics
            phase_profiler.log_epoch_metric("train_loss", train_loss)
            phase_profiler.log_epoch_metric("val_loss", val_metrics["val_loss"])
            phase_profiler.log_epoch_metric("auroc", val_metrics["ROC-AUC"])
            phase_profiler.log_epoch_metric("f1", val_metrics["F1 Score"])

            # Track best
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                best_metrics = dict(val_metrics)
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{args.probe_num_epochs} — "
                    f"train_loss={train_loss:.4f}, val_loss={val_metrics['val_loss']:.4f}, "
                    f"AUROC={val_metrics['ROC-AUC']:.4f}, F1={val_metrics['F1 Score']:.4f}"
                )

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, best_metrics


def _validate_probe(model, val_loader, criterion, device, halu_threshold=0.5):
    """Run probe validation and compute all metrics (mirrors ICRProbeTrainer._validate_epoch)."""
    model.eval()
    val_losses = []
    val_preds = []
    val_preds_continuous = []
    val_labels = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            val_losses.append(loss.item())
            outputs = outputs.squeeze()
            preds = outputs.round()
            val_preds.extend(preds.cpu().numpy().tolist())
            val_preds_continuous.extend(outputs.cpu().numpy().tolist())
            val_labels.extend(y_batch.cpu().numpy().tolist())

    # Metrics (same logic as ICRProbeTrainer._validate_epoch)
    TP = FP = FN = TN = 0
    for i in range(len(val_preds_continuous)):
        if val_preds_continuous[i] >= halu_threshold:
            if val_labels[i] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if val_labels[i] == 1:
                FN += 1
            else:
                TN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    f1 = f1_score(val_labels, val_preds, zero_division=0)

    try:
        fpr, tpr, thresholds = roc_curve(val_labels, val_preds_continuous)
        roc_auc = float(roc_auc_score(val_labels, val_preds_continuous))
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = float(thresholds[optimal_idx])
    except ValueError:
        roc_auc = 0.0
        optimal_threshold = 0.5

    try:
        pcc = float(np.corrcoef(val_labels, val_preds_continuous)[0, 1])
    except (ValueError, IndexError):
        pcc = 0.0

    neg_preds = [val_preds_continuous[i] for i in range(len(val_preds_continuous)) if val_labels[i] == 0]
    pos_preds = [val_preds_continuous[i] for i in range(len(val_preds_continuous)) if val_labels[i] == 1]

    return {
        "val_loss": float(np.mean(val_losses)),
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PCC": pcc if not np.isnan(pcc) else 0.0,
        "optimal_threshold": optimal_threshold,
        "mean_neg_pred": float(np.mean(neg_preds)) if neg_preds else 0.0,
        "mean_pos_pred": float(np.mean(pos_preds)) if pos_preds else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve torch dtype
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.torch_dtype]

    profiler = ICRProfiler(output_dir=args.profile_output_dir)

    # ===== Phase 1: Model Loading =====
    with profiler.phase("model_loading", args.gpu_util_interval) as p:
        logger.info(f"Loading model: {args.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            output_hidden_states=True,
            output_attentions=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        profiler.set_model_info(model, args.model_name)

    logger.info(f"Model loaded. Layers: {model.config.num_hidden_layers}, "
                f"Hidden: {model.config.hidden_size}")

    # ===== Phase 2: Data Loading =====
    with profiler.phase("data_loading", args.gpu_util_interval) as p:
        loader = RAGTruthICRLoader(
            data_dir=args.ragtruth_data_dir,
            task_types=args.ragtruth_task_types,
            model_filter=args.ragtruth_model_filter,
        )
        (train_samples, train_labels), (test_samples, test_labels), stats = loader.load_splits()

        # Add sequence length info to stats
        stats["max_token_length_setting"] = args.ragtruth_max_length
        profiler.set_dataset_info(stats)

    logger.info(f"Data loaded: train={len(train_samples)}, test={len(test_samples)}")

    # ===== Phase 3: ICR Extraction =====
    # Extract train and test features separately to preserve split
    with profiler.phase("icr_extraction", args.gpu_util_interval) as p:
        logger.info(f"Extracting ICR features: {len(train_samples)} train + {len(test_samples)} test")
        icr_device = next(model.parameters()).device

        train_features_list = []
        train_valid_labels = []
        for i, sample in enumerate(tqdm(train_samples, desc="ICR Train")):
            with p.operation("forward_pass"):
                feature, success = extract_icr_features_for_sample(
                    model, tokenizer, sample, args.ragtruth_max_length,
                    icr_device,
                    args.icr_top_k, args.icr_top_p, args.icr_pooling, args.icr_use_induction_head,
                )
            if success and feature is not None:
                train_features_list.append(feature)
                train_valid_labels.append(train_labels[i])
            if (i + 1) % 10 == 0:
                clear_gpu_memory()

        test_features_list = []
        test_valid_labels = []
        for i, sample in enumerate(tqdm(test_samples, desc="ICR Test")):
            with p.operation("forward_pass"):
                feature, success = extract_icr_features_for_sample(
                    model, tokenizer, sample, args.ragtruth_max_length,
                    icr_device,
                    args.icr_top_k, args.icr_top_p, args.icr_pooling, args.icr_use_induction_head,
                )
            if success and feature is not None:
                test_features_list.append(feature)
                test_valid_labels.append(test_labels[i])
            if (i + 1) % 10 == 0:
                clear_gpu_memory()

    if not train_features_list or not test_features_list:
        raise RuntimeError("Insufficient valid samples after ICR extraction.")

    train_features = np.stack(train_features_list)
    train_valid_labels = np.array(train_valid_labels, dtype=np.int32)
    test_features = np.stack(test_features_list)
    test_valid_labels = np.array(test_valid_labels, dtype=np.int32)

    logger.info(f"ICR features: train={train_features.shape}, test={test_features.shape}")
    logger.info(f"Train labels: truthful={train_valid_labels.sum()}, "
                f"hallucinated={len(train_valid_labels) - train_valid_labels.sum()}")
    logger.info(f"Test labels: truthful={test_valid_labels.sum()}, "
                f"hallucinated={len(test_valid_labels) - test_valid_labels.sum()}")

    # Free LLM from GPU — no longer needed
    del model
    clear_gpu_memory()

    # ===== Phase 4: Probe Training =====
    with profiler.phase("probe_training", args.gpu_util_interval) as p:
        probe_model, best_train_metrics = run_probe_training(
            train_features, train_valid_labels,
            test_features, test_valid_labels,
            args, p,
        )

    logger.info(f"Probe training complete. Best val metrics:")
    for k, v in best_train_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # ===== Phase 5: Evaluation =====
    with profiler.phase("evaluation", args.gpu_util_interval) as p:
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        probe_model.eval()

        with p.operation("test_inference"):
            test_tensor = torch.tensor(test_features, dtype=torch.float32).to(device)
            with torch.no_grad():
                test_preds = probe_model(test_tensor).squeeze().cpu().numpy()

        with p.operation("metric_computation"):
            test_preds_binary = (test_preds >= args.halu_threshold).astype(int)

            try:
                final_auroc = float(roc_auc_score(test_valid_labels, test_preds))
            except ValueError:
                final_auroc = 0.0

            final_f1 = float(f1_score(test_valid_labels, test_preds_binary, zero_division=0))

            try:
                final_pcc = float(np.corrcoef(test_valid_labels, test_preds)[0, 1])
                if np.isnan(final_pcc):
                    final_pcc = 0.0
            except (ValueError, IndexError):
                final_pcc = 0.0

            TP = int(((test_preds >= args.halu_threshold) & (test_valid_labels == 1)).sum())
            FP = int(((test_preds >= args.halu_threshold) & (test_valid_labels == 0)).sum())
            FN = int(((test_preds < args.halu_threshold) & (test_valid_labels == 1)).sum())
            TN = int(((test_preds < args.halu_threshold) & (test_valid_labels == 0)).sum())

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    logger.info("=" * 60)
    logger.info("Final Test Results:")
    logger.info(f"  AUROC:     {final_auroc:.4f}")
    logger.info(f"  F1:        {final_f1:.4f}")
    logger.info(f"  PCC:       {final_pcc:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    logger.info("=" * 60)

    # ===== Save profiling results =====
    profiler.save()
    logger.info(f"Profiling results saved to {args.profile_output_dir}")

    # Save ICR features for reproducibility
    features_dir = os.path.join(args.profile_output_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    np.save(os.path.join(features_dir, "train_features.npy"), train_features)
    np.save(os.path.join(features_dir, "train_labels.npy"), train_valid_labels)
    np.save(os.path.join(features_dir, "test_features.npy"), test_features)
    np.save(os.path.join(features_dir, "test_labels.npy"), test_valid_labels)
    logger.info(f"ICR features saved to {features_dir}")


if __name__ == "__main__":
    main()
