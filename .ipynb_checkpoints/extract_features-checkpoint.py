#!/usr/bin/env python3
"""Extract ICR scores from LLM for RAGTruth samples — faithful to original ICRScore.

Faithfully reproduces src/icr_score.py's ICRScore.compute_icr() including:
  - Induction head detection (skewness >= 3, entropy <= 3)
  - Induction head pooling (only selected heads, mean pooling)
  - Top-k token selection (k=20 by default, or top_p override)
  - Hidden state projection + JS divergence

Tracks GPU memory: peak + average allocated per sample.

Usage:
    CUDA_VISIBLE_DEVICES=0 python extract_features.py \
        --model_name /gz-fs/models/Meta-Llama-3.1-8B-Instruct \
        --ragtruth_data_dir ../../data/RAGTruth/dataset \
        --ragtruth_model_filter llama-2-7b-chat \
        --ragtruth_task_types Summary \
        --output_dir /gz-data/icr_features/llama3.1-8B/Summary
"""

import os, argparse, json, logging, gc, time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# GPU Memory Tracker
# ===========================================================================

class GPUMemoryTracker:
    """Track per-sample GPU memory: peak and average allocated."""

    def __init__(self):
        self.peak_mb_list = []
        self.avg_mb_list = []
        self.sample_times = []
        self._current_samples = []

    def begin_sample(self):
        """Call before processing a sample."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self._current_samples = []
        self._start_time = time.time()

    def snapshot(self):
        """Call during processing to record a memory snapshot for averaging."""
        if torch.cuda.is_available():
            self._current_samples.append(torch.cuda.memory_allocated() / (1024 ** 2))

    def end_sample(self):
        """Call after processing a sample. Records peak and average."""
        elapsed = time.time() - self._start_time
        self.sample_times.append(elapsed)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self.peak_mb_list.append(peak)

            if self._current_samples:
                self.avg_mb_list.append(np.mean(self._current_samples))
            else:
                self.avg_mb_list.append(torch.cuda.memory_allocated() / (1024 ** 2))

    def summary(self):
        """Return summary dict."""
        if not self.peak_mb_list:
            return {}
        return {
            "num_samples": len(self.peak_mb_list),
            "gpu_peak_mb_max": round(max(self.peak_mb_list), 1),
            "gpu_peak_mb_mean": round(np.mean(self.peak_mb_list), 1),
            "gpu_peak_mb_min": round(min(self.peak_mb_list), 1),
            "gpu_avg_allocated_mb_mean": round(np.mean(self.avg_mb_list), 1),
            "gpu_avg_allocated_mb_max": round(max(self.avg_mb_list), 1),
            "total_time_seconds": round(sum(self.sample_times), 1),
            "avg_time_per_sample_seconds": round(np.mean(self.sample_times), 2),
            "gpu_peak_gb_max": round(max(self.peak_mb_list) / 1024, 2),
            "gpu_avg_allocated_gb_mean": round(np.mean(self.avg_mb_list) / 1024, 2),
        }


# ===========================================================================
# Faithful ICR Score computation (matches src/icr_score.py exactly)
# ===========================================================================

def calculate_skewness_entropy(attn_map):
    """Exact replica of ICRScore._calculate_skewness_entropy."""
    sequence_size = attn_map.size(0)
    row_sums = attn_map.sum(dim=1, keepdim=True)
    row_normalized = attn_map / (row_sums + 1e-12)
    indices = torch.arange(1, sequence_size + 1, device=attn_map.device, dtype=attn_map.dtype).view(1, -1)
    mean_indices = (row_normalized * indices).sum(dim=1)
    variance = ((indices - mean_indices.unsqueeze(1)) ** 2 * row_normalized).sum(dim=1)
    third_moment = ((indices - mean_indices.unsqueeze(1)) ** 3 * row_normalized).sum(dim=1)
    skewness = third_moment / (variance ** 1.5 + 1e-12)
    entropy = -torch.sum(row_normalized * torch.log2(row_normalized + 1e-12), dim=1)
    valid_rows = row_sums.squeeze() > 0
    avg_skewness = skewness[valid_rows].mean().item() if valid_rows.any() else 0.0
    avg_entropy = entropy[valid_rows].mean().item() if valid_rows.any() else 0.0
    return avg_skewness, avg_entropy


def detect_induction_heads(attn_all, skew_threshold=3, entropy_threshold=3):
    """Exact replica of ICRScore._is_induction_head."""
    n_layers, n_heads = attn_all.shape[0], attn_all.shape[1]
    is_induction = []
    for layer_idx in range(n_layers):
        skewness_entropy = torch.zeros(n_heads, 2, device=attn_all.device)
        for head_idx in range(n_heads):
            s, e = calculate_skewness_entropy(attn_all[layer_idx, head_idx])
            skewness_entropy[head_idx] = torch.tensor([s, e], device=attn_all.device)
        skewness = skewness_entropy[:, 0]
        entropy = skewness_entropy[:, 1]
        heads = (skewness >= skew_threshold) & (entropy <= entropy_threshold)
        if heads.sum() < n_heads // 8:
            top_heads = skewness.topk(n_heads // 8, largest=True).indices
            heads[:] = False
            heads[top_heads] = True
        is_induction.append(heads.tolist())
    return is_induction


def pool_attentions_with_induction(output_attn, induction_heads, pooling="mean"):
    """Exact replica of ICRScore._pooling_attn."""
    n_layers = output_attn.shape[0]
    n_heads = output_attn.shape[1]
    pooled = []
    for layer_idx in range(n_layers):
        selected = []
        for head_idx in range(n_heads):
            if induction_heads[layer_idx][head_idx]:
                selected.append(output_attn[layer_idx, head_idx])
        if selected:
            stacked = torch.stack(selected)
            if pooling == "mean":
                pooled.append(torch.mean(stacked, dim=0))
            elif pooling == "max":
                pooled.append(torch.max(stacked, dim=0)[0])
            elif pooling == "min":
                pooled.append(torch.min(stacked, dim=0)[0])
        else:
            pooled.append(torch.zeros_like(output_attn[layer_idx, 0]))
    return pooled


def js_divergence(p, q):
    """Exact replica of src/icr_score.py js_divergence."""
    p = (p - p.mean()) / max(p.std().item(), 1e-8)
    q = (q - q.mean()) / max(q.std().item(), 1e-8)
    p = F.softmax(p, dim=0)
    q = F.softmax(q, dim=0)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum().item()
    kl_qm = (q * (q / m).log()).sum().item()
    return 0.5 * kl_pm + 0.5 * kl_qm


def compute_icr_faithful(
    hidden_states_all, attentions_all, prompt_len, response_len,
    top_k=20, top_p=None, pooling="mean",
    use_induction_head=True, skew_threshold=3, entropy_threshold=3,
):
    """Faithful reproduction of ICRScore.compute_icr()."""
    n_layers = attentions_all.shape[0]

    output_hs = hidden_states_all[:, prompt_len:prompt_len + response_len, :]
    output_attn = attentions_all[:, :, prompt_len:prompt_len + response_len, :]

    if use_induction_head:
        induction_heads = detect_induction_heads(attentions_all, skew_threshold, entropy_threshold)
    else:
        induction_heads = [[True] * attentions_all.shape[1]] * n_layers

    pooled_attn = pool_attentions_with_induction(output_attn, induction_heads, pooling)

    icr_scores = []
    for layer_idx in range(n_layers):
        layer_scores = []
        for token_idx in range(min(response_len, len(pooled_attn[layer_idx]))):
            current_attn = pooled_attn[layer_idx][token_idx]
            valid_len = prompt_len + token_idx + 1
            current_attn = current_attn[:valid_len]

            if len(current_attn) == 0:
                layer_scores.append(0.0)
                continue

            k = min(top_k, len(current_attn)) if top_k is not None else len(current_attn)
            if top_p is not None:
                k = int(top_p * len(current_attn))
            k = max(1, k)
            topk_values, topk_indices = torch.topk(current_attn, k=k)

            current_hs = output_hs[layer_idx + 1, token_idx]
            previous_hs = output_hs[layer_idx, token_idx]
            hs_diff = current_hs - previous_hs

            prev_layer_all_hs = hidden_states_all[layer_idx]
            attended_hs = prev_layer_all_hs[topk_indices]
            w_i = torch.sum(hs_diff * attended_hs, dim=1) / (torch.norm(attended_hs, dim=1) + 1e-8)

            try:
                score = js_divergence(w_i, topk_values)
                if np.isnan(score) or np.isinf(score):
                    score = 0.0
            except:
                score = 0.0
            layer_scores.append(score)
        icr_scores.append(layer_scores)

    return np.array(icr_scores, dtype=np.float32)


# ===========================================================================
# RAGTruth data loading
# ===========================================================================

def load_ragtruth(data_dir, task_types=None, model_filter=None, split_filter=None):
    """Load RAGTruth samples."""
    source_map = {}
    with open(os.path.join(data_dir, "source_info.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            item = json.loads(line)
            source_map[str(item["source_id"])] = item

    samples = []
    with open(os.path.join(data_dir, "response.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            item = json.loads(line)
            if split_filter and item.get("split", "") != split_filter: continue
            if item.get("quality", "good") in {"incorrect_refusal", "truncated"}: continue
            source_id = str(item.get("source_id", ""))
            source_info = source_map.get(source_id)
            if source_info is None: continue
            task_type = source_info.get("task_type", "QA")
            if task_types and task_type not in task_types: continue
            item_model = item.get("model", "")
            if model_filter and not any(mf.lower() in item_model.lower() for mf in model_filter): continue

            prompt = source_info.get("prompt", "")
            if not prompt:
                src = source_info.get("source_info", "")
                if task_type == "QA" and isinstance(src, dict):
                    q = src.get("question", ""); p = src.get("passages", "")
                    if isinstance(p, list): p = "\n\n".join(str(x) for x in p)
                    prompt = f"Q: {q}\nContext: {p}"
                elif task_type == "Summary":
                    prompt = f"Summarize: {src if isinstance(src, str) else str(src)}"
                elif task_type == "Data2txt" and isinstance(src, dict):
                    prompt = f"Describe: {json.dumps(src, ensure_ascii=False)}"
                else:
                    prompt = str(src) if src else ""

            labels = item.get("labels", [])
            label = 0 if labels else 1

            samples.append({
                "id": item.get("id", f"{source_id}_{item_model}"),
                "prompt_text": prompt.strip(),
                "response_text": item.get("response", ""),
                "label": label,
                "split": item.get("split", "train"),
                "task_type": task_type,
            })
    return samples


# ===========================================================================
# Feature extraction + ICR computation
# ===========================================================================

def extract_and_save(model, tokenizer, samples, output_dir, device, top_k=20, top_p=None):
    """Forward pass → ICR scores on GPU → save only scores to disk.
    Also tracks per-sample GPU peak and average allocated memory."""
    os.makedirs(output_dir, exist_ok=True)
    features_dir = os.path.join(output_dir, "features_individual")
    os.makedirs(features_dir, exist_ok=True)

    tracker = GPUMemoryTracker()
    answers = []
    skipped = 0

    for sample in tqdm(samples, desc="Extracting"):
        sample_id = sample["id"]

        prompt_ids = tokenizer(sample["prompt_text"], return_tensors="pt", add_special_tokens=True).input_ids
        prompt_len = prompt_ids.shape[1]
        full_text = sample["prompt_text"] + "\n" + sample["response_text"]
        full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=True).input_ids
        total_len = full_ids.shape[1]
        response_len = total_len - prompt_len

        if response_len < 2:
            skipped += 1
            continue

        # --- Begin tracking ---
        tracker.begin_sample()

        # Forward pass
        with torch.no_grad():
            output = model(full_ids.to(device), output_hidden_states=True, output_attentions=True)

        tracker.snapshot()  # after forward pass

        # Stack as float32 on GPU
        hs = torch.stack([h.squeeze(0) for h in output.hidden_states]).float()
        attn = torch.stack([a.squeeze(0) for a in output.attentions]).float()

        del output
        gc.collect()
        torch.cuda.empty_cache()

        tracker.snapshot()  # after del output, before ICR

        # Compute ICR scores faithfully on GPU
        try:
            icr_scores = compute_icr_faithful(
                hs, attn, prompt_len, response_len,
                top_k=top_k, top_p=top_p,
            )
        except Exception as e:
            logger.warning(f"ICR failed for {sample_id} (seq={total_len}): {e}")
            del hs, attn
            gc.collect(); torch.cuda.empty_cache()
            tracker.end_sample()
            skipped += 1
            continue

        del hs, attn
        gc.collect()
        torch.cuda.empty_cache()

        tracker.snapshot()  # after cleanup
        tracker.end_sample()

        # Save only ICR scores (~4KB per sample)
        torch.save({"icr_scores": torch.tensor(icr_scores)},
                    os.path.join(features_dir, f"{sample_id}.pt"))

        answers.append({
            "id": sample_id,
            "prompt_len": prompt_len,
            "response_len": response_len,
            "label": sample["label"],
            "split": sample["split"],
            "task_type": sample["task_type"],
        })

    # Save answers.json (merge if sharded)
    answers_path = os.path.join(output_dir, "answers.json")
    if os.path.exists(answers_path):
        with open(answers_path) as f:
            existing = json.load(f)
        existing_ids = {a["id"] for a in existing}
        for a in answers:
            if a["id"] not in existing_ids:
                existing.append(a)
        answers = existing
    with open(answers_path, "w") as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)

    # Save GPU memory stats
    mem_stats = tracker.summary()
    if mem_stats:
        stats_path = os.path.join(output_dir, "gpu_memory_stats.json")
        # Merge if exists (for sharded runs)
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                existing_stats = json.load(f)
            if isinstance(existing_stats, list):
                existing_stats.append(mem_stats)
            else:
                existing_stats = [existing_stats, mem_stats]
            mem_stats_save = existing_stats
        else:
            mem_stats_save = mem_stats
        with open(stats_path, "w") as f:
            json.dump(mem_stats_save, f, indent=2)

        logger.info(f"GPU Memory Stats:")
        logger.info(f"  Peak (max across samples):    {mem_stats['gpu_peak_gb_max']:.2f} GB")
        logger.info(f"  Peak (mean across samples):   {mem_stats['gpu_peak_mb_mean']/1024:.2f} GB")
        logger.info(f"  Avg allocated (mean):         {mem_stats['gpu_avg_allocated_gb_mean']:.2f} GB")
        logger.info(f"  Total time:                   {mem_stats['total_time_seconds']:.1f}s")
        logger.info(f"  Avg time/sample:              {mem_stats['avg_time_per_sample_seconds']:.2f}s")

    logger.info(f"Saved {len(answers)} samples (skipped {skipped})")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--ragtruth_data_dir", type=str, required=True)
    parser.add_argument("--ragtruth_task_types", type=str, nargs="+", default=None)
    parser.add_argument("--ragtruth_model_filter", type=str, nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--torch_dtype", type=str, default="float16")
    parser.add_argument("--top_k", type=int, default=20, help="Paper Appendix B.3: k=20")
    parser.add_argument("--top_p", type=float, default=None, help="If set, overrides top_k")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

    logger.info(f"Loading: {args.model_name} (shard {args.shard_id}/{args.num_shards})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=dtype_map[args.torch_dtype],
        device_map={"": 0}, low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    device = next(model.parameters()).device

    train = load_ragtruth(args.ragtruth_data_dir, args.ragtruth_task_types,
                          args.ragtruth_model_filter, "train")
    test = load_ragtruth(args.ragtruth_data_dir, args.ragtruth_task_types,
                         args.ragtruth_model_filter, "test")
    all_samples = train + test
    logger.info(f"Total: {len(all_samples)} (train={len(train)}, test={len(test)})")

    if args.num_shards > 1:
        all_samples = [s for i, s in enumerate(all_samples) if i % args.num_shards == args.shard_id]
        logger.info(f"Shard {args.shard_id}: {len(all_samples)} samples")

    train_sub = [s for s in all_samples if s["split"] == "train"]
    test_sub = [s for s in all_samples if s["split"] == "test"]

    if train_sub:
        logger.info(f"--- Train split ({len(train_sub)} samples) ---")
        extract_and_save(model, tokenizer, train_sub,
                         os.path.join(args.output_dir, "train"), device,
                         top_k=args.top_k, top_p=args.top_p)
    if test_sub:
        logger.info(f"--- Test split ({len(test_sub)} samples) ---")
        extract_and_save(model, tokenizer, test_sub,
                         os.path.join(args.output_dir, "test"), device,
                         top_k=args.top_k, top_p=args.top_p)
    logger.info("Done!")


if __name__ == "__main__":
    main()