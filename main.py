#!/usr/bin/env python3
"""ICR Probe 主训练/评估脚本

直接读取 HD 框架提取的特征，计算 ICR Score，训练 Probe。

安装依赖:
    pip install torch numpy scikit-learn tqdm

支持:
- sample-level: 对每个样本，将 [n_layers, n_tokens] 的 ICR scores 聚合为 [n_layers]
- token-level: 直接使用 [n_layers] 维的每个 token 特征

Usage:
    # Sample-level 训练
    python main.py --features_dir /path/to/features --level sample --output_dir ./outputs
    
    # Token-level 训练
    python main.py --features_dir /path/to/features --level token --output_dir ./outputs
    
    # 评估
    python main.py --features_dir /path/to/features --level sample --eval_only --model_path ./outputs/model.pth
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, roc_curve, auc
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent))
from src.utils import ICRProbe
from src.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# GPU Memory & Time Tracker
# =============================================================================

class PipelineTracker:
    """Track GPU peak/avg memory and timing across the full pipeline."""
    
    def __init__(self):
        self.phases = {}
        self._current_phase = None
        self._phase_start = 0
        self._snapshots = []
        self.global_start = time.time()
    
    def begin_phase(self, name):
        self._current_phase = name
        self._phase_start = time.time()
        self._snapshots = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    
    def snapshot(self):
        if torch.cuda.is_available():
            self._snapshots.append(torch.cuda.memory_allocated() / (1024**2))
    
    def end_phase(self):
        elapsed = time.time() - self._phase_start
        phase_info = {"time_seconds": round(elapsed, 2)}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            phase_info["gpu_peak_mb"] = round(peak, 1)
            phase_info["gpu_peak_gb"] = round(peak / 1024, 2)
            if self._snapshots:
                avg = sum(self._snapshots) / len(self._snapshots)
                phase_info["gpu_avg_allocated_mb"] = round(avg, 1)
                phase_info["gpu_avg_allocated_gb"] = round(avg / 1024, 2)
        self.phases[self._current_phase] = phase_info
    
    def summary(self):
        total_time = round(time.time() - self.global_start, 2)
        overall_peak = max((p.get("gpu_peak_mb", 0) for p in self.phases.values()), default=0)
        overall_avg_list = [p["gpu_avg_allocated_mb"] for p in self.phases.values() if "gpu_avg_allocated_mb" in p]
        overall_avg = round(sum(overall_avg_list) / len(overall_avg_list), 1) if overall_avg_list else 0
        return {
            "total_time_seconds": total_time,
            "gpu_peak_mb": round(overall_peak, 1),
            "gpu_peak_gb": round(overall_peak / 1024, 2),
            "gpu_avg_allocated_mb": overall_avg,
            "gpu_avg_allocated_gb": round(overall_avg / 1024, 2),
            "phases": self.phases,
        }
    
    def log_summary(self):
        s = self.summary()
        logger.info("=" * 60)
        logger.info("Pipeline Profiling Summary")
        logger.info("=" * 60)
        logger.info(f"Total time:           {s['total_time_seconds']}s")
        logger.info(f"GPU peak (overall):   {s['gpu_peak_gb']} GB")
        logger.info(f"GPU avg (overall):    {s['gpu_avg_allocated_gb']} GB")
        for name, info in s["phases"].items():
            peak_str = f"{info.get('gpu_peak_gb', 'N/A')} GB" if 'gpu_peak_gb' in info else 'N/A'
            avg_str = f"{info.get('gpu_avg_allocated_gb', 'N/A')} GB" if 'gpu_avg_allocated_gb' in info else 'N/A'
            logger.info(f"  {name:30s} {info['time_seconds']:>8.1f}s  peak={peak_str}  avg={avg_str}")
        logger.info("=" * 60)
        return s




# =============================================================================
# ICR Score 计算 (简化版，适配 HD 框架的 tensor 格式)
# =============================================================================

def js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """计算 JS 散度"""
    p = (p - p.mean()) / max(p.std().item(), 1e-8)
    q = (q - q.mean()) / max(q.std().item(), 1e-8)
    p = F.softmax(p, dim=0)
    q = F.softmax(q, dim=0)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum().item()
    kl_qm = (q * (q / m).log()).sum().item()
    return 0.5 * kl_pm + 0.5 * kl_qm


def compute_icr_scores(
    hidden_states: torch.Tensor,  # [n_layers, seq_len, hidden_dim]
    attentions: torch.Tensor,     # [n_layers, n_heads, seq_len, seq_len]
    prompt_len: int,
    response_len: int,
    top_p: float = 0.1,
) -> np.ndarray:
    """计算 ICR Scores
    
    Args:
        hidden_states: [n_layers, seq_len, hidden_dim]
        attentions: [n_layers, n_heads, seq_len, seq_len]
        prompt_len: prompt 长度
        response_len: response 长度
        top_p: top-p 采样比例
        
    Returns:
        [n_layers, n_response_tokens] 的 ICR 分数矩阵
    """
    n_layers = attentions.shape[0]
    seq_len = hidden_states.shape[1]
    response_start = prompt_len
    
    # 池化注意力头: [n_layers, seq_len, seq_len]
    pooled_attn = attentions.mean(dim=1)
    
    icr_scores = []
    
    for layer_idx in range(n_layers):
        layer_scores = []
        
        for token_idx in range(response_start, min(response_start + response_len, seq_len)):
            # 当前 token 的注意力分布 (因果注意力)
            current_attn = pooled_attn[layer_idx, token_idx, :token_idx + 1]
            
            if len(current_attn) == 0:
                layer_scores.append(0.0)
                continue
            
            # Top-p 选择
            top_k = max(1, int(top_p * len(current_attn)))
            top_k = min(top_k, len(current_attn))
            topk_values, topk_indices = torch.topk(current_attn, k=top_k)
            
            # 隐藏状态变化
            if layer_idx == 0:
                hs_diff = hidden_states[layer_idx, token_idx]
            else:
                hs_diff = hidden_states[layer_idx, token_idx] - hidden_states[layer_idx - 1, token_idx]
            
            # 被注意位置的隐藏状态
            prev_layer_hs = hidden_states[max(0, layer_idx - 1)]
            attended_hs = prev_layer_hs[topk_indices]
            
            # 信息贡献权重
            w_i = torch.sum(hs_diff * attended_hs, dim=1) / (torch.norm(attended_hs, dim=1) + 1e-8)
            
            # JS 散度
            try:
                icr_score = js_divergence(w_i, topk_values)
            except:
                icr_score = 0.0
            
            layer_scores.append(icr_score)
        
        icr_scores.append(layer_scores)
    
    return np.array(icr_scores)


# =============================================================================
# 数据加载
# =============================================================================

def load_hd_features(features_dir: Path) -> Tuple[Dict[str, dict], Dict[str, int]]:
    """加载 HD 框架的特征
    
    Args:
        features_dir: 特征目录，包含 features/, answers.json, metadata.json
        
    Returns:
        (sample_info_dict, labels_dict)
    """
    features_dir = Path(features_dir)
    
    # 加载 answers.json (包含 prompt_len, response_len, label)
    answers_path = features_dir / "answers.json"
    if not answers_path.exists():
        raise FileNotFoundError(f"answers.json not found: {answers_path}")
    
    with open(answers_path) as f:
        answers_data = json.load(f)
    
    # 构建样本信息
    sample_info = {}
    labels = {}
    
    # 支持两种格式：
    # 1. 列表格式: [{"id": "xxx", "prompt_len": ..., "label": ...}, ...]
    # 2. 字典格式: {"sample_id": {"prompt_len": ..., "label": ...}, ...}
    if isinstance(answers_data, list):
        for item in answers_data:
            sample_id = str(item.get("id", item.get("sample_id", "")))
            if not sample_id:
                continue
            sample_info[sample_id] = {
                "prompt_len": item.get("prompt_len", item.get("input_len", 0)),
                "response_len": item.get("response_len", item.get("output_len", 0)),
                "label": item.get("label", 0),
            }
            labels[sample_id] = item.get("label", 0)
    else:
        # 字典格式
        for sample_id, info in answers_data.items():
            sample_info[sample_id] = {
                "prompt_len": info.get("prompt_len", info.get("input_len", 0)),
                "response_len": info.get("response_len", info.get("output_len", 0)),
                "label": info.get("label", 0),
            }
            labels[sample_id] = info.get("label", 0)
    
    logger.info(f"Loaded {len(sample_info)} samples from {answers_path}")
    return sample_info, labels


def load_sample_features(
    features_dir: Path,
    sample_id: str,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """加载单个样本的 hidden_states 和 full_attentions
    
    支持两种格式:
    1. 单文件格式: features_individual/{sample_id}.pt (包含所有特征)
    2. 目录格式: features_individual/{sample_id}/hidden_states.pt + full_attentions.pt
    
    Returns:
        (hidden_states, attentions) 或 (None, None)
    """
    features_dir = Path(features_dir)
    
    hidden_states = None
    attentions = None
    
    # ============ 格式1: 单文件 features_individual/{sample_id}.pt ============
    single_file_paths = [
        features_dir / "features_individual" / f"{sample_id}.pt",
        features_dir / "features" / f"{sample_id}.pt",
        features_dir / f"{sample_id}.pt",
    ]
    
    for pt_path in single_file_paths:
        if pt_path.exists():
            try:
                data = torch.load(pt_path, map_location="cpu", weights_only=False)
                
                # 解析文件内容
                if isinstance(data, dict):
                    # 可能的结构: {"info": {...}, "tensors": {"hidden_states": ..., "full_attentions": ...}}
                    # 或者: {"hidden_states": ..., "full_attentions": ...}
                    # 或者: {"features": {"hidden_states": ..., "full_attentions": ...}}
                    
                    tensors = data.get("tensors", data.get("features", data))
                    
                    if isinstance(tensors, dict):
                        hidden_states = tensors.get("hidden_states")
                        attentions = tensors.get("full_attentions", tensors.get("full_attention"))
                
                if hidden_states is not None and attentions is not None:
                    logger.debug(f"Loaded {sample_id} from single file: hs={hidden_states.shape}, attn={attentions.shape}")
                    return hidden_states, attentions
                    
            except Exception as e:
                logger.debug(f"Failed to load single file {pt_path}: {e}")
    
    # ============ 格式2: 目录 features_individual/{sample_id}/ ============
    dir_paths = [
        features_dir / "features_individual" / sample_id,
        features_dir / "features" / sample_id,
        features_dir / sample_id,
    ]
    
    sample_dir = None
    for p in dir_paths:
        if p.exists() and p.is_dir():
            sample_dir = p
            break
    
    if sample_dir is not None:
        # 加载 hidden_states
        for hs_name in ["hidden_states.pt", "hiddens.pt"]:
            hs_path = sample_dir / hs_name
            if hs_path.exists():
                try:
                    data = torch.load(hs_path, map_location="cpu", weights_only=False)
                    if isinstance(data, dict):
                        hidden_states = data.get("features", {}).get("hidden_states", data.get("hidden_states"))
                    else:
                        hidden_states = data
                    if hidden_states is not None:
                        break
                except Exception as e:
                    logger.debug(f"Failed to load {hs_path}: {e}")
        
        # 加载 full_attentions
        for attn_name in ["full_attentions.pt", "full_attention.pt", "attentions.pt"]:
            attn_path = sample_dir / attn_name
            if attn_path.exists():
                try:
                    data = torch.load(attn_path, map_location="cpu", weights_only=False)
                    if isinstance(data, dict):
                        attentions = data.get("features", {}).get("full_attentions", data.get("full_attentions"))
                    else:
                        attentions = data
                    if attentions is not None:
                        break
                except Exception as e:
                    logger.debug(f"Failed to load {attn_path}: {e}")
    
    if hidden_states is not None and attentions is not None:
        logger.debug(f"Loaded {sample_id} from dir: hs={hidden_states.shape}, attn={attentions.shape}")
    
    return hidden_states, attentions


# =============================================================================
# 特征提取
# =============================================================================

def extract_icr_features(
    features_dir: Path,
    sample_info: Dict[str, dict],
    labels: Dict[str, int],
    level: str = "sample",
    top_p: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """提取 ICR 特征
    
    Args:
        features_dir: 特征目录
        sample_info: 样本信息
        labels: 标签
        level: "sample" 或 "token"
        top_p: ICR 计算的 top-p 参数
        
    Returns:
        (X, y, sample_ids)
        - sample level: X shape [n_samples, n_layers], y shape [n_samples]
        - token level: X shape [n_tokens, n_layers], y shape [n_tokens]
    """
    X_list = []
    y_list = []
    sample_ids = []
    
    # 统计
    stats = {"total": 0, "no_dir": 0, "no_hs": 0, "no_attn": 0, "icr_failed": 0, "success": 0}
    
    for sample_id in tqdm(sample_info.keys(), desc="Extracting ICR features"):
        stats["total"] += 1
        info = sample_info[sample_id]
        label = labels[sample_id]
        
        # Try loading pre-computed ICR scores (from extract_features.py)
        icr_scores = None
        for pt_path in [
            features_dir / "features_individual" / f"{sample_id}.pt",
            features_dir / "features" / f"{sample_id}.pt",
            features_dir / f"{sample_id}.pt",
        ]:
            if pt_path.exists():
                try:
                    data = torch.load(pt_path, map_location="cpu", weights_only=False)
                    if isinstance(data, dict) and "icr_scores" in data:
                        icr_scores = data["icr_scores"].numpy()
                        stats["success"] += 1
                        break
                except:
                    pass
        
        if icr_scores is None:
            # Fall back to raw features + compute ICR on the fly
            hidden_states, attentions = load_sample_features(features_dir, sample_id)
            if hidden_states is None and attentions is None:
                stats["no_dir"] += 1; continue
            if hidden_states is None:
                stats["no_hs"] += 1; continue
            if attentions is None:
                stats["no_attn"] += 1; continue
            try:
                icr_scores = compute_icr_scores(
                    hidden_states=hidden_states, attentions=attentions,
                    prompt_len=info["prompt_len"], response_len=info["response_len"],
                    top_p=top_p,
                )
            except Exception as e:
                logger.warning(f"Failed to compute ICR for {sample_id}: {e}")
                continue
        
        if icr_scores.size == 0:
            continue
        
        if level == "sample":
            # Sample-level: 对 token 维度取平均 -> [n_layers]
            sample_feature = np.mean(icr_scores, axis=1)
            X_list.append(sample_feature)
            y_list.append(label)
            sample_ids.append(sample_id)
        else:
            # Token-level: 每个 token 一个特征向量 [n_layers]
            n_tokens = icr_scores.shape[1]
            for t in range(n_tokens):
                X_list.append(icr_scores[:, t])
                y_list.append(label)  # 使用 sample label（如果没有 token-level 标签）
            sample_ids.append(sample_id)
        
        # 清理内存
        try: del hidden_states, attentions
        except NameError: pass
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    logger.info(f"Extracted features: X shape {X.shape}, y shape {y.shape}")
    logger.info(f"Positive samples: {int(y.sum())}, Negative: {len(y) - int(y.sum())}")
    
    return X, y, sample_ids


# =============================================================================
# 训练
# =============================================================================

def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Config,
    device: str = "cuda",
) -> Tuple[ICRProbe, Dict]:
    """训练 ICR Probe
    
    Returns:
        (model, metrics)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # 转换为 tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    
    # DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # 模型
    input_dim = X_train.shape[1]
    model = ICRProbe(input_dim=input_dim).to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_factor, patience=config.lr_patience)
    
    best_val_auc = 0
    best_state = None
    
    for epoch in range(config.num_epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t.to(device)).squeeze().cpu().numpy()
        
        val_auc = roc_auc_score(y_val, val_outputs)
        
        scheduler.step(train_loss)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{config.num_epochs}: train_loss={train_loss:.4f}, val_auc={val_auc:.4f}")
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Final validation metrics
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t.to(device)).squeeze().cpu().numpy()
    
    val_preds = (val_outputs > 0.5).astype(int)
    
    metrics = {
        "auroc": roc_auc_score(y_val, val_outputs),
        "f1": f1_score(y_val, val_preds),
        "precision": precision_score(y_val, val_preds, zero_division=0),
        "recall": recall_score(y_val, val_preds, zero_division=0),
        "accuracy": accuracy_score(y_val, val_preds),
    }
    
    logger.info(f"Training complete. Best val AUROC: {metrics['auroc']:.4f}")
    
    return model, metrics


# =============================================================================
# 评估
# =============================================================================

def evaluate(
    model: ICRProbe,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = "cuda",
) -> Dict:
    """评估模型"""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        outputs = model(X_test_t).squeeze().cpu().numpy()
    
    preds = (outputs > 0.5).astype(int)
    
    fpr, tpr, thresholds = roc_curve(y_test, outputs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    metrics = {
        "auroc": roc_auc_score(y_test, outputs),
        "aupr": auc(*precision_recall_curve(y_test, outputs)[:2][::-1]),
        "f1": f1_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "accuracy": accuracy_score(y_test, preds),
        "optimal_threshold": optimal_threshold,
        "n_samples": len(y_test),
        "n_positive": int(y_test.sum()),
        "n_negative": len(y_test) - int(y_test.sum()),
    }
    
    return metrics


def precision_recall_curve(y_true, y_scores):
    """简单实现 precision-recall curve"""
    from sklearn.metrics import precision_recall_curve as sk_pr_curve
    return sk_pr_curve(y_true, y_scores)


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ICR Probe Training & Evaluation")
    
    # 数据路径
    parser.add_argument("--features_dir", type=str, required=True,
                        help="HD 框架特征目录路径")
    parser.add_argument("--test_features_dir", type=str, default=None,
                        help="测试集特征目录（可选，默认从 train 中划分）")
    
    # 模式
    parser.add_argument("--level", type=str, default="sample", choices=["sample", "token"],
                        help="检测级别: sample 或 token")
    parser.add_argument("--eval_only", action="store_true",
                        help="仅评估模式")
    parser.add_argument("--model_path", type=str, default=None,
                        help="预训练模型路径（eval_only 模式必需）")
    
    # ICR 参数
    parser.add_argument("--top_p", type=float, default=0.1,
                        help="ICR 计算的 top-p 参数")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--test_size", type=float, default=0.2)
    
    # 输出
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置
    config = Config(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        test_size=args.test_size,
        save_dir=str(output_dir),
    )
    
    tracker = PipelineTracker()
    
    logger.info("=" * 60)
    logger.info("ICR Probe Training & Evaluation")
    logger.info("=" * 60)
    logger.info(f"Features dir: {args.features_dir}")
    logger.info(f"Level: {args.level}")
    logger.info(f"Top-p: {args.top_p}")
    logger.info(f"Output dir: {args.output_dir}")
    
    # 诊断目录结构
    features_path = Path(args.features_dir)
    logger.info(f"\n--- 目录结构诊断 ---")
    logger.info(f"features_dir exists: {features_path.exists()}")
    if features_path.exists():
        subdirs = list(features_path.iterdir())
        logger.info(f"Top-level contents: {[p.name for p in subdirs[:10]]}")
        
        # 检查 features_individual
        fi_dir = features_path / "features_individual"
        if fi_dir.exists():
            samples = list(fi_dir.iterdir())[:3]
            logger.info(f"features_individual/ has {len(list(fi_dir.iterdir()))} items")
            for s in samples:
                if s.is_dir():
                    files = list(s.glob("*.pt"))
                    logger.info(f"  Sample {s.name}: {[f.name for f in files]}")
        
        # 检查 features
        f_dir = features_path / "features"
        if f_dir.exists():
            samples = list(f_dir.iterdir())[:3]
            logger.info(f"features/ has {len(list(f_dir.iterdir()))} items")
            for s in samples:
                if s.is_dir():
                    files = list(s.glob("*.pt"))
                    logger.info(f"  Sample {s.name}: {[f.name for f in files]}")
    logger.info("--- 诊断结束 ---\n")
    
    # 加载数据
    tracker.begin_phase("data_loading")
    sample_info, labels = load_hd_features(Path(args.features_dir))
    tracker.end_phase()
    
    # 提取 ICR 特征
    tracker.begin_phase("icr_feature_extraction")
    tracker.snapshot()
    X, y, sample_ids = extract_icr_features(
        features_dir=Path(args.features_dir),
        sample_info=sample_info,
        labels=labels,
        level=args.level,
        top_p=args.top_p,
    )
    
    if len(X) == 0:
        logger.error("No features extracted!")
        return
    
    # Filter NaN features
    valid = ~np.isnan(X).any(axis=1)
    if not valid.all():
        logger.warning(f"Removing {(~valid).sum()} samples with NaN features")
        X = X[valid]
        y = y[valid]
    tracker.snapshot()
    tracker.end_phase()
    
    if args.eval_only:
        # 仅评估
        if args.model_path is None:
            logger.error("--model_path required for eval_only mode")
            return
        
        model = ICRProbe(input_dim=X.shape[1])
        model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
        
        metrics = evaluate(model, X, y, device=args.device)
        
        logger.info("=" * 60)
        logger.info("Evaluation Results:")
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.4f}")
            else:
                logger.info(f"  {k}: {v}")
    else:
        # 训练
        # 划分训练/验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
        
        # 训练
        tracker.begin_phase("probe_training")
        tracker.snapshot()
        model, train_metrics = train_probe(
            X_train, y_train, X_val, y_val,
            config=config,
            device=args.device,
        )
        
        tracker.snapshot()
        tracker.end_phase()
        
        # 保存模型
        model_path = output_dir / "model.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # 保存配置
        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump({
                "level": args.level,
                "top_p": args.top_p,
                "input_dim": X.shape[1],
                **config.__dict__,
            }, f, indent=2)
        
        # 如果有测试集，评估
        if args.test_features_dir:
            test_sample_info, test_labels = load_hd_features(Path(args.test_features_dir))
            X_test, y_test, _ = extract_icr_features(
                features_dir=Path(args.test_features_dir),
                sample_info=test_sample_info,
                labels=test_labels,
                level=args.level,
                top_p=args.top_p,
            )
            
            if len(X_test) > 0:
                tracker.begin_phase("test_evaluation")
                tracker.snapshot()
                test_metrics = evaluate(model, X_test, y_test, device=args.device)
                tracker.snapshot()
                tracker.end_phase()
                
                logger.info("=" * 60)
                logger.info("Test Results:")
                for k, v in test_metrics.items():
                    if isinstance(v, float):
                        logger.info(f"  {k}: {v:.4f}")
                    else:
                        logger.info(f"  {k}: {v}")
                
                # 保存结果
                results_path = output_dir / "results.json"
                with open(results_path, "w") as f:
                    json.dump({
                        "train_metrics": train_metrics,
                        "test_metrics": test_metrics,
                    }, f, indent=2)
        else:
            # 用验证集作为最终评估
            val_metrics = evaluate(model, X_val, y_val, device=args.device)
            
            logger.info("=" * 60)
            logger.info("Validation Results:")
            for k, v in val_metrics.items():
                if isinstance(v, float):
                    logger.info(f"  {k}: {v:.4f}")
                else:
                    logger.info(f"  {k}: {v}")
            
            results_path = output_dir / "results.json"
            with open(results_path, "w") as f:
                json.dump({
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                }, f, indent=2)
    
    # Save pipeline profiling
    profiling = tracker.log_summary()
    profiling_path = output_dir / "pipeline_profiling.json"
    with open(profiling_path, "w") as f:
        json.dump(profiling, f, indent=2)
    logger.info(f"Pipeline profiling saved to {profiling_path}")
    
    logger.info("=" * 60)
    logger.info("Done!")


if __name__ == "__main__":
    main()