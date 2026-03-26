"""Phase-aware profiling engine for ICR Probe.

Tracks time, GPU memory, CPU memory, GPU utilization, and training metrics
at three granularities:
- Phase level (model_loading, data_loading, icr_extraction, probe_training, evaluation)
- Epoch level (per-epoch timing + metrics within a phase)
- Operation level (tokenization, forward_pass, icr_computation, etc.)

Usage:
    profiler = ICRProfiler(output_dir="./profiling_results/")

    with profiler.phase("icr_extraction") as p:
        for i, sample in enumerate(samples):
            with p.operation("forward_pass"):
                output = model(...)
            with p.operation("icr_computation"):
                scores = compute_icr(...)

    with profiler.phase("probe_training") as p:
        for epoch in range(num_epochs):
            with p.epoch_scope():
                ...
                p.log_epoch_metric("auroc", auroc)

    profiler.save()
"""

import os
import gc
import time
import json
import threading
import tracemalloc
import logging
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OperationStats:
    """Aggregated stats for a single operation type."""
    name: str
    total_seconds: float = 0.0
    count: int = 0
    avg_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_seconds": round(self.total_seconds, 4),
            "count": self.count,
            "avg_ms": round(self.avg_ms, 3),
            "min_ms": round(self.min_ms, 3) if self.min_ms != float("inf") else 0.0,
            "max_ms": round(self.max_ms, 3),
        }


@dataclass
class PhaseMetrics:
    """Metrics collected for a single phase."""
    phase_name: str
    total_time_seconds: float = 0.0
    epoch_times: List[float] = field(default_factory=list)
    epoch_metrics: List[Dict[str, float]] = field(default_factory=list)
    operations: Dict[str, OperationStats] = field(default_factory=dict)
    gpu_peak_allocated_mb: float = 0.0
    gpu_peak_reserved_mb: float = 0.0
    cpu_peak_mb: float = 0.0
    gpu_utilization_avg: float = 0.0
    gpu_memory_timeline: List[Tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_name": self.phase_name,
            "total_time_seconds": round(self.total_time_seconds, 4),
            "epoch_times": [round(t, 4) for t in self.epoch_times],
            "epoch_metrics": self.epoch_metrics,
            "operations": {k: v.to_dict() for k, v in self.operations.items()},
            "gpu_peak_allocated_mb": round(self.gpu_peak_allocated_mb, 1),
            "gpu_peak_reserved_mb": round(self.gpu_peak_reserved_mb, 1),
            "cpu_peak_mb": round(self.cpu_peak_mb, 1),
            "gpu_utilization_avg": round(self.gpu_utilization_avg, 1),
            "gpu_memory_timeline_samples": len(self.gpu_memory_timeline),
        }


@dataclass
class ProfilingResults:
    """Aggregate results across all phases."""
    phases: Dict[str, PhaseMetrics] = field(default_factory=dict)
    total_time_seconds: float = 0.0
    model_name: str = ""
    total_params: int = 0
    trainable_params: int = 0
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "model": self.model_name,
                "timestamp": self.timestamp,
                "total_params": self.total_params,
                "trainable_params": self.trainable_params,
                "hardware": self.hardware_info,
            },
            "total_time_seconds": round(self.total_time_seconds, 4),
            "phases": {k: v.to_dict() for k, v in self.phases.items()},
            "dataset_info": self.dataset_info,
        }

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "ICR Probe Profiling Results Summary",
            "=" * 70,
            f"Model: {self.model_name}",
            f"Total params: {self.total_params:,} | Trainable: {self.trainable_params:,}",
            f"Total time: {self.total_time_seconds:.2f}s",
            "",
        ]
        for name, pm in self.phases.items():
            pct = (pm.total_time_seconds / self.total_time_seconds * 100) if self.total_time_seconds > 0 else 0
            lines.append(f"  {name:<25} {pm.total_time_seconds:>8.2f}s ({pct:>5.1f}%)  "
                         f"GPU peak: {pm.gpu_peak_allocated_mb:>8.1f}MB  "
                         f"CPU peak: {pm.cpu_peak_mb:>8.1f}MB")
        lines.append("=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# GPU utilization sampler (background thread)
# ---------------------------------------------------------------------------

class GPUUtilSampler:
    """Background thread that periodically samples GPU utilization and memory."""

    def __init__(self, device_id: int = 0, interval: float = 2.0):
        self.device_id = device_id
        self.interval = interval
        self._utilization_samples: List[float] = []
        self._memory_timeline: List[Tuple[float, float]] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time = 0.0

    def start(self) -> None:
        if not torch.cuda.is_available():
            return
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                util = torch.cuda.utilization(self.device_id)
                self._utilization_samples.append(float(util))
            except Exception:
                pass
            try:
                allocated = torch.cuda.memory_allocated(self.device_id) / (1024 ** 2)
                elapsed = time.time() - self._start_time
                self._memory_timeline.append((round(elapsed, 2), round(allocated, 1)))
            except Exception:
                pass
            self._stop_event.wait(timeout=self.interval)

    def get_avg_utilization(self) -> float:
        if not self._utilization_samples:
            return 0.0
        return sum(self._utilization_samples) / len(self._utilization_samples)

    def get_memory_timeline(self) -> List[Tuple[float, float]]:
        return list(self._memory_timeline)


# ---------------------------------------------------------------------------
# Phase profiler (context manager for a single phase)
# ---------------------------------------------------------------------------

class PhaseProfiler:
    """Profiles a single phase (time, memory, operations, metrics)."""

    def __init__(self, name: str, sample_interval: float = 2.0):
        self.name = name
        self._sample_interval = sample_interval
        self._start_time = 0.0
        self._epoch_times: List[float] = []
        self._epoch_metrics: List[Dict[str, float]] = []
        self._current_epoch_metrics: Dict[str, float] = {}
        self._op_records: Dict[str, List[float]] = {}
        self._sampler = GPUUtilSampler(interval=sample_interval)
        self._metrics: Optional[PhaseMetrics] = None
        self._tracemalloc_active = False

    def __enter__(self) -> "PhaseProfiler":
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._tracemalloc_active = True
        self._sampler.start()
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = time.time() - self._start_time

        cpu_peak_mb = 0.0
        if self._tracemalloc_active and tracemalloc.is_tracing():
            _, cpu_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self._tracemalloc_active = False
            cpu_peak_mb = cpu_peak / (1024 ** 2)

        gpu_peak_allocated = 0.0
        gpu_peak_reserved = 0.0
        if torch.cuda.is_available():
            gpu_peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
            gpu_peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)

        self._sampler.stop()

        operations = {}
        for op_name, durations in self._op_records.items():
            durations_ms = [d * 1000 for d in durations]
            total_s = sum(durations)
            count = len(durations)
            operations[op_name] = OperationStats(
                name=op_name,
                total_seconds=total_s,
                count=count,
                avg_ms=sum(durations_ms) / count if count > 0 else 0.0,
                min_ms=min(durations_ms) if durations_ms else 0.0,
                max_ms=max(durations_ms) if durations_ms else 0.0,
            )

        self._metrics = PhaseMetrics(
            phase_name=self.name,
            total_time_seconds=total_time,
            epoch_times=list(self._epoch_times),
            epoch_metrics=list(self._epoch_metrics),
            operations=operations,
            gpu_peak_allocated_mb=gpu_peak_allocated,
            gpu_peak_reserved_mb=gpu_peak_reserved,
            cpu_peak_mb=cpu_peak_mb,
            gpu_utilization_avg=self._sampler.get_avg_utilization(),
            gpu_memory_timeline=self._sampler.get_memory_timeline(),
        )

    @contextmanager
    def epoch_scope(self):
        """Track a single epoch's duration."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._current_epoch_metrics = {}
        start = time.time()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._epoch_times.append(time.time() - start)
            self._epoch_metrics.append(dict(self._current_epoch_metrics))

    def log_epoch_metric(self, name: str, value: float) -> None:
        """Log a metric for the current epoch (e.g., AUROC, loss)."""
        self._current_epoch_metrics[name] = round(value, 6)

    @contextmanager
    def operation(self, name: str):
        """Track a single operation (forward_pass, icr_computation, etc.)."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            duration = time.time() - start
            self._op_records.setdefault(name, []).append(duration)

    def get_metrics(self) -> Optional[PhaseMetrics]:
        return self._metrics


# ---------------------------------------------------------------------------
# Top-level profiler
# ---------------------------------------------------------------------------

class ICRProfiler:
    """Orchestrates profiling across all ICR pipeline phases."""

    def __init__(self, output_dir: str = "./profiling_results/"):
        self.output_dir = output_dir
        self._phases: Dict[str, PhaseProfiler] = {}
        self._phase_order: List[str] = []
        self._model_name = ""
        self._total_params = 0
        self._trainable_params = 0
        self._dataset_info: Dict[str, Any] = {}
        self._global_start = time.time()

    def phase(self, name: str, sample_interval: float = 2.0) -> PhaseProfiler:
        """Create a PhaseProfiler context manager for a named phase."""
        profiler = PhaseProfiler(name, sample_interval)
        self._phases[name] = profiler
        self._phase_order.append(name)
        return profiler

    def set_model_info(self, model: torch.nn.Module, model_name: str = "") -> None:
        self._model_name = model_name
        self._total_params = sum(p.numel() for p in model.parameters())
        self._trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def set_dataset_info(self, info: Dict[str, Any]) -> None:
        self._dataset_info = info

    def _get_hardware_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"cuda_available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            info["gpu_memory_total_gb"] = round(total_mem, 2)
        return info

    def finalize(self) -> ProfilingResults:
        total_time = time.time() - self._global_start
        phases: Dict[str, PhaseMetrics] = {}
        for name in self._phase_order:
            profiler = self._phases[name]
            metrics = profiler.get_metrics()
            if metrics is not None:
                phases[name] = metrics

        return ProfilingResults(
            phases=phases,
            total_time_seconds=total_time,
            model_name=self._model_name,
            total_params=self._total_params,
            trainable_params=self._trainable_params,
            dataset_info=self._dataset_info,
            hardware_info=self._get_hardware_info(),
            timestamp=datetime.now().isoformat(),
        )

    def save(self) -> str:
        results = self.finalize()
        os.makedirs(self.output_dir, exist_ok=True)
        from profile_report import save_results
        save_results(results, self.output_dir)
        logger.info(f"Profiling results saved to {self.output_dir}")
        return self.output_dir


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def clear_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
