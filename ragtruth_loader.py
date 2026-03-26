"""RAGTruth dataset loader for ICR Probe hallucination detection.

Loads RAGTruth data (response.jsonl + source_info.jsonl) and returns
prompt/response text pairs for ICR extraction, plus binary labels.

Unlike the TSV loader which concatenates prompt+response for teacher-forcing
tokenization, this loader keeps them separate because ICR needs token boundaries
(core_positions) to distinguish prompt vs response regions.

Label convention (aligned with ICR Probe):
  - 1 = truthful  (RAGTruth: labels == [])
  - 0 = hallucinated  (RAGTruth: labels contains span annotations)
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EXCLUDE_QUALITY = {"incorrect_refusal", "truncated"}


class RAGTruthICRLoader:
    """Load RAGTruth dataset into ICR-compatible format.

    Returns sample dicts with prompt_text and response_text kept separate,
    so the caller can tokenize and compute core_positions for ICRScore.

    Args:
        data_dir: Path to directory containing response.jsonl and source_info.jsonl
        task_types: List of task types to include (None = all). Options: "QA", "Summary", "Data2txt"
        exclude_quality: Set of quality tags to exclude
        model_filter: Only include samples from these source models (partial match). None = all.
        exclude_implicit_true: If True, treat samples where ALL hallucination spans
            are implicit_true as truthful.
    """

    def __init__(
        self,
        data_dir: str,
        task_types: Optional[List[str]] = None,
        exclude_quality: Optional[set] = None,
        model_filter: Optional[List[str]] = None,
        exclude_implicit_true: bool = False,
    ):
        self.data_dir = data_dir
        self.task_types = set(task_types) if task_types else None
        self.exclude_quality = exclude_quality if exclude_quality is not None else DEFAULT_EXCLUDE_QUALITY
        self.model_filter = model_filter
        self.exclude_implicit_true = exclude_implicit_true

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_source_info(self) -> Dict[str, Dict[str, Any]]:
        """Load source_info.jsonl into a dict keyed by source_id."""
        source_map = {}
        source_file = os.path.join(self.data_dir, "source_info.jsonl")

        with open(source_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                source_id = item.get("source_id")
                if source_id is not None:
                    source_map[str(source_id)] = item

        logger.info(f"Loaded {len(source_map)} source entries from {source_file}")
        return source_map

    def _build_prompt_text(self, source_info: Dict[str, Any]) -> str:
        """Extract prompt text from source_info.

        Uses the `prompt` field which contains the exact instruction sent to the LLM,
        including context passages, questions, data tables, etc.
        """
        prompt = source_info.get("prompt", "")

        if not prompt:
            task_type = source_info.get("task_type", "")
            source_data = source_info.get("source_info", "")

            if task_type == "QA" and isinstance(source_data, dict):
                question = source_data.get("question", "")
                passages = source_data.get("passages", "")
                if isinstance(passages, list):
                    passages = "\n\n".join(str(p) for p in passages)
                prompt = f"Q: {question}\nContext: {passages}"
            elif task_type == "Summary":
                text = source_data if isinstance(source_data, str) else str(source_data)
                prompt = f"Summarize: {text}"
            elif task_type == "Data2txt" and isinstance(source_data, dict):
                prompt = f"Describe: {json.dumps(source_data, ensure_ascii=False)}"
            else:
                prompt = str(source_data) if source_data else ""

        return prompt.strip()

    def _get_label(self, item: Dict[str, Any]) -> int:
        """Determine binary label. Returns 1=truthful, 0=hallucinated."""
        labels = item.get("labels", [])

        if not labels:
            return 1

        if self.exclude_implicit_true:
            real_hallucinations = [
                span for span in labels
                if not span.get("implicit_true", False)
            ]
            if not real_hallucinations:
                return 1

        return 0

    def _should_include(self, item: Dict[str, Any], task_type: str) -> bool:
        """Check if a response item passes all filters."""
        if item.get("quality", "good") in self.exclude_quality:
            return False
        if self.task_types and task_type not in self.task_types:
            return False
        if self.model_filter:
            item_model = item.get("model", "")
            if not any(mf.lower() in item_model.lower() for mf in self.model_filter):
                return False
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, split_filter: Optional[str] = None) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, Any]]:
        """Load RAGTruth data as sample dicts.

        Args:
            split_filter: Only include samples with this split value ("train" or "test").
                          None = load all splits.

        Returns:
            samples: List of dicts, each with keys:
                - prompt_text: str (the source prompt)
                - response_text: str (the model response)
                - task_type: str
                - source_id: str
            labels: numpy int32 array, 1=truthful 0=hallucinated
            stats: Dataset statistics dict
        """
        source_map = self._load_source_info()
        response_file = os.path.join(self.data_dir, "response.jsonl")

        samples: List[Dict[str, Any]] = []
        labels: List[int] = []
        task_counts: Dict[str, int] = {}
        label_counts = {0: 0, 1: 0}

        with open(response_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                item = json.loads(line)

                if split_filter and item.get("split", "") != split_filter:
                    continue

                source_id = str(item.get("source_id", ""))
                source_info = source_map.get(source_id)
                if source_info is None:
                    continue

                task_type = source_info.get("task_type", "QA")

                if not self._should_include(item, task_type):
                    continue

                prompt_text = self._build_prompt_text(source_info)
                response_text = item.get("response", "")

                samples.append({
                    "prompt_text": prompt_text,
                    "response_text": response_text,
                    "task_type": task_type,
                    "source_id": source_id,
                })

                label = self._get_label(item)
                labels.append(label)
                label_counts[label] += 1
                task_counts[task_type] = task_counts.get(task_type, 0) + 1

        labels_arr = np.array(labels, dtype=np.int32)

        stats = {
            "dataset": "RAGTruth",
            "total_samples": len(samples),
            "task_type_distribution": task_counts,
            "label_distribution": {
                "truthful": label_counts[1],
                "hallucinated": label_counts[0],
            },
            "split_filter": split_filter,
        }

        logger.info(
            f"Loaded {len(samples)} samples from RAGTruth "
            f"(truthful={label_counts[1]}, hallucinated={label_counts[0]})"
        )
        return samples, labels_arr, stats

    def load_splits(
        self,
        seed: int = 42,
    ) -> Tuple[
        Tuple[List[Dict[str, Any]], np.ndarray],  # train
        Tuple[List[Dict[str, Any]], np.ndarray],  # test
        Dict[str, Any],  # stats
    ]:
        """Load RAGTruth using its built-in train/test splits.

        Returns:
            (train_samples, train_labels),
            (test_samples, test_labels),
            stats
        """
        train_samples, train_labels, train_stats = self.load(split_filter="train")
        test_samples, test_labels, test_stats = self.load(split_filter="test")

        stats = {
            "dataset": "RAGTruth",
            "total_samples": len(train_samples) + len(test_samples),
            "train_samples": len(train_samples),
            "test_samples": len(test_samples),
            "task_type_distribution": {
                k: train_stats["task_type_distribution"].get(k, 0)
                + test_stats["task_type_distribution"].get(k, 0)
                for k in set(
                    list(train_stats["task_type_distribution"].keys())
                    + list(test_stats["task_type_distribution"].keys())
                )
            },
            "label_distribution": {
                "truthful": train_stats["label_distribution"]["truthful"]
                + test_stats["label_distribution"]["truthful"],
                "hallucinated": train_stats["label_distribution"]["hallucinated"]
                + test_stats["label_distribution"]["hallucinated"],
            },
            "split_mode": "builtin_train_test",
        }

        logger.info(
            f"RAGTruth splits: train={len(train_samples)}, test={len(test_samples)}"
        )

        return (train_samples, train_labels), (test_samples, test_labels), stats
