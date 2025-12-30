"""
Evaluation utilities for LLM training.

Provides functions for computing evaluation metrics including
perplexity, loss, and other language modeling metrics.
"""

import logging
import math
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel

logger = logging.getLogger("llm_training")


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.

    Args:
        loss: Cross-entropy loss value

    Returns:
        Perplexity value
    """
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


@torch.no_grad()
def evaluate(
    model: PreTrainedModel,
    eval_dataloader: DataLoader,
    device: torch.device,
    max_eval_batches: int | None = None,
) -> dict[str, float]:
    """
    Evaluate model on evaluation dataset.

    Args:
        model: Model to evaluate
        eval_dataloader: Evaluation data loader
        device: Device to run evaluation on
        max_eval_batches: Maximum number of batches to evaluate (for quick eval)

    Returns:
        Dictionary of evaluation metrics including loss and perplexity
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    num_eval_batches = (
        min(len(eval_dataloader), max_eval_batches)
        if max_eval_batches
        else len(eval_dataloader)
    )

    progress_bar = tqdm(
        eval_dataloader,
        total=num_eval_batches,
        desc="Evaluating",
        disable=not logger.isEnabledFor(logging.INFO),
    )

    for batch_idx, batch in enumerate(progress_bar):
        if max_eval_batches and batch_idx >= max_eval_batches:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        batch_tokens = batch["attention_mask"].sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        num_batches += 1

        progress_bar.set_postfix({"loss": loss.item()})

    model.train()

    if total_tokens == 0:
        logger.warning("No tokens evaluated, returning inf loss")
        return {"eval_loss": float("inf"), "eval_perplexity": float("inf")}

    avg_loss = total_loss / total_tokens
    perplexity = compute_perplexity(avg_loss)

    metrics = {
        "eval_loss": avg_loss,
        "eval_perplexity": perplexity,
        "eval_batches": num_batches,
        "eval_tokens": total_tokens,
    }

    logger.info(f"Evaluation complete: loss={avg_loss:.4f}, perplexity={perplexity:.4f}")

    return metrics


def quick_evaluate(
    model: PreTrainedModel,
    eval_dataloader: DataLoader,
    device: torch.device,
    num_batches: int = 10,
) -> dict[str, float]:
    """
    Quick evaluation on a subset of batches.

    Useful for frequent evaluation during training without
    evaluating on the entire dataset.

    Args:
        model: Model to evaluate
        eval_dataloader: Evaluation data loader
        device: Device to run evaluation on
        num_batches: Number of batches to evaluate

    Returns:
        Dictionary of evaluation metrics
    """
    return evaluate(model, eval_dataloader, device, max_eval_batches=num_batches)


class MetricsTracker:
    """Track and aggregate training metrics over time."""

    def __init__(self) -> None:
        """Initialize metrics tracker."""
        self.metrics_history: list[dict[str, Any]] = []
        self.current_epoch_metrics: dict[str, list[float]] = {}

    def update(self, metrics: dict[str, float], step: int) -> None:
        """
        Update metrics with new values.

        Args:
            metrics: Dictionary of metric names and values
            step: Current training step
        """
        record = {"step": step, **metrics}
        self.metrics_history.append(record)

        for key, value in metrics.items():
            if key not in self.current_epoch_metrics:
                self.current_epoch_metrics[key] = []
            self.current_epoch_metrics[key].append(value)

    def get_average(self, metric_name: str, last_n: int | None = None) -> float:
        """
        Get average value for a metric.

        Args:
            metric_name: Name of the metric
            last_n: Average over last N values (None for all)

        Returns:
            Average metric value
        """
        if metric_name not in self.current_epoch_metrics:
            return 0.0

        values = self.current_epoch_metrics[metric_name]
        if last_n:
            values = values[-last_n:]

        return sum(values) / len(values) if values else 0.0

    def reset_epoch(self) -> dict[str, float]:
        """
        Reset epoch metrics and return averages.

        Returns:
            Dictionary of average metrics for the epoch
        """
        averages = {k: sum(v) / len(v) for k, v in self.current_epoch_metrics.items() if v}
        self.current_epoch_metrics = {}
        return averages

    def get_best(self, metric_name: str, mode: str = "min") -> tuple[float, int]:
        """
        Get best value and step for a metric.

        Args:
            metric_name: Name of the metric
            mode: "min" for lower is better, "max" for higher is better

        Returns:
            Tuple of (best_value, step)
        """
        relevant = [m for m in self.metrics_history if metric_name in m]
        if not relevant:
            return (float("inf") if mode == "min" else float("-inf"), 0)

        if mode == "min":
            best = min(relevant, key=lambda x: x[metric_name])
        else:
            best = max(relevant, key=lambda x: x[metric_name])

        return best[metric_name], best["step"]


def log_metrics(
    metrics: dict[str, float],
    step: int,
    prefix: str = "",
) -> None:
    """
    Log metrics to console and optionally to tracking service.

    Args:
        metrics: Dictionary of metric names and values
        step: Current training step
        prefix: Prefix for metric names
    """
    formatted = " | ".join(f"{prefix}{k}: {v:.4f}" for k, v in metrics.items())
    logger.info(f"Step {step}: {formatted}")
