"""
Utility functions for LLM training pipeline.

Provides helper functions for logging, checkpoint management, 
configuration loading, and other common operations.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("llm_training")
    logger.setLevel(getattr(logging, log_level.upper()))

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is not supported
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r") as f:
        if path.suffix in (".yaml", ".yml"):
            config = yaml.safe_load(f)
        elif path.suffix == ".json":
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    return config


def get_default_config() -> dict[str, Any]:
    """
    Get default training configuration.

    Returns:
        Default configuration dictionary
    """
    return {
        # Model configuration
        "model_name_or_path": "gpt2",
        "use_gradient_checkpointing": True,
        # Data configuration
        "dataset_name": "wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "max_seq_length": 512,
        "preprocessing_num_workers": 4,
        # Training configuration
        "output_dir": "./output",
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "max_train_steps": 1000,
        "warmup_steps": 100,
        "lr_scheduler_type": "cosine",
        "save_steps": 500,
        "eval_steps": 100,
        "logging_steps": 10,
        "seed": 42,
        # Mixed precision
        "bf16": True,
        # Checkpoint
        "resume_from_checkpoint": None,
    }


def merge_configs(
    default_config: dict[str, Any],
    file_config: dict[str, Any] | None,
    cli_args: argparse.Namespace | None,
) -> dict[str, Any]:
    """
    Merge configurations with priority: CLI args > file config > defaults.

    Args:
        default_config: Default configuration
        file_config: Configuration loaded from file
        cli_args: Command line arguments

    Returns:
        Merged configuration dictionary
    """
    config = default_config.copy()

    if file_config:
        for key, value in file_config.items():
            if value is not None:
                config[key] = value

    if cli_args:
        for key, value in vars(cli_args).items():
            if value is not None and key != "config":
                config[key] = value

    return config


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    output_dir: str,
    is_distributed: bool = False,
) -> str:
    """
    Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        scheduler: Learning rate scheduler state to save
        step: Current training step
        output_dir: Directory to save checkpoint
        is_distributed: Whether training is distributed

    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = Path(output_dir) / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if is_distributed:
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    torch.save(
        {
            "step": step,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        checkpoint_dir / "checkpoint.pt",
    )

    return str(checkpoint_dir)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: torch.device | None = None,
) -> int:
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory or file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load checkpoint to

    Returns:
        Training step from checkpoint

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
    """
    path = Path(checkpoint_path)
    if path.is_dir():
        checkpoint_file = path / "checkpoint.pt"
    else:
        checkpoint_file = path

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("step", 0)


def get_device() -> torch.device:
    """
    Get the appropriate device for training.

    Returns:
        torch.device for training (CUDA, MPS, or CPU)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """
    Count total and trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_metrics(metrics: dict[str, float]) -> str:
    """
    Format metrics dictionary for logging.

    Args:
        metrics: Dictionary of metric names and values

    Returns:
        Formatted string representation
    """
    return " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())


def get_local_rank() -> int:
    """
    Get local rank for distributed training.

    Returns:
        Local rank (0 if not distributed)
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """
    Check if this is the main process (rank 0).

    Returns:
        True if main process
    """
    return get_local_rank() == 0
