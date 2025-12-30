"""
Model loading and wrapping utilities for LLM training.

Provides functions for loading pre-trained models, applying gradient
checkpointing, and preparing models for distributed training.
"""

import logging
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("llm_training")


def load_model_and_tokenizer(
    model_name_or_path: str,
    use_gradient_checkpointing: bool = True,
    torch_dtype: torch.dtype | None = None,
    device_map: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a pre-trained language model and tokenizer.

    Args:
        model_name_or_path: HuggingFace model identifier or path to local model
        use_gradient_checkpointing: Whether to enable gradient checkpointing
        torch_dtype: Data type for model weights (e.g., torch.bfloat16)
        device_map: Device placement strategy (e.g., "auto", "cuda:0")

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ValueError: If model cannot be loaded
    """
    logger.info(f"Loading model: {model_name_or_path}")

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    model_kwargs: dict[str, Any] = {
        "config": config,
        "trust_remote_code": True,
    }

    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    if device_map is not None:
        model_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    if use_gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        else:
            logger.warning(
                "Model does not support gradient checkpointing, skipping..."
            )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    logger.info(f"Model loaded with {count_model_params(model)} parameters")

    return model, tokenizer


def count_model_params(model: torch.nn.Module) -> str:
    """
    Count and format model parameters.

    Args:
        model: PyTorch model

    Returns:
        Formatted string with parameter count (e.g., "124.44M")
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if total_params >= 1e9:
        total_str = f"{total_params / 1e9:.2f}B"
    elif total_params >= 1e6:
        total_str = f"{total_params / 1e6:.2f}M"
    else:
        total_str = f"{total_params / 1e3:.2f}K"

    return f"{total_str} total ({trainable_params:,} trainable)"


def prepare_model_for_training(
    model: AutoModelForCausalLM,
    device: torch.device,
    bf16: bool = True,
) -> AutoModelForCausalLM:
    """
    Prepare model for training by moving to device and setting up precision.

    Args:
        model: Model to prepare
        device: Target device
        bf16: Whether to use bfloat16 precision

    Returns:
        Prepared model
    """
    if bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported():
        model = model.to(device=device, dtype=torch.bfloat16)
        logger.info("Model converted to bfloat16")
    elif bf16 and device.type == "cpu":
        model = model.to(device=device, dtype=torch.bfloat16)
        logger.info("Model converted to bfloat16 on CPU")
    else:
        model = model.to(device=device)
        logger.info(f"Model moved to {device}")

    model.train()
    return model


def wrap_model_for_ddp(
    model: AutoModelForCausalLM,
    device_ids: list[int] | None = None,
    find_unused_parameters: bool = False,
) -> torch.nn.parallel.DistributedDataParallel:
    """
    Wrap model for Distributed Data Parallel training.

    Args:
        model: Model to wrap
        device_ids: List of GPU device IDs
        find_unused_parameters: Whether to find unused parameters

    Returns:
        DDP-wrapped model
    """
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters,
    )
    logger.info("Model wrapped with DistributedDataParallel")
    return ddp_model


def get_model_input_names(model: AutoModelForCausalLM) -> list[str]:
    """
    Get the expected input names for a model.

    Args:
        model: HuggingFace model

    Returns:
        List of input tensor names
    """
    if hasattr(model, "forward"):
        import inspect

        sig = inspect.signature(model.forward)
        return [p.name for p in sig.parameters.values() if p.name != "self"]
    return ["input_ids", "attention_mask", "labels"]
