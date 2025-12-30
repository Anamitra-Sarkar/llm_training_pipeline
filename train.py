"""
Main training script for LLM pretraining/fine-tuning.

This script provides a production-ready training pipeline with support for:
- Mixed precision training (bf16)
- Gradient accumulation and checkpointing
- Distributed training (DDP-ready)
- Checkpoint resumption
- Evaluation and perplexity tracking
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from transformers import get_scheduler

from data import prepare_datasets
from evaluate import MetricsTracker, evaluate, log_metrics
from model import load_model_and_tokenizer, prepare_model_for_training, wrap_model_for_ddp
from utils import (
    get_default_config,
    get_device,
    get_local_rank,
    is_main_process,
    load_checkpoint,
    load_config,
    merge_configs,
    save_checkpoint,
    set_seed,
    setup_logging,
)

logger = logging.getLogger("llm_training")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML/JSON configuration file",
    )

    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        default=None,
        help="Enable gradient checkpointing",
    )

    # Data arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Dataset configuration name",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="Number of preprocessing workers",
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=None,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=None,
        help="Evaluation batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="Weight decay",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        default=None,
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log every N steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )

    # Mixed precision
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=None,
        help="Use bfloat16 mixed precision",
    )
    parser.add_argument(
        "--no_bf16",
        action="store_true",
        help="Disable bfloat16 mixed precision",
    )

    # Checkpoint
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    return parser.parse_args()


def setup_distributed() -> tuple[bool, int, int]:
    """
    Set up distributed training if available.

    Returns:
        Tuple of (is_distributed, local_rank, world_size)
    """
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        return True, local_rank, world_size

    return False, 0, 1


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> AdamW:
    """
    Create AdamW optimizer with weight decay.

    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay factor

    Returns:
        Configured optimizer
    """
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    return AdamW(optimizer_grouped_parameters, lr=learning_rate)


def create_scheduler(
    optimizer: AdamW,
    scheduler_type: str,
    num_training_steps: int,
    warmup_steps: int,
) -> LRScheduler:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler
        num_training_steps: Total training steps
        warmup_steps: Number of warmup steps

    Returns:
        Configured scheduler
    """
    return get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )


def training_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    use_bf16: bool,
) -> float:
    """
    Perform a single training step.

    Args:
        model: Model to train
        batch: Input batch
        device: Training device
        use_bf16: Whether to use bf16

    Returns:
        Loss value
    """
    batch = {k: v.to(device) for k, v in batch.items()}

    # Handle autocast - only CUDA supports bf16 autocast properly
    if use_bf16 and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss
    else:
        outputs = model(**batch)
        loss = outputs.loss

    loss.backward()

    return loss.item()


def train(config: dict[str, Any]) -> None:
    """
    Main training loop.

    Args:
        config: Training configuration dictionary
    """
    is_distributed, local_rank, world_size = setup_distributed()

    if is_main_process():
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / "training.log"
        setup_logging(log_level="INFO", log_file=str(log_file))
    else:
        setup_logging(log_level="WARNING")

    logger.info("=" * 60)
    logger.info("Starting LLM Training")
    logger.info("=" * 60)

    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)

    set_seed(config["seed"])

    device = get_device()
    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")

    logger.info(f"Using device: {device}")
    logger.info(f"Distributed: {is_distributed} (world_size={world_size})")

    use_bf16 = config["bf16"]
    if use_bf16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        logger.warning("bf16 not supported on this GPU, falling back to fp32")
        use_bf16 = False

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=config["model_name_or_path"],
        use_gradient_checkpointing=config["use_gradient_checkpointing"],
    )

    model = prepare_model_for_training(model, device, bf16=use_bf16)

    if is_distributed:
        model = wrap_model_for_ddp(model, device_ids=[local_rank])

    train_dataloader, eval_dataloader = prepare_datasets(
        dataset_name=config["dataset_name"],
        dataset_config=config["dataset_config"],
        tokenizer=tokenizer,
        max_seq_length=config["max_seq_length"],
        train_batch_size=config["per_device_train_batch_size"],
        eval_batch_size=config["per_device_eval_batch_size"],
        num_workers=config["preprocessing_num_workers"],
    )

    optimizer = create_optimizer(
        model=model,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_type=config["lr_scheduler_type"],
        num_training_steps=config["max_train_steps"],
        warmup_steps=config["warmup_steps"],
    )

    start_step = 0
    if config["resume_from_checkpoint"]:
        logger.info(f"Resuming from checkpoint: {config['resume_from_checkpoint']}")
        start_step = load_checkpoint(
            checkpoint_path=config["resume_from_checkpoint"],
            model=model.module if is_distributed else model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        logger.info(f"Resumed from step {start_step}")

    metrics_tracker = MetricsTracker()
    global_step = start_step
    accumulation_loss = 0.0

    logger.info("Starting training loop...")

    progress_bar = tqdm(
        total=config["max_train_steps"] - start_step,
        desc="Training",
        disable=not is_main_process(),
    )

    model.train()
    data_iter = iter(train_dataloader)

    while global_step < config["max_train_steps"]:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)

        loss = training_step(model, batch, device, use_bf16)
        accumulation_loss += loss

        if (global_step + 1) % config["gradient_accumulation_steps"] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            avg_loss = accumulation_loss / config["gradient_accumulation_steps"]
            metrics_tracker.update({"train_loss": avg_loss, "lr": scheduler.get_last_lr()[0]}, global_step)
            accumulation_loss = 0.0

        global_step += 1
        progress_bar.update(1)

        if global_step % config["logging_steps"] == 0 and is_main_process():
            avg_loss = metrics_tracker.get_average("train_loss", last_n=config["logging_steps"])
            current_lr = scheduler.get_last_lr()[0]
            log_metrics(
                {"train_loss": avg_loss, "lr": current_lr},
                step=global_step,
            )

        if (
            global_step % config["eval_steps"] == 0
            and eval_dataloader is not None
            and is_main_process()
        ):
            eval_metrics = evaluate(model, eval_dataloader, device)
            metrics_tracker.update(eval_metrics, global_step)
            log_metrics(eval_metrics, step=global_step, prefix="eval_")

        if global_step % config["save_steps"] == 0 and is_main_process():
            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=global_step,
                output_dir=config["output_dir"],
                is_distributed=is_distributed,
            )
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    progress_bar.close()

    if is_main_process():
        final_checkpoint = save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=global_step,
            output_dir=config["output_dir"],
            is_distributed=is_distributed,
        )
        logger.info(f"Final checkpoint saved: {final_checkpoint}")

        if eval_dataloader is not None:
            logger.info("Running final evaluation...")
            final_metrics = evaluate(model, eval_dataloader, device)
            log_metrics(final_metrics, step=global_step, prefix="final_")

    if is_distributed:
        dist.destroy_process_group()

    logger.info("Training complete!")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    default_config = get_default_config()

    file_config = None
    if args.config:
        file_config = load_config(args.config)

    if args.no_bf16:
        args.bf16 = False

    config = merge_configs(default_config, file_config, args)

    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)

    try:
        train(config)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
