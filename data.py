"""
Dataset and DataLoader utilities for LLM training.

Provides functions for loading, preprocessing, and batching datasets
for language model training.
"""

import logging
from typing import Any

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

logger = logging.getLogger("llm_training")


def load_training_data(
    dataset_name: str,
    dataset_config: str | None = None,
    split: str = "train",
    streaming: bool = False,
) -> Dataset:
    """
    Load a dataset from HuggingFace datasets.

    Args:
        dataset_name: Name of the dataset on HuggingFace Hub
        dataset_config: Optional dataset configuration name
        split: Dataset split to load
        streaming: Whether to use streaming mode

    Returns:
        Loaded dataset

    Raises:
        ValueError: If dataset cannot be loaded
    """
    logger.info(f"Loading dataset: {dataset_name} (config: {dataset_config})")

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        streaming=streaming,
    )

    logger.info(f"Dataset loaded: {len(dataset) if not streaming else 'streaming'} examples")
    return dataset


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    text_column: str = "text",
    num_workers: int = 4,
) -> Dataset:
    """
    Tokenize and chunk a text dataset for language modeling.

    Args:
        dataset: Input dataset
        tokenizer: Tokenizer to use
        max_seq_length: Maximum sequence length for chunks
        text_column: Name of the text column in dataset
        num_workers: Number of workers for parallel processing

    Returns:
        Tokenized and chunked dataset
    """
    logger.info(f"Tokenizing dataset with max_seq_length={max_seq_length}")

    def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            examples[text_column],
            truncation=False,
            add_special_tokens=True,
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    def group_texts(examples: dict[str, list]) -> dict[str, list]:
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])

        if total_length < max_seq_length:
            return {k: [] for k in examples.keys()}

        total_length = (total_length // max_seq_length) * max_seq_length

        result = {
            k: [
                concatenated[k][i : i + max_seq_length]
                for i in range(0, total_length, max_seq_length)
            ]
            for k in concatenated.keys()
        }

        result["labels"] = result["input_ids"].copy()
        return result

    chunked = tokenized.map(
        group_texts,
        batched=True,
        num_proc=num_workers,
        desc="Grouping texts",
    )

    logger.info(f"Dataset tokenized: {len(chunked)} sequences")
    return chunked


def create_data_collator(tokenizer: PreTrainedTokenizer) -> callable:
    """
    Create a data collator for language modeling.

    Args:
        tokenizer: Tokenizer to use for padding

    Returns:
        Data collator function
    """

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
        attention_mask = torch.stack(
            [torch.tensor(item.get("attention_mask", [1] * len(item["input_ids"]))) for item in batch]
        )
        labels = torch.stack([torch.tensor(item["labels"]) for item in batch])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate_fn


def create_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for training or evaluation.

    Args:
        dataset: Input dataset
        tokenizer: Tokenizer for the collator
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of dataloader workers
        drop_last: Whether to drop the last incomplete batch

    Returns:
        DataLoader instance
    """
    collator = create_data_collator(tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        f"DataLoader created: {len(dataloader)} batches (batch_size={batch_size})"
    )
    return dataloader


def prepare_datasets(
    dataset_name: str,
    dataset_config: str | None,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    train_batch_size: int,
    eval_batch_size: int,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader | None]:
    """
    Prepare train and evaluation dataloaders.

    Args:
        dataset_name: Name of the dataset
        dataset_config: Dataset configuration
        tokenizer: Tokenizer to use
        max_seq_length: Maximum sequence length
        train_batch_size: Training batch size
        eval_batch_size: Evaluation batch size
        num_workers: Number of preprocessing workers

    Returns:
        Tuple of (train_dataloader, eval_dataloader or None)
    """
    train_dataset = load_training_data(dataset_name, dataset_config, split="train")
    train_tokenized = tokenize_dataset(
        train_dataset, tokenizer, max_seq_length, num_workers=num_workers
    )
    train_loader = create_dataloader(
        train_tokenized, tokenizer, train_batch_size, shuffle=True
    )

    eval_loader = None
    try:
        eval_dataset = load_training_data(
            dataset_name, dataset_config, split="validation"
        )
        eval_tokenized = tokenize_dataset(
            eval_dataset, tokenizer, max_seq_length, num_workers=num_workers
        )
        eval_loader = create_dataloader(
            eval_tokenized, tokenizer, eval_batch_size, shuffle=False, drop_last=False
        )
    except ValueError:
        logger.warning("No validation split found, skipping evaluation dataloader")

    return train_loader, eval_loader
