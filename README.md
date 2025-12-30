# LLM Training Pipeline

A production-ready training script for LLM pretraining and fine-tuning using PyTorch and Hugging Face Transformers.

## Features

- **PyTorch + Hugging Face Transformers** integration
- **bf16 mixed precision** (enabled by default for modern GPUs)
- **AdamW optimizer** with weight decay
- **Learning rate schedulers** (cosine by default, linear, constant, etc.)
- **Gradient accumulation** for effective larger batch sizes
- **Gradient checkpointing** (enabled by default for memory efficiency)
- **Distributed training** ready (DDP-safe)
- **Checkpoint resumption** support
- **Periodic checkpoint saving** (enabled by default)
- **Evaluation with perplexity** (enabled by default)
- **Flexible configuration** via CLI arguments or YAML/JSON files
- **Dry run mode** for verification without training

## Production Defaults

This script is designed for modern GPU environments (A100/H100/TPUv5e) with sensible production defaults:

| Feature | Default | Opt-out Flag |
|---------|---------|--------------|
| bf16 mixed precision | **ENABLED** | `--no_bf16` |
| Gradient checkpointing | **ENABLED** | `--no_gradient_checkpointing` |
| Cosine LR scheduler | **ACTIVE** | Change with `--lr_scheduler_type` |
| Evaluation loop | **ENABLED** | - |
| Checkpoint saving | **ENABLED** | - |

## File Structure

```
.
├── train.py              # Main entry point
├── model.py              # Model loading and wrapping
├── data.py               # Dataset and dataloader logic
├── evaluate.py           # Evaluation and perplexity computation
├── utils.py              # Helper utilities
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Anamitra-Sarkar/llm_training_pipeline.git
cd llm_training_pipeline

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Verify Installation (Dry Run)

```bash
# Verify all components initialize correctly without training
python train.py --dry_run
```

This will:
- Load the model and tokenizer
- Prepare datasets
- Create optimizer and scheduler
- Run a forward pass
- Report status of all production features

### Basic Training

```bash
# Train with default configuration (GPT-2 on WikiText-2)
python train.py

# Train with custom parameters
python train.py \
    --model_name_or_path gpt2-medium \
    --dataset_name wikitext \
    --dataset_config wikitext-103-raw-v1 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --max_train_steps 5000 \
    --output_dir ./output
```

### Using Configuration File

Create a `config.yaml` file:

```yaml
# Model
model_name_or_path: gpt2-medium
use_gradient_checkpointing: true

# Data
dataset_name: wikitext
dataset_config: wikitext-103-raw-v1
max_seq_length: 1024
preprocessing_num_workers: 4

# Training
output_dir: ./output
per_device_train_batch_size: 4
per_device_eval_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 5e-5
weight_decay: 0.01
max_train_steps: 10000
warmup_steps: 500
lr_scheduler_type: cosine
save_steps: 1000
eval_steps: 500
logging_steps: 50
seed: 42

# Mixed precision
bf16: true
```

Then run:

```bash
python train.py --config config.yaml
```

### Multi-GPU Training (Distributed)

```bash
# Using torchrun
torchrun --nproc_per_node=4 train.py --config config.yaml

# Using distributed launch
python -m torch.distributed.launch --nproc_per_node=4 train.py --config config.yaml
```

### Resume from Checkpoint

```bash
python train.py --resume_from_checkpoint ./output/checkpoint-1000
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name_or_path` | str | gpt2 | HuggingFace model name or local path |
| `use_gradient_checkpointing` | bool | true | Enable gradient checkpointing |
| `dataset_name` | str | wikitext | HuggingFace dataset name |
| `dataset_config` | str | wikitext-2-raw-v1 | Dataset configuration |
| `max_seq_length` | int | 512 | Maximum sequence length |
| `preprocessing_num_workers` | int | 4 | Number of data preprocessing workers |
| `output_dir` | str | ./output | Output directory |
| `per_device_train_batch_size` | int | 4 | Training batch size per device |
| `per_device_eval_batch_size` | int | 4 | Evaluation batch size per device |
| `gradient_accumulation_steps` | int | 4 | Gradient accumulation steps |
| `learning_rate` | float | 5e-5 | Learning rate |
| `weight_decay` | float | 0.01 | Weight decay |
| `max_train_steps` | int | 1000 | Maximum training steps |
| `warmup_steps` | int | 100 | Warmup steps |
| `lr_scheduler_type` | str | cosine | LR scheduler type |
| `save_steps` | int | 500 | Save checkpoint every N steps |
| `eval_steps` | int | 100 | Evaluate every N steps |
| `logging_steps` | int | 10 | Log every N steps |
| `seed` | int | 42 | Random seed |
| `bf16` | bool | true | Use bfloat16 mixed precision |
| `resume_from_checkpoint` | str | None | Path to checkpoint to resume from |

## Hardware Support

The training script automatically detects and uses the best available hardware:

- **NVIDIA GPU** with CUDA (recommended)
- **Apple Silicon** with MPS
- **CPU** (for testing and debugging)

## Metrics and Logging

The training script logs the following metrics:

- **Training loss** (per step and averaged)
- **Learning rate** (per step)
- **Evaluation loss** (at eval_steps intervals)
- **Perplexity** (at eval_steps intervals)

Logs are written to both console and `{output_dir}/training.log`.

## Checkpoints

Checkpoints are saved to `{output_dir}/checkpoint-{step}/` and include:

- Model weights
- Optimizer state
- Learning rate scheduler state
- Training step

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.35+
- datasets 2.14+

## License

This project is licensed under the GNU AGPL v3 License - see the [LICENSE](LICENSE) file for details.