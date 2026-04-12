# SLM Fine-Tuning Project

## Project Overview
This project focuses on fine-tuning Small Language Models (SLMs) and Large Language Models (LLMs). Work includes dataset preparation, training configuration, evaluation, and deployment of fine-tuned models.

## Tech Stack
- **Language**: Python 3.10+
- **ML Frameworks**: PyTorch, Hugging Face Transformers, PEFT (LoRA/QLoRA), TRL
- **Data**: Hugging Face Datasets, pandas
- **Experiment Tracking**: Weights & Biases (wandb) or TensorBoard
- **Compute**: CUDA-enabled GPUs
- **Environment**: Conda env `slm-finetune`, Jupyter notebooks for experimentation, Python scripts for pipelines
- **Activate**: `conda activate slm-finetune`

## Project Structure
```
SLM_Finetuning/
├── CLAUDE.md              # This file - project context for Claude
├── data/                  # Datasets (raw, processed, splits)
│   ├── raw/
│   └── processed/
├── configs/               # Training and model configs (YAML)
├── scripts/               # Runnable training/eval/export scripts
├── notebooks/             # Jupyter notebooks for exploration
├── models/                # Saved model checkpoints (gitignored)
├── results/               # Evaluation results and metrics
├── src/                   # Reusable Python modules
│   ├── data/              # Data loading and preprocessing
│   ├── training/          # Training loops and utilities
│   └── evaluation/        # Eval metrics and benchmarks
└── tests/                 # Unit tests
```

## Conventions
- Use snake_case for all Python files and functions.
- Config files use YAML format in `configs/`.
- Notebooks are for exploration; production code goes in `src/` and `scripts/`.
- All training runs should be reproducible — always set and log random seeds.
- Pin dependency versions in `requirements.txt`.
- Model checkpoints and large files go in `models/` (gitignored).
- Datasets larger than 10MB go in `data/` (gitignored) — document download instructions instead.

## Common Commands
```bash
# Activate environment
conda activate slm-finetune

# Install dependencies (if needed)
pip install -r requirements.txt

# Run a fine-tuning job
python scripts/train.py --config configs/lora_config.yaml

# Evaluate a model
python scripts/evaluate.py --model models/<checkpoint> --dataset data/processed/eval.jsonl

# Export model for inference
python scripts/export.py --model models/<checkpoint> --format gguf
```

## Key Decisions
- Prefer LoRA/QLoRA for parameter-efficient fine-tuning to reduce GPU memory requirements.
- Use 4-bit quantization (bitsandbytes) when working with models >7B parameters.
- Alpaca/ShareGPT format for instruction-tuning datasets.
- Evaluation uses both automated metrics (perplexity, BLEU/ROUGE) and manual inspection.
