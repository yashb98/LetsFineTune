# LetsFineTune

A collection of fine-tuned language models built with LoRA/QLoRA, Unsloth, and Hugging Face Transformers.

Each model gets its own training pipeline — from dataset preparation to evaluation — with reproducible configs and notebooks.

## Current Models

| Model | Base | Dataset | Method | Status |
|-------|------|---------|--------|--------|
| Qwen3.5-9B Reasoning | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) | LoRA (bf16, r=32, alpha=64) | Training |

## Project Structure

```
LetsFineTune/
├── configs/           # Training configs (YAML)
├── scripts/           # Training, evaluation, and data prep scripts
├── notebooks/         # Jupyter notebooks for experimentation
├── src/               # Reusable Python modules
│   ├── data/          # Data loading and preprocessing
│   ├── training/      # Training loops and utilities
│   └── evaluation/    # Eval metrics and benchmarks
├── models/            # Saved checkpoints (gitignored)
├── results/           # Evaluation results and metrics
└── data/              # Datasets (gitignored, see download instructions)
```

## Quick Start

```bash
# Set up environment
conda activate slm-finetune
pip install -r requirements.txt

# Run a fine-tuning job
python scripts/train.py --config configs/lora_8bit_config.yaml

# Evaluate a checkpoint
python scripts/evaluate.py --model models/<checkpoint> --dataset data/processed/eval.jsonl
```

## Tech Stack

- **Frameworks:** PyTorch, Hugging Face Transformers, PEFT, TRL, Unsloth
- **Training:** LoRA/QLoRA with bitsandbytes quantization
- **Tracking:** Weights & Biases
- **Compute:** CUDA-enabled GPUs

## License

MIT
