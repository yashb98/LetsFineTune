# GitHub Copilot Instructions

## Project Context
This is a machine learning project for fine-tuning Small Language Models (SLMs) using PyTorch and Hugging Face libraries.

## Code Generation Preferences
- Use Hugging Face Transformers API for model loading and training.
- Use PEFT library for LoRA/QLoRA adapters.
- Use TRL (Transformer Reinforcement Learning) for SFTTrainer.
- Always include type hints in function signatures.
- Use logging module instead of print statements in scripts.
- Prefer pathlib.Path over os.path for file operations.
- Use argparse or hydra for CLI argument parsing in scripts.
- Include proper CUDA device management (device_map="auto").

## Common Patterns
- Model loading: `AutoModelForCausalLM.from_pretrained()` with quantization config
- Tokenizer: always set `padding_side="right"` for causal LM training
- LoRA config: target modules vary by architecture — check model's named modules
- Dataset: use `datasets.load_dataset()` with proper formatting functions
- Training: `SFTTrainer` from TRL with `TrainingArguments`

## Avoid
- Hardcoded file paths — use config files or CLI args.
- Training without validation split.
- Loading full-precision models when quantization is available.
- Ignoring tokenizer special tokens (pad, eos, bos).
