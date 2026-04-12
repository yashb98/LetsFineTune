---
paths:
  - "configs/**"
  - "scripts/train.py"
  - "src/training/**"
---

# Training Rules

- All training hyperparameters go in YAML config files under `configs/`, never hardcoded.
- Always set and log random seeds for reproducibility.
- Include gradient accumulation, mixed precision (bf16), and gradient checkpointing by default.
- Log metrics: loss, learning rate, GPU memory usage, tokens/second.
- Save checkpoints at configurable intervals with `save_total_limit` to manage disk.
- Always include a validation split for monitoring overfitting.
- Handle CUDA OOM gracefully — catch the error and suggest reducing batch size.
- Prefer LoRA/QLoRA (8-bit default) over full fine-tuning unless config specifies otherwise.
- Always validate dataset format and tokenizer compatibility before starting training.
