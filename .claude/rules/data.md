---
paths:
  - "src/data/**"
  - "data/**"
  - "notebooks/**"
---

# Data Handling Rules

- Support Alpaca and ShareGPT conversation formats for instruction tuning.
- Always report dataset statistics before training: size, token length distribution, field coverage.
- Validate required fields exist (instruction, output for Alpaca; conversations for ShareGPT).
- Check for and remove duplicates, empty responses, and malformed entries.
- Split datasets into train/val/test. Default: 90/5/5.
- Store raw data in `data/raw/`, processed data in `data/processed/`.
- Large datasets are gitignored — document download/generation steps in code or README.
- Use Hugging Face `datasets` library with `num_proc` for parallel preprocessing.
