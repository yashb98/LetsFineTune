---
name: train
description: Run a fine-tuning job with a specified config
allowed-tools: Bash Read Grep Glob
---

# Train Skill

Run a fine-tuning training job.

## Steps
1. Check GPU availability with `nvidia-smi`.
2. Verify the config file exists and is valid YAML.
3. Validate that the dataset is accessible.
4. Run: `python scripts/train.py --config <config_path>`
5. Monitor initial training steps for errors (OOM, data issues).
6. Report training start confirmation with estimated time.
