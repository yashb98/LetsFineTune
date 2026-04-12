---
name: evaluate
description: Evaluate a fine-tuned model checkpoint
allowed-tools: Bash Read Grep Glob
---

# Evaluate Skill

Evaluate a fine-tuned model checkpoint against a dataset.

## Steps
1. Verify the model checkpoint exists at the given path.
2. Check that the evaluation dataset exists and is valid JSONL.
3. Run: `python scripts/evaluate.py --model <model_path> --dataset <dataset_path>`
4. Read and summarize the results from `results/eval_results.json`.
5. Show key metrics and a few sample outputs.
