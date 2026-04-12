---
name: data-prep
description: Validate, analyze, and prepare datasets for fine-tuning
allowed-tools: Bash Read Write Edit Grep Glob
---

# Data Preparation Skill

Validate a dataset's format, analyze token distributions, and prepare train/val/test splits.

## Arguments
- Dataset path (local file or HuggingFace Hub name)
- Format: `alpaca` or `sharegpt` (default: alpaca)
- Output directory (default: `data/processed/`)

## Steps

1. **Load the dataset** — from local JSONL/JSON or HuggingFace Hub.
2. **Validate schema**:
   - Alpaca: require `instruction` and `output` fields, `input` optional.
   - ShareGPT: require `conversations` field with `from`/`value` pairs.
   - Report any rows with missing or empty required fields.
3. **Quality checks**:
   - Count total examples.
   - Detect and report duplicates (by instruction or conversation).
   - Flag empty or very short responses (< 10 characters).
   - Flag very long examples that may exceed `max_length`.
4. **Token analysis** (using the target model's tokenizer if available):
   - Report min / median / mean / max / p95 token lengths.
   - Plot or print the token length distribution.
   - Recommend `max_length` setting based on p95.
5. **Split dataset**:
   - Default: 90% train / 5% validation / 5% test.
   - Stratify if a category/label column exists.
   - Save splits as JSONL to `data/processed/{train,val,test}.jsonl`.
6. **Report summary**:
   - Total examples, split sizes.
   - Quality issues found (duplicates removed, short responses flagged).
   - Recommended training config adjustments.

## Example Usage
```
/data-prep yahma/alpaca-cleaned --format alpaca
/data-prep data/raw/my_dataset.jsonl --format sharegpt --output data/processed/
```
