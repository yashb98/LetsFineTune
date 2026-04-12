"""Dataset preparation and validation for fine-tuning.

Usage:
    python scripts/data_prep.py --dataset yahma/alpaca-cleaned --format alpaca
    python scripts/data_prep.py --dataset data/raw/custom.jsonl --format sharegpt --output data/processed/
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ALPACA_REQUIRED = {"instruction", "output"}
SHAREGPT_REQUIRED = {"conversations"}


def validate_alpaca(dataset) -> list[str]:
    """Validate Alpaca format dataset."""
    issues = []
    empty_count = 0
    short_count = 0

    for i, example in enumerate(dataset):
        for field in ALPACA_REQUIRED:
            if field not in example or not example[field]:
                issues.append(f"Row {i}: missing or empty '{field}'")
        if example.get("output") and len(example["output"].strip()) < 10:
            short_count += 1

    if short_count:
        issues.append(f"{short_count} examples have very short outputs (< 10 chars)")
    return issues


def validate_sharegpt(dataset) -> list[str]:
    """Validate ShareGPT format dataset."""
    issues = []
    for i, example in enumerate(dataset):
        convos = example.get("conversations", [])
        if not convos:
            issues.append(f"Row {i}: empty conversations")
            continue
        for j, turn in enumerate(convos):
            if "from" not in turn or "value" not in turn:
                issues.append(f"Row {i}, turn {j}: missing 'from' or 'value'")
    return issues


def find_duplicates(dataset, format_type: str) -> int:
    """Count duplicate entries."""
    if format_type == "alpaca":
        keys = [ex["instruction"] for ex in dataset]
    else:
        keys = [str(ex.get("conversations", "")) for ex in dataset]

    counts = Counter(keys)
    return sum(c - 1 for c in counts.values() if c > 1)


def compute_token_stats(dataset, format_type: str) -> dict:
    """Compute character-level length statistics (proxy for tokens)."""
    if format_type == "alpaca":
        lengths = [
            len(ex.get("instruction", "")) + len(ex.get("input", "")) + len(ex.get("output", ""))
            for ex in dataset
        ]
    else:
        lengths = [
            sum(len(t.get("value", "")) for t in ex.get("conversations", []))
            for ex in dataset
        ]

    arr = np.array(lengths)
    return {
        "count": len(arr),
        "min_chars": int(arr.min()),
        "max_chars": int(arr.max()),
        "mean_chars": int(arr.mean()),
        "median_chars": int(np.median(arr)),
        "p95_chars": int(np.percentile(arr, 95)),
        "estimated_p95_tokens": int(np.percentile(arr, 95) / 4),
    }


def split_and_save(dataset, output_dir: str, train_ratio: float = 0.9, val_ratio: float = 0.05, seed: int = 42):
    """Split dataset and save as JSONL files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    split1 = dataset.train_test_split(test_size=(1 - train_ratio), seed=seed)
    train_ds = split1["train"]
    remaining = split1["test"]

    val_fraction = val_ratio / (1 - train_ratio)
    split2 = remaining.train_test_split(test_size=(1 - val_fraction), seed=seed)
    val_ds = split2["train"]
    test_ds = split2["test"]

    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        path = output / f"{name}.jsonl"
        ds.to_json(str(path))
        logger.info("Saved %s: %d examples -> %s", name, len(ds), path)

    return len(train_ds), len(val_ds), len(test_ds)


def main():
    parser = argparse.ArgumentParser(description="Prepare and validate a dataset for fine-tuning")
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset name or local path")
    parser.add_argument("--format", type=str, default="alpaca", choices=["alpaca", "sharegpt"])
    parser.add_argument("--output", type=str, default="data/processed/")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("Loading dataset: %s", args.dataset)
    if Path(args.dataset).exists():
        dataset = load_dataset("json", data_files=args.dataset, split="train")
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    logger.info("Total examples: %d", len(dataset))

    # Validate
    logger.info("Validating %s format...", args.format)
    if args.format == "alpaca":
        issues = validate_alpaca(dataset)
    else:
        issues = validate_sharegpt(dataset)

    if issues:
        logger.warning("Validation issues found:")
        for issue in issues[:20]:
            logger.warning("  - %s", issue)
        if len(issues) > 20:
            logger.warning("  ... and %d more", len(issues) - 20)
    else:
        logger.info("Validation passed - no issues found")

    # Duplicates
    dup_count = find_duplicates(dataset, args.format)
    if dup_count:
        logger.warning("Found %d duplicate entries", dup_count)
    else:
        logger.info("No duplicates found")

    # Token stats
    stats = compute_token_stats(dataset, args.format)
    logger.info("Length statistics:")
    for key, value in stats.items():
        logger.info("  %s: %s", key, value)
    logger.info("Recommended max_length: %d tokens (based on p95)", min(stats["estimated_p95_tokens"] * 2, 4096))

    # Split and save
    train_n, val_n, test_n = split_and_save(dataset, args.output, seed=args.seed)
    logger.info("Done! Train: %d, Val: %d, Test: %d", train_n, val_n, test_n)


if __name__ == "__main__":
    main()
