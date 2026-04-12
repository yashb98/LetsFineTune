"""Evaluate a fine-tuned model against a dataset.

Usage:
    python scripts/evaluate.py --model models/phi2-lora/final --dataset data/processed/eval.jsonl
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_path: str):
    """Load fine-tuned PEFT model and tokenizer."""
    logger.info("Loading model from %s", model_path)
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate a response for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_config)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def evaluate_dataset(model, tokenizer, dataset_path: str, num_samples: int | None = None) -> list[dict]:
    """Run evaluation on a JSONL dataset."""
    results = []
    with open(dataset_path) as f:
        examples = [json.loads(line) for line in f]

    if num_samples:
        examples = examples[:num_samples]

    for i, example in enumerate(examples):
        prompt = example.get("prompt", example.get("instruction", ""))
        expected = example.get("expected", example.get("output", ""))

        response = generate_response(model, tokenizer, prompt)
        results.append({
            "prompt": prompt,
            "expected": expected,
            "generated": response,
        })

        if (i + 1) % 10 == 0:
            logger.info("Evaluated %d/%d examples", i + 1, len(examples))

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model")
    parser.add_argument("--model", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to evaluation dataset (JSONL)")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--output", type=str, default="results/eval_results.json", help="Output file")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    results = evaluate_dataset(model, tokenizer, args.dataset, args.num_samples)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
