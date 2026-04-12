"""Fine-tuning script for causal language models with LoRA/QLoRA.

Usage:
    python scripts/train.py --config configs/lora_config.yaml
"""

import argparse
import logging
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_quantization_config(config: dict) -> BitsAndBytesConfig | None:
    """Create quantization config if enabled."""
    quant_cfg = config.get("quantization", {})
    if not quant_cfg.get("enabled", False):
        return None

    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg["bits"] == 4,
        load_in_8bit=quant_cfg["bits"] == 8,
        bnb_4bit_quant_type=quant_cfg.get("quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant_cfg.get("double_quant", True),
        bnb_4bit_compute_dtype=getattr(torch, config["model"].get("torch_dtype", "bfloat16")),
    )


def format_alpaca(example: dict) -> dict:
    """Format dataset example in Alpaca instruction format."""
    if example.get("input"):
        text = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    else:
        text = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return {"text": text}


def main(config_path: str) -> None:
    config = load_config(config_path)
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    data_cfg = config["dataset"]
    train_cfg = config["training"]

    logger.info("Loading tokenizer: %s", model_cfg["name"])
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Loading model: %s", model_cfg["name"])
    quant_config = get_quantization_config(config)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        quantization_config=quant_config,
        device_map=model_cfg.get("device_map", "auto"),
        torch_dtype=getattr(torch, model_cfg.get("torch_dtype", "bfloat16")),
        attn_implementation=model_cfg.get("attn_implementation"),
    )

    if quant_config:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    logger.info("Loading dataset: %s", data_cfg["name"])
    dataset = load_dataset(data_cfg["name"], split=data_cfg.get("split", "train"))

    if data_cfg.get("format") == "alpaca":
        dataset = dataset.map(format_alpaca, num_proc=data_cfg.get("num_proc", 4))

    if data_cfg.get("val_split"):
        split = dataset.train_test_split(test_size=data_cfg["val_split"], seed=train_cfg.get("seed", 42))
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        weight_decay=train_cfg.get("weight_decay", 0.001),
        max_grad_norm=train_cfg.get("max_grad_norm", 0.3),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        optim=train_cfg.get("optim", "paged_adamw_32bit"),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 100),
        eval_steps=train_cfg.get("eval_steps", 100),
        eval_strategy="steps" if eval_dataset else "no",
        save_total_limit=train_cfg.get("save_total_limit", 3),
        seed=train_cfg.get("seed", 42),
        report_to=train_cfg.get("report_to", "none"),
        load_best_model_at_end=True if eval_dataset else False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=data_cfg.get("max_length", 2048),
    )

    logger.info("Starting training...")
    trainer.train()

    output_dir = Path(train_cfg["output_dir"]) / "final"
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Model saved to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a language model with LoRA/QLoRA")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
