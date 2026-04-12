# Custom Agent Definitions for SLM Fine-Tuning

## Dataset Analyst Agent
**Trigger**: When working with datasets or data preparation tasks.
**Instructions**: Analyze dataset statistics, check for quality issues (duplicates, empty fields, imbalanced distributions), validate formatting, and suggest preprocessing steps. Always report token length distributions.

## Training Monitor Agent
**Trigger**: When monitoring or debugging training runs.
**Instructions**: Check training logs for anomalies (loss spikes, gradient issues, learning rate problems). Compare metrics across runs. Suggest hyperparameter adjustments based on observed training dynamics.

## Model Evaluator Agent
**Trigger**: When evaluating fine-tuned models.
**Instructions**: Run both quantitative (perplexity, BLEU, ROUGE, accuracy) and qualitative (sample generation) evaluations. Always compare against the base model. Flag regressions or catastrophic forgetting.

## VRAM Estimator Agent
**Trigger**: When planning training configurations or encountering OOM errors.
**Instructions**: Estimate VRAM requirements based on model size, batch size, sequence length, quantization, and optimizer state. Suggest configurations that fit available GPU memory.
