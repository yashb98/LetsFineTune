---
paths:
  - "src/evaluation/**"
  - "scripts/evaluate.py"
  - "results/**"
---

# Evaluation Rules

- Always compare fine-tuned model against the base (non-fine-tuned) model.
- Report quantitative metrics: perplexity, BLEU, ROUGE, task-specific accuracy.
- Include qualitative samples: show 5-10 prompt/response pairs for manual inspection.
- Save all evaluation results as JSON in `results/`.
- Flag signs of catastrophic forgetting (degraded performance on general tasks).
- Test inference latency and memory usage alongside quality metrics.
