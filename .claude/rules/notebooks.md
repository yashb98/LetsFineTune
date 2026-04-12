---
paths:
  - "notebooks/**/*.ipynb"
---

# Notebook Rules

- Notebooks are for exploration and prototyping only.
- Production-ready code must be moved to `src/` or `scripts/`.
- Clear all outputs before committing (or use pre-commit hook).
- Include markdown cells explaining what each section does.
- Pin random seeds in notebooks for reproducible experiments.
- Use `%load_ext autoreload` and `%autoreload 2` when importing from `src/`.
