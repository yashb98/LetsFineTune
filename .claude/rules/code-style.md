---
paths:
  - "**/*.py"
  - "scripts/**"
  - "src/**"
---

# Python Code Style

- Python 3.10+ with type hints on all function signatures.
- Google-style docstrings for public functions.
- Use `pathlib.Path` over `os.path`.
- Use `logging` module, never `print()` in scripts.
- Use `argparse` for CLI argument parsing.
- snake_case for files, functions, variables. PascalCase for classes.
- Line length: 120 characters (configured in pyproject.toml via ruff).
- Imports ordered: stdlib, third-party, local (enforced by ruff isort).
