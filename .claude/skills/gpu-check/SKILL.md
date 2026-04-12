---
name: gpu-check
description: Check GPU status, VRAM, and CUDA availability
allowed-tools: Bash
---

# GPU Check Skill

Check the current GPU status and estimate capacity.

## Steps
1. Run `nvidia-smi` to show GPU model, VRAM usage, temperature.
2. Run `python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"`.
3. Report available VRAM and recommend max model size / batch size.
