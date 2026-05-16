# rsm-gen â€” Generation Service

Local music generation for Rockband Simulator/Manager.

## Setup (first time)

From this directory (`generation-service/`):

```
uv sync
```

Models auto-download from Hugging Face on first run (~2â€“9 GB depending on variant).
Default cache lives at `~/.cache/huggingface/hub`.

## Slice 1: Smoke Test

```
uv run python -m rsm_gen.prototype
```

Generates one 30-second rock track to `./output/`. Adjust prompt, duration, model variant
in the CONFIG block at the top of `src/rsm_gen/prototype.py`.

## Architecture Note

Python is the prototyping tool. Shipping target is `acestep.cpp` (C++17/GGML),
swappable backend behind the same HTTP contract.
