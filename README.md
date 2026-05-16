# Rockband Simulator/Manager

Go from no-name loser to the biggest name in music.

## Repo Layout

```
rsm/
â”œâ”€â”€ generation-service/   # Python + ACE-Step prototype, will become local FastAPI service
â””â”€â”€ game/                 # Godot project (placeholder until prototype validates)
```

The two halves communicate only over HTTP. Different toolchains, same repo.

## Status

Slice 1 â€” validate ACE-Step output quality on a generic rock baseline. No cleanup
pipeline, no UI, no game integration yet.

See `generation-service/README.md` for setup.

## Production Note

The Python stack is for prototyping. Shipping runtime targets `acestep.cpp` (C++17/GGML,
~50MB binary, runs on CUDA/ROCm/Metal/Vulkan/CPU). Same HTTP contract, swappable backend.
