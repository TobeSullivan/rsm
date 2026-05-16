"""
Rockband Simulator/Manager - Generation Prototype (Slice 1)

Single-stage: pre-defined caption + lyrics -> DiT renders audio.

NOTE: create_sample() (LM-generated lyrics from a natural-language query) needs
correct LLMHandler initialization, which we have not figured out yet. Deferred
to slice 3 - probably a checkpoint path issue. For now we hardcode lyrics so
we can validate that vocals work at all.

Adjust the CONFIG block below to change lyrics, duration, or model variant.
"""

from __future__ import annotations

import logging
from pathlib import Path

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.llm_inference import LLMHandler

# ─── CONFIG ──────────────────────────────────────────────────────────────
CAPTION = "energetic rock anthem with powerful vocals, electric guitars, driving drums"
LYRICS = """[Verse 1]
Stand up tall, the night is calling
Lights are bright and walls are falling
[Chorus]
We are the fire, we are the sound
Tear it down and lift it off the ground"""
VOCAL_LANGUAGE = "en"
DURATION_SECONDS = 30

DEVICE = "cuda"                          # "cuda" | "mps" | "cpu"
DIT_CONFIG = "acestep-v15-turbo"         # fast variant, 8-step diffusion
LM_MODEL = "acestep-5Hz-lm-0.6B"         # smallest LM, fastest startup
LM_BACKEND = "pytorch"                   # "vllm" | "pytorch" - pytorch is more portable on Windows
AUDIO_FORMAT = "flac"                    # lossless so we hear what the model actually produced

# Turbo-specific recommendations from ACE-Step docs
INFERENCE_STEPS = 8
SHIFT = 3.0                              # Recommended for turbo models

# Paths
SERVICE_ROOT = Path(__file__).resolve().parents[2]   # generation-service/
CHECKPOINT_DIR = Path.home() / ".cache" / "huggingface" / "hub"
OUTPUT_DIR = SERVICE_ROOT / "output"

# ─── MAIN ────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
    )
    log = logging.getLogger("rsm-gen")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", OUTPUT_DIR)

    # ─── Initialize handlers ─────────────────────────────────────────────
    log.info("Initializing ACE-Step DiT handler (device=%s, config=%s)...", DEVICE, DIT_CONFIG)
    dit_handler = AceStepHandler()
    dit_handler.initialize_service(
        project_root=str(SERVICE_ROOT),
        config_path=DIT_CONFIG,
        device=DEVICE,
    )

    log.info("Initializing LLM handler (model=%s, backend=%s)...", LM_MODEL, LM_BACKEND)
    llm_handler = LLMHandler()
    llm_handler.initialize(
        checkpoint_dir=str(CHECKPOINT_DIR),
        lm_model_path=LM_MODEL,
        backend=LM_BACKEND,
        device=DEVICE,
    )

    # ─── Generate audio ──────────────────────────────────────────────────
    log.info("Generating %ds audio: %r", DURATION_SECONDS, CAPTION)
    log.info("Lyrics:\n%s", LYRICS)

    params = GenerationParams(
        task_type="text2music",
        caption=CAPTION,
        lyrics=LYRICS,
        vocal_language=VOCAL_LANGUAGE,
        duration=DURATION_SECONDS,
        inference_steps=INFERENCE_STEPS,
        shift=SHIFT,
        thinking=True,
    )
    config = GenerationConfig(
        batch_size=1,
        audio_format=AUDIO_FORMAT,
    )

    result = generate_music(
        dit_handler,
        llm_handler,
        params,
        config,
        save_dir=str(OUTPUT_DIR),
    )

    if not result.success:
        log.error("generate_music failed: %s", result.error)
        return

    # ─── Report ──────────────────────────────────────────────────────────
    for audio in result.audios:
        log.info("Generated: %s (seed=%s)", audio["path"], audio["params"].get("seed"))

    times = result.extra_outputs.get("time_costs", {})
    if times:
        log.info(
            "Time costs - LM p1: %.1fs | LM p2: %.1fs | DiT: %.1fs | total: %.1fs",
            times.get("lm_phase1_time", 0),
            times.get("lm_phase2_time", 0),
            times.get("dit_total_time_cost", 0),
            times.get("pipeline_total_time", 0),
        )

    log.info("Done. Listen to the output in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()