"""
Rockband Simulator/Manager - Generation Prototype (Phase 2a)

Pipeline: caption + lyrics -> LM generates audio codes -> DiT renders audio.

Phase 2a goal: get LLMHandler.initialize() working so `thinking=True` actually
produces 5Hz audio codes and vocals render real phonemes instead of "oohs."

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
LM_MODEL = "acestep-5Hz-lm-1.7B"         # matches what's downloaded in ./checkpoints/
LM_BACKEND = "pytorch"                   # "vllm" | "pytorch" - pytorch is portable on Windows
AUDIO_FORMAT = "flac"                    # lossless so we hear what the model actually produced

# Turbo-specific recommendations from ACE-Step docs
INFERENCE_STEPS = 8
SHIFT = 3.0                              # Recommended for turbo models

# Paths
SERVICE_ROOT = Path(__file__).resolve().parents[2]   # generation-service/
CHECKPOINT_DIR = SERVICE_ROOT / "checkpoints"        # local cache for ACE-Step models
OUTPUT_DIR = SERVICE_ROOT / "output"

# ─── MAIN ────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
    )
    log = logging.getLogger("rsm-gen")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("SERVICE_ROOT: %s", SERVICE_ROOT)
    log.info("Output directory: %s", OUTPUT_DIR)
    log.info("Checkpoint directory: %s", CHECKPOINT_DIR)

    # ─── Verify checkpoint dir before init ──────────────────────────────
    if not CHECKPOINT_DIR.exists():
        log.error("Checkpoint directory does not exist: %s", CHECKPOINT_DIR)
        return

    log.info("Contents of checkpoint dir:")
    for entry in sorted(CHECKPOINT_DIR.iterdir()):
        log.info("  - %s%s", entry.name, "/" if entry.is_dir() else "")

    lm_model_path = CHECKPOINT_DIR / LM_MODEL
    if not lm_model_path.exists():
        log.error("LM model directory not found: %s", lm_model_path)
        log.error("Expected to find subdir %r inside %s", LM_MODEL, CHECKPOINT_DIR)
        return

    log.info("LM model dir verified: %s", lm_model_path)
    log.info("Contents of LM model dir:")
    for entry in sorted(lm_model_path.iterdir()):
        if entry.is_file():
            log.info("  - %s (%d bytes)", entry.name, entry.stat().st_size)
        else:
            log.info("  - %s/", entry.name)

    # ─── Initialize DiT handler ─────────────────────────────────────────
    log.info("Initializing ACE-Step DiT handler (device=%s, config=%s)...", DEVICE, DIT_CONFIG)
    dit_handler = AceStepHandler()
    dit_handler.initialize_service(
        project_root=str(SERVICE_ROOT),
        config_path=DIT_CONFIG,
        device=DEVICE,
    )
    log.info("DiT handler initialized.")

    # ─── Initialize LLM handler ─────────────────────────────────────────
    log.info("Initializing LLM handler (model=%s, backend=%s)...", LM_MODEL, LM_BACKEND)
    llm_handler = LLMHandler()

    try:
        init_result = llm_handler.initialize(
            checkpoint_dir=str(CHECKPOINT_DIR),
            lm_model_path=LM_MODEL,
            backend=LM_BACKEND,
            device=DEVICE,
        )
        log.info("LLMHandler.initialize() returned: %r", init_result)
    except Exception as exc:
        log.exception("LLMHandler.initialize() raised an exception: %s", exc)
        return

    # Introspect handler state. We do not know what attributes initialize() sets
    # or clears, so dump everything public and non-callable. If init silently
    # failed, the missing/None values will tell us where.
    log.info("LLM handler attributes after init:")
    for attr in sorted(a for a in dir(llm_handler) if not a.startswith("_")):
        try:
            value = getattr(llm_handler, attr)
            if callable(value):
                continue
            log.info("  %s = %r", attr, value)
        except Exception as exc:
            log.info("  %s = <error reading: %s>", attr, exc)

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
        lm_p1 = times.get("lm_phase1_time", 0)
        lm_p2 = times.get("lm_phase2_time", 0)
        log.info(
            "Time costs - LM p1: %.1fs | LM p2: %.1fs | DiT: %.1fs | total: %.1fs",
            lm_p1,
            lm_p2,
            times.get("dit_total_time_cost", 0),
            times.get("pipeline_total_time", 0),
        )
        if lm_p1 < 0.1 and lm_p2 < 0.1:
            log.warning(
                "LM phase times are ~0s. LM likely did not run despite thinking=True. "
                "Vocals will sound like 'oohs' rather than real words."
            )
        else:
            log.info("LM ran. Audio codes were generated. Vocals should have phonemes.")
    else:
        log.warning("No time_costs in extra_outputs. Could not verify LM ran.")

    log.info("Done. Listen to the output in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()