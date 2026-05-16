"""
Smoke test for split-call architecture validation (v2).

What changed from v1: this no longer calls create_sample(). v1's slowness
turned out to be create_sample doing a much bigger generation than the
fused generate_music Phase 1 (writing lyrics + caption from scratch vs.
just generating metadata around user-provided text). create_sample's
timing is a real Phase 2b UX concern, but it's orthogonal to the
architecture question we need to answer first.

What v2 validates:

1. generate_music with thinking=True + use_cot_caption=False respects
   the user's caption. In 2a, the same caption ("energetic rock anthem
   with powerful vocals") got rewritten into "wordless gang vocals
   ('Oh-oh-oh')" — the opposite of what the user asked for. With
   use_cot_caption=False, the rewrite should not happen and vocals should
   have real words.

2. Phase 2 (vocalist audio codes) still runs when thinking=True, even
   with all use_cot_* flags off. Confirmed if the output has real phonemes
   (not just melodic "oohs").

3. Warm-run timing. Cold was 167s in 2a; what's the second run in the
   same process?

The test inputs mirror your existing prototype exactly so timing is
directly comparable.

Run from <service-root>:  uv run scripts/smoke_split_call.py
"""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger

# --- Configuration ---------------------------------------------------------
SERVICE_ROOT = Path(__file__).parent.parent.resolve()
CHECKPOINT_DIR = SERVICE_ROOT / "checkpoints"
OUTPUT_DIR = SERVICE_ROOT / "outputs" / "smoke_split_call"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LM_MODEL = "acestep-5Hz-lm-1.7B"
DIT_CONFIG = "acestep-v15-turbo"
DEVICE = "cuda"
LM_BACKEND = "pytorch"

# Match the existing prototype's inputs verbatim so we can compare audio
# character side-by-side with the 2a output ("wordless gang vocals").
CAPTION = (
	"energetic rock anthem with powerful vocals, "
	"electric guitars, driving drums"
)
LYRICS = """[Verse 1]
Stand up tall, the night is calling
Lights are bright and walls are falling
[Chorus]
We are the fire, we are the sound
Tear it down and lift it off the ground"""

# Run 2 uses different inputs so we're measuring warm timing, not cache.
CAPTION_RUN2 = "upbeat indie pop, jangly guitars, bright melodic vocals"
LYRICS_RUN2 = """[Verse 1]
Sun is rising over the sea
Coffee cup is calling for me
[Chorus]
This is summer, this is the song
Sing it loud and sing it along"""

logger.info(f"SERVICE_ROOT:   {SERVICE_ROOT}")
logger.info(f"CHECKPOINT_DIR: {CHECKPOINT_DIR}")
logger.info(f"OUTPUT_DIR:     {OUTPUT_DIR}")

from acestep.handler import AceStepHandler  # noqa: E402
from acestep.llm_inference import LLMHandler  # noqa: E402
from acestep.inference import (  # noqa: E402
	GenerationParams,
	GenerationConfig,
	generate_music,
)


# --- Handler init ----------------------------------------------------------
def init_handlers() -> tuple[AceStepHandler, LLMHandler]:
	"""Match the existing prototype's init exactly."""
	logger.info("Initializing DiT handler...")
	t0 = time.time()
	dit = AceStepHandler()
	dit.initialize_service(
		project_root=str(SERVICE_ROOT),
		config_path=DIT_CONFIG,
		device=DEVICE,
	)
	logger.info(f"DiT init: {time.time() - t0:.2f}s")

	logger.info("Initializing LM handler...")
	t0 = time.time()
	lm = LLMHandler()
	lm.initialize(
		checkpoint_dir=str(CHECKPOINT_DIR),
		lm_model_path=LM_MODEL,
		backend=LM_BACKEND,
		device=DEVICE,
	)
	logger.info(f"LM init: {time.time() - t0:.2f}s")

	return dit, lm


# --- generate_music with CoT disabled --------------------------------------
def run_generate(
	dit_handler: AceStepHandler,
	lm_handler: LLMHandler,
	caption: str,
	lyrics: str,
	label: str,
	save_subdir: str,
	bpm: int = 140,
	duration: float = 30.0,
	keyscale: str = "E minor",
	vocal_language: str = "en",
	seed: int = 42,
) -> dict:
	"""generate_music with the lab's intended flag combination:

	  thinking=True            -> Phase 2 vocalist runs, vocals get phonemes
	  use_cot_caption=False    -> LM does NOT rewrite the caption
	  use_cot_metas=False      -> LM does NOT regenerate bpm/key/etc.
	  use_cot_language=False   -> LM does NOT re-detect language

	If this works, the lab's edit-and-rerun-from-here mechanic is real.
	"""
	logger.info(f"[{label}] caption: {caption!r}")
	logger.info(f"[{label}] lyrics: {lyrics!r}")
	logger.info(
		f"[{label}] generate_music "
		f"(thinking=True, use_cot_caption/metas/language=False)"
	)

	params = GenerationParams(
		task_type="text2music",
		caption=caption,
		lyrics=lyrics,
		bpm=bpm,
		duration=duration,
		keyscale=keyscale,
		vocal_language=vocal_language,
		instrumental=False,
		thinking=True,
		use_cot_caption=False,
		use_cot_metas=False,
		use_cot_language=False,
		inference_steps=8,
		shift=3.0,
		infer_method="ode",
		seed=seed,
	)

	config = GenerationConfig(
		batch_size=1,
		use_random_seed=False,
		audio_format="flac",
	)

	save_dir = OUTPUT_DIR / save_subdir
	save_dir.mkdir(parents=True, exist_ok=True)

	t0 = time.time()
	result = generate_music(
		dit_handler,
		lm_handler,
		params,
		config,
		save_dir=str(save_dir),
	)
	wall = time.time() - t0
	logger.info(f"[{label}] wall-clock: {wall:.2f}s")

	if not result.success:
		logger.error(f"[{label}] failed: {result.error}")
		raise RuntimeError(f"generate_music failed: {result.error}")

	time_costs = result.extra_outputs.get("time_costs", {}) or {}
	logger.info(f"[{label}] time_costs:")
	if not time_costs:
		logger.warning(f"[{label}]   (empty)")
	for k, v in time_costs.items():
		logger.info(f"  {k}: {v}")

	for i, audio in enumerate(result.audios):
		logger.info(f"[{label}] Output[{i}]: {audio['path']}")

	return {"wall": wall, "time_costs": time_costs, "audios": result.audios}


# --- Main ------------------------------------------------------------------
def main() -> None:
	logger.info("=" * 70)
	logger.info("Smoke test v2: use_cot_*=False respect + warm-run timing")
	logger.info("=" * 70)

	dit, lm = init_handlers()

	logger.info("")
	logger.info("### RUN 1 (COLD) — caption-rewrite-disable test ###")
	logger.info("Same inputs as your 2a prototype, but with use_cot_caption=False.")
	logger.info("2a output had LM-rewritten 'wordless gang vocals (Oh-oh-oh)'.")
	logger.info("This run should produce vocals with REAL WORDS from the lyrics.")
	result_1 = run_generate(
		dit, lm,
		caption=CAPTION,
		lyrics=LYRICS,
		label="run1.cold",
		save_subdir="run1_cold_cot_off",
		seed=42,
	)

	logger.info("")
	logger.info("### RUN 2 (WARM) — timing measurement ###")
	result_2 = run_generate(
		dit, lm,
		caption=CAPTION_RUN2,
		lyrics=LYRICS_RUN2,
		label="run2.warm",
		save_subdir="run2_warm",
		bpm=128,
		duration=30.0,
		keyscale="C Major",
		seed=123,
	)

	# --- Summary ---
	logger.info("")
	logger.info("=" * 70)
	logger.info("SUMMARY")
	logger.info("=" * 70)

	def _summarize(label: str, result: dict) -> None:
		tc = result["time_costs"]
		logger.info(f"{label}:")
		logger.info(f"  wall-clock:      {result['wall']:.2f}s")
		logger.info(f"  lm_phase1_time:  {tc.get('lm_phase1_time', 'n/a')}")
		logger.info(f"  lm_phase2_time:  {tc.get('lm_phase2_time', 'n/a')}")
		logger.info(f"  dit_total_time:  {tc.get('dit_total_time_cost', 'n/a')}")
		logger.info(f"  pipeline_total:  {tc.get('pipeline_total_time', 'n/a')}")

	_summarize("Run 1 (COLD)", result_1)
	_summarize("Run 2 (WARM)", result_2)

	logger.info("")
	logger.info("LISTEN TEST:")
	logger.info(f"  Run 1: {OUTPUT_DIR / 'run1_cold_cot_off'}")
	logger.info(f"    Compare to your 2a output. Vocals should have")
	logger.info(f"    REAL WORDS from the lyrics, NOT 'oh-oh-oh' chants.")
	logger.info(f"  Run 2: {OUTPUT_DIR / 'run2_warm'}")
	logger.info(f"    Should sound like upbeat indie pop, not the 2a rock track.")
	logger.info("")
	logger.info("Smoke test complete.")


if __name__ == "__main__":
	main()