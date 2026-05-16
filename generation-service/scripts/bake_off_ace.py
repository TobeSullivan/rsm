"""
Phase 2c — Music Model Bake-off — ACE-Step harness (Turbo + XL).

Headless. Generates N seeds for a single prompt file, writes audio + a
manifest.json that pairs cleanly with rsm-bakeoff-scorecard.xlsx rows.

Smallest viable harness:
	- One script per model family. ACE-Step Turbo + XL share this script
	  (same pipeline, --variant swaps checkpoint).
	- DiffRhythm 2 and SongGeneration v2 get their own scripts later.
	- No "model-agnostic harness" abstraction. Refactor common patterns
	  AFTER they prove common.

Apples-to-apples with the lab:
	- Same handler init pattern as scripts/lab_v0.py
	- Same GenerationParams contract (split-call: thinking=True,
	  use_cot_*=False so user-edited caption/lyrics/meta are respected)
	- Same Turbo defaults (inference_steps=8, shift=3.0, infer_method=ode)
	- audio_format="flac" (matches lab; lossless beats wav for scoring)

Apples-to-apples across models:
	- Same lyrics block to every model
	- Same seeds [1, 2, 3] across every model
	- Same content in the caption; syntax tuned per model in the JSON

Run from <service-root>:
	uv run scripts/bake_off_ace.py --variant turbo
	uv run scripts/bake_off_ace.py --variant xl --prompt scripts/bake_off_prompts/r1_pop_baseline.json

Output:
	outputs/r1-pop/
		ace-step-turbo_seed1.flac
		ace-step-turbo_seed2.flac
		ace-step-turbo_seed3.flac
		manifest_ace-step-turbo.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

import torch

# --- Configuration ---------------------------------------------------------
SERVICE_ROOT = Path(__file__).parent.parent.resolve()
CHECKPOINT_DIR = SERVICE_ROOT / "checkpoints"
OUTPUT_ROOT = SERVICE_ROOT / "outputs"

DEFAULT_PROMPT_FILE = SERVICE_ROOT / "scripts" / "bake_off_prompts" / "r1_pop_baseline.json"

LM_MODEL = "acestep-5Hz-lm-1.7B"
LM_BACKEND = "pytorch"  # pytorch portable across Windows/Mac; vllm is Linux-first

# Per-variant DiT config strings.
# Turbo: confirmed by lab v0.2 (scripts/lab_v0.py uses "acestep-v15-turbo").
# XL: best guess — needs verification against ACE-Step docs when you first
# run XL on the M4 Max. If wrong, fix here and the rest of the harness is fine.
DIT_CONFIGS = {
	"turbo": "acestep-v15-turbo",
	"xl": "acestep-v15",  # TODO verify XL config name against ACE-Step docs
}

# Per-variant inference defaults.
# Turbo defaults match the lab (and ACE-Step docs).
# XL defaults are placeholders — standard ACE-Step typically uses 30+ steps and
# different shift. Verify when running XL.
VARIANT_DEFAULTS = {
	"turbo": {
		"inference_steps": 8,
		"shift": 3.0,
		"infer_method": "ode",
		"guidance_scale": 15.0,
	},
	"xl": {
		"inference_steps": 30,         # TODO verify XL default inference_steps
		"shift": 3.0,                  # TODO verify XL default shift
		"infer_method": "ode",         # TODO verify XL default infer_method
		"guidance_scale": 15.0,        # TODO verify XL default guidance_scale
	},
}


# --- Device auto-detection -------------------------------------------------
def detect_device() -> str:
	"""CUDA on Windows/Linux, MPS on Mac, CPU fallback. Logged loudly."""
	if torch.cuda.is_available():
		device = "cuda"
		logger.info(f"Device: cuda ({torch.cuda.get_device_name(0)})")
	elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		device = "mps"
		logger.info("Device: mps (Apple Silicon)")
	else:
		device = "cpu"
		logger.warning("Device: cpu — generation will be very slow")
	return device


def peak_memory_gb(device: str) -> float | None:
	"""Best-effort peak memory readout. None if unavailable on this device."""
	try:
		if device == "cuda":
			return torch.cuda.max_memory_allocated() / (1024 ** 3)
		# MPS does not expose a peak-allocated API equivalent in stable torch.
		# Skip rather than guess.
		return None
	except Exception as e:
		logger.warning(f"Peak memory readout failed: {e}")
		return None


def reset_peak_memory(device: str) -> None:
	if device == "cuda":
		try:
			torch.cuda.reset_peak_memory_stats()
		except Exception as e:
			logger.warning(f"reset_peak_memory_stats failed: {e}")


# --- Handler init (mirrors lab_v0.py exactly) ------------------------------
def init_handlers(variant: str, device: str):
	"""Initialize DiT and LM handlers. Imports deferred so loguru prints
	the config block first."""
	from acestep.handler import AceStepHandler
	from acestep.llm_inference import LLMHandler

	dit_config = DIT_CONFIGS[variant]

	logger.info(f"Initializing DiT handler (variant={variant}, config={dit_config})...")
	t0 = time.time()
	dit = AceStepHandler()
	dit.initialize_service(
		project_root=str(SERVICE_ROOT),
		config_path=dit_config,
		device=device,
	)
	logger.info(f"DiT init: {time.time() - t0:.2f}s")

	logger.info(f"Initializing LM handler (model={LM_MODEL}, backend={LM_BACKEND})...")
	t0 = time.time()
	lm = LLMHandler()
	lm.initialize(
		checkpoint_dir=str(CHECKPOINT_DIR),
		lm_model_path=LM_MODEL,
		backend=LM_BACKEND,
		device=device,
	)
	logger.info(f"LM init: {time.time() - t0:.2f}s")

	return dit, lm


# --- One generation --------------------------------------------------------
def run_one_seed(
	dit_handler,
	lm_handler,
	prompt: dict,
	variant: str,
	seed: int,
	save_dir: Path,
	device: str,
) -> dict:
	"""Generate one .flac for one seed. Returns a manifest entry dict."""
	from acestep.inference import GenerationParams, GenerationConfig, generate_music

	caption = prompt["captions"]["ace_step"]
	lyrics = prompt["lyrics"]
	gen = prompt["generation"]
	defaults = VARIANT_DEFAULTS[variant]

	logger.info(f"--- variant={variant} seed={seed} ---")
	logger.info(f"caption: {caption!r}")

	params = GenerationParams(
		task_type="text2music",
		caption=caption,
		lyrics=lyrics,
		bpm=int(gen["bpm"]),
		duration=float(gen["duration_seconds"]),
		keyscale=gen.get("keyscale", ""),
		vocal_language=gen.get("vocal_language", "en"),
		instrumental=bool(gen.get("instrumental", False)),
		# Split-call contract: user-edited caption/lyrics/meta respected,
		# vocalist phase (phoneme articulation) still runs.
		thinking=True,
		use_cot_caption=False,
		use_cot_metas=False,
		use_cot_language=False,
		# Per-variant inference knobs.
		inference_steps=defaults["inference_steps"],
		shift=defaults["shift"],
		infer_method=defaults["infer_method"],
		guidance_scale=defaults["guidance_scale"],
		seed=seed,
	)

	config = GenerationConfig(
		batch_size=1,
		use_random_seed=False,
		audio_format="flac",
	)

	# Run.
	reset_peak_memory(device)
	t0 = time.time()
	result = generate_music(
		dit_handler,
		lm_handler,
		params,
		config,
		save_dir=str(save_dir),
	)
	wall = time.time() - t0

	if not result.success:
		logger.error(f"generate_music failed for seed={seed}: {result.error}")
		return {
			"seed": seed,
			"success": False,
			"error": str(result.error),
			"wall_clock_seconds": wall,
		}

	# ACE-Step writes to its own filename inside save_dir; rename to our
	# scorecard-friendly convention so we don't have to read manifest to
	# know which file is which.
	original_path = Path(result.audios[0]["path"])
	target_name = f"ace-step-{variant}_seed{seed}.flac"
	target_path = save_dir / target_name
	if original_path != target_path:
		try:
			original_path.rename(target_path)
		except Exception as e:
			logger.warning(f"Rename failed ({e}); keeping original name {original_path.name}")
			target_path = original_path

	tc = result.extra_outputs.get("time_costs", {}) if hasattr(result, "extra_outputs") else {}
	peak_gb = peak_memory_gb(device)

	logger.info(f"  -> {target_path.name} (wall={wall:.2f}s, peak={peak_gb}GB)")

	return {
		"seed": seed,
		"success": True,
		"output_filename": target_path.name,
		"wall_clock_seconds": round(wall, 2),
		"peak_memory_gb": round(peak_gb, 3) if peak_gb is not None else None,
		"time_costs": dict(tc) if tc else {},
		"params": {
			"caption": caption,
			"bpm": gen["bpm"],
			"duration_seconds": gen["duration_seconds"],
			"inference_steps": defaults["inference_steps"],
			"shift": defaults["shift"],
			"infer_method": defaults["infer_method"],
			"guidance_scale": defaults["guidance_scale"],
			"seed": seed,
		},
	}


# --- Main ------------------------------------------------------------------
def main() -> None:
	ap = argparse.ArgumentParser(description="ACE-Step bake-off harness (Turbo + XL).")
	ap.add_argument(
		"--variant",
		choices=["turbo", "xl"],
		default="turbo",
		help="Which ACE-Step checkpoint to load. Default: turbo.",
	)
	ap.add_argument(
		"--prompt",
		type=Path,
		default=DEFAULT_PROMPT_FILE,
		help=f"Path to prompt JSON. Default: {DEFAULT_PROMPT_FILE.relative_to(SERVICE_ROOT)}",
	)
	args = ap.parse_args()

	logger.info("=" * 70)
	logger.info("Phase 2c bake-off — ACE-Step")
	logger.info("=" * 70)
	logger.info(f"SERVICE_ROOT:   {SERVICE_ROOT}")
	logger.info(f"CHECKPOINT_DIR: {CHECKPOINT_DIR}")
	logger.info(f"variant:        {args.variant}")
	logger.info(f"prompt file:    {args.prompt}")

	if not args.prompt.exists():
		logger.error(f"Prompt file not found: {args.prompt}")
		raise SystemExit(1)

	prompt = json.loads(args.prompt.read_text(encoding="utf-8"))
	output_subdir = prompt["_meta"]["output_subdir"]
	save_dir = OUTPUT_ROOT / output_subdir
	save_dir.mkdir(parents=True, exist_ok=True)
	logger.info(f"save_dir:       {save_dir}")

	device = detect_device()
	dit, lm = init_handlers(args.variant, device)

	seeds = prompt["generation"]["seeds"]
	logger.info(f"Generating {len(seeds)} seed(s): {seeds}")

	entries = []
	for seed in seeds:
		entry = run_one_seed(
			dit_handler=dit,
			lm_handler=lm,
			prompt=prompt,
			variant=args.variant,
			seed=seed,
			save_dir=save_dir,
			device=device,
		)
		entries.append(entry)

	# Manifest: per-model file in save_dir. Multiple manifests can coexist
	# in r1-pop/ (one per model family / variant). Cross-reference with
	# scorecard rows by (round, model_family, variant, seed).
	manifest = {
		"round": prompt["_meta"]["round"],
		"model_family": "ace-step",
		"variant": args.variant,
		"dit_config": DIT_CONFIGS[args.variant],
		"lm_model": LM_MODEL,
		"lm_backend": LM_BACKEND,
		"device": device,
		"host": {
			"platform": platform.platform(),
			"python": platform.python_version(),
			"torch": torch.__version__,
		},
		"prompt_file": str(args.prompt.relative_to(SERVICE_ROOT)) if args.prompt.is_absolute() else str(args.prompt),
		"prompt_caption_ace_step": prompt["captions"]["ace_step"],
		"generated_at_utc": datetime.now(timezone.utc).isoformat(),
		"generations": entries,
	}
	manifest_path = save_dir / f"manifest_ace-step-{args.variant}.json"
	manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
	logger.info(f"Manifest: {manifest_path}")

	successes = sum(1 for e in entries if e.get("success"))
	logger.info(f"Done: {successes}/{len(entries)} successful generations.")
	if successes < len(entries):
		raise SystemExit(2)


if __name__ == "__main__":
	main()
