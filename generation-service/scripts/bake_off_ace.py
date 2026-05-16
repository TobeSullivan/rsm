"""
Phase 2c — Music Model Bake-off — ACE-Step harness (Turbo + XL Turbo/SFT/Base).

Headless. Generates N seeds for a single prompt file, writes audio + a
manifest.json that pairs cleanly with rsmbakeoffscorecard.xlsx rows.

Smallest viable harness:
	- One script per model family. All four ACE-Step variants share this script
	  (same pipeline, --variant swaps checkpoint + inference defaults).
	- DiffRhythm 2 and SongGeneration v2 get their own scripts later.
	- No "model-agnostic harness" abstraction. Refactor common patterns
	  AFTER they prove common.

Apples-to-apples with the lab:
	- Same handler init pattern as scripts/lab_v0.py
	- Same GenerationParams contract (split-call: thinking=True,
	  use_cot_*=False so user-edited caption/lyrics/meta are respected)
	- audio_format="flac" (matches lab; lossless beats wav for scoring)

Apples-to-apples across models:
	- Same lyrics block to every model
	- Same seeds [1, 2, 3] across every model
	- Same content in the caption; syntax tuned per model in the JSON

Variant inference defaults (verified against ACE-Step docs/en/INFERENCE.md):
	- Distilled (turbo, xl-turbo): 8 inference steps, shift=3.0.
	  guidance_scale is documented as ignored for turbo models; we still
	  set the doc default 7.0 for consistency.
	- Non-distilled (xl-sft, xl-base): 50 inference steps. shift=1.0 (the
	  documented default for non-distilled — earlier shift=3.0 across the
	  board produced pure-noise output on xl-sft and xl-base, fixed here).
	  guidance_scale=7.0 (doc default; typical range 5.0-9.0).

CLI overrides (added for the 2026-05-16 xl-sft/xl-base noise investigation):
	--shift, --guidance-scale, --inference-steps, --use-adg let us A/B
	individual knobs against a variant's defaults without editing source.
	--seeds limits the run to a subset for cheap experiments. --output-suffix
	tags filenames so debug runs don't clobber prior good outputs.

Diagnostic stats:
	Every generation logs crest factor (peak / abs_mean of the audio
	waveform) into the manifest. Music sits around 5-20, gaussian noise
	sits around ~3. Single number telling us at-a-glance whether a
	generation produced music or noise. Latent stats also captured when
	ACE-Step exposes them via extra_outputs.

Run from <service-root>:
	uv run scripts/bake_off_ace.py --variant turbo
	uv run scripts/bake_off_ace.py --variant xl-turbo
	uv run scripts/bake_off_ace.py --variant xl-sft
	uv run scripts/bake_off_ace.py --variant xl-base

Debugging examples (2026-05-16 noise investigation):
	# A/B test the shift fix on a single seed (~3 min on Mac):
	uv run scripts/bake_off_ace.py --variant xl-base --shift 1.0 \\
		--seeds 1 --output-suffix _shift1
	# Also try APG, if shift fix alone isn't enough:
	uv run scripts/bake_off_ace.py --variant xl-base --shift 1.0 \\
		--use-adg --seeds 1 --output-suffix _shift1_apg

Output:
	outputs/r1-pop/
		ace-step-turbo_seed1.flac
		ace-step-turbo_seed2.flac
		ace-step-turbo_seed3.flac
		manifest_ace-step-turbo.json
	(and equivalents for ace-step-xl-turbo, ace-step-xl-sft, ace-step-xl-base)
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
LM_BACKEND = "pytorch"  # pytorch portable across Windows/Mac; vllm is Linux-first.
                        # NOTE: ACE-Step also supports an "mlx" LM backend on Apple
                        # Silicon which may give additional speedup on Mac. Left as
                        # pytorch for now since current Mac wall-clock is acceptable;
                        # revisit if non-distilled XL feels too slow.

# Per-variant DiT config strings. Verified against the ACE-Step 1.5 README
# (Model Zoo) and docs/en/INFERENCE.md.
DIT_CONFIGS = {
	"turbo":    "acestep-v15-turbo",
	"xl-turbo": "acestep-v15-xl-turbo",
	"xl-sft":   "acestep-v15-xl-sft",
	"xl-base":  "acestep-v15-xl-base",
}

# Per-variant inference defaults.
# - shift: 3.0 for distilled (turbo, xl-turbo) per ACE-Step docs.
#   1.0 (the documented default) for non-distilled (xl-sft, xl-base) — using
#   shift=3.0 on non-distilled variants produced pure-noise output, confirmed
#   empirically 2026-05-16 and consistent with a known HF issue on
#   acestep-v15-sft re: shift/timestep schedule mismatch.
# - guidance_scale: 7.0 (doc default). Turbo models ignore it (ACE-Step logs
#   "Turbo model detected: overriding guidance_scale 7.0 -> 1.0").
# - use_adg: False everywhere by default. xl-base ships its own apg_guidance.py
#   and may benefit from --use-adg, but this is a tuning knob with several
#   sub-parameters (eta, norm_thresh, momentum) we haven't dialed in.
VARIANT_DEFAULTS = {
	"turbo": {
		"inference_steps": 8,
		"shift": 3.0,
		"infer_method": "ode",
		"guidance_scale": 7.0,
		"use_adg": False,
	},
	"xl-turbo": {
		"inference_steps": 8,
		"shift": 3.0,
		"infer_method": "ode",
		"guidance_scale": 7.0,
		"use_adg": False,
	},
	"xl-sft": {
		"inference_steps": 50,
		"shift": 1.0,
		"infer_method": "ode",
		"guidance_scale": 7.0,
		"use_adg": False,
	},
	"xl-base": {
		"inference_steps": 50,
		"shift": 1.0,
		"infer_method": "ode",
		"guidance_scale": 7.0,
		"use_adg": False,
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


# --- Diagnostic stats ------------------------------------------------------
def audio_stats(audio_tensor) -> dict:
	"""Compute cheap diagnostic stats on the generated audio waveform.

	Key field is `crest_factor` = peak / abs_mean. After ACE-Step's
	normalize-to-peak step, this is dominated by the abs_mean denominator,
	which reflects how "loud-on-average" the signal is.

	Rough buckets (post ACE-Step normalization, peak ~ 0.89):
		crest_factor < 4   → noise-like (consistently loud, no dynamics)
		crest_factor 5-10  → low-dynamic-range music (compressed pop)
		crest_factor 10-25 → typical music with vocals + dynamics
		crest_factor > 25  → very dynamic (mostly-quiet with peaks)
	"""
	if audio_tensor is None:
		return {"error": "audio_tensor is None"}
	try:
		t = audio_tensor.detach().float().cpu()
		abs_t = t.abs()
		peak = float(abs_t.max())
		abs_mean = float(abs_t.mean())
		return {
			"shape": list(t.shape),
			"mean": float(t.mean()),
			"std": float(t.std()),
			"abs_mean": abs_mean,
			"peak": peak,
			"crest_factor": peak / (abs_mean + 1e-9),
		}
	except Exception as e:
		return {"error": f"audio_stats failed: {e}"}


def latent_stats(extra_outputs) -> dict | None:
	"""Try to capture pre-VAE latent stats if ACE-Step exposes them.

	Healthy denoising → latents with structured variance.
	Failed denoising  → latents that look ~N(0,1) (std ≈ 1.0, mean ≈ 0).
	Field name varies by ACE-Step version (docs say 'latents', logs show
	'pred_latents'); we probe both.
	"""
	if not extra_outputs:
		return None
	for key in ("latents", "pred_latents"):
		t = extra_outputs.get(key) if hasattr(extra_outputs, "get") else None
		if t is None:
			continue
		try:
			t_f = t.detach().float().cpu()
			abs_t = t_f.abs()
			return {
				"source_key": key,
				"shape": list(t_f.shape),
				"mean": float(t_f.mean()),
				"std": float(t_f.std()),
				"abs_mean": float(abs_t.mean()),
				"abs_max": float(abs_t.max()),
			}
		except Exception as e:
			return {"source_key": key, "error": str(e)}
	return None


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
	effective_params: dict,
	output_suffix: str,
) -> dict:
	"""Generate one .flac for one seed. Returns a manifest entry dict."""
	from acestep.inference import GenerationParams, GenerationConfig, generate_music

	caption = prompt["captions"]["ace_step"]
	lyrics = prompt["lyrics"]
	gen = prompt["generation"]

	logger.info(f"--- variant={variant} seed={seed} ---")
	logger.info(
		f"effective: steps={effective_params['inference_steps']}, "
		f"shift={effective_params['shift']}, "
		f"infer_method={effective_params['infer_method']}, "
		f"guidance_scale={effective_params['guidance_scale']}, "
		f"use_adg={effective_params['use_adg']}"
	)
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
		# Per-variant inference knobs (CLI-overrideable).
		inference_steps=effective_params["inference_steps"],
		shift=effective_params["shift"],
		infer_method=effective_params["infer_method"],
		guidance_scale=effective_params["guidance_scale"],
		use_adg=effective_params["use_adg"],
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
	target_name = f"ace-step-{variant}_seed{seed}{output_suffix}.flac"
	target_path = save_dir / target_name
	if original_path != target_path:
		try:
			original_path.rename(target_path)
		except Exception as e:
			logger.warning(f"Rename failed ({e}); keeping original name {original_path.name}")
			target_path = original_path

	# Diagnostic stats. Audio tensor is in result.audios[0]["tensor"]
	# per ACE-Step's documented audio dict structure.
	tc = result.extra_outputs.get("time_costs", {}) if hasattr(result, "extra_outputs") else {}
	a_stats = audio_stats(result.audios[0].get("tensor"))
	l_stats = latent_stats(result.extra_outputs if hasattr(result, "extra_outputs") else None)
	peak_gb = peak_memory_gb(device)

	cf = a_stats.get("crest_factor")
	if isinstance(cf, (int, float)):
		cf_str = f"{cf:.2f}"
		if cf < 4.0:
			cf_hint = " ⚠️  noise-like (crest < 4)"
		elif cf < 5.0:
			cf_hint = " ⚠️  borderline (crest 4-5)"
		else:
			cf_hint = ""
	else:
		cf_str = str(cf)
		cf_hint = ""

	logger.info(
		f"  -> {target_path.name} "
		f"(wall={wall:.2f}s, peak={peak_gb}GB, crest_factor={cf_str}){cf_hint}"
	)

	return {
		"seed": seed,
		"success": True,
		"output_filename": target_path.name,
		"wall_clock_seconds": round(wall, 2),
		"peak_memory_gb": round(peak_gb, 3) if peak_gb is not None else None,
		"time_costs": dict(tc) if tc else {},
		"audio_stats": a_stats,
		"latent_stats": l_stats,
		"params": {
			"caption": caption,
			"bpm": gen["bpm"],
			"duration_seconds": gen["duration_seconds"],
			"inference_steps": effective_params["inference_steps"],
			"shift": effective_params["shift"],
			"infer_method": effective_params["infer_method"],
			"guidance_scale": effective_params["guidance_scale"],
			"use_adg": effective_params["use_adg"],
			"seed": seed,
		},
	}


def parse_seeds_arg(seeds_arg: str | None, prompt_seeds: list) -> list:
	"""If --seeds was passed, parse it; otherwise fall back to prompt's seeds."""
	if not seeds_arg:
		return list(prompt_seeds)
	return [int(s.strip()) for s in seeds_arg.split(",") if s.strip()]


def resolve_effective_params(variant: str, args) -> dict:
	"""Apply CLI overrides on top of the variant's defaults. None override = keep default."""
	eff = dict(VARIANT_DEFAULTS[variant])  # shallow copy
	if args.inference_steps is not None:
		eff["inference_steps"] = args.inference_steps
	if args.shift is not None:
		eff["shift"] = args.shift
	if args.guidance_scale is not None:
		eff["guidance_scale"] = args.guidance_scale
	if args.use_adg:
		eff["use_adg"] = True
	return eff


# --- Main ------------------------------------------------------------------
def main() -> None:
	ap = argparse.ArgumentParser(description="ACE-Step bake-off harness (Turbo + XL Turbo/SFT/Base).")
	ap.add_argument(
		"--variant",
		choices=["turbo", "xl-turbo", "xl-sft", "xl-base"],
		default="turbo",
		help="Which ACE-Step checkpoint to load. Default: turbo.",
	)
	ap.add_argument(
		"--prompt",
		type=Path,
		default=DEFAULT_PROMPT_FILE,
		help=f"Path to prompt JSON. Default: {DEFAULT_PROMPT_FILE.relative_to(SERVICE_ROOT)}",
	)
	# Inference knob overrides (None = use variant default from VARIANT_DEFAULTS).
	ap.add_argument(
		"--shift",
		type=float,
		default=None,
		help="Override shift. Variant defaults: turbo/xl-turbo=3.0, xl-sft/xl-base=1.0.",
	)
	ap.add_argument(
		"--guidance-scale",
		type=float,
		default=None,
		help="Override guidance_scale. Default 7.0 across variants (ignored by turbo).",
	)
	ap.add_argument(
		"--inference-steps",
		type=int,
		default=None,
		help="Override inference_steps. Variant defaults: turbo/xl-turbo=8, xl-sft/xl-base=50.",
	)
	ap.add_argument(
		"--use-adg",
		action="store_true",
		help="Enable Adaptive Projected Guidance (base-only feature). Default off across variants.",
	)
	# Run-shape overrides.
	ap.add_argument(
		"--seeds",
		type=str,
		default=None,
		help="Comma-separated seeds to run (e.g. '1' or '1,2,3'). Default: use prompt's seeds.",
	)
	ap.add_argument(
		"--output-suffix",
		type=str,
		default="",
		help="Suffix appended to output filenames and manifest (e.g. '_shift1'). Default: none.",
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

	# Resolve effective params (variant defaults + CLI overrides).
	effective_params = resolve_effective_params(args.variant, args)
	defaults = VARIANT_DEFAULTS[args.variant]
	overrides_applied = {k: v for k, v in effective_params.items() if defaults.get(k) != v}
	if overrides_applied:
		logger.info(f"overrides:      {overrides_applied}")
	else:
		logger.info("overrides:      (none — using variant defaults)")
	if args.output_suffix:
		logger.info(f"output suffix:  {args.output_suffix!r}")

	device = detect_device()
	dit, lm = init_handlers(args.variant, device)

	seeds = parse_seeds_arg(args.seeds, prompt["generation"]["seeds"])
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
			effective_params=effective_params,
			output_suffix=args.output_suffix,
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
		"effective_params": effective_params,
		"overrides_applied": overrides_applied,
		"output_suffix": args.output_suffix,
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
	manifest_path = save_dir / f"manifest_ace-step-{args.variant}{args.output_suffix}.json"
	manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
	logger.info(f"Manifest: {manifest_path}")

	# Crest-factor summary across seeds. Cheap at-a-glance read of whether
	# this run produced music or noise.
	successes = sum(1 for e in entries if e.get("success"))
	cfs = [
		e.get("audio_stats", {}).get("crest_factor")
		for e in entries if e.get("success")
	]
	cfs = [c for c in cfs if isinstance(c, (int, float))]
	if cfs:
		logger.info(f"Crest factors: {[round(c, 2) for c in cfs]} (>5 = music, <4 = noise)")

	logger.info(f"Done: {successes}/{len(entries)} successful generations.")
	if successes < len(entries):
		raise SystemExit(2)


if __name__ == "__main__":
	main()