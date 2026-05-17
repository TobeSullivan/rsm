"""
DiffRhythm 2 bake-off harness.

Lives at the ROOT of the DiffRhythm2 vendored repo (sibling to inference.py).
Imports DR2's modules directly — no sys.path hacking required since we're
inside their package.

Patches vs upstream inference.py:
  1. Device detection: MPS-aware (was cuda-or-cpu)
  2. FP16 gated to CUDA only (FP16 on MPS produces silent NaNs in some ops)
  3. lyric tag transformer (ACE-Step uppercase → DR2 lowercase)
  4. Reproducible seeds via torch.manual_seed before each generation
  5. Crest factor + audio_stats + latent_stats in manifest
  6. CLI overrides for inference knobs (steps, cfg, max_secs, etc.)
  7. Reads prompts from shared bake_off_prompts/*.json so we stay apples-to-apples
     with ACE-Step's harness

Run:
  cd <repo>/diffrhythm2
  uv run bake_off.py --prompt-file ../generation-service/scripts/bake_off_prompts/r1_pop_baseline.json --seeds 1
  # ^ smoke test with one seed before committing to 3-seed R1

  # full R1 run:
  uv run bake_off.py --prompt-file ../generation-service/scripts/bake_off_prompts/r1_pop_baseline.json

Outputs:
  ../generation-service/bake_off_outputs/diffrhythm/<prompt_name>/
    seed1.mp3
    seed2.mp3
    seed3.mp3
    manifest_<timestamp>.json

On Mac, if MPS fails for an unsupported op, re-run with:
  PYTORCH_ENABLE_MPS_FALLBACK=1 uv run bake_off.py ...
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Import from DR2's own inference module (we are sibling to it).
# This must work because bake_off.py lives at the root of the DR2 clone.
import inference as dr2_inference
from inference import (
    STRUCT_INFO,
    parse_lyrics,
    make_fake_stereo,
    prepare_model,
)


# ----------------------------- Tag transformer -----------------------------

# Map ACE-Step style tags (uppercase) to DiffRhythm 2 tags (lowercase).
# DR2's STRUCT_INFO has: start, end, intro, verse, chorus, outro, inst, solo,
# bridge, hook, break, stop, space. No [pre-chorus] — map to [verse] since
# they're structurally adjacent in nearly all pop songs.
#
# Note: DR2's parse_lyrics() actually skips most tags (only [start] produces a
# token); the rest get `continue`'d. So this transform is mostly cosmetic — for
# manifest cleanliness and to avoid the g2p tokenizer choking on uppercase
# bracketed text mixed into lyric lines. We do it anyway because it's cheap.
ACE_TO_DR2_TAG_MAP = {
    "[intro]": "[intro]",
    "[verse]": "[verse]",
    "[pre-chorus]": "[verse]",   # no DR2 equivalent, fall back to verse
    "[prechorus]": "[verse]",
    "[chorus]": "[chorus]",
    "[bridge]": "[bridge]",
    "[outro]": "[outro]",
    "[hook]": "[hook]",
    "[instrumental]": "[inst]",
    "[inst]": "[inst]",
    "[solo]": "[solo]",
    "[break]": "[break]",
}

STRUCT_LINE_PATTERN = re.compile(r"^\[.*?\]$")


def transform_lyrics_for_dr2(lyrics: str) -> tuple[str, list[str]]:
    """
    Transform ACE-Step uppercase structure tags to DR2 lowercase tags.
    Returns (transformed_lyrics, warnings_list).
    """
    warnings = []
    out_lines = []
    for line in lyrics.split("\n"):
        stripped = line.strip()
        if not stripped:
            out_lines.append(line)
            continue
        if STRUCT_LINE_PATTERN.match(stripped):
            key = stripped.lower()
            mapped = ACE_TO_DR2_TAG_MAP.get(key)
            if mapped is None:
                # Unknown tag — pass through lowercased; DR2 will skip if not in STRUCT_INFO
                warnings.append(f"Unknown tag '{stripped}' — passing through as '{key}'")
                out_lines.append(key)
            else:
                out_lines.append(mapped)
        else:
            out_lines.append(line)
    return "\n".join(out_lines), warnings


# ----------------------------- Diagnostics -----------------------------


def compute_audio_stats(audio_np: np.ndarray, sample_rate: int) -> dict:
    """
    Crest factor + basic stats. Computed on the MONO pre-stereo signal so the
    fake-stereo delay doesn't inflate peak/abs_mean. Matches ACE-Step harness's
    crest factor semantics.
    """
    flat = audio_np.flatten().astype(np.float64)
    peak = float(np.max(np.abs(flat)))
    abs_mean = float(np.mean(np.abs(flat)))
    rms = float(np.sqrt(np.mean(flat ** 2)))
    crest_factor = peak / abs_mean if abs_mean > 1e-12 else float("inf")
    duration_seconds = flat.shape[0] / sample_rate
    return {
        "duration_seconds": round(duration_seconds, 3),
        "sample_rate": sample_rate,
        "peak": round(peak, 6),
        "abs_mean": round(abs_mean, 6),
        "rms": round(rms, 6),
        "crest_factor": round(crest_factor, 3),
    }


def compute_latent_stats(latent: torch.Tensor) -> dict:
    """Stats on the model's pre-decoder latent. Useful for the noise/garbage debug pattern."""
    flat = latent.detach().float().cpu().numpy().flatten()
    return {
        "shape": list(latent.shape),
        "mean": round(float(np.mean(flat)), 6),
        "std": round(float(np.std(flat)), 6),
        "min": round(float(np.min(flat)), 6),
        "max": round(float(np.max(flat)), 6),
    }


# ----------------------------- Device selection -----------------------------


def pick_device() -> torch.device:
    """Prefer MPS on Apple Silicon, then CUDA, then CPU. Matches ACE-Step harness pattern."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def should_use_fp16(device: torch.device) -> bool:
    """
    FP16 only on CUDA. MPS half-precision has known silent-NaN issues in
    several ops (esp. some conv variants and reductions). FP32 on MPS is the
    safe path; we eat the memory cost.
    """
    return device.type == "cuda"


# ----------------------------- Main -----------------------------


def main():
    parser = argparse.ArgumentParser(description="DiffRhythm 2 bake-off harness")
    parser.add_argument(
        "--prompt-file",
        type=str,
        required=True,
        help="Path to bake-off prompt JSON (e.g. ../generation-service/scripts/bake_off_prompts/r1_pop_baseline.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: ../generation-service/bake_off_outputs/diffrhythm/<prompt_name>/)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="ASLP-lab/DiffRhythm2",
        help="HuggingFace repo ID for DR2 weights",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Seeds to run (default: 1 2 3)",
    )
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument(
        "--max-secs",
        type=float,
        default=90.0,
        help="Track length in seconds (default 90, matches R1 pop baseline)",
    )
    parser.add_argument(
        "--no-fake-stereo",
        action="store_true",
        help="Disable DR2's fake-stereo post-processing; output mono",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix appended to output filenames (e.g. '_test'). Use for A/B without clobbering.",
    )
    parser.add_argument(
        "--prompt-key",
        type=str,
        default="diffrhythm",
        help="Key in prompt JSON's 'prompts' dict to use (default: diffrhythm)",
    )
    args = parser.parse_args()

    # ---- Load prompt ----
    prompt_path = Path(args.prompt_file).resolve()
    if not prompt_path.exists():
        print(f"[FATAL] prompt file not found: {prompt_path}")
        sys.exit(1)
    with open(prompt_path) as f:
        prompt_data = json.load(f)

    prompt_name = prompt_data.get("name", prompt_path.stem)
    lyrics_raw = prompt_data.get("lyrics")
    prompts_dict = prompt_data.get("prompts", {})
    style_prompt_text = prompts_dict.get(args.prompt_key)

    if lyrics_raw is None:
        print("[FATAL] prompt JSON has no 'lyrics' field")
        sys.exit(1)
    if style_prompt_text is None:
        print(
            f"[FATAL] prompt JSON has no 'prompts.{args.prompt_key}' field. "
            f"Available keys: {list(prompts_dict.keys())}"
        )
        sys.exit(1)

    # ---- Transform lyrics (ACE-Step caps → DR2 lowercase) ----
    lyrics_transformed, tag_warnings = transform_lyrics_for_dr2(lyrics_raw)
    for w in tag_warnings:
        print(f"[lyric-warn] {w}")

    # ---- Output dir ----
    if args.output_dir is None:
        # Default lives under generation-service (sibling repo) so scoring tools
        # find DR2 outputs in the same root as ACE-Step outputs.
        out_dir = (
            Path(__file__).resolve().parent.parent
            / "generation-service"
            / "bake_off_outputs"
            / "diffrhythm"
            / prompt_name
        )
    else:
        out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output-dir] {out_dir}")

    # ---- Device + dtype ----
    device = pick_device()
    use_fp16 = should_use_fp16(device)
    print(f"[device] {device} | fp16={use_fp16}")

    # ---- Capture DR2 git revision for manifest traceability ----
    try:
        dr2_rev = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).resolve().parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        dr2_rev = "unknown"

    # ---- Load model ----
    print(f"[load] preparing model from {args.repo_id}")
    t_load_start = time.time()
    diffrhythm2, mulan, lrc_tokenizer, decoder = prepare_model(args.repo_id, device)
    # CRITICAL: set the module-level lrc_tokenizer so parse_lyrics() can find it.
    # Upstream relies on the __main__ block's destructured assignment doing this
    # implicitly; since we're not running their main, we set it ourselves.
    dr2_inference.lrc_tokenizer = lrc_tokenizer
    load_seconds = round(time.time() - t_load_start, 2)
    print(f"[load] done in {load_seconds}s")

    # ---- Embed style prompt (text → MuQ-MuLan) ----
    with torch.no_grad():
        style_prompt_embed = mulan(texts=[style_prompt_text])
    style_prompt_embed = style_prompt_embed.to(device).squeeze(0)

    # ---- Tokenize lyrics ----
    lyrics_tokens_nested = parse_lyrics(lyrics_transformed)
    lyrics_token = torch.tensor(
        sum(lyrics_tokens_nested, []), dtype=torch.long, device=device
    )
    print(f"[lyrics] {len(lyrics_token)} tokens after parse")

    # ---- FP16 cast (CUDA only) ----
    if use_fp16:
        diffrhythm2 = diffrhythm2.half()
        decoder = decoder.half()
        style_prompt_embed = style_prompt_embed.half()
        print("[dtype] models cast to fp16")

    # ---- Manifest scaffold ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = {
        "run_id": timestamp,
        "model": "diffrhythm2",
        "model_repo": args.repo_id,
        "model_revision": dr2_rev,
        "harness_version": "1.0",
        "device": str(device),
        "dtype": "fp16" if use_fp16 else "fp32",
        "prompt_file": str(prompt_path),
        "prompt_name": prompt_name,
        "prompt_key": args.prompt_key,
        "style_prompt": style_prompt_text,
        "lyrics_raw": lyrics_raw,
        "lyrics_transformed": lyrics_transformed,
        "lyrics_tag_warnings": tag_warnings,
        "inference_params": {
            "cfg_strength": args.cfg_strength,
            "steps": args.steps,
            "max_secs": args.max_secs,
            "fake_stereo": not args.no_fake_stereo,
        },
        "load_seconds": load_seconds,
        "seeds": [],
    }

    fake_stereo = not args.no_fake_stereo
    sample_rate = decoder.h.sampling_rate

    # ---- Generation loop ----
    # Imported here to defer the pedalboard import until after model load
    # (faster fail-fast on model issues).
    import pedalboard

    for seed in args.seeds:
        print(f"\n[seed {seed}] starting")
        torch.manual_seed(seed)
        np.random.seed(seed)

        t_gen_start = time.time()
        with torch.inference_mode():
            latent = diffrhythm2.sample_block_cache(
                text=lyrics_token.unsqueeze(0),
                duration=int(args.max_secs * 5),  # DR2 uses 5 Hz frame rate
                style_prompt=style_prompt_embed.unsqueeze(0),
                steps=args.steps,
                cfg_strength=args.cfg_strength,
                process_bar=True,
            )
            latent_for_decode = latent.transpose(1, 2)
            audio = decoder.decode_audio(latent_for_decode, overlap=5, chunk_size=20)
        gen_seconds = round(time.time() - t_gen_start, 2)

        # Mono numpy for stats (true crest factor on the model's actual output)
        audio_np_mono = audio.float().cpu().numpy().squeeze()[None, :]
        audio_stats = compute_audio_stats(audio_np_mono, sample_rate)
        latent_stats = compute_latent_stats(latent)

        # Apply fake stereo if requested (DR2's default behavior)
        if fake_stereo:
            audio_for_writing = make_fake_stereo(audio_np_mono, sample_rate)
        else:
            audio_for_writing = audio_np_mono
        num_channels = audio_for_writing.shape[0]

        # Write file
        filename = f"seed{seed}{args.output_suffix}.mp3"
        output_path = out_dir / filename
        with pedalboard.io.AudioFile(
            str(output_path), "w", sample_rate, num_channels
        ) as f:
            f.write(audio_for_writing)

        # Per-seed log line — format matches ACE-Step harness
        print(
            f"[seed {seed}] done in {gen_seconds}s | "
            f"crest_factor={audio_stats['crest_factor']} | "
            f"peak={audio_stats['peak']} | rms={audio_stats['rms']} | "
            f"output={output_path.name}"
        )

        manifest["seeds"].append(
            {
                "seed": seed,
                "output_path": str(output_path),
                "output_filename": filename,
                "generation_time_seconds": gen_seconds,
                "audio_stats": audio_stats,
                "latent_stats": latent_stats,
            }
        )

    # ---- Write manifest ----
    manifest_path = out_dir / f"manifest_{timestamp}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[manifest] written to {manifest_path}")

    # ---- Summary ----
    crest_factors = [s["audio_stats"]["crest_factor"] for s in manifest["seeds"]]
    gen_times = [s["generation_time_seconds"] for s in manifest["seeds"]]
    print(
        f"\n[summary] {len(manifest['seeds'])} seeds | "
        f"crest_range={min(crest_factors):.2f}-{max(crest_factors):.2f} | "
        f"avg_gen_time={sum(gen_times)/len(gen_times):.1f}s"
    )


if __name__ == "__main__":
    main()
