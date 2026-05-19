"""TTS generator for RSM TTS prototype.

Takes a directory of patter text outputs (from generate_patter.py) and produces
WAV files using Chatterbox TTS, cloning voices from reference clips.

Usage:
    python generate_tts.py --input patter\phase_a_qwen2.5_7b --voices-dir voices

Output layout:
    generated/
        phase_a_qwen2.5_7b/
            altrock_seattle_1997/
                cold_open.wav
                between_songs.wav
                ...

Reference voice clips must exist under voices/<voice_id>/reference.wav.
See voices/README.md for sourcing.

Emotion stress test:
    python generate_tts.py --emotion-stress
This generates the same fixed line in three emotional registers (deadpan / excited / somber)
for each voice to test Chatterbox's emotion exaggeration knob.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Chatterbox API surface — verify against the installed package before assuming.
# This is the documented usage pattern from Chatterbox README as of mid-2025.
# If the import or call signature has changed, the harness will fail loudly
# and tell you to update it. Per project rules: verify the API surface before calling it.
try:
    from chatterbox.tts import ChatterboxTTS  # type: ignore
except ImportError as e:
    raise ImportError(
        "chatterbox-tts not installed or import path changed. "
        "Run install.ps1 first. If install succeeded, check the chatterbox README — "
        "the import path may have moved."
    ) from e

import soundfile as sf  # type: ignore

from prompt_template import PROJECT_ROOT, VOICE_PROFILES


# Lines for the emotion stress test — same content, three emotional intents
EMOTION_STRESS_LINE = (
    "And we're back with another hour of music. "
    "Coming up next, a song that means a lot to a lot of people in this town."
)
EMOTION_STRESS_INTENTS = ["deadpan", "excited", "somber"]


def load_chatterbox(device: str = "cuda") -> object:
    """Load the Chatterbox TTS model. Device should be 'cuda' or 'cpu'."""
    print(f"Loading Chatterbox on {device}...")
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device=device)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return model


def synthesize(
    model: object,
    text: str,
    reference_audio: Path,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
) -> tuple["object", int, float]:
    """Generate one TTS clip. Returns (audio_array, sample_rate, latency_seconds).

    exaggeration: 0.0-1.0+, Chatterbox's emotion exaggeration knob.
    cfg_weight: classifier-free guidance strength.
    """
    t0 = time.time()
    wav = model.generate(  # type: ignore[attr-defined]
        text,
        audio_prompt_path=str(reference_audio),
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )
    latency = time.time() - t0
    # ChatterboxTTS returns a torch tensor; convert to numpy for soundfile
    if hasattr(wav, "cpu"):
        wav = wav.cpu().numpy()
    if wav.ndim > 1:
        wav = wav.squeeze()
    sample_rate = getattr(model, "sr", 24000)
    return wav, sample_rate, latency


def _safe_model_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def run_patter_tts(
    patter_dir: Path,
    voices_dir: Path,
    device: str = "cuda",
    exaggeration: float = 0.5,
) -> None:
    """Run TTS over all patter outputs in patter_dir."""
    if not patter_dir.exists():
        raise FileNotFoundError(f"Patter dir not found: {patter_dir}")

    out_root = PROJECT_ROOT / "generated" / patter_dir.name
    out_root.mkdir(parents=True, exist_ok=True)

    model = load_chatterbox(device=device)

    latencies: list[dict] = []
    voice_dirs = sorted([d for d in patter_dir.iterdir() if d.is_dir()])
    total = sum(len(list(vd.glob("*.txt"))) for vd in voice_dirs)
    done = 0

    for voice_dir in voice_dirs:
        voice_id = voice_dir.name
        if voice_id not in VOICE_PROFILES:
            print(f"  Skipping unknown voice_id directory: {voice_id}")
            continue

        reference = voices_dir / voice_id / "reference.wav"
        if not reference.exists():
            print(f"  Skipping {voice_id} — no reference.wav at {reference}")
            continue

        voice_out = out_root / voice_id
        voice_out.mkdir(parents=True, exist_ok=True)

        for patter_file in sorted(voice_dir.glob("*.txt")):
            done += 1
            scenario = patter_file.stem
            text = patter_file.read_text(encoding="utf-8")
            print(f"[{done}/{total}] {voice_id} :: {scenario}...", end=" ", flush=True)
            try:
                wav, sr, latency = synthesize(model, text, reference, exaggeration=exaggeration)
                out_path = voice_out / f"{scenario}.wav"
                sf.write(out_path, wav, sr)
                latencies.append({
                    "voice_id": voice_id,
                    "scenario": scenario,
                    "latency_seconds": round(latency, 3),
                    "char_count": len(text),
                    "exaggeration": exaggeration,
                })
                print(f"{latency:.2f}s")
            except Exception as e:  # noqa: BLE001
                print(f"FAILED: {e}")

    log_path = out_root / "_tts_latency.json"
    log_path.write_text(json.dumps(latencies, indent=2), encoding="utf-8")
    print()
    print(f"TTS latency log: {log_path}")
    if latencies:
        avg = sum(x["latency_seconds"] for x in latencies) / len(latencies)
        print(f"Average TTS latency: {avg:.2f}s per clip")


def run_emotion_stress(
    voices_dir: Path,
    device: str = "cuda",
) -> None:
    """Generate the same line in three emotional registers per voice."""
    out_root = PROJECT_ROOT / "generated" / "emotion_stress"
    out_root.mkdir(parents=True, exist_ok=True)

    model = load_chatterbox(device=device)

    # Map intent to a Chatterbox exaggeration value.
    # These are starting points; the test exists to figure out what values produce
    # which emotional registers cleanly.
    intent_to_exag = {
        "deadpan": 0.2,
        "excited": 0.9,
        "somber": 0.4,
    }

    latencies: list[dict] = []
    for voice_id in VOICE_PROFILES:
        reference = voices_dir / voice_id / "reference.wav"
        if not reference.exists():
            print(f"  Skipping {voice_id} — no reference.wav at {reference}")
            continue
        voice_out = out_root / voice_id
        voice_out.mkdir(parents=True, exist_ok=True)
        for intent in EMOTION_STRESS_INTENTS:
            exag = intent_to_exag[intent]
            print(f"  {voice_id} :: {intent} (exag={exag})...", end=" ", flush=True)
            try:
                wav, sr, latency = synthesize(model, EMOTION_STRESS_LINE, reference, exaggeration=exag)
                out_path = voice_out / f"{intent}.wav"
                sf.write(out_path, wav, sr)
                latencies.append({
                    "voice_id": voice_id,
                    "intent": intent,
                    "exaggeration": exag,
                    "latency_seconds": round(latency, 3),
                })
                print(f"{latency:.2f}s")
            except Exception as e:  # noqa: BLE001
                print(f"FAILED: {e}")

    log_path = out_root / "_emotion_latency.json"
    log_path.write_text(json.dumps(latencies, indent=2), encoding="utf-8")
    print()
    print(f"Emotion-stress log: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TTS audio for RSM prototype.")
    parser.add_argument("--input", type=str,
                        help="Patter directory to process (e.g. patter\\phase_a_qwen2.5_7b)")
    parser.add_argument("--voices-dir", type=str, default="voices",
                        help="Directory of reference voice clips")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu", "mps"],
                        help="Inference device. mps for Mac M-series.")
    parser.add_argument("--exaggeration", type=float, default=0.5,
                        help="Chatterbox emotion exaggeration (0.0-1.0+). Default 0.5.")
    parser.add_argument("--emotion-stress", action="store_true",
                        help="Run emotion stress test instead of full patter run")
    args = parser.parse_args()

    voices_dir = (PROJECT_ROOT / args.voices_dir).resolve()

    if args.emotion_stress:
        run_emotion_stress(voices_dir, device=args.device)
        return

    if not args.input:
        parser.error("--input is required unless --emotion-stress")

    patter_dir = Path(args.input).resolve()
    run_patter_tts(patter_dir, voices_dir, device=args.device, exaggeration=args.exaggeration)


if __name__ == "__main__":
    main()
