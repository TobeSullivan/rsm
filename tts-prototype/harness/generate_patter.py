"""Patter generator for RSM TTS prototype.

Runs all configured voices x scenarios through the LM (via Ollama), with or without
RAG context, and writes outputs to patter/<phase>/<voice>/<scenario>.txt.

Usage:
    python generate_patter.py --phase A --model qwen2.5:7b
    python generate_patter.py --phase B --model qwen2.5:7b
    python generate_patter.py --phase B --model qwen2.5:14b   # Phase C is just a bigger model in B mode
    python generate_patter.py --phase B --model llama3.2:3b   # Console floor probe

Output layout:
    patter/
        phase_a_qwen2.5_7b/
            altrock_seattle_1997/
                cold_open.txt
                between_songs.txt
                ...
            country_nashville_1997/
                ...
        phase_b_qwen2.5_7b/
            ...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import ollama  # type: ignore

from prompt_template import (
    PROJECT_ROOT,
    SCENARIOS,
    VOICE_PROFILES,
    build_prompt,
)


def _safe_model_name(model: str) -> str:
    """Make a model name safe for use in a directory name."""
    return model.replace(":", "_").replace("/", "_")


def generate_one(
    voice_id: str,
    scenario: str,
    phase: str,
    model: str,
    temperature: float = 0.7,
) -> tuple[str, float]:
    """Generate patter for one (voice, scenario) and return (text, latency_seconds)."""
    prompt = build_prompt(voice_id, scenario, phase)

    t0 = time.time()
    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={"temperature": temperature},
    )
    latency = time.time() - t0

    # ollama-python returns a dict-like; the generated text is under "response"
    text = response["response"].strip() if isinstance(response, dict) else str(response).strip()
    return text, latency


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LM patter for RSM TTS prototype.")
    parser.add_argument("--phase", required=True, choices=["A", "B", "C"],
                        help="Phase: A=ungrounded, B=RAG, C=same as B with bigger model")
    parser.add_argument("--model", required=True,
                        help="Ollama model name (e.g. qwen2.5:7b, llama3.1:8b, llama3.2:3b)")
    parser.add_argument("--voices", nargs="*",
                        help="Subset of voice_ids to run (default: all)")
    parser.add_argument("--scenarios", nargs="*",
                        help="Subset of scenarios to run (default: all)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="LM sampling temperature")
    args = parser.parse_args()

    voices_to_run = args.voices if args.voices else list(VOICE_PROFILES.keys())
    scenarios_to_run = args.scenarios if args.scenarios else SCENARIOS

    out_root = PROJECT_ROOT / "patter" / f"phase_{args.phase.lower()}_{_safe_model_name(args.model)}"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Output root: {out_root}")
    print(f"Voices: {voices_to_run}")
    print(f"Scenarios: {scenarios_to_run}")
    print(f"Model: {args.model}")
    print(f"Phase: {args.phase}")
    print(f"Temperature: {args.temperature}")
    print()

    latencies: list[dict] = []
    total = len(voices_to_run) * len(scenarios_to_run)
    done = 0

    for voice_id in voices_to_run:
        voice_dir = out_root / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)

        for scenario in scenarios_to_run:
            done += 1
            print(f"[{done}/{total}] {voice_id} :: {scenario}...", end=" ", flush=True)
            try:
                text, latency = generate_one(
                    voice_id, scenario, args.phase, args.model,
                    temperature=args.temperature,
                )
            except Exception as e:  # noqa: BLE001
                print(f"FAILED: {e}")
                continue

            # Write the patter
            out_path = voice_dir / f"{scenario}.txt"
            out_path.write_text(text, encoding="utf-8")

            latencies.append({
                "voice_id": voice_id,
                "scenario": scenario,
                "latency_seconds": round(latency, 3),
                "char_count": len(text),
            })
            print(f"{latency:.2f}s ({len(text)} chars)")

    # Write a latency log
    log_path = out_root / "_latency.json"
    log_path.write_text(json.dumps(latencies, indent=2), encoding="utf-8")
    print()
    print(f"Latency log: {log_path}")

    # Summary
    if latencies:
        avg_lat = sum(x["latency_seconds"] for x in latencies) / len(latencies)
        avg_chars = sum(x["char_count"] for x in latencies) / len(latencies)
        print(f"Average latency: {avg_lat:.2f}s per generation")
        print(f"Average length: {avg_chars:.0f} chars per generation")

    print()
    print("Done. Next:")
    print(f"  Listen to / read the outputs in {out_root}")
    print("  Then run TTS: python harness\\generate_tts.py --input {out_root}")


if __name__ == "__main__":
    main()
