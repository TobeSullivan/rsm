# RSM TTS + LM Coupled Prototype

**This is not a TTS test. It is an architecture test.**

The RSM canon says: local-first, zero ongoing infrastructure cost, dynamic DJ patter generated on the player's machine. That entire architecture rests on a 7-8B-class LM running on a minimum-spec gaming PC producing *era-authentic, genre-authentic, scene-fluent* DJ patter, then a local TTS speaking that patter in a cloned era-appropriate voice with sub-200ms latency.

If the LM can't write the patter, voice quality doesn't matter — the patter itself is the failure. If the TTS can't speak it well, patter quality doesn't matter — players hear a robot. **Both must clear the bar or the architecture fails.** This prototype tests them as a coupled system.

## What this answers

1. Can a 7-8B-class local LM, with RAG context from a decade-pack scene cell, generate DJ patter that's era-authentic, genre-authentic, and scene-fluent for the 1990-2000 demo era?
2. Can Chatterbox (Resemble AI, MIT) clone era-appropriate DJ voices and speak generated patter with usable fidelity, naturalness, and latency?
3. Does it run acceptably on the Windows 4070 (8GB VRAM) — i.e., does it survive minimum-spec gaming PC, which is the console-floor proxy?

## Test phases

The phases are gates. Each phase's outcome determines whether to proceed, fall back, or change the architecture.

### Phase A — LM ungrounded
Can a 7-8B model write the patter from base knowledge alone, no RAG?
- **Pass** → architecture works, ship it.
- **Fail** → proceed to B.

### Phase B — LM with RAG grounding
Same 7-8B model, but with a decade-pack scene cell injected as context.
- **Pass** → architecture works via the RAG-driven decade-pack pattern already planned. Ship.
- **Fail** → proceed to C.

### Phase C — Bigger model with RAG
14B-class with RAG. Acceptable cost: lose console (probably). Console becomes a v2+ port.
- **Pass** → decision point: PC-only launch acceptable?
- **Fail** → proceed to D.

### Phase D — Authored templates with LM variation
Pre-write patter skeletons per scenario per persona; LM fills variable slots only. Living-world promise gets quieter. Defensive fallback.

### Stretch — 4B-class probe
If Phase A or B passes, run the same test against a 4B model (Phi-3.5, Llama-3.2-3B, Qwen2.5-3B) to see if Series-S-class console is reachable.

## Test substrate

Seven DJ voices spanning eras and genres:

| Voice | Era | Genre / Format |
|---|---|---|
| 1920s radio newsreel | 1920s | News, public domain |
| 1970s FM rock DJ | 1970s | FM rock |
| 1990s alt-rock DJ | 1997 | Alternative / Seattle scene |
| 1990s country DJ | 1997 | Country / Nashville |
| 1990s hip-hop mixshow host | 1997 | Hip-hop / NYC |
| 2010s indie podcast | 2010s | Indie / podcast-flavored |
| 2025 K-pop MC | 2025 | K-pop |

Five of seven are inside the demo era. The 1920s and 2010s are range stress.

Six scenarios per voice = 42 LM generations to score per phase:

1. **Cold open** — DJ takes the air, generic patter to start the show
2. **Between songs** — coming out of a real (in-era) track
3. **News break** — era-appropriate breaking news (Cobain dies 4/5/94, Biggie shot 3/9/97, etc.)
4. **Local color** — name-drop a scene, venue, or regional band
5. **Caller interaction** — DJ reacts to a hypothetical call-in
6. **Genre moment** — country state fair plug, rap mixshow hype, K-pop comeback announcement, etc.

## Scoring

Two rubrics, both 1-5 with explicit level definitions (see `rubric.md`):

**LM patter text:**
- Era authenticity
- Genre authenticity
- Scene fluency
- Voice consistency (does the same DJ persona hold across multiple lines?)
- No-hallucination (does it make up bands/events/venues that didn't exist?)

**TTS audio output:**
- Voice fidelity (does it sound like the reference clone?)
- Era-appropriateness (would this voice fit on a station from that era?)
- Latency (wall-clock to generate ~10s of audio)
- Emotion expressiveness (does the emotion knob actually do something?)
- Naturalness (human or robot?)

## Pass bar

Soft target: average ≥4.0 across all axes for the 1997 voices on Phase A or B. The bookend voices (1920s, 2010s, K-pop) are diagnostic, not gating — they tell us where the model's edges are.

Hard fail: any axis averages <3.0 on the 1997 voices across phases A through C. That's the architecture-needs-to-change signal.

## What's in this repo

```
tts-prototype/
├── README.md                          # this file
├── rubric.md                          # scoring rubric with level definitions
├── context/                           # decade-pack-style RAG context blocks
│   ├── 1997-altrock-seattle.md
│   ├── 1997-country-nashville.md
│   ├── 1997-rap-nyc.md
│   ├── 1920s-radio-newsreel.md
│   ├── 2010s-indie-podcast.md
│   └── 2025-kpop-mc.md
├── scenarios/                         # the six scenario prompt templates
│   ├── cold_open.txt
│   ├── between_songs.txt
│   ├── news_break.txt
│   ├── local_color.txt
│   ├── caller_interaction.txt
│   └── genre_moment.txt
├── harness/                           # install + run scripts
│   ├── install.ps1                    # Windows install
│   ├── pull_models.ps1                # Ollama model pulls
│   ├── generate_patter.py             # LM patter generation
│   ├── generate_tts.py                # Chatterbox TTS generation
│   ├── latency_log.py                 # shared latency logging
│   └── prompt_template.py             # master patter prompt builder
├── voices/                            # reference audio clips (you populate)
│   └── README.md                      # sourcing guide
├── patter/                            # generated patter text outputs (gitignored)
├── generated/                         # generated audio (gitignored)
└── results/                           # scored results (your output)
    └── _template.md                   # results scoring template per voice
```

## How to run (high level)

From a Windows PowerShell prompt at `C:\dev\rsm\tts-prototype\`:

```powershell
# One-time setup
.\harness\install.ps1
.\harness\pull_models.ps1

# Populate voices/ with reference audio per the sourcing guide

# Phase A: ungrounded LM patter
python .\harness\generate_patter.py --phase A --model qwen2.5:7b

# Phase B: RAG-grounded LM patter
python .\harness\generate_patter.py --phase B --model qwen2.5:7b

# Generate TTS for both phases' patter
python .\harness\generate_tts.py --input .\patter\phase_a\
python .\harness\generate_tts.py --input .\patter\phase_b\

# Listen, score, write to .\results\{voice}.md
```

## Hardware

- **Windows 4070 laptop, 8GB VRAM** — primary dev surface per the RSM hardware rule. This is where we test. If it passes here, it passes the gaming-PC minimum spec.
- **M4 Max** — used only when speed or model size demands. Larger-model probes (14B with comfort, 70B for upper-bound sanity checks) run here.

## Reference audio sourcing

See `voices/README.md` for the sourcing guide. Public domain or your-own-voice only. No copyrighted DJ recordings — even for testing.

## Status

Initial scaffold. Reference audio not yet populated. Tests not yet run.
