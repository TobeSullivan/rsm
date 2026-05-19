"""Prompt template builder for RSM TTS prototype.

Composes the LM prompt by combining:
- The scenario prompt template (from scenarios/)
- Voice persona parameters (from voices_config.yaml)
- Optional context block (from context/) — only injected in Phase B/C

The output is a single string ready to send to the LM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Project root is two levels up from this file (harness/prompt_template.py -> tts-prototype/)
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass(frozen=True)
class VoiceProfile:
    """Per-voice parameters injected into the prompt template."""
    voice_id: str            # short id, e.g. "altrock_seattle_1997"
    voice_persona: str       # human-readable description, e.g. "alternative-rock DJ in their mid-30s"
    station_descriptor: str  # e.g. "107.7 The End (KNDD), the dominant Seattle alt-rock station"
    date: str                # e.g. "April 9, 1997"
    location: str            # e.g. "Seattle, Washington"
    context_file: str        # filename in context/, e.g. "1997-altrock-seattle.md"
    extra: dict              # scenario-specific extras (e.g. previous_song, next_song)


# Hardcoded voice profiles for the seven test voices.
# These could move to YAML later; keeping inline so the harness is self-contained for v0.
VOICE_PROFILES: dict[str, VoiceProfile] = {
    "altrock_seattle_1997": VoiceProfile(
        voice_id="altrock_seattle_1997",
        voice_persona="alternative-rock DJ in their mid-30s, came up through college radio, knows the local bands personally, has a chip about MTV gentrifying the scene",
        station_descriptor="107.7 The End (KNDD), the dominant Seattle commercial alt-rock station",
        date="April 9, 1997",
        location="Seattle, Washington",
        context_file="1997-altrock-seattle.md",
        extra={
            "between_songs": {
                "previous_song": "'Semi-Charmed Life' by Third Eye Blind",
                "next_song": "'Paranoid Android' by Radiohead",
            },
            "news_break": {
                "event_description": "Soundgarden has just announced their breakup — the band is calling it quits after 13 years",
            },
        },
    ),
    "country_nashville_1997": VoiceProfile(
        voice_id="country_nashville_1997",
        voice_persona="country radio DJ in their late 40s, grew up in middle Tennessee, real Southern cadence, has been on country radio 20 years, knows the artists personally enough to swap stories, sponsor reads second nature",
        station_descriptor="WSIX 97.9 Nashville, 'Nash 97.9'",
        date="October 13, 1997",
        location="Nashville, Tennessee",
        context_file="1997-country-nashville.md",
        extra={
            "between_songs": {
                "previous_song": "'It's Your Love' by Tim McGraw and Faith Hill",
                "next_song": "'Carrying Your Love with Me' by George Strait",
            },
            "news_break": {
                "event_description": "John Denver has died in a plane crash off the California coast yesterday, October 12th",
            },
        },
    ),
    "rap_nyc_1997": VoiceProfile(
        voice_id="rap_nyc_1997",
        voice_persona="hip-hop DJ in their late 20s, Brooklyn-born, comes up out of college radio and the club DJ circuit, has personal relationships with the artists, mourns Biggie hard",
        station_descriptor="Hot 97 (WQHT 97.1), New York's hip-hop and R&B station",
        date="March 25, 1997",
        location="New York City",
        context_file="1997-rap-nyc.md",
        extra={
            "between_songs": {
                "previous_song": "'Hypnotize' by The Notorious B.I.G.",
                "next_song": "'Can't Nobody Hold Me Down' by Puff Daddy featuring Mase",
            },
            "news_break": {
                "event_description": "Today is the release of Notorious B.I.G.'s Life After Death — the album drops two weeks after his murder in Los Angeles",
            },
        },
    ),
    "newsreel_1928": VoiceProfile(
        voice_id="newsreel_1928",
        voice_persona="formal radio newsreel announcer with a mid-Atlantic accent, oratorical and declarative, reading from wire copy",
        station_descriptor="WJZ New York, part of the NBC Blue Network",
        date="May 22, 1927",
        location="New York City",
        context_file="1920s-radio-newsreel.md",
        extra={
            "news_break": {
                "event_description": "Charles Lindbergh has just landed The Spirit of St. Louis in Paris, completing the first solo nonstop transatlantic flight",
            },
        },
    ),
    "fm_rock_1977": VoiceProfile(
        voice_id="fm_rock_1977",
        voice_persona="late-30s FM rock DJ, has been on the air since the underground era of the late 60s, voice is the texture as much as the words, laid-back, hip, plays album sides",
        station_descriptor="WNEW-FM 102.7 New York",
        date="August 17, 1977",
        location="New York City",
        context_file="1970s-fm-rock-dj.md",
        extra={
            "between_songs": {
                "previous_song": "side two of 'Rumours' by Fleetwood Mac",
                "next_song": "'Heroes' by David Bowie — the new one off the album",
            },
            "news_break": {
                "event_description": "Elvis Presley has died at Graceland, August 16th, age 42",
            },
        },
    ),
    "indie_podcast_2017": VoiceProfile(
        voice_id="indie_podcast_2017",
        voice_persona="indie-music podcast host in their mid-30s, conversational, music-critic vocabulary, self-aware about the medium, reads listener emails and Patreon questions",
        station_descriptor="an independent music-criticism podcast in its third year",
        date="June 23, 2017",
        location="Brooklyn, New York",
        context_file="2010s-indie-podcast.md",
        extra={
            "between_songs": {
                "previous_song": "'Cut to the Feeling' by Carly Rae Jepsen",
                "next_song": "'Nobody' by Mitski",
            },
            "news_break": {
                "event_description": "Prodigy of Mobb Deep has died at age 42, complications from sickle cell anemia, just days after performing in Las Vegas",
            },
        },
    ),
    "kpop_mc_2025": VoiceProfile(
        voice_id="kpop_mc_2025",
        voice_persona="bilingual K-pop music show MC in their late 20s, professional warmth, knows all the fandom names by heart, tracks chart positions in real time",
        station_descriptor="M Countdown on Mnet, broadcasting live to global audiences",
        date="November 13, 2025",
        location="Seoul, South Korea",
        context_file="2025-kpop-mc.md",
        extra={
            "between_songs": {
                "previous_song": "'JUMP' by BLACKPINK",
                "next_song": "'NOT CUTE ANYMORE' by ILLIT",
            },
            "news_break": {
                "event_description": "BTS has officially confirmed their full-group comeback for spring 2026 — their first album since Proof in 2022, all seven members having completed military service",
            },
        },
    ),
}


SCENARIOS = [
    "cold_open",
    "between_songs",
    "news_break",
    "local_color",
    "caller_interaction",
    "genre_moment",
]


def load_scenario_template(scenario: str) -> str:
    """Load a scenario prompt template from disk."""
    path = PROJECT_ROOT / "scenarios" / f"{scenario}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Scenario template not found: {path}")
    return path.read_text(encoding="utf-8")


def load_context_block(context_file: str) -> str:
    """Load a context block from disk."""
    path = PROJECT_ROOT / "context" / context_file
    if not path.exists():
        raise FileNotFoundError(f"Context file not found: {path}")
    return path.read_text(encoding="utf-8")


def build_prompt(voice_id: str, scenario: str, phase: str) -> str:
    """Build the full LM prompt for a given (voice, scenario, phase) combo.

    Phases:
      A — ungrounded: no context block injected
      B — RAG-grounded: full context block injected
      C — same as B but caller varies which model is used (downstream of this fn)
    """
    if voice_id not in VOICE_PROFILES:
        raise ValueError(f"Unknown voice_id: {voice_id}")
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    voice = VOICE_PROFILES[voice_id]
    template = load_scenario_template(scenario)

    # Pull the prompt template body out of the markdown — everything below
    # "## Prompt template" up to the next H2 or EOF.
    body = _extract_prompt_body(template)

    # Substitute the standard params
    params = {
        "voice_persona": voice.voice_persona,
        "station_descriptor": voice.station_descriptor,
        "date": voice.date,
        "location": voice.location,
    }

    # Scenario-specific extras
    extras = voice.extra.get(scenario, {})
    params.update(extras)

    # Context block injection
    if phase in ("B", "C"):
        context_block = load_context_block(voice.context_file)
        context_injected = (
            "\n\n# Persona context (use this to ground your patter — names, "
            "venues, vocabulary, cultural moments are all real and era-accurate)\n\n"
            + context_block
        )
    else:
        context_injected = ""
    params["context_block_if_phase_B_else_blank"] = context_injected

    # Naive {placeholder} substitution. If a needed placeholder is missing from extras,
    # leave it as a visible placeholder so the test reveals the gap instead of silently failing.
    out = body
    for key, val in params.items():
        out = out.replace("{" + key + "}", str(val))
    return out


def _extract_prompt_body(template_text: str) -> str:
    """Pull the body of the prompt template out of the markdown file.

    The scenario .txt files use markdown headers. We want everything from
    '## Prompt template' onward, dropping any trailing '## Suggested' sections.
    """
    lines = template_text.splitlines()
    in_body = False
    out_lines: list[str] = []
    for line in lines:
        if line.strip().startswith("## Prompt template"):
            in_body = True
            continue
        if in_body and line.strip().startswith("## "):
            # Hit the next H2 — stop collecting
            break
        if in_body:
            out_lines.append(line)
    return "\n".join(out_lines).strip()


if __name__ == "__main__":
    # Smoke test: print one prompt
    import sys
    voice_id = sys.argv[1] if len(sys.argv) > 1 else "altrock_seattle_1997"
    scenario = sys.argv[2] if len(sys.argv) > 2 else "cold_open"
    phase = sys.argv[3] if len(sys.argv) > 3 else "A"
    print(build_prompt(voice_id, scenario, phase))
