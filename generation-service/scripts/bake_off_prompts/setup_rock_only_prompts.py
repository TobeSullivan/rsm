#!/usr/bin/env python3
"""
Rock-only R2 prompt materialization for XL-Turbo bake-off.

Generates rock-focused R2A (sonic fingerprint) and R2B (artist-name diagnostic)
JSON files. Single source of truth: re-run this to regenerate all rock prompt JSONs.

Output layout (alongside this script):
  rock_r2a_<slug>.json   (3 seeds, no artist name in caption)
  rock_r2b_<slug>.json   (1 seed,  "in the style of X" appended)

Genre-only entries (punk, classic_rock) get R2A only.

Run from anywhere:
    uv run scripts/bake_off_prompts/setup_rock_only_prompts.py
"""

import json
from pathlib import Path

OUT_DIR = Path(__file__).parent
SEEDS_R2A = [1, 2, 3]
SEEDS_R2B = [1]

SHARED_LYRICS = """[Verse]
Lights flicker low in the empty hall
Memory of a name I can't recall
Walking out with nothing left to lose
Take it or leave it, I'm gonna choose

[Chorus]
Tear it down, watch it fall
Nothing's gonna last at all
Burn the page and start again
Same old story but different end

[Verse]
Sirens calling in the dead of night
Shadows dancing in the broken light
Took a wrong turn somewhere down the line
Lost my way but I'm doing fine

[Chorus]
Tear it down, watch it fall
Nothing's gonna last at all
Burn the page and start again
Same old story but different end"""

# (slug, display_name, sonic_fingerprint_caption, artist_name_or_None)
# artist_name=None => genre-only entry, no R2B file written
ENTRIES = [
    (
        "punk",
        "Hardcore Punk",
        "Hardcore punk, 180 BPM, raw and aggressive, fast distorted guitars and frantic drums, shouted male vocals, lo-fi gritty mix",
        None,
    ),
    (
        "classic_rock",
        "Classic Rock",
        "Classic rock, 115 BPM, anthemic and powerful, crunchy electric guitar riff with Hammond organ, soaring male rock vocals, warm analog production",
        None,
    ),
    (
        "tool",
        "Tool",
        "Progressive metal, 132 BPM, dark and atmospheric, drop-D distorted guitar and complex polyrhythmic drums, melodic baritone male vocals, expansive cinematic production",
        "Tool",
    ),
    (
        "pantera",
        "Pantera",
        "Groove metal, 120 BPM, aggressive and heavy, downtuned crunching guitar and syncopated double bass drums, alternating shouted and sung male vocals, tight modern metal production",
        "Pantera",
    ),
    (
        "metallica",
        "Metallica",
        "Thrash metal, 180 BPM, intense and driving, palm-muted distorted guitar with melodic leads and fast double bass drums, melodic shouted male vocals, full-range polished production",
        "Metallica",
    ),
    (
        "acid_bath",
        "Acid Bath",
        "Sludge metal, 80 BPM, swampy and ominous, heavily distorted slow downtuned guitar and doomy bass, alternating clean and harsh male vocals, lo-fi murky Southern production",
        "Acid Bath",
    ),
    (
        "led_zeppelin",
        "Led Zeppelin",
        "Classic hard rock, 110 BPM, bluesy and powerful, electric guitar with wah and slide, hard-hitting drums, soaring male vocals with falsetto wails, warm analog 1970s production",
        "Led Zeppelin",
    ),
    (
        "beatles",
        "The Beatles",
        "1960s rock and roll, 120 BPM, melodic and bright, jangly electric guitars and tight snappy drums, harmonized male vocal trio, vintage mid-fidelity mix",
        "The Beatles",
    ),
]


def make_prompt(name, caption, seeds):
    """Schema matches r1_pop_baseline.json. ACE-Step is the only candidate left;
    other model variant keys stay empty for future-proofing."""
    return {
        "name": name,
        "captions": {
            "ace_step": caption,
            "diffrhythm": "",
            "songgen": "",
            "suno": "",
        },
        "lyrics": SHARED_LYRICS,
        "seeds": seeds,
    }


def main():
    written = []
    for slug, display, fingerprint_caption, artist_name in ENTRIES:
        # R2A: sonic fingerprint, no artist name in caption
        r2a_path = OUT_DIR / f"rock_r2a_{slug}.json"
        r2a = make_prompt(
            name=f"{display} - Sonic Fingerprint",
            caption=fingerprint_caption,
            seeds=SEEDS_R2A,
        )
        r2a_path.write_text(json.dumps(r2a, indent=2))
        written.append(r2a_path.name)

        # R2B: artist-name diagnostic (skip for genre-only entries)
        if artist_name is not None:
            r2b_path = OUT_DIR / f"rock_r2b_{slug}.json"
            r2b_caption = f"{fingerprint_caption}, in the style of {artist_name}"
            r2b = make_prompt(
                name=f"{display} - Artist Name Diagnostic",
                caption=r2b_caption,
                seeds=SEEDS_R2B,
            )
            r2b_path.write_text(json.dumps(r2b, indent=2))
            written.append(r2b_path.name)

    print(f"Wrote {len(written)} prompt files to {OUT_DIR}:")
    for name in sorted(written):
        print(f"  {name}")


if __name__ == "__main__":
    main()
