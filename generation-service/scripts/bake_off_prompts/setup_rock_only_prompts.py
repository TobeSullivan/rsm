#!/usr/bin/env python3
"""
Rock-only R2 prompt materialization for XL-Turbo stress test.

Schema mirrors setup_r2_prompts.py (validated 2026-05-17).
Genre-only entries (punk, classic_rock) get R2A only; named artists get both.

Run from generation-service/scripts/bake_off_prompts/:
    python setup_rock_only_prompts.py
or from generation-service/:
    uv run scripts/bake_off_prompts/setup_rock_only_prompts.py

Creates 14 files: 8 rock_r2a_<slug>.json + 6 rock_r2b_<slug>.json.

All entries share one rock-flavored lyrics block — this is a style-prompt
stress test, not a per-artist authenticity test. If a song-by-song authenticity
pass is later wanted, swap SHARED_LYRICS for per-entry custom lyrics like
setup_r2_prompts.py does.
"""

import json
from pathlib import Path

DATE_LOCKED = "2026-05-17"

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

# artist_name=None => genre-only entry, no R2B written
ENTRIES = [
    {
        "id": "punk",
        "display": "Hardcore Punk",
        "genre_summary": "Hardcore punk",
        "difficulty": "Easy",
        "bpm": 180,
        "caption_fingerprint": (
            "Hardcore punk, 180 BPM, raw and aggressive, fast distorted guitars "
            "and frantic drums, shouted male vocals, lo-fi gritty mix"
        ),
        "artist_name": None,
    },
    {
        "id": "classic_rock",
        "display": "Classic Rock",
        "genre_summary": "Classic rock",
        "difficulty": "Easy",
        "bpm": 115,
        "caption_fingerprint": (
            "Classic rock, 115 BPM, anthemic and powerful, crunchy electric "
            "guitar riff with Hammond organ, soaring male rock vocals, warm "
            "analog production"
        ),
        "artist_name": None,
    },
    {
        "id": "tool",
        "display": "Tool",
        "genre_summary": "Progressive metal",
        "difficulty": "Hard",
        "bpm": 132,
        "caption_fingerprint": (
            "Progressive metal, 132 BPM, dark and atmospheric, drop-D distorted "
            "guitar and complex polyrhythmic drums, melodic baritone male vocals, "
            "expansive cinematic production"
        ),
        "artist_name": "Tool",
    },
    {
        "id": "pantera",
        "display": "Pantera",
        "genre_summary": "Groove metal",
        "difficulty": "Hard",
        "bpm": 120,
        "caption_fingerprint": (
            "Groove metal, 120 BPM, aggressive and heavy, downtuned crunching "
            "guitar and syncopated double bass drums, alternating shouted and "
            "sung male vocals, tight modern metal production"
        ),
        "artist_name": "Pantera",
    },
    {
        "id": "metallica",
        "display": "Metallica",
        "genre_summary": "Thrash metal",
        "difficulty": "Hard",
        "bpm": 180,
        "caption_fingerprint": (
            "Thrash metal, 180 BPM, intense and driving, palm-muted distorted "
            "guitar with melodic leads and fast double bass drums, melodic "
            "shouted male vocals, full-range polished production"
        ),
        "artist_name": "Metallica",
    },
    {
        "id": "acid_bath",
        "display": "Acid Bath",
        "genre_summary": "Sludge metal",
        "difficulty": "Very Hard",
        "bpm": 80,
        "caption_fingerprint": (
            "Sludge metal, 80 BPM, swampy and ominous, heavily distorted slow "
            "downtuned guitar and doomy bass, alternating clean and harsh male "
            "vocals, lo-fi murky Southern production"
        ),
        "artist_name": "Acid Bath",
    },
    {
        "id": "led_zeppelin",
        "display": "Led Zeppelin",
        "genre_summary": "Classic hard rock",
        "difficulty": "Medium",
        "bpm": 110,
        "caption_fingerprint": (
            "Classic hard rock, 110 BPM, bluesy and powerful, electric guitar "
            "with wah and slide, hard-hitting drums, soaring male vocals with "
            "falsetto wails, warm analog 1970s production"
        ),
        "artist_name": "Led Zeppelin",
    },
    {
        "id": "beatles",
        "display": "The Beatles",
        "genre_summary": "1960s rock and roll",
        "difficulty": "Medium",
        "bpm": 120,
        "caption_fingerprint": (
            "1960s rock and roll, 120 BPM, melodic and bright, jangly electric "
            "guitars and tight snappy drums, harmonized male vocal trio, vintage "
            "mid-fidelity mix"
        ),
        "artist_name": "The Beatles",
    },
]


def build_file(entry: dict, test_type: str) -> dict:
    """test_type: 'fingerprint' (R2A, 3 seeds) or 'name' (R2B, 1 seed)."""
    if test_type == "fingerprint":
        round_id = f"rock_r2a_{entry['id']}"
        caption = entry["caption_fingerprint"]
        seeds = [1, 2, 3]
        description = (
            f"Rock R2A sonic fingerprint: {entry['display']} "
            f"({entry['genre_summary']}). No artist name in caption. 3 seeds. "
            f"Locked {DATE_LOCKED}."
        )
        output_subdir = f"rock-r2a-{entry['id']}"
        test_type_meta = "sonic_fingerprint"
    elif test_type == "name":
        round_id = f"rock_r2b_{entry['id']}"
        caption = (
            f"{entry['caption_fingerprint']}, in the style of {entry['artist_name']}"
        )
        seeds = [1]
        description = (
            f"Rock R2B artist-name diagnostic: {entry['display']} "
            f"({entry['genre_summary']}). Artist named in caption. 1 seed. "
            f"Locked {DATE_LOCKED}."
        )
        output_subdir = f"rock-r2b-{entry['id']}"
        test_type_meta = "artist_name"
    else:
        raise ValueError(f"unknown test_type: {test_type}")

    return {
        "_meta": {
            "round": round_id,
            "description": description,
            "output_subdir": output_subdir,
            "test_type": test_type_meta,
            "artist_display": entry["display"],
            "artist_genre": entry["genre_summary"],
            "artist_difficulty": entry["difficulty"],
        },
        "captions": {
            "ace_step": caption,
            "diffrhythm": "",
            "songgeneration": "",
            "suno": "",
        },
        "lyrics": SHARED_LYRICS,
        "generation": {
            "bpm": entry["bpm"],
            "duration_seconds": 90.0,
            "vocal_language": "en",
            "instrumental": False,
            "keyscale": "",
            "seeds": seeds,
        },
    }


def main():
    out_dir = Path(__file__).resolve().parent
    written = []
    for entry in ENTRIES:
        # R2A always
        data = build_file(entry, "fingerprint")
        filename = f"rock_r2a_{entry['id']}.json"
        with open(out_dir / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        written.append(filename)

        # R2B only for entries with a named artist
        if entry["artist_name"] is not None:
            data = build_file(entry, "name")
            filename = f"rock_r2b_{entry['id']}.json"
            with open(out_dir / filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            written.append(filename)

    print(f"Wrote {len(written)} files to {out_dir}:")
    print()
    print("R2A (sonic fingerprint, 3 seeds each):")
    for f in sorted(written):
        if f.startswith("rock_r2a_"):
            print(f"  {f}")
    print()
    print("R2B (artist-name diagnostic, 1 seed each):")
    for f in sorted(written):
        if f.startswith("rock_r2b_"):
            print(f"  {f}")


if __name__ == "__main__":
    main()
