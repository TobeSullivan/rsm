#!/usr/bin/env python3
"""
Materialize 14 R2 prompt JSON files for ACE-Step Turbo overnight run.

Run from generation-service/scripts/bake_off_prompts/:
    python setup_r2_prompts.py

Creates 14 files:
    r2a_<artist>.json — sonic fingerprint test (3 seeds, no artist name in caption)
    r2b_<artist>.json — artist-name diagnostic (1 seed, "in the style of X" in caption)

Schema matches r1_pop_baseline.json. Only captions.ace_step is filled in;
other model variants left empty since only ACE-Step is still in the bake-off.
"""

import json
from pathlib import Path

DATE_LOCKED = "2026-05-17"

ARTISTS = [
    {
        "id": "luke_combs",
        "display": "Luke Combs",
        "genre_summary": "Modern country",
        "difficulty": "Easy",
        "bpm": 88,
        "vocal_language": "en",
        "instrumental": False,
        "caption_fingerprint": (
            "Modern country, 88 BPM, mid-tempo radio-friendly arrangement, "
            "strummed acoustic guitar, clean electric guitar with mild compression, "
            "pedal steel guitar fills, fiddle accents, simple kick-and-snare with "
            "brushed touches, male baritone lead vocal with slight Southern rasp, "
            "nostalgic and sincere mood, polished Nashville studio production, "
            "warm bass, present vocal mix"
        ),
        "caption_name": (
            "Modern country song in the style of Luke Combs, 88 BPM, mid-tempo "
            "radio-friendly arrangement, strummed acoustic guitar, pedal steel, "
            "fiddle, male baritone vocal with Southern rasp, nostalgic sincere "
            "mood, polished Nashville production"
        ),
        "lyrics": (
            "[Intro]\n\n"
            "[Verse]\n"
            "Tail lights fading down the county road\n"
            "Pickup empty where she used to be\n"
            "Mama's wedding ring still on the dresser\n"
            "Heaven sometimes looks a lot like leaving\n\n"
            "[Pre-Chorus]\n"
            "And I keep telling everyone I'm alright\n"
            "Telling everybody time will heal\n\n"
            "[Chorus]\n"
            "But the front porch swing won't swing the same\n"
            "The kitchen radio knows her name\n"
            "I built this house with both our hands\n"
            "Now I'm sleeping in the wreckage of our plans\n"
            "She took the dog, she took the years\n"
            "She left me everything I can't bear\n\n"
            "[Verse]\n"
            "Sister called and asked if I'd been eating\n"
            "I told her yeah, then I poured another drink\n"
            "Pastor said come Sunday I should join him\n"
            "I been singing to myself a little less each week\n\n"
            "[Chorus]\n"
            "But the front porch swing won't swing the same\n"
            "The kitchen radio knows her name\n"
            "I built this house with both our hands\n"
            "Now I'm sleeping in the wreckage of our plans"
        ),
    },
    {
        "id": "usher",
        "display": "Usher",
        "genre_summary": "Contemporary R&B",
        "difficulty": "Easy-Medium",
        "bpm": 92,
        "vocal_language": "en",
        "instrumental": False,
        "caption_fingerprint": (
            "Contemporary R&B, 92 BPM, smooth slow groove, lush layered synth pads, "
            "syncopated hip-hop drum programming with crisp snares, deep sub-bass "
            "with rounded 808 hits, finger snap percussion, male tenor lead vocal "
            "with melismatic runs and falsetto adlibs, stacked background vocal "
            "harmonies, sensual yearning mood, polished urban production with "
            "vocal sheen and reverb tail"
        ),
        "caption_name": (
            "Contemporary R&B song in the style of Usher, 92 BPM, smooth slow groove, "
            "layered synth pads, hip-hop drum programming, 808 sub-bass, male tenor "
            "with melismatic runs and falsetto, stacked harmonies, sensual yearning "
            "mood, polished urban production"
        ),
        "lyrics": (
            "[Intro]\n\n"
            "[Verse]\n"
            "Curtains still drawn when the sun came up\n"
            "You were tracing letters on my chest\n"
            "Saying something I was too tired to answer\n"
            "Now I'm playing it back in my head\n\n"
            "[Pre-Chorus]\n"
            "And I should call but I keep waiting\n"
            "For something I can't put into words\n\n"
            "[Chorus]\n"
            "You're the one I left for nothing\n"
            "You're the one I lost for nothing\n"
            "I had everything I wanted\n"
            "And I traded it for nothing at all\n"
            "You're the silence in the morning\n"
            "You're the question in the evening\n"
            "I had everything I wanted\n"
            "Yeah I lost it all for nothing at all\n\n"
            "[Verse]\n"
            "I see your face in every crowded restaurant\n"
            "Hear your laugh in every borrowed song\n"
            "My pride is heavier than my pillow\n"
            "But I'm too far gone to apologize\n\n"
            "[Chorus]\n"
            "You're the one I left for nothing\n"
            "You're the one I lost for nothing\n"
            "I had everything I wanted\n"
            "And I traded it for nothing at all"
        ),
    },
    {
        "id": "tool",
        "display": "Tool",
        "genre_summary": "Progressive metal",
        "difficulty": "Hard",
        "bpm": 132,
        "vocal_language": "en",
        "instrumental": False,
        "caption_fingerprint": (
            "Progressive metal, 132 BPM in shifting odd time signatures, downtuned "
            "distorted electric guitar with palm-muted polyrhythmic riffing, complex "
            "tom-heavy drumming with intricate cymbal work, fretless electric bass "
            "with audible string slides, baritone male lead vocal alternating between "
            "low spoken cadence and anguished melodic shouts, brooding philosophical "
            "introspective mood, dense atmospheric production with reverb and tape "
            "saturation"
        ),
        "caption_name": (
            "Progressive metal song in the style of Tool, 132 BPM odd time signatures, "
            "downtuned distorted guitar, polyrhythmic palm-muted riffing, tom-heavy "
            "drumming, fretless bass, baritone male vocal alternating spoken and "
            "shouted, brooding philosophical mood, dense atmospheric production"
        ),
        "lyrics": (
            "[Intro]\n\n"
            "[Verse]\n"
            "Counting backwards from a number that I don't remember\n"
            "Watching shadows learn the shape of my own face\n"
            "Every prophet selling something I already paid for\n"
            "Every god I built is asking for my name\n\n"
            "[Chorus]\n"
            "Tear it down\n"
            "Build it again from the bones that bury us\n"
            "Tear it down\n"
            "Nothing sacred ever stays the way we wanted\n\n"
            "[Verse]\n"
            "Mirror tells the truth in a language I keep losing\n"
            "Repetition is the only honest prayer\n"
            "I confessed to every angel and they all said the same thing\n"
            "You were never lost you only learned to disappear\n\n"
            "[Bridge]\n"
            "Strip it all back to the silence underneath\n"
            "Strip it all back to the ringing in the deep\n\n"
            "[Chorus]\n"
            "Tear it down\n"
            "Build it again from the bones that bury us\n"
            "Tear it down\n"
            "Nothing sacred ever stays the way we wanted"
        ),
    },
    {
        "id": "nirvana",
        "display": "Nirvana",
        "genre_summary": "Grunge / alt-rock",
        "difficulty": "Hard",
        "bpm": 116,
        "vocal_language": "en",
        "instrumental": False,
        "caption_fingerprint": (
            "1990s grunge alt-rock, 116 BPM, loud-quiet-loud dynamic shifts, clean "
            "chorused electric guitar in verses transitioning to distorted overdriven "
            "guitar in choruses, simple powerful four-on-the-floor drumming with "
            "crash cymbal accents, distorted electric bass following the guitar root, "
            "throat-strained male vocal alternating mumbled verse with shouted "
            "melodic chorus, angsty disaffected disillusioned mood, raw lo-fi analog "
            "production with intentional roughness"
        ),
        "caption_name": (
            "Grunge alt-rock song in the style of Nirvana, 116 BPM, loud-quiet "
            "dynamics, clean chorused verse guitar to distorted chorus guitar, "
            "simple powerful drumming, throat-strained male vocal mumbled verse "
            "and shouted chorus, angsty disaffected mood, raw lo-fi production"
        ),
        "lyrics": (
            "[Intro]\n\n"
            "[Verse]\n"
            "Bought the magazine again to feel less alone\n"
            "All the smiling people sold me back my own face\n"
            "Television says my generation found a meaning\n"
            "I missed the meeting must have been asleep again\n\n"
            "[Chorus]\n"
            "Hey hey hey I'm not the type to figure out\n"
            "Hey hey hey I'm not the type to make it count\n"
            "Hey hey hey I never asked for anything\n"
            "And I got all of it just the same\n\n"
            "[Verse]\n"
            "Sister's voice through the apartment wall reminds me\n"
            "I forgot to call my mother on her birthday\n"
            "Counting all the people who I owe an explanation\n"
            "Counting all the explanations I won't give\n\n"
            "[Chorus]\n"
            "Hey hey hey I'm not the type to figure out\n"
            "Hey hey hey I'm not the type to make it count\n"
            "Hey hey hey I never asked for anything\n"
            "And I got all of it just the same\n\n"
            "[Bridge]\n"
            "And I got all of it the same\n"
            "And I got all of it the same"
        ),
    },
    {
        "id": "wu_tang",
        "display": "Wu-Tang Clan",
        "genre_summary": "East Coast hip-hop",
        "difficulty": "Very Hard",
        "bpm": 92,
        "vocal_language": "en",
        "instrumental": False,
        "caption_fingerprint": (
            "1990s East Coast boom-bap hip-hop, 92 BPM, dusty soul sample loop with "
            "vinyl crackle, sparse upright piano stab on the two and four, raw "
            "kick-snare drum break sample with prominent hi-hat, deep upright bass "
            "under the loop, multiple male rapper voices trading verses with distinct "
            "intricate flows, occasional kung fu film sample interlude, gritty New "
            "York street narrative mood, lo-fi sample-based production with mono "
            "compression"
        ),
        "caption_name": (
            "East Coast hip-hop song in the style of Wu-Tang Clan, 92 BPM, dusty "
            "soul sample loop, sparse piano stab, boom-bap drums, multiple male "
            "rapper voices with distinct flows, kung fu sample interludes, gritty "
            "street mood, lo-fi sample-based production"
        ),
        "lyrics": (
            "[Intro]\n\n"
            "[Verse]\n"
            "Concrete teaching mathematics that the schoolbooks never had\n"
            "Brother on the corner cooking up a different kind of math\n"
            "Sirens in the distance like an alarm clock for the kingdom\n"
            "Every rooftop holding council, every alley holding court\n\n"
            "[Chorus]\n"
            "We the ones the city built around\n"
            "We the ones the city tried to drown\n"
            "We the ones still standing in the sound\n"
            "And we ain't going nowhere right now\n\n"
            "[Verse]\n"
            "Pops told me real recognize real before I knew his real name\n"
            "Mama held the family on a wage that wouldn't fit the frame\n"
            "Every cousin got a story that the news ain't never telling\n"
            "Every block remembers names that nobody else is spelling\n\n"
            "[Chorus]\n"
            "We the ones the city built around\n"
            "We the ones the city tried to drown\n"
            "We the ones still standing in the sound\n"
            "And we ain't going nowhere right now"
        ),
    },
    {
        "id": "calvin_harris",
        "display": "Calvin Harris",
        "genre_summary": "Stadium EDM",
        "difficulty": "Easy-Medium",
        "bpm": 128,
        "vocal_language": "en",
        "instrumental": False,
        "caption_fingerprint": (
            "Stadium EDM, 128 BPM, driving four-on-the-floor kick drum, bright "
            "supersaw synth lead, plucked synth arpeggio in verses, heavy sidechain "
            "compression pumping under the kick, female pop vocal feature with "
            "bright top-end and processed harmonies, anthemic build-up with snare "
            "roll into euphoric main drop, uplifting festival energy mood, polished "
            "arena production with wide stereo imaging"
        ),
        "caption_name": (
            "Stadium EDM track in the style of Calvin Harris, 128 BPM, "
            "four-on-the-floor kick, supersaw synth lead, sidechain compression, "
            "plucked synth arpeggio, female pop vocal feature with processed "
            "harmonies, anthemic build and drop, euphoric festival mood, polished "
            "arena production"
        ),
        "lyrics": (
            "[Intro]\n\n"
            "[Verse]\n"
            "Friday night the city's just beginning\n"
            "You and me we move like we already won\n"
            "Hold my hand and don't pretend you're listening\n"
            "Tonight the only word is gone\n\n"
            "[Pre-Chorus]\n"
            "We can lose the morning we can lose the year\n"
            "Everything I needed everything is here\n\n"
            "[Chorus]\n"
            "Hold on hold on hold on\n"
            "We don't need to know where we're going\n"
            "Hold on hold on hold on\n"
            "The whole world's right here where we're standing\n"
            "Hold on hold on hold on\n"
            "This is everything I waited for\n"
            "Hold on hold on\n"
            "Don't let go don't let go\n\n"
            "[Verse]\n"
            "Sunrise will arrive without permission\n"
            "Concrete is forgiving when you dance\n"
            "Every face I love is in the building\n"
            "Tonight the only word is yes\n\n"
            "[Chorus]\n"
            "Hold on hold on hold on\n"
            "We don't need to know where we're going\n"
            "Hold on hold on hold on\n"
            "The whole world's right here where we're standing"
        ),
    },
    {
        "id": "einaudi",
        "display": "Ludovico Einaudi",
        "genre_summary": "Neoclassical piano (instrumental)",
        "difficulty": "Hard (instrumental)",
        "bpm": 72,
        "vocal_language": "en",
        "instrumental": True,
        "caption_fingerprint": (
            "Contemporary neoclassical piano composition, 72 BPM, solo grand piano "
            "with intimate close-mic recording, sparse single-note motifs in the "
            "upper register building gradually into rolling arpeggiated harmonic "
            "patterns across both hands, subtle sustained string section entering "
            "quietly underneath, melancholic introspective contemplative mood, "
            "minimalist repetitive structure with slow harmonic shifts, no vocals, "
            "no drums, no bass, no percussion, pure acoustic instrumental composition"
        ),
        "caption_name": (
            "Contemporary neoclassical piano composition in the style of Ludovico "
            "Einaudi, 72 BPM, solo grand piano with intimate recording, sparse "
            "single-note motifs building into arpeggiated patterns, subtle strings "
            "underneath, melancholic introspective mood, minimalist structure, no "
            "vocals, no drums, instrumental"
        ),
        "lyrics": "[Instrumental]",
    },
]


def build_file(artist: dict, test_type: str) -> dict:
    """test_type: 'fingerprint' (R2A, 3 seeds) or 'name' (R2B, 1 seed)."""
    if test_type == "fingerprint":
        round_id = f"r2a_{artist['id']}"
        caption = artist["caption_fingerprint"]
        seeds = [1, 2, 3]
        description = (
            f"R2A sonic fingerprint test: {artist['display']} "
            f"({artist['genre_summary']}, difficulty: {artist['difficulty']}). "
            f"No artist name in caption. 3 seeds. Locked {DATE_LOCKED}."
        )
        output_subdir = f"r2a-{artist['id']}"
    elif test_type == "name":
        round_id = f"r2b_{artist['id']}"
        caption = artist["caption_name"]
        seeds = [1]
        description = (
            f"R2B artist-name diagnostic: {artist['display']} "
            f"({artist['genre_summary']}). Artist named in caption. 1 seed. "
            f"Locked {DATE_LOCKED}."
        )
        output_subdir = f"r2b-{artist['id']}"
    else:
        raise ValueError(f"unknown test_type: {test_type}")

    return {
        "_meta": {
            "round": round_id,
            "description": description,
            "output_subdir": output_subdir,
            "test_type": "sonic_fingerprint" if test_type == "fingerprint" else "artist_name",
            "artist_display": artist["display"],
            "artist_genre": artist["genre_summary"],
            "artist_difficulty": artist["difficulty"],
        },
        "captions": {
            "ace_step": caption,
            "diffrhythm": "",
            "songgeneration": "",
            "suno": "",
        },
        "lyrics": artist["lyrics"],
        "generation": {
            "bpm": artist["bpm"],
            "duration_seconds": 90.0,
            "vocal_language": artist["vocal_language"],
            "instrumental": artist["instrumental"],
            "keyscale": "",
            "seeds": seeds,
        },
    }


def main():
    out_dir = Path(__file__).resolve().parent
    written = []
    for artist in ARTISTS:
        for test_type, prefix in [("fingerprint", "r2a"), ("name", "r2b")]:
            data = build_file(artist, test_type)
            filename = f"{prefix}_{artist['id']}.json"
            path = out_dir / filename
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            written.append(filename)

    print(f"Wrote {len(written)} files to {out_dir}:")
    for f in written:
        print(f"  {f}")
    print()
    print("R2A files (sonic fingerprint, 3 seeds each):")
    for f in written:
        if f.startswith("r2a_"):
            print(f"  {f}")
    print()
    print("R2B files (artist-name diagnostic, 1 seed each):")
    for f in written:
        if f.startswith("r2b_"):
            print(f"  {f}")


if __name__ == "__main__":
    main()
