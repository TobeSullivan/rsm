# Reference Audio Sourcing Guide

## What goes here

One directory per voice_id, each containing one `reference.wav` (15-30 seconds, clean, mono, 24kHz preferred).

Layout:
```
voices/
├── altrock_seattle_1997/reference.wav
├── country_nashville_1997/reference.wav
├── rap_nyc_1997/reference.wav
├── newsreel_1928/reference.wav
├── fm_rock_1977/reference.wav
├── indie_podcast_2017/reference.wav
└── kpop_mc_2025/reference.wav
```

## Sourcing rules

**Public domain, your own voice, or properly-licensed only.**

- The 1920s newsreel is public domain because of age (pre-1929 US recordings are PD as of 2024, and most 1928-1932 will be PD by 2027-2028 per the rolling Copyright Term).
- Everything else needs to be safe to clone for testing.
- **No copyrighted DJ recordings.** Not Marco Collins, not Funkmaster Flex, not Allison Steele. Even for prototyping, this is the rule.

## Safe sources

### Public domain
- **Internet Archive** — extensive 1920s-1950s radio collection, mostly PD
- **Library of Congress National Recording Registry** — selected PD items
- **Pathé / Movietone newsreels** — audio only, PD due to age

### Creative Commons / freely licensed
- **NPR podcast clips** — many under CC-BY licenses
- **PRX / Public Radio Exchange** — various licenses, check each
- **Wikipedia audio samples** — usually CC-BY-SA

### Your own voice
- Easiest by far. Record yourself reading a 20-second passage in the rough cadence of each target era/genre.
- Quality requirement: clean recording, no background music, no other speakers, 16kHz or higher.
- This is the recommended path for v0 testing — fastest, zero licensing concern, and reveals whether Chatterbox can clone a voice it has never heard before (which is the actual product test).

### Synthetic but human-sounding
- Generate a reference voice with a different TTS (or just record one) — Chatterbox is meant to clone arbitrary voices in 5-second cuts, so any clean source works for testing.

## What to capture per voice

Try to match cadence, accent, and energy to the target persona. You don't need to *be* a 1997 alt-rock DJ — you need to provide Chatterbox a voice that *plausibly could be* one, then the era texture comes from the patter content + Chatterbox's emotion knob.

Reference samples should be 15-30 seconds. Chatterbox can technically clone from 5 seconds but longer references produce more stable output.

## What NOT to do

- Don't use samples of known DJs from copyrighted broadcasts. Aside from the legal exposure, this also makes the test less valuable — we want to test "can we clone an arbitrary voice well" not "can we re-create a specific famous voice."
- Don't use music in the reference clip. Pure speech, please.
- Don't use phone-quality or low-bitrate audio. Garbage in, garbage out.

## Quick recording recipe (your own voice)

If you're producing references yourself:

1. Open Audacity (free) or any DAW.
2. Set sample rate to 24kHz, mono.
3. Record ~20 seconds of yourself reading the appropriate passage below at the appropriate energy level.
4. Export as `voices/<voice_id>/reference.wav`.

### Suggested reading passages

**altrock_seattle_1997** (mid-30s, conversational, slightly tired energy):
> "You're listening to The End, 107-7 KNDD. Coming up after the break we've got the new Foo Fighters in heavy rotation, plus the live recording from the Crocodile last Thursday with that band I keep telling you about. Stick around."

**country_nashville_1997** (50s-ish, warm Southern, unhurried):
> "Good morning, neighbors. It's a fine Tennessee fall day out there. We've got some Tim McGraw coming up, brand-new George Strait right after that, and a tribute to John Denver up at the top of the hour. Y'all drive safe out there."

**rap_nyc_1997** (late 20s, fast, energetic):
> "Yo, what's up Brooklyn, what's up Queens, what's up the BX. Hot 97, this is your boy with the new joint about to drop. Salute to Biggie this whole hour, Life After Death just hit shelves today. We coming back with that exclusive after this."

**newsreel_1928** (formal, oratorical, mid-Atlantic accent):
> "Ladies and gentlemen of the radio audience. We bring you tonight, by wireless dispatch from Paris, the most remarkable news this correspondent has had occasion to report. Colonel Charles Lindbergh has landed safely at Le Bourget Field, completing his solitary flight across the Atlantic Ocean."

**fm_rock_1977** (laid-back, drawn out, slightly stoned cadence):
> "And that was side two of the new Fleetwood Mac, Rumours, in case you missed it the first time around. Heavy stuff, beautiful record. Coming up, we got the brand new Bowie, off the new one, just landed on my desk this morning. Stick with me."

**indie_podcast_2017** (conversational, qualifying clauses):
> "Welcome back to the podcast, this is, you know, our long-running conversation about indie music in the streaming era. I want to start this episode by talking about the new Mitski record, because I think it does something really interesting with how it sequences its singles versus the album experience."

**kpop_mc_2025** (warm, professional, slight Korean accent if you can do it without being a caricature):
> "Welcome back to M Countdown. We have an amazing lineup for you today. Up next is the comeback stage from ILLIT, with the title track from their newest mini-album. Make some noise for our LUVITY in the audience tonight."

These are starting points. Adjust to your own voice; the test is meaningful even if your imitations aren't perfect.
