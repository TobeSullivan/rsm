# Scoring Rubric

All scores are 1-5. Definitions below make 3 vs 4 decidable rather than vibes-based.

## LM patter text scoring

### Era authenticity
*Does this sound like radio patter from the target era, or a 2026 model's idea of it?*
- **5** — Indistinguishable from a real DJ transcript of the period. Uses period-correct cadence, sponsor mentions, phrasing, references current to the era.
- **4** — Reads as era-correct with one or two minor anachronisms (a phrase too modern, a reference slightly out of place). Would pass a casual ear.
- **3** — Era is identifiable but generic. No glaring anachronisms, but no era-specific texture either. Could be any DJ from a 20-year window.
- **2** — Clear anachronisms (phrases, references, sensibilities from outside the era). Reads as "AI's idea of the era."
- **1** — Era-agnostic or wrong era entirely. No texture at all.

### Genre authenticity
*Country DJ patter ≠ rap mixshow ≠ K-pop MC. Is this voice distinctly the right genre?*
- **5** — Genre-specific patter unmistakably. Country DJ talks about church on Sunday and the state fair. Rap host hypes a freestyle and shouts out the borough. Indistinguishable from a real practitioner.
- **4** — Right genre with strong texture. One or two moments where it could plausibly be a different genre's DJ.
- **3** — Genre-correct but generic. Music DJ patter, no genre-specific flavor.
- **2** — Mostly generic with the wrong genre's flavor leaking in.
- **1** — Wrong genre entirely, or genre-blind generic patter.

### Scene fluency
*Does it reference the right (real) acts, scenes, venues, cultural moments for the era? Is it situated in the world?*
- **5** — Drops 3+ real-and-correct references (acts charting, venues, cultural events, scene figures). All era-accurate. Sounds embedded.
- **4** — 1-2 correct references, no incorrect ones. Sounds informed.
- **3** — No specific references, but nothing wrong. Generic but plausible.
- **2** — References made but at least one is wrong (act in wrong era, venue that didn't exist, etc.).
- **1** — References made and multiple are wrong, or aggressively generic with hallucinated context.

### Voice consistency
*Does the same DJ persona hold across multiple scenarios? Is "DJ Marcus" the same person in cold-open and news-break?*

Scored across the six scenarios per voice, not per scenario.
- **5** — Persona holds entirely. Same cadence, same catchphrases, same attitude. Reads as one DJ.
- **4** — Mostly consistent with one scenario drifting slightly.
- **3** — Recognizably the same persona but with noticeable inconsistency.
- **2** — Drifts to a different persona in 2+ scenarios.
- **1** — Six different DJs across six scenarios.

### No-hallucination
*Does it make up bands, events, venues, or facts that didn't exist?*
- **5** — Every specific claim is verifiable. No invented bands presented as real, no invented dates, no fake events.
- **4** — One minor invention (a venue name that sounds real but isn't, etc.) but nothing damaging.
- **3** — One real hallucination (a band that didn't exist, a date that's wrong) but only one.
- **2** — Multiple hallucinations.
- **1** — Patter is mostly inventions presented as fact.

## TTS audio output scoring

### Voice fidelity
*Does the generated audio sound like the reference clip we cloned from?*
- **5** — Indistinguishable in a blind test from a different recording by the same speaker.
- **4** — Clearly the same voice, with minor differences in timbre or cadence.
- **3** — Recognizably the same voice but with audible cloning artifacts.
- **2** — Some resemblance, but clearly a different voice.
- **1** — No resemblance to the reference.

### Era-appropriateness (audio)
*Does the voice's sound (not just identity) fit the target era's broadcast quality and style?*
- **5** — Would pass for a real broadcast of the era. Compression, mic character, delivery style all era-correct.
- **4** — Mostly era-correct, one element off (too clean for the 1920s, etc.).
- **3** — Era-neutral. Wouldn't stand out but doesn't enhance.
- **2** — Wrong era's sound character (a 1990s DJ sounding like a 2020s podcaster, etc.).
- **1** — Aggressively wrong era texture.

Note: Post-processing (broadcast EQ, era-appropriate compression, vinyl crackle) can be added in production. Score the raw TTS output, but note where post-processing would lift it.

### Latency
*Wall-clock time from text submission to audio playback start, for ~10 seconds of generated audio.*

Latency is logged automatically by the harness. Score is a translation, not a judgment:
- **5** — <200ms (Chatterbox spec target)
- **4** — 200-500ms
- **3** — 500ms-1s
- **2** — 1-3s
- **1** — >3s

### Emotion expressiveness
*Test the same line in three emotional registers (deadpan / excited / somber). Does the emotion knob actually change the delivery?*
- **5** — Three clearly different deliveries, each appropriate to the requested emotion. Indistinguishable from a voice actor trying three takes.
- **4** — Three different deliveries, all appropriate, with minor inconsistency in one.
- **3** — Two distinct registers, third is muddled. Or all three differ but one is wrong for its label.
- **2** — Slight variation between registers. Knob barely moves the output.
- **1** — All three registers sound the same.

### Naturalness
*Does the speech sound like a human, or like an AI?*
- **5** — Human. Casual listener would not flag this as synthetic.
- **4** — Human with one or two synthetic tells (a vowel oddly held, a sibilant artifact).
- **3** — Plausibly human in short bursts, synthetic over a full passage.
- **2** — Clearly synthetic but listenable.
- **1** — Aggressively synthetic, distracting.

## Scoring procedure

For each voice:
1. Generate patter via LM for all 6 scenarios (Phase A first, then B if A fails)
2. Score the LM text on the 5 LM axes — voice consistency is scored once across all 6 scenarios; the other 4 are scored per scenario then averaged
3. Run Chatterbox on the LM patter text (and on a hand-written control set)
4. Score the audio on the 5 TTS axes per scenario, then average
5. Write results to `results/{voice}.md` using the template

## Pass bar

**Soft target:** average ≥4.0 across all 10 axes for the three 1997 voices (alt-rock, country, rap). These are the architecture-deciding cases.

**Hard fail:** any axis averages <3.0 on the 1997 voices across phases A through C. Architecture must change.

The bookend voices (1920s, 2010s, K-pop) are diagnostic, not gating. They tell us where the model's edges are without blocking the architecture decision.
