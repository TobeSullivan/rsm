# Scoring Results: {VOICE_ID}

*Fill out one of these per voice tested. Copy to results/{voice_id}.md.*

Tested on: {DATE}
Hardware: {Windows 4070 / M4 Max / CPU-only}
Model(s) tested: {qwen2.5:7b / llama3.1:8b / llama3.2:3b / qwen2.5:14b / ...}
Chatterbox version: {version string}

---

## Phase A — LM ungrounded ({model})

### LM patter text scores (1-5)

| Scenario | Era authenticity | Genre authenticity | Scene fluency | No-hallucination |
|---|---|---|---|---|
| cold_open | | | | |
| between_songs | | | | |
| news_break | | | | |
| local_color | | | | |
| caller_interaction | | | | |
| genre_moment | | | | |

Voice consistency (across all 6 scenarios): **/5**

### Notes
- What worked:
- What didn't:
- Most egregious miss:
- Best generated line (quote):

---

## Phase B — LM RAG-grounded ({model})

### LM patter text scores (1-5)

| Scenario | Era authenticity | Genre authenticity | Scene fluency | No-hallucination |
|---|---|---|---|---|
| cold_open | | | | |
| between_songs | | | | |
| news_break | | | | |
| local_color | | | | |
| caller_interaction | | | | |
| genre_moment | | | | |

Voice consistency: **/5**

### Notes
- Did grounding help? Where most? Where least?
- What did the LM ignore from the context block?
- What did the LM lift verbatim that should have been paraphrased?

---

## Phase C — Bigger model RAG (only if needed)

### Model used: {qwen2.5:14b or other}

Scores as above. Notes focus on: did the bigger model do what the smaller couldn't?

---

## TTS scores (1-5) — Chatterbox on {phase}'s patter

| Scenario | Voice fidelity | Era-appropriateness | Latency | Emotion expressiveness | Naturalness |
|---|---|---|---|---|---|
| cold_open | | | | | |
| between_songs | | | | | |
| news_break | | | | | |
| local_color | | | | | |
| caller_interaction | | | | | |
| genre_moment | | | | | |

### Emotion stress test (separate)

| Intent | Exaggeration | Audible difference? | Notes |
|---|---|---|---|
| deadpan | 0.2 | | |
| excited | 0.9 | | |
| somber | 0.4 | | |

### Notes
- Where did the voice break character?
- Was latency consistent or did some lines stall?
- Did the reference clip choice matter? (Try a different reference, re-score)

---

## Pass / fail call

**LM (this voice):** PASS / SOFT PASS / FAIL — at phase {A/B/C}
**TTS (this voice):** PASS / SOFT PASS / FAIL

**Recommendation:**
{What does this voice's result imply for the architecture? Ship as-is? Need more grounding? Need a bigger model? Need a different TTS path?}

## What I'd want for the demo
{Specific things that would close the gap between current quality and what would be acceptable for a player to actually hear in the shipped game.}
