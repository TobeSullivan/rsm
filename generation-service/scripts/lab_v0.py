"""
RSM generation-service lab — Phase 2b v0.2.

Long-lived Gradio app for capability discovery.

v0.2 changes over v0.1:
- Generation history: every completed generation appends to a dropdown.
  Pick a past generation to replay it. Fixes three v0.1 issues at once —
  Audio player state stuck across generations, "last song is gone" once
  a new one generates, and "play-again-at-end doesn't replay."
- Progress bar tracks ACE-Step's internal tqdm via gr.Progress(track_tqdm=True),
  so the bar actually fills as LM Phase 2 + DiT + VAE decode progress.

Phase 2 capability findings recorded in v0.1 testing (parked for handoff):
- Reference-artist text tags ("in the style of Pantera", "in the style of
  Led Zeppelin") produce audio that hits the genre but doesn't match the
  artist. ACE-Step is trained on genre/mood/instrument labels, not artist
  labels. Text-only artist matching is a weak signal here. The real path
  for artist-style matching is ACE-Step's Cover mode (reference audio +
  transform caption) — different mechanic, needs the game to source audio.
- Era specificity is loose. "1990s metal" reads more like 1980s metal in
  the audio. Genre + era is a weaker combination than the genre alone.

What this lab is FOR:
- Discovering which structured fields move the audio in predictable ways.
- The eventual game composes prompts from band/career state, not free-form
  text — the lab tests that shape.

What this lab is NOT:
- A prompt engine.
- A "make me a song" button.
- A lyrics writer (lyrics generation belongs in a separate small text LLM —
  ACE-Step's LM is for audio code generation and is way too slow for text).

Run from <service-root>:  uv run scripts/lab_v0.py
"""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr
from loguru import logger

# --- Configuration ---------------------------------------------------------
SERVICE_ROOT = Path(__file__).parent.parent.resolve()
CHECKPOINT_DIR = SERVICE_ROOT / "checkpoints"
OUTPUT_DIR = SERVICE_ROOT / "outputs" / "lab_v0"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LM_MODEL = "acestep-5Hz-lm-1.7B"
DIT_CONFIG = "acestep-v15-turbo"
DEVICE = "cuda"
LM_BACKEND = "pytorch"

logger.info(f"SERVICE_ROOT:   {SERVICE_ROOT}")
logger.info(f"CHECKPOINT_DIR: {CHECKPOINT_DIR}")
logger.info(f"OUTPUT_DIR:     {OUTPUT_DIR}")

from acestep.handler import AceStepHandler  # noqa: E402
from acestep.llm_inference import LLMHandler  # noqa: E402
from acestep.inference import (  # noqa: E402
	GenerationParams,
	GenerationConfig,
	generate_music,
)


# --- Schema definitions ---------------------------------------------------
GENRES = [
	"rock", "pop", "hip-hop", "R&B", "country", "folk",
	"jazz", "blues", "electronic", "metal", "classical", "world",
]
ERAS = [
	"1920s", "1930s", "1940s", "1950s", "1960s", "1970s",
	"1980s", "1990s", "2000s", "2010s", "contemporary",
]
ENERGIES = ["mellow", "moderate", "energetic", "intense"]
MOODS = [
	"joyful", "melancholy", "angry", "romantic", "nostalgic",
	"defiant", "contemplative", "playful",
]

LEAD_INSTRUMENTS = [
	"electric guitar", "acoustic guitar", "piano", "synth",
	"strings", "brass", "vocals", "none",
]
OTHER_INSTRUMENTS = [
	"acoustic drums", "programmed drums", "bass", "organ",
	"synth pads", "strings section", "brass section",
	"harmonica", "saxophone", "horns", "accordion",
	"violin", "cello", "banjo", "mandolin",
]

VOCAL_STYLES = [
	"lead male", "lead female", "duet", "multi-voice harmony",
	"call-and-response", "spoken word", "instrumental (no vocals)",
]

LANGUAGES = ["en", "es", "fr", "de", "ja", "ko", "zh", "hi", "pt", "unknown"]
LANGUAGE_NAMES = {
	"en": "English", "es": "Spanish", "fr": "French", "de": "German",
	"ja": "Japanese", "ko": "Korean", "zh": "Chinese", "hi": "Hindi",
	"pt": "Portuguese", "unknown": "",
}

FIDELITY_TIERS = [
	"bedroom demo", "cassette 4-track", "garage band",
	"club-quality", "studio polished", "radio-ready",
]
FIDELITY_DESCRIPTIONS = {
	"bedroom demo": "lo-fi bedroom recording",
	"cassette 4-track": "cassette 4-track recording with tape hiss",
	"garage band": "garage band recording with room sound",
	"club-quality": "club-quality mix",
	"studio polished": "studio polished production",
	"radio-ready": "radio-ready commercial production",
}

GENRE_DEFAULT_LEAD = {
	"rock": "electric guitar",
	"pop": "vocals",
	"hip-hop": "vocals",
	"R&B": "vocals",
	"country": "acoustic guitar",
	"folk": "acoustic guitar",
	"jazz": "piano",
	"blues": "electric guitar",
	"electronic": "synth",
	"metal": "electric guitar",
	"classical": "strings",
	"world": "vocals",
}

DEFAULTS = {
	"genre": "rock",
	"era": "1980s",
	"energy": "energetic",
	"mood": "defiant",
	"reference_artist": "",
	"lead_instrument": "electric guitar",
	"other_instruments": ["acoustic drums", "bass"],
	"vocal_style": "lead male",
	"language": "en",
	"fidelity": "studio polished",
	"bpm": 140,
	"key": "E minor",
	"duration": 30.0,
	"theme": "",
	"lyrics": """[Verse 1]
Stand up tall, the night is calling
Lights are bright and walls are falling
[Chorus]
We are the fire, we are the sound
Tear it down and lift it off the ground""",
}

HISTORY_MAX = 20  # cap history list to avoid runaway dropdown


# --- Caption assembly -----------------------------------------------------
def _join_with_and(items: list[str]) -> str:
	items = [i for i in items if i]
	if not items:
		return ""
	if len(items) == 1:
		return items[0]
	if len(items) == 2:
		return f"{items[0]} and {items[1]}"
	return f"{', '.join(items[:-1])}, and {items[-1]}"


def assemble_caption(
	genre: str,
	era: str,
	energy: str,
	mood: str,
	reference_artist: str,
	lead_instrument: str,
	other_instruments: list[str],
	vocal_style: str,
	language: str,
	fidelity: str,
	theme: str,
) -> str:
	parts: list[str] = []

	identity = " ".join(filter(None, [energy or "", era or "", genre or ""])).strip()
	if identity:
		parts.append(identity)

	if fidelity and fidelity in FIDELITY_DESCRIPTIONS:
		parts.append(FIDELITY_DESCRIPTIONS[fidelity])

	if lead_instrument and lead_instrument != "none":
		instr = f"featuring {lead_instrument}"
		if other_instruments:
			instr += f" with {_join_with_and(other_instruments)}"
		parts.append(instr)
	elif other_instruments:
		parts.append(f"featuring {_join_with_and(other_instruments)}")

	if vocal_style and vocal_style != "instrumental (no vocals)":
		vocal_part = f"{vocal_style} vocals"
		if language and language != "unknown":
			lang_name = LANGUAGE_NAMES.get(language, language)
			if lang_name:
				vocal_part += f" in {lang_name}"
		parts.append(vocal_part)

	if mood:
		parts.append(f"{mood} feel")

	if reference_artist and reference_artist.strip():
		parts.append(f"in the style of {reference_artist.strip()}")

	if theme and theme.strip():
		parts.append(f"about {theme.strip()}")

	return ", ".join(parts)


# --- Handler init ---------------------------------------------------------
logger.info("Booting handlers (cold start ~25s)...")
_boot_t0 = time.time()

logger.info("Initializing DiT handler...")
_dit_t0 = time.time()
DIT_HANDLER = AceStepHandler()
DIT_HANDLER.initialize_service(
	project_root=str(SERVICE_ROOT),
	config_path=DIT_CONFIG,
	device=DEVICE,
)
logger.info(f"DiT init: {time.time() - _dit_t0:.2f}s")

logger.info("Initializing LM handler...")
_lm_t0 = time.time()
LM_HANDLER = LLMHandler()
LM_HANDLER.initialize(
	checkpoint_dir=str(CHECKPOINT_DIR),
	lm_model_path=LM_MODEL,
	backend=LM_BACKEND,
	device=DEVICE,
)
logger.info(f"LM init: {time.time() - _lm_t0:.2f}s")

logger.info(f"Boot complete in {time.time() - _boot_t0:.2f}s")


# --- History helpers ------------------------------------------------------
def format_history_label(item: dict) -> str:
	"""Compact one-line label for the history dropdown."""
	cap = item.get("caption", "(no caption)")
	if len(cap) > 60:
		cap = cap[:57] + "..."
	return f"{item.get('time', '??:??:??')} | {cap}"


def history_to_choices(history: list[dict]) -> list[tuple[str, str]]:
	"""Build (label, value) pairs for the dropdown. value = audio path."""
	return [(format_history_label(item), item["path"]) for item in history]


# --- Helpers --------------------------------------------------------------
def format_time_costs(tc: dict, wall: float) -> str:
	if not tc:
		return f"(no time_costs)\nwall-clock: {wall:.2f}s"
	lines = []
	if "lm_phase1_time" in tc:
		lines.append(f"LM Phase 1: {tc['lm_phase1_time']:.2f}s")
	if "lm_phase2_time" in tc:
		lines.append(f"LM Phase 2: {tc['lm_phase2_time']:.2f}s")
	if "dit_total_time_cost" in tc:
		lines.append(f"DiT total:  {tc['dit_total_time_cost']:.2f}s")
	if "dit_vae_decode_time_cost" in tc:
		lines.append(f"  (VAE decode: {tc['dit_vae_decode_time_cost']:.2f}s)")
	if "pipeline_total_time" in tc:
		lines.append(f"PIPELINE:   {tc['pipeline_total_time']:.2f}s")
	lines.append(f"wall-clock: {wall:.2f}s")
	return "\n".join(lines)


# --- Generate audio -------------------------------------------------------
def do_generate(
	caption_preview: str,
	lyrics: str,
	vocal_style: str,
	language: str,
	bpm: int,
	key: str,
	duration: float,
	inference_steps: int,
	guidance_scale: float,
	seed: int,
	shift: float,
	infer_method: str,
	thinking: bool,
	use_cot_caption: bool,
	use_cot_metas: bool,
	use_cot_language: bool,
	history: list,
	progress=gr.Progress(track_tqdm=True),
):
	"""Generator. Yields tuples matching the outputs list:
	  (audio_path, status, time_costs, history_state, history_dropdown)

	track_tqdm=True lets the Gradio progress bar follow ACE-Step's
	internal tqdm loops (LM CFG Generation, VAE chunk decode) instead
	of sitting at 10% for 70s.
	"""
	history = list(history or [])

	if not caption_preview or not caption_preview.strip():
		yield (
			None, "Caption preview is empty. Set fields or type a caption.",
			"", history, gr.update(),
		)
		return

	is_instrumental = vocal_style == "instrumental (no vocals)"
	effective_lyrics = "[Instrumental]" if is_instrumental else lyrics

	yield None, "Building generation params...", "", history, gr.update()

	params = GenerationParams(
		task_type="text2music",
		caption=caption_preview.strip(),
		lyrics=effective_lyrics,
		bpm=int(bpm) if bpm else None,
		duration=float(duration) if duration else -1.0,
		keyscale=key.strip() if key else "",
		vocal_language=(
			language if language and language != "unknown" else "unknown"
		),
		instrumental=is_instrumental,
		thinking=bool(thinking),
		use_cot_caption=bool(use_cot_caption),
		use_cot_metas=bool(use_cot_metas),
		use_cot_language=bool(use_cot_language),
		inference_steps=int(inference_steps),
		guidance_scale=float(guidance_scale),
		seed=int(seed) if seed is not None else -1,
		shift=float(shift),
		infer_method=infer_method,
	)

	config = GenerationConfig(
		batch_size=1,
		use_random_seed=(int(seed) < 0),
		audio_format="flac",
	)

	save_dir = OUTPUT_DIR / f"gen_{int(time.time())}"
	save_dir.mkdir(parents=True, exist_ok=True)

	yield None, "Running pipeline (~70s)...", "", history, gr.update()

	t0 = time.time()
	try:
		result = generate_music(
			DIT_HANDLER, LM_HANDLER, params, config,
			save_dir=str(save_dir),
		)
	except Exception as e:
		logger.exception("generate_music threw")
		yield (
			None, f"Pipeline failed with exception: {e}",
			"", history, gr.update(),
		)
		return

	wall = time.time() - t0

	if not result.success:
		logger.error(f"generate_music failed: {result.error}")
		yield (
			None, f"Pipeline failed: {result.error}",
			"", history, gr.update(),
		)
		return

	tc = result.extra_outputs.get("time_costs", {}) or {}
	tc_text = format_time_costs(tc, wall)

	audio_path = str(result.audios[0]["path"])
	seed_used = result.audios[0]["params"].get("seed", "?")

	# Append to history (newest first), cap length.
	new_item = {
		"path": audio_path,
		"caption": caption_preview.strip(),
		"seed": str(seed_used),
		"time": time.strftime("%H:%M:%S"),
	}
	history.insert(0, new_item)
	history = history[:HISTORY_MAX]

	choices = history_to_choices(history)
	# Set the dropdown to the new generation so user sees it selected.
	dropdown_update = gr.Dropdown(choices=choices, value=audio_path)

	logger.info(
		f"Generated {audio_path} (seed={seed_used}, wall={wall:.2f}s, "
		f"history_size={len(history)})"
	)
	yield (
		audio_path,
		f"Done. Seed used: {seed_used}",
		tc_text,
		history,
		dropdown_update,
	)


def load_from_history(selected_path: str) -> str:
	"""When user picks a history item, return its path to load into player."""
	if not selected_path:
		return None
	logger.debug(f"Loading from history: {selected_path}")
	return selected_path


# --- UI helpers -----------------------------------------------------------
def on_genre_change(genre: str) -> str:
	default = GENRE_DEFAULT_LEAD.get(genre, "vocals")
	logger.debug(f"Genre={genre!r} -> lead_instrument={default!r}")
	return default


# --- UI -------------------------------------------------------------------
def build_ui() -> gr.Blocks:
	with gr.Blocks(title="RSM Lab v0.2") as app:
		gr.Markdown(
			"# RSM generation-service lab — Phase 2b v0.2\n"
			"Structured-prompt lab for capability discovery. "
			"Edit fields on the left; the **caption preview** on "
			"the right updates automatically. The preview is what "
			"gets sent to the model — edit directly to tune."
		)

		# Session-level history of generated audio. Keyed by path.
		history_state = gr.State([])

		with gr.Row():
			# === LEFT: structured fields ===
			with gr.Column(scale=1):
				gr.Markdown(
					"### Identity\n"
					"_Who is this band, where, when. Affects overall character._"
				)
				genre = gr.Dropdown(
					GENRES, value=DEFAULTS["genre"], label="Genre",
					info="Broad style category.",
				)
				era = gr.Dropdown(
					ERAS, value=DEFAULTS["era"], label="Era",
					info=(
						"Decade or period. v0.1 finding: era is a weaker "
						"signal than genre. Don't rely on it for precision."
					),
				)
				energy = gr.Dropdown(
					ENERGIES, value=DEFAULTS["energy"], label="Energy",
					info=(
						"Overall intensity. Distinct from BPM — a slow "
						"song can still feel intense."
					),
				)
				mood = gr.Dropdown(
					MOODS, value=DEFAULTS["mood"], label="Mood",
					info=(
						"Emotional character. Combine freely with "
						"energy ('joyful intense' vs 'joyful mellow')."
					),
				)
				reference_artist = gr.Textbox(
					value=DEFAULTS["reference_artist"],
					label="Reference artist (optional)",
					placeholder="e.g. 'NWA', 'Robert Plant', 'early Beatles'",
					info=(
						"v0.1 finding: text-only artist matching is weak. "
						"Genre comes through, artist style does not. "
						"Real artist-style matching needs Cover mode "
						"(reference audio), not text tags."
					),
				)

				gr.Markdown(
					"### Instrumentation\n"
					"_What the song sounds like physically._"
				)
				lead_instrument = gr.Dropdown(
					LEAD_INSTRUMENTS,
					value=DEFAULTS["lead_instrument"],
					label="Lead instrument",
					info=(
						"The instrument carrying the main melody. "
						"Auto-suggested based on genre — override "
						"freely. Set to 'vocals' for vocal-led tracks."
					),
				)
				other_instruments = gr.Dropdown(
					OTHER_INSTRUMENTS,
					value=DEFAULTS["other_instruments"],
					label="Other instruments",
					multiselect=True,
					info="Supporting instruments. Stack as many as fits.",
				)

				gr.Markdown(
					"### Vocals\n"
					"_Vocal style and language. Set to instrumental "
					"to skip vocals entirely._"
				)
				vocal_style = gr.Dropdown(
					VOCAL_STYLES,
					value=DEFAULTS["vocal_style"],
					label="Vocal style",
					info=(
						"Arrangement of vocals. 'Instrumental' will "
						"replace lyrics with [Instrumental] tag."
					),
				)
				language = gr.Dropdown(
					LANGUAGES, value=DEFAULTS["language"], label="Language",
					info="ISO code for the vocals. 'unknown' = auto-detect.",
				)

				gr.Markdown(
					"### Production — Path A surface\n"
					"_The Phase 2 verdict-critical axis. If these tiers "
					"produce audibly different audio, fidelity-via-tags "
					"(Path A) is a viable gameplay mechanic._"
				)
				fidelity = gr.Dropdown(
					FIDELITY_TIERS,
					value=DEFAULTS["fidelity"],
					label="Fidelity tier",
					info=(
						"Production quality tier appended to caption. "
						"Compare two generations differing ONLY on this "
						"to see what it does."
					),
				)

				gr.Markdown(
					"### Metas\n"
					"_Numeric musical metadata. Leave defaults for "
					"genre-typical values._"
				)
				bpm = gr.Number(
					value=DEFAULTS["bpm"], label="BPM", precision=0,
					info=(
						"Beats per minute (30-300). Genre-typical: "
						"60-90 ballad, 110-130 pop, 140+ rock/dance."
					),
				)
				key = gr.Textbox(
					value=DEFAULTS["key"], label="Key",
					info=(
						"Musical key, e.g. 'E minor', 'C Major'. "
						"Leave empty for model choice."
					),
				)
				duration = gr.Number(
					value=DEFAULTS["duration"], label="Duration (s)",
					info=(
						"Length in seconds. 30s = fast iteration; "
						"120s = full song. Cost scales with duration."
					),
				)

				gr.Markdown(
					"### Lyrics\n"
					"_Free-form. [Verse]/[Chorus] tags help the model "
					"find structure. Lyrics generation belongs in a "
					"separate small text LLM (Phase 3 work); ACE-Step's "
					"LM is for audio codes, not text — too slow for "
					"text generation._"
				)
				theme = gr.Textbox(
					value=DEFAULTS["theme"],
					label="Lyrical theme (optional)",
					placeholder="e.g. 'rebellion against authority', 'missing someone'",
					info=(
						"One-line description of what the song is about. "
						"Folds into the caption."
					),
				)
				lyrics = gr.Textbox(
					value=DEFAULTS["lyrics"],
					label="Lyrics",
					lines=12,
					info="4-8 lines per section is typical.",
				)

			# === RIGHT: caption preview, knobs, output ===
			with gr.Column(scale=1):
				gr.Markdown(
					"### Caption preview\n"
					"_What gets sent to the model. Auto-fills from "
					"the fields on the left. Editable — direct edits "
					"get clobbered if you change a field after, so "
					"generate first or copy your edit somewhere._"
				)
				caption_preview = gr.Textbox(
					value=assemble_caption(
						DEFAULTS["genre"], DEFAULTS["era"],
						DEFAULTS["energy"], DEFAULTS["mood"],
						DEFAULTS["reference_artist"],
						DEFAULTS["lead_instrument"],
						DEFAULTS["other_instruments"],
						DEFAULTS["vocal_style"], DEFAULTS["language"],
						DEFAULTS["fidelity"], DEFAULTS["theme"],
					),
					label="Caption",
					lines=4,
					buttons=["copy"],
				)

				gr.Markdown(
					"### DiT knobs\n"
					"_Advanced. Defaults are tuned for turbo + 8GB VRAM._"
				)
				with gr.Accordion("Advanced", open=False):
					inference_steps = gr.Slider(
						1, 64, value=8, step=1,
						label="Inference steps",
						info=(
							"Diffusion steps. Higher = better quality, "
							"slower. Turbo's sweet spot is 8."
						),
					)
					guidance_scale = gr.Slider(
						1.0, 15.0, value=7.0, step=0.5,
						label="Guidance scale",
						info=(
							"How strongly to follow the caption. Turbo "
							"ignores this (forces 1.0). For non-turbo, "
							"5-9 is typical; higher = more rigid."
						),
					)
					seed = gr.Number(
						value=-1, label="Seed", precision=0,
						info=(
							"-1 = random. Same seed + same inputs = "
							"same output. Use to reproduce a result."
						),
					)
					shift = gr.Slider(
						1.0, 5.0, value=3.0, step=0.1,
						label="Timestep shift",
						info=(
							"Timestep distribution. 3.0 recommended for "
							"turbo. Subtle effect."
						),
					)
					infer_method = gr.Dropdown(
						["ode", "sde"], value="ode",
						label="Inference method",
						info=(
							"ODE = deterministic, faster. SDE = "
							"stochastic, varies between runs."
						),
					)
					thinking = gr.Checkbox(
						value=True, label="thinking",
						info=(
							"Master LM switch. ON: vocals get phonemes "
							"from lyrics. OFF: vocals come out as "
							"melodic 'oohs' (no real words)."
						),
					)
					use_cot_caption = gr.Checkbox(
						value=False, label="use_cot_caption",
						info=(
							"If ON, LM rewrites your caption before "
							"generating. Usually OFF so your edits stick."
						),
					)
					use_cot_metas = gr.Checkbox(
						value=False, label="use_cot_metas",
						info="If ON, LM regenerates BPM/key/duration.",
					)
					use_cot_language = gr.Checkbox(
						value=False, label="use_cot_language",
						info="If ON, LM re-detects vocal language.",
					)

				gr.Markdown(
					"### Generate\n"
					"_~70s per click on this hardware. Each generation "
					"appends to the history dropdown below for replay "
					"and side-by-side comparison._"
				)
				generate_btn = gr.Button(
					"Generate audio (~70s)",
					variant="primary",
					size="lg",
				)
				status = gr.Textbox(
					label="Status",
					interactive=False,
					lines=2,
					value="Ready. Click Generate to start.",
				)
				time_costs_out = gr.Textbox(
					label="Timing",
					interactive=False,
					lines=6,
				)

				gr.Markdown(
					"### Now playing / history\n"
					"_Generations from this session. Pick one to load it "
					"into the player below. Newest first; max 20 kept._"
				)
				history_dropdown = gr.Dropdown(
					label="Select a generation",
					choices=[],
					interactive=True,
					value=None,
				)
				audio_out = gr.Audio(
					label="Audio",
					interactive=False,
					type="filepath",
				)

		# === Wiring: caption preview auto-update ===
		structured_inputs = [
			genre, era, energy, mood, reference_artist,
			lead_instrument, other_instruments,
			vocal_style, language,
			fidelity,
			theme,
		]
		for component in structured_inputs:
			component.change(
				fn=assemble_caption,
				inputs=structured_inputs,
				outputs=caption_preview,
				show_progress="hidden",
			)

		# === Wiring: genre cascade ===
		genre.change(
			fn=on_genre_change,
			inputs=genre,
			outputs=lead_instrument,
			show_progress="hidden",
		)

		# === Wiring: generate button ===
		generate_btn.click(
			fn=do_generate,
			inputs=[
				caption_preview, lyrics,
				vocal_style, language,
				bpm, key, duration,
				inference_steps, guidance_scale, seed,
				shift, infer_method,
				thinking, use_cot_caption, use_cot_metas, use_cot_language,
				history_state,
			],
			outputs=[
				audio_out, status, time_costs_out,
				history_state, history_dropdown,
			],
			show_progress="full",
		)

		# === Wiring: history selection → load into player ===
		history_dropdown.change(
			fn=load_from_history,
			inputs=history_dropdown,
			outputs=audio_out,
			show_progress="hidden",
		)

	return app


if __name__ == "__main__":
	app = build_ui()
	logger.info("Launching Gradio app at http://127.0.0.1:7860 ...")
	app.queue().launch(
		server_name="127.0.0.1",
		server_port=7860,
		inbrowser=True,
		show_error=True,
		theme=gr.themes.Default(),
	)