from __future__ import annotations

from io import BytesIO
from datetime import datetime
from math import gcd
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import tempfile
import unicodedata
import uuid

import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf
import streamlit as st
from scipy.signal import resample_poly
from vocal_balance_ui import render_vocal_balance, prepare_full
from isolated_vocal_balance.core import measure as measure_balance
from isolated_vocal_balance.proposals import suggest as suggest_balance
from moss_memory import MEMORY_PATH, TOPICS, KINDS, add_note, delete_note, load_notes, select_notes
from moss_descriptions import DESCRIPTION_TASKS
from moss_audit import audit_description
from moss_linked import check_worker_response, resolve_explanations, audit_linked_rows
from moss_measurements import measurement_snapshot, context_for_clip, context_fingerprint, validate_context, bind_history_tables


SUPPORTED_FORMATS = [
    "wav", "wave", "flac", "aif", "aiff", "aifc", "mp3", "m4a", "aac",
    "ogg", "oga", "opus", "wma", "caf",
]
FORMAT_LABEL = ", ".join(f".{extension}" for extension in SUPPORTED_FORMATS)

# Broad coverage for useful mix clues, with enough detail to identify vocal regions.
# These are analysis bands, not EQ targets.
EQ_BANDS = [
    ("Rumble (30–60 Hz)", 30, 60, 42),
    ("Bass (60–120 Hz)", 60, 120, 85),
    ("Low body (120–250 Hz)", 120, 250, 173),
    ("Low-mid (250–500 Hz)", 250, 500, 354),
    ("Mid (500 Hz–1 kHz)", 500, 1000, 707),
    ("Upper-mid (1–2.5 kHz)", 1000, 2500, 1581),
    ("Presence (2.5–5 kHz)", 2500, 5000, 3536),
    ("Sibilance/edge (5–8 kHz)", 5000, 8000, 6325),
    ("Air (8–12 kHz)", 8000, 12000, 9800),
]

PLUGIN_PROFILE_PATH = Path(__file__).resolve().parent / "plugin_profile.json"
PLUGIN_FEEDBACK_PATH = Path(__file__).resolve().parent / "plugin_feedback.json"
FFMPEG_TIMEOUT_SECONDS = 300
DEFAULT_PLUGIN_PROFILE = {
    "De-essers": [
        "Waves DeEsser", "Waves RDeEsser", "Waves Sibilance", "Waves Sibilance-Live",
    ],
    "Equalizers / channel strips": [
        "UAD SSL E Channel Strip", "UAD SSL E Channel Strip Legacy",
        "UAD API Vision Channel Strip", "UAD API Vision Channel Strip Legacy",
        "UAD API 550A", "UAD API 560", "UAD Avalon VT-737sp",
        "UAD Century Tube Channel Strip", "UAD Precision Channel Strip",
        "UAD Pultec HLF-3C", "UAD Pultec MEQ-5", "UAD Pultec-Pro Legacy",
        "UAD Neve 1073", "UAD Neve 1073 Legacy", "UAD UA 610-A", "UAD UA 610-B",
        "Waves AudioTrack", "Waves LinEQ Broadband", "Waves LinEQ Lowband",
        "Waves PuigTec EQP1A", "Waves PuigTec MEQ5", "Waves Q1", "Waves Q2",
        "Waves Q3", "Waves Q4", "Waves Q6", "Waves Q8", "Waves Q10",
        "Waves RChannel", "Waves REQ 2", "Waves REQ 4", "Waves REQ 6",
        "Waves VEQ3", "Waves VEQ4",
    ],
    "Multiband dynamics": ["Waves C4", "Waves LinMB"],
    "Other UAD processors": ["UAD A-Type Multiband Enhancer"],
    "Compressors": [
        "UAD API 2500", "UAD API Vision Channel Strip", "UAD Avalon VT-737sp",
        "UAD Century Tube Channel Strip", "UAD dbx 160", "UAD Empirical Labs Distressor",
        "UAD Empirical Labs FATSO Jr", "UAD Empirical Labs FATSO Sr",
        "UAD Fairchild 660", "UAD Fairchild 670", "UAD Fairchild 670 Legacy",
        "UAD LA-6176 Channel Strip", "UAD Precision Channel Strip",
        "UAD SSL E Channel Strip", "UAD SSL E Channel Strip Legacy",
        "UAD Teletronix LA-2A", "UAD Teletronix LA-2A Gray", "UAD Teletronix LA-2A Legacy",
        "UAD Teletronix LA-2A Silver", "UAD Teletronix LA-3A", "UAD UA 175-B", "UAD UA 176",
        "UAD UA 1176 Rev A", "UAD UA 1176AE", "UAD UA 1176LN Legacy",
        "UAD UA 1176LN Rev E", "UAD UA 1176SE Legacy",
        "Waves C1 comp", "Waves C1 comp-gate", "Waves C1 gate", "Waves CLA-2A",
        "Waves CLA-3A", "Waves CLA-76", "Waves H-Comp", "Waves L1 limiter", "Waves L2",
        "Waves L3 MultiMaximizer", "Waves L3 UltraMaximizer", "Waves L3-LL Multi",
        "Waves L3-LL Ultra", "Waves L316", "Waves MV2", "Waves PuigChild 660",
        "Waves RCompressor", "Waves Renaissance Axx", "Waves RVox", "Waves VComp",
    ],
    "Sidechain compressors": ["Waves C1 comp-sc", "Waves C4"],
    "Level riders / vocal control": [
        "Waves Bass Rider", "Waves Bass Rider Live", "Waves MaxxVolume",
        "Waves Vocal Rider", "Waves Vocal Rider Live",
    ],
    "Other Waves tools": [
        "Waves Butch Vig Vocals", "Waves CLA Guitars", "Waves DeBreath", "Waves Doppler",
        "Waves Doubler2", "Waves Doubler4", "Waves EKramer DR", "Waves EMO-F2",
        "Waves EMO-Generator", "Waves EMO-Q4", "Waves Enigma", "Waves GTR Amp",
        "Waves GTR Amp 2Cab", "Waves GTR Stomp 2", "Waves GTR Stomp 4", "Waves GTR Stomp 6",
        "Waves GTR Tool Rack", "Waves GTR Tuner", "Waves GW ToneCentric", "Waves H-Delay",
        "Waves IR-L", "Waves Kramer HLS", "Waves Kramer PIE", "Waves Kramer Tape",
        "Waves LoAir", "Waves Maserati DRM", "Waves MaxxBass", "Waves MetaFilter",
        "Waves MetaFlanger", "Waves MondoMod", "Waves Morphoder", "Waves OneKnob Driver",
        "Waves OneKnob Filter", "Waves OneKnob Pumper", "Waves PAZ-Frequency",
        "Waves PAZ-Meters", "Waves PS22 Split", "Waves PS22 Spread", "Waves PS22 Spread(10)",
        "Waves PS22 XSplit", "Waves Q-Capture", "Waves Q-Clone", "Waves RBass",
        "Waves Smack Attack", "Waves SoundShifter Pitch", "Waves Submarine",
        "Waves SuperTap 2-Taps", "Waves SuperTap 6-Taps", "Waves TransX Multi",
        "Waves TransX Wide", "Waves TrueVerb", "Waves UltraPitch 3 Voices",
        "Waves UltraPitch 6 Voices", "Waves UltraPitch Shift", "Waves Vitamin",
        "Waves VU Meter", "Waves Tune LT", "Waves WLM Meter", "Waves WLM Plus",
    ],
}


def load_plugin_profile() -> dict:
    if PLUGIN_PROFILE_PATH.is_file():
        try:
            data = json.loads(PLUGIN_PROFILE_PATH.read_text(encoding="utf-8"))
            saved = data.get("plugins", data) if isinstance(data, dict) else {}
            if not isinstance(saved, dict):
                saved = {}
            profile = {}
            for category, defaults in DEFAULT_PLUGIN_PROFILE.items():
                names = saved.get(category, defaults.copy())
                if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
                    names = defaults.copy()
                elif not data.get("version"):
                    names = list(names) + [name for name in defaults if name not in names]
                profile[category] = list(dict.fromkeys(name.strip() for name in names if name.strip()))
            if not data.get("version"):
                save_plugin_profile(profile)
            return profile
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    return {category: names.copy() for category, names in DEFAULT_PLUGIN_PROFILE.items()}


def save_plugin_profile(profile: dict) -> None:
    data = {"version": 1, "plugins": profile}
    write_json_atomic(PLUGIN_PROFILE_PATH, data)


def load_plugin_feedback() -> list[dict]:
    if not PLUGIN_FEEDBACK_PATH.is_file():
        return []
    try:
        data = json.loads(PLUGIN_FEEDBACK_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        return [
            item for item in data[-500:]
            if isinstance(item, dict)
            and isinstance(item.get("category"), str)
            and isinstance(item.get("plugin"), str)
            and isinstance(item.get("helpful"), bool)
        ]
    except (OSError, json.JSONDecodeError):
        return []


def save_plugin_feedback(feedback: list[dict]) -> None:
    write_json_atomic(PLUGIN_FEEDBACK_PATH, feedback[-500:])


def write_json_atomic(path: Path, data: object) -> None:
    """Replace a local JSON file atomically so interruptions cannot leave partial state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(data, temporary_file, ensure_ascii=False, indent=2, allow_nan=False)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


PLUGIN_FIT_TERMS = {
    ("De-essers", "sibilance"): ["sibilance", "rdeesser", "deesser"],
    ("Equalizers / channel strips", "eq_presence"): ["q3", "q6", "req", "lin eq", "ssl e", "api 550"],
    ("Equalizers / channel strips", "eq_body"): ["pultec", "veq", "avalon", "1073", "610", "ssl e", "api 550"],
    ("Compressors", "compress_fast"): ["1176", "cla-76", "distressor", "dbx 160"],
    ("Compressors", "compress_level"): ["la-2a", "la-3a", "rcompressor", "rvox", "avalon", "fairchild"],
    ("Sidechain compressors", "sidechain"): ["c1 comp-sc"],
    ("Multiband dynamics", "multiband_presence"): ["c4", "linmb"],
}


def choose_plugin(category: str, profile: dict, feedback: list[dict], issue_key: str = "") -> str | None:
    choices = profile.get(category, [])
    if not choices:
        return None
    preferred_terms = PLUGIN_FIT_TERMS.get((category, issue_key), [])
    scores = {name: (3 if any(term in name.casefold() for term in preferred_terms) else 0) for name in choices}
    for item in feedback:
        if item.get("category") == category and item.get("plugin") in scores:
            weight = 3 if item.get("issue") == issue_key else 1
            scores[item["plugin"]] += weight if item.get("helpful") else -weight
    return max(choices, key=lambda name: scores[name])


def render_plugin_feedback(category: str, plugin: str, issue_key: str, session_id: str, voice_name: str) -> None:
    with st.form(f"plugin_feedback_{session_id}_{voice_name}_{issue_key}"):
        result = st.selectbox(
            "Cuando lo pruebes, ¿te ayudó?",
            ["Aún no lo probé", "Sí, me ayudó", "No me ayudó"],
            key=f"plugin_feedback_choice_{session_id}_{voice_name}_{issue_key}",
        )
        submitted = st.form_submit_button("Guardar valoración")
    if submitted and result != "Aún no lo probé":
        feedback = load_plugin_feedback()
        feedback.append({
            "category": category,
            "plugin": plugin,
            "helpful": result == "Sí, me ayudó",
            "issue": issue_key,
            "voice": voice_name,
            "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        })
        save_plugin_feedback(feedback)
        st.success("Guardé tu valoración. La usaré para priorizar plugins que te han funcionado.")


def render_plugin_recommendations(report: dict, profile: dict) -> None:
    feedback = load_plugin_feedback()
    voice_comparisons = report.get("voice_comparisons") or {}
    if not voice_comparisons:
        return
    st.subheader("Sugerencias con tus plugins")
    st.caption("Elijo herramientas según el tipo de señal y el patrón medido, y priorizo tus valoraciones anteriores. Las mediciones son pistas para escuchar, no órdenes de ajuste.")
    suggestions_shown = False
    for voice_name, comparison in voice_comparisons.items():
        eq_rows = comparison.get("eq_insights") or []
        sibilance_rows = [
            row for row in eq_rows
            if row.get("Band") == "Sibilance/edge (5–8 kHz)"
            and isinstance(row.get("Relative offset vs section median (dB)"), (int, float))
            and row["Relative offset vs section median (dB)"] > 0
        ]
        sibilance_strength = [row["Relative offset vs section median (dB)"] for row in sibilance_rows]
        if sibilance_rows and (len(sibilance_rows) >= 2 or max(sibilance_strength) >= 8):
            plugin = choose_plugin("De-essers", profile, feedback, "sibilance")
            if plugin:
                suggestions_shown = True
                st.markdown(f"**{voice_name} · Sibilancia:** la banda de 5–8 kHz destaca en {len(sibilance_rows)} tramo(s); prueba **{plugin}** solo si las ‘s’ o ‘sh’ molestan. Empieza suave y evita que la voz cecee o pierda brillo.")
                render_plugin_feedback("De-essers", plugin, "sibilance", report.get("session_id", "current"), voice_name)

        non_sibilance_rows = [row for row in eq_rows if row.get("Band") != "Sibilance/edge (5–8 kHz)"]
        repeated_bands = {}
        for row in non_sibilance_rows:
            band = row.get("Band", "")
            offset = row.get("Relative offset vs section median (dB)")
            if band and isinstance(offset, (int, float)):
                sign = "stronger" if offset > 0 else "weaker"
                repeated_bands.setdefault((band, sign), []).append(offset)
        persistent = [(key, values) for key, values in repeated_bands.items() if len(values) >= 2]
        if persistent:
            band_order = {name: index for index, (name, *_rest) in enumerate(EQ_BANDS)}
            persistent.sort(key=lambda item: (-len(item[1]), band_order.get(item[0][0], 99)))
            selected_band, direction = persistent[0][0]
            issue_key = "eq_presence" if selected_band in {"Presence (2.5–5 kHz)", "Upper-mid (1–2.5 kHz)"} else "eq_body"
            plugin = choose_plugin("Equalizers / channel strips", profile, feedback, issue_key)
            if plugin:
                suggestions_shown = True
                direction_text = "la voz destaca frente al acompañamiento" if direction == "stronger" else "la voz queda relativamente por detrás del acompañamiento"
                st.markdown(f"**{voice_name} · EQ persistente ({selected_band}):** en {len(repeated_bands[(selected_band, direction)])} tramos {direction_text}. Si también lo oyes así, prueba **{plugin}** con un cambio pequeño y compara con bypass. El offset no es la cantidad de EQ.")
                render_plugin_feedback("Equalizers / channel strips", plugin, issue_key, report.get("session_id", "current"), voice_name)
        elif non_sibilance_rows:
            st.caption(f"{voice_name}: los avisos de EQ aparecen en tramos aislados; antes de cambiar la EQ fija, escucha si el cambio es puntual y considera automatización o control dinámico.")

        compression_rows = comparison.get("compression_insights") or []
        section_spreads = [
            row.get("Active vocal level spread (p90–p10 dB)")
            for row in comparison.get("sections", [])
            if isinstance(row.get("Active vocal level spread (p90–p10 dB)"), (int, float))
        ]
        wide_count = sum(value >= 8 for value in section_spreads)
        moderate_count = sum(5.5 <= value < 8 for value in section_spreads)
        if compression_rows:
            fast_peaks = wide_count >= 2 or (section_spreads and max(section_spreads) >= 12)
            issue_key = "compress_fast" if fast_peaks else "compress_level"
            plugin = choose_plugin("Compressors", profile, feedback, issue_key)
            if plugin:
                suggestions_shown = True
                if fast_peaks:
                    behavior = f"hay {wide_count} tramo(s) con dispersión amplia"
                    starting_point = "para sujetar picos, con ataque y reducción moderados"
                else:
                    behavior = f"hay {moderate_count} tramo(s) con dispersión moderada"
                    starting_point = "para nivelar suavemente"
                st.markdown(f"**{voice_name} · Dinámica ({behavior}):** si al escuchar las frases todavía saltan, prueba **{plugin}** {starting_point}. Primero ajusta clip gain/automatización; la app no calcula la reducción de ganancia del compresor.")
                render_plugin_feedback("Compressors", plugin, issue_key, report.get("session_id", "current"), voice_name)

        section_deltas = [
            row.get("Vocal-to-instrumental RMS (dB)")
            for row in comparison.get("sections", [])
            if row.get("Section context") == "Accompaniment active"
            and row.get("Vocal-to-instrumental RMS (dB)") is not None
        ]
        if len(section_deltas) >= 3 and float(np.median(section_deltas)) <= -3.0:
            plugin = choose_plugin("Sidechain compressors", profile, feedback, "sidechain")
            if plugin:
                suggestions_shown = True
                st.markdown(f"**{voice_name} · Balance de voz:** en {len(section_deltas)} tramos con acompañamiento la mediana está en {np.median(section_deltas):.1f} dB. Si la voz se pierde al escuchar, prueba **{plugin}** en la instrumental con ducking muy sutil, disparado por la voz.")
                render_plugin_feedback("Sidechain compressors", plugin, "sidechain", report.get("session_id", "current"), voice_name)
    if not suggestions_shown:
        st.caption("No hay avisos que activen una sugerencia de plugin para este análisis.")

st.set_page_config(page_title="Vocal Mix Check", page_icon="🎚️", layout="wide")
def read_audio(upload, measure_loudness: bool = False):
    """Decode common audio formats with FFmpeg; optionally return EBU R128 loudness data."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    audio_bytes = upload.getvalue()
    if not audio_bytes:
        raise ValueError("The selected file is empty. Choose an audio file with a nonzero size.")

    def decode(command):
        try:
            return subprocess.run(
                command,
                input=audio_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=FFMPEG_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError(
                "Audio decoding took longer than five minutes. Export this track as WAV or FLAC and try again."
            ) from exc
        except OSError as exc:
            raise ValueError("The bundled FFmpeg could not start. Restart the app and try again.") from exc

    command = [ffmpeg, "-hide_banner", "-nostats", "-loglevel", "info" if measure_loudness else "error", "-i", "pipe:0"]
    if measure_loudness:
        # The filter passes audio through while logging integrated loudness and LRA,
        # so the mix is decoded only once for both the RMS and LUFS measurements.
        command.extend(["-af", "ebur128=framelog=quiet"])
    command.extend(["-map", "0:a:0", "-vn", "-c:a", "pcm_f32le", "-f", "wav", "pipe:1"])
    result = decode(command)
    loudness = {}
    warning = None
    if measure_loudness and result.returncode == 0 and result.stdout:
        log = result.stderr.decode("utf-8", errors="replace")
        integrated = re.search(r"\bI:\s*(-?\d+(?:\.\d+)?)\s+LUFS", log)
        loudness_range = re.search(r"\bLRA:\s*(-?\d+(?:\.\d+)?)\s+LU\b", log)
        if integrated:
            loudness["Integrated loudness (LUFS)"] = float(integrated.group(1))
        if loudness_range:
            loudness["Loudness range (LRA, LU)"] = float(loudness_range.group(1))
        if not loudness:
            warning = "FFmpeg decoded the mix, but did not return EBU R128 loudness values."
    elif measure_loudness:
        warning = "The bundled FFmpeg could not measure EBU R128 loudness; the app will still analyze the mix's other metrics."
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
            "-map", "0:a:0", "-vn", "-c:a", "pcm_f32le", "-f", "wav", "pipe:1",
        ]
        result = decode(command)
    if result.returncode != 0 or not result.stdout:
        details = result.stderr.decode("utf-8", errors="replace").strip()
        details = details[-1200:] if details else ""
        raise ValueError(details or "The file could not be decoded. Try exporting it as WAV or FLAC.")
    try:
        data, sr = sf.read(BytesIO(result.stdout), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError("The audio was decoded, but its samples could not be read.") from exc
    if data.size == 0:
        raise ValueError("The uploaded file contains no audio samples.")
    if sr <= 0 or not np.isfinite(data).all():
        raise ValueError("The decoded audio contains invalid sample data. Re-export it as WAV or FLAC and try again.")
    if measure_loudness:
        return data.T, int(sr), loudness, warning
    return data.T, int(sr)


def mono(audio: np.ndarray) -> np.ndarray:
    return np.mean(audio, axis=0, dtype=np.float32)


def combine_vocal_uploads(uploads) -> tuple[np.ndarray, int, list[str]]:
    """Sum vocal tracks that share a common start point."""
    _, native, rate = prepare_full(uploads, read_audio)
    return native.T, rate, []


def db(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), 1e-8)))


def resample(audio: np.ndarray, old_sr: int, new_sr: int) -> np.ndarray:
    if old_sr == new_sr:
        return audio
    common = gcd(old_sr, new_sr)
    return resample_poly(audio, new_sr // common, old_sr // common, axis=1).astype(np.float32)


def mix_metrics(audio: np.ndarray) -> dict[str, float]:
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    metrics = {
        "Peak (dBFS)": db(peak),
        "Average RMS (dBFS)": db(rms),
        "Crest factor (dB)": db(peak) - db(rms),
        "Samples at/above -0.1 dBFS (%)": 100.0 * float(np.mean(np.abs(audio) >= 0.9886)),
    }
    if audio.shape[0] >= 2:
        left, right = audio[0], audio[1]
        mid = (left + right) * 0.5
        side = (left - right) * 0.5
        metrics["Side-to-mid level (dB)"] = db(np.sqrt(np.mean(side**2))) - db(np.sqrt(np.mean(mid**2)))
    return metrics


def available_audio_inputs() -> tuple[list[dict], str | None]:
    """List local input devices; audio remains on this machine."""
    try:
        import sounddevice as sd

        devices = sd.query_devices()
    except Exception as exc:
        return [], str(exc)
    inputs = []
    for index, device in enumerate(devices):
        try:
            channels = int(device.get("max_input_channels", 0))
            if channels > 0:
                inputs.append({
                    "index": index,
                    "name": str(device.get("name", f"Audio input {index}")),
                    "channels": channels,
                    "sample_rate": int(round(float(device.get("default_samplerate", 44100)))),
                })
        except (TypeError, ValueError, OverflowError):
            continue
    return inputs, None


def capture_audio_input(device_index: int, seconds: int) -> tuple[np.ndarray, int]:
    """Record a short input buffer in memory; no audio file is created."""
    try:
        import sounddevice as sd

        device = sd.query_devices(device_index)
        channels = min(2, int(device["max_input_channels"]))
        sample_rate = int(round(float(device["default_samplerate"])))
        if channels < 1 or sample_rate < 8000:
            raise ValueError("The selected device does not report a usable audio input.")
        sd.check_input_settings(device=device_index, channels=channels, samplerate=sample_rate)
        frames = int(seconds * sample_rate)
        audio = sd.rec(
            frames,
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            device=device_index,
            blocking=True,
        )
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        raise ValueError(
            "Could not capture from that input. Check that the device is connected, selected in Logic, "
            "and allowed in macOS microphone privacy settings. " + message[:500]
        ) from exc
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("The captured input was empty or invalid. Check the Logic output routing and try again.")
    return audio.T.copy(), sample_rate


def band_levels(
    audio_mono: np.ndarray,
    sr: int,
    n_fft: int | None = None,
    hop: int | None = None,
    frame_mask: np.ndarray | None = None,
    spectrum: np.ndarray | None = None,
) -> dict[str, float]:
    if n_fft is None:
        n_fft = min(8192, max(1024, 2 ** int(np.floor(np.log2(max(1024, len(audio_mono)))))))
    if hop is None:
        hop = n_fft // 2
    if spectrum is None:
        spectrum = np.abs(librosa.stft(audio_mono, n_fft=n_fft, hop_length=hop))
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    frame_count = spectrum.shape[1]
    selected_frames = np.ones(frame_count, dtype=bool) if frame_mask is None else frame_mask[:frame_count]
    out: dict[str, float] = {}
    for name, low, high, _center in EQ_BANDS:
        frequency_mask = (frequencies >= low) & (frequencies < min(high, sr / 2))
        if not np.any(frequency_mask) or not np.any(selected_frames):
            out[name] = float("nan")
        else:
            band = spectrum[frequency_mask, :frame_count][:, selected_frames]
            out[name] = db(float(np.sqrt(np.mean(band ** 2))))
    return out


def inspect_stems(vocal: np.ndarray, vocal_sr: int, music: np.ndarray, music_sr: int) -> tuple[dict, list[str]]:
    notes: list[str] = []
    # Use exactly the same channel handling, resampling and frame measurements as A/B.
    balance_voice, _, _ = prepare_full([Path("voice")],
                                       lambda _: (vocal, vocal_sr))
    balance_music, _, _ = prepare_full([Path("instrumental")],
                                       lambda _: (music, music_sr))
    balance = measure_balance(balance_voice, balance_music)
    balance_proposals = {p["start"]: p for p in suggest_balance(balance_voice, balance_music)}
    if vocal_sr != music_sr:
        music = resample(music, music_sr, vocal_sr)
        notes.append(f"Instrumental was resampled from {music_sr} Hz to match the vocal at {vocal_sr} Hz.")
    v, m = mono(vocal), mono(music)
    duration_v, duration_m = len(v) / vocal_sr, len(m) / vocal_sr
    if abs(duration_v - duration_m) > 0.10:
        notes.append(f"Stem durations differ by {abs(duration_v-duration_m):.2f} s; analysis uses their shared length. Check that both exports start at the same point.")
    n = min(len(v), len(m))
    v, m = v[:n], m[:n]

    frame = min(8192, max(1024, 2 ** int(np.floor(np.log2(max(1024, int(vocal_sr * 0.25)))))))
    hop = frame // 2
    # Build each full-track spectrum once. Section EQ summaries below reuse these
    # arrays instead of recomputing two full STFTs for every 15-second section.
    vocal_spectrum = np.abs(librosa.stft(v, n_fft=frame, hop_length=hop))
    music_spectrum = np.abs(librosa.stft(m, n_fft=frame, hop_length=hop))
    vr = librosa.feature.rms(y=v, frame_length=frame, hop_length=hop, center=True)[0]
    mr = librosa.feature.rms(y=m, frame_length=frame, hop_length=hop, center=True)[0]
    count = min(len(vr), len(mr))
    vr, mr = vr[:count], mr[:count]
    vocal_db = 20 * np.log10(np.maximum(vr, 1e-8))
    music_db = 20 * np.log10(np.maximum(mr, 1e-8))

    vocal_active = vr > max(float(vr.max()) * 0.035, 1e-6)
    music_active = mr > max(float(mr.max()) * 0.01, 1e-6)
    if np.any(music_active):
        typical_music_db = float(np.percentile(music_db[music_active], 75))
        music_present = music_active & (music_db >= typical_music_db - 10.0)
    else:
        typical_music_db = None
        music_present = np.zeros(count, dtype=bool)

    delta = vocal_db - music_db
    comparable = vocal_active & music_present
    segment_frames = max(1, int(15.0 * vocal_sr / hop))
    segment_rows = []
    reduced_sections = np.zeros(count, dtype=bool)
    eq_insights = []
    compression_insights = []

    for start in range(0, count, segment_frames):
        stop = min(count, start + segment_frames)
        section = slice(start, stop)
        voice_frames = vocal_active[section]
        if not np.any(voice_frames):
            continue

        voice_values = vocal_db[section][voice_frames]
        voice_spread = float(np.percentile(voice_values, 90) - np.percentile(voice_values, 10))
        section_music_present = music_present[section]
        reduced_fraction = float(np.mean(~section_music_present[voice_frames]))
        accompaniment_reduced = reduced_fraction >= 0.5
        if accompaniment_reduced:
            reduced_sections[section] = True

        compare_frames = comparable[section]
        level_delta = float(np.median(delta[section][compare_frames])) if np.any(compare_frames) and not accompaniment_reduced else None
        t0 = start * hop / vocal_sr
        t1 = min(n / vocal_sr, stop * hop / vocal_sr)
        timestamp = f"{int(t0//60):02d}:{int(t0%60):02d}–{int(t1//60):02d}:{int(t1%60):02d}"
        instrumental_activity = float(np.mean(music_active[section]))
        section_music_levels = music_db[section][voice_frames & music_active[section]]
        level_vs_typical = (
            float(np.median(section_music_levels) - typical_music_db)
            if typical_music_db is not None and len(section_music_levels)
            else None
        )
        segment_rows.append({
            "Time": timestamp,
            "Section context": "Reduced / solo accompaniment" if accompaniment_reduced else "Accompaniment active",
            "Detectable instrumental frames (%)": round(instrumental_activity * 100, 1),
            "Instrumental level vs typical active level (dB)": None if level_vs_typical is None else round(level_vs_typical, 1),
            "Vocal-to-instrumental RMS (dB)": None if level_delta is None else round(level_delta, 1),
            "Active vocal level spread (p90–p10 dB)": round(voice_spread, 1),
        })

        if voice_spread >= 8.0:
            move = (
                "Wide level spread: check clip gain/automation first. If phrases still jump, try 2:1–3:1 compression; "
                "set threshold for about 2–3 dB gain reduction on loud phrases."
            )
        elif voice_spread >= 5.5:
            move = (
                "Moderate level spread: audition light compression around 2:1, aiming for about 1–2 dB gain reduction; "
                "skip it if the phrase already sits steadily."
            )
        else:
            move = None
        if move:
            compression_insights.append({
                "Time": timestamp,
                "Section context": segment_rows[-1]["Section context"],
                "Active vocal level spread (p90–p10 dB)": round(voice_spread, 1),
                "Starting move": move,
            })

        if accompaniment_reduced:
            continue

        section_frame_mask = np.zeros(count, dtype=bool)
        section_frame_mask[start:stop] = comparable[section]
        section_vocal_bands = band_levels(v, vocal_sr, frame, hop, section_frame_mask, vocal_spectrum)
        section_music_bands = band_levels(m, vocal_sr, frame, hop, section_frame_mask, music_spectrum)
        section_balance = {
            name: section_vocal_bands[name] - section_music_bands[name]
            for name in section_vocal_bands
            if np.isfinite(section_vocal_bands[name])
            and np.isfinite(section_music_bands[name])
            and section_vocal_bands[name] > -90
            and section_music_bands[name] > -90
        }
        if len(section_balance) >= 3:
            balance_midpoint = float(np.median(list(section_balance.values())))
            deviations = [
                (name, value - balance_midpoint)
                for name, value in section_balance.items()
                if abs(value - balance_midpoint) >= 4.0
                and next(item[3] for item in EQ_BANDS if item[0] == name) >= 120
            ]
            for name, deviation in sorted(deviations, key=lambda item: abs(item[1]), reverse=True)[:2]:
                band_center = next(item[3] for item in EQ_BANDS if item[0] == name)
                if deviation > 0:
                    if 5000 <= band_center < 8000:
                        move = f"If sibilant or edgy, try a de-esser near {band_center:,} Hz (about 1–2 dB reduction)."
                    else:
                        move = f"If this band sounds too strong, try a broad −1 dB cut near {band_center:,} Hz."
                else:
                    move = f"If the vocal is masked here, try a broad +1 dB vocal boost near {band_center:,} Hz or a small instrumental cut."
                eq_insights.append({
                    "Time": timestamp,
                    "Band": name,
                    "Relative offset vs section median (dB)": round(float(deviation), 1),
                    "Audition this move": move,
                })

    analysis_frames = comparable & ~reduced_sections
    compare_deltas = delta[analysis_frames]
    active_voice_values = vocal_db[vocal_active]
    if len(compare_deltas):
        median_delta = float(np.median(compare_deltas))
        p10_delta = float(np.percentile(compare_deltas, 10))
        p90_delta = float(np.percentile(compare_deltas, 90))
        vb = band_levels(v, vocal_sr, frame, hop, analysis_frames, vocal_spectrum)
        mb = band_levels(m, vocal_sr, frame, hop, analysis_frames, music_spectrum)
        bands = {
            name: round(vb[name] - mb[name], 1)
            for name in vb
            if np.isfinite(vb[name]) and np.isfinite(mb[name])
        }
    else:
        median_delta = p10_delta = p90_delta = None
        bands = {}
        notes.append("No sections had enough accompaniment for a reliable vocal-to-instrumental comparison. Balance and frequency comparisons are omitted; vocal dynamics are still measured.")

    vocal_dynamic_spread = (
        float(np.percentile(active_voice_values, 90) - np.percentile(active_voice_values, 10))
        if len(active_voice_values) else None
    )
    result = {
        "duration": n / vocal_sr,
        "balance_method": "stereo_rms_50ms_5s_v1",
        "median_delta": balance["baseline_db"],
        "p10_delta": balance["p10_db"],
        "p90_delta": balance["p90_db"],
        "balance_rows": [
            {"Inicio (s)": row["start"], "Fin (s)": row["end"],
             "Balance medido (dB)": row["ratio_db"],
             "Referencia (dB)": balance["baseline_db"],
             "Desvío (dB)": row["difference_db"],
             "Ajuste propuesto (dB)": balance_proposals.get(row["start"], {}).get("gain_db"),
             "Comparable (%)": row["comparable_pct"]}
            for row in balance["rows"]
        ],
        "vocal_dynamic_spread": vocal_dynamic_spread,
        "sections": [{key: value for key, value in row.items()
                      if key != "Vocal-to-instrumental RMS (dB)"} for row in segment_rows],
        "bands": bands,
        "eq_insights": eq_insights,
        "compression_insights": compression_insights,
    }
    return result, notes


def metric_table(metrics: dict[str, float]):
    st.dataframe(
        [{"Measurement": key, "Value": round(value, 2)} for key, value in metrics.items()],
        hide_index=True,
        width="stretch",
    )


SESSIONS_DIR = Path(__file__).resolve().parent / "saved_sessions"


def safe_filename(name: str) -> str:
    base = Path(name).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return cleaned or "audio_file"


def track_identity(upload) -> dict:
    """Identify a mix without retaining its audio: normalized filename plus a content digest."""
    stem = Path(upload.name).stem
    normalized = unicodedata.normalize("NFKD", stem.casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    name_key = re.sub(r"[^a-z0-9]+", "", normalized)
    return {
        "name_key": name_key,
        "sha256": hashlib.sha256(upload.getvalue()).hexdigest(),
    }


def find_previous_track_report(identity: dict, sessions: list[tuple[Path, dict]]):
    """Prefer exact audio matches; fall back to a matching normalized mix filename."""
    exact_matches = []
    name_matches = []
    for session_dir, report in sessions:
        old_identity = report.get("track_identity") or {}
        if identity.get("sha256") and identity["sha256"] == old_identity.get("sha256"):
            exact_matches.append((session_dir, report))
        elif identity.get("name_key") and identity["name_key"] == old_identity.get("name_key"):
            name_matches.append((session_dir, report))
        elif not old_identity and identity.get("name_key"):
            old_name = report.get("mix", {}).get("name", "")
            old_stem = unicodedata.normalize("NFKD", Path(old_name).stem.casefold())
            old_stem = "".join(char for char in old_stem if not unicodedata.combining(char))
            if identity["name_key"] == re.sub(r"[^a-z0-9]+", "", old_stem):
                name_matches.append((session_dir, report))
    if exact_matches:
        return exact_matches[0], "audio"
    if name_matches:
        return name_matches[0], "filename"
    return None, None


def comparison_rows(current: dict, previous: dict) -> list[dict]:
    """Build a compact numeric diff of two reports, keyed by stable section/band labels."""
    rows = []

    def add(label, old_value, new_value, suffix=""):
        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            old_rounded, new_rounded = round(float(old_value), 1), round(float(new_value), 1)
            change = round(new_rounded - old_rounded, 1)
            if abs(change) >= 0.1:
                rows.append({
                    "Measurement": label,
                    "Previous": old_rounded,
                    "Current": new_rounded,
                    "Change": change,
                    "Unit": suffix,
                })

    old_mix = previous.get("mix", {})
    new_mix = current.get("mix", {})
    add("Stereo mix · Duration", old_mix.get("duration_seconds"), new_mix.get("duration_seconds"), "s")
    add("Stereo mix · Sample rate", old_mix.get("sample_rate"), new_mix.get("sample_rate"), "Hz")
    add("Stereo mix · Channels", old_mix.get("channels"), new_mix.get("channels"))
    old_metrics, new_metrics = old_mix.get("metrics", {}), new_mix.get("metrics", {})
    for key in sorted(set(old_metrics) & set(new_metrics)):
        add(f"Stereo mix · {key}", old_metrics[key], new_metrics[key])

    old_voices = previous.get("voice_comparisons") or {}
    new_voices = current.get("voice_comparisons") or {}
    voice_fields = [
        ("median_delta", "Median vocal-to-instrumental RMS", "dB"),
        ("p10_delta", "Quieter comparable moments", "dB"),
        ("p90_delta", "Louder comparable moments", "dB"),
        ("vocal_dynamic_spread", "Vocal level spread (voice only)", "dB"),
    ]
    for voice_name in sorted(set(old_voices) & set(new_voices)):
        old_voice, new_voice = old_voices[voice_name], new_voices[voice_name]
        for field, label, unit in voice_fields:
            add(f"{voice_name} · {label}", old_voice.get(field), new_voice.get(field), unit)
        for band in sorted(set(old_voice.get("bands", {})) & set(new_voice.get("bands", {}))):
            add(f"{voice_name} · Band {band}", old_voice["bands"][band], new_voice["bands"][band], "dB")

        def section_map(voice):
            result = {}
            for section in voice.get("sections", []):
                time = section.get("Time")
                if not time:
                    continue
                for field in (
                    "Instrumental level vs typical active level (dB)",
                    "Vocal-to-instrumental RMS (dB)",
                    "Active vocal level spread (p90–p10 dB)",
                ):
                    result[(time, field)] = section.get(field)
            return result

        old_sections, new_sections = section_map(old_voice), section_map(new_voice)
        for time, field in sorted(set(old_sections) & set(new_sections)):
            add(f"{voice_name} · {time} · {field}", old_sections[(time, field)], new_sections[(time, field)], "dB")

        def eq_map(voice):
            return {
                (item.get("Time", ""), item.get("Band", "")): item.get("Relative offset vs section median (dB)")
                for item in voice.get("eq_insights", [])
            }

        old_eq, new_eq = eq_map(old_voice), eq_map(new_voice)
        for time, band in sorted(set(old_eq) & set(new_eq)):
            add(f"{voice_name} · EQ offset · {time} · {band}", old_eq[(time, band)], new_eq[(time, band)], "dB")
    return rows


def render_change_comparison(report: dict) -> None:
    comparison = report.get("change_comparison")
    if not comparison:
        return
    st.subheader("Cambios desde el análisis anterior de esta pista")
    if comparison.get("matched_by") == "audio":
        st.caption("La app reconoció el mismo audio. No guarda el archivo: conserva solo una huella SHA-256 y las mediciones del informe.")
    else:
        st.caption("La app encontró el mismo nombre de mezcla. Confirma que sea la misma canción; guarda la huella y las mediciones, no el audio.")
    st.caption(f"Comparación con {comparison.get('created_at', 'el informe anterior')} · {comparison.get('mix_name', 'análisis anterior')}. Las filas rojas cambiaron; la diferencia es valor actual menos valor anterior.")
    rows = comparison.get("rows", [])
    if not rows:
        st.info("No hay cambios medibles con la precisión mostrada (0,1 unidades).")
        return
    import pandas as pd
    frame = pd.DataFrame(rows)
    frame["Previous"] = frame.apply(lambda row: f"{row['Previous']:g} {row['Unit']}".strip(), axis=1)
    frame["Current"] = frame.apply(lambda row: f"{row['Current']:g} {row['Unit']}".strip(), axis=1)
    frame["Change"] = frame.apply(lambda row: f"{row['Change']:+g} {row['Unit']}".strip(), axis=1)
    frame = frame.drop(columns=["Unit"])
    def mark_changed(_row):
        return ["background-color: #4a1820; color: #ffb3bd; font-weight: 600"] * len(_row)
    st.dataframe(frame.style.apply(mark_changed, axis=1), hide_index=True, width="stretch")


def describe_upload(upload, label: str, index: int | None = None) -> dict:
    display_label = label if index is None else f"{label} {index}"
    return {"label": display_label, "name": safe_filename(upload.name)}


def save_analysis(report: dict, mix_file, voice_1_files, voice_2_files, instrumental_file, reference_file) -> Path:
    """Save report metadata only; uploaded audio is never copied to disk."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(exist_ok=False)

    files = [describe_upload(mix_file, "mix")]
    for index, upload in enumerate(voice_1_files, start=1):
        files.append(describe_upload(upload, "Voice 1", index))
    for index, upload in enumerate(voice_2_files, start=1):
        files.append(describe_upload(upload, "Voice 2", index))
    if instrumental_file is not None:
        files.append(describe_upload(instrumental_file, "instrumental"))
    if reference_file is not None:
        files.append(describe_upload(reference_file, "reference"))

    report["session_id"] = session_id
    report["created_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    report["files"] = files
    report["audio_retained"] = False
    try:
        write_json_atomic(session_dir / "report.json", report)
    except Exception:
        try:
            session_dir.rmdir()
        except OSError:
            pass
        raise
    return session_dir


def saved_audio_available(session_dir: Path, report: dict) -> bool:
    files = report.get("files", [])
    if not isinstance(files, list) or not files:
        return False
    for item in files:
        if not isinstance(item, dict):
            return False
        saved_name = item.get("saved_name")
        if not isinstance(saved_name, str) or not saved_name or Path(saved_name).name != saved_name:
            return False
        if not (session_dir / saved_name).is_file():
            return False
    return True


def load_saved_sessions() -> list[tuple[Path, dict]]:
    if not SESSIONS_DIR.exists():
        return []
    sessions = []
    try:
        session_dirs = list(SESSIONS_DIR.iterdir())
    except OSError:
        return []
    for session_dir in session_dirs:
        report_path = session_dir / "report.json"
        if session_dir.is_dir() and report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if not isinstance(report, dict):
                    continue
                mix = report.get("mix")
                if not isinstance(mix, dict) or not all(
                    key in mix for key in ("duration_seconds", "sample_rate", "channels", "metrics")
                ) or not isinstance(mix.get("metrics"), dict):
                    continue
                report.setdefault("session_id", session_dir.name)
                if not isinstance(report.get("session_id"), str):
                    report["session_id"] = session_dir.name
                if not isinstance(report.get("files"), list):
                    report["files"] = []
                if not isinstance(report.get("warnings"), list):
                    report["warnings"] = []
                if not isinstance(report.get("voice_comparisons"), dict):
                    report["voice_comparisons"] = {}
                sessions.append((session_dir, report))
            except (OSError, json.JSONDecodeError, TypeError):
                continue
    return sorted(
        sessions,
        key=lambda item: item[1].get("last_analyzed_at", item[1].get("created_at", "")),
        reverse=True,
    )


class SavedUpload:
    """Small file-like wrapper so saved local audio can use the same analysis code."""
    def __init__(self, path: Path, original_name: str):
        self.path = path
        self.name = original_name

    def getvalue(self) -> bytes:
        return self.path.read_bytes()


def restore_saved_inputs(session_dir: Path, report: dict) -> dict:
    inputs = {
        "mix": None,
        "voice_1": [],
        "voice_2": [],
        "instrumental": None,
        "reference": None,
    }
    for saved_file in report.get("files", []):
        label = str(saved_file.get("label", "")).lower()
        saved_name = Path(saved_file.get("saved_name", "")).name
        path = session_dir / saved_name
        if not path.is_file():
            continue
        item = SavedUpload(path, saved_file.get("name", path.name))
        if label == "mix":
            inputs["mix"] = item
        elif label.startswith(("voice 1", "vocal 1", "vocal_1")):
            inputs["voice_1"].append(item)
        elif label.startswith(("voice 2", "vocal 2", "vocal_2")):
            inputs["voice_2"].append(item)
        elif label == "instrumental":
            inputs["instrumental"] = item
        elif label == "reference":
            inputs["reference"] = item
    return inputs


def reanalyze_saved_session(session_dir: Path, previous_report: dict) -> dict:
    inputs = restore_saved_inputs(session_dir, previous_report)
    if inputs["mix"] is None:
        raise ValueError("The saved session does not contain its mix file.")
    report = make_report(
        inputs["mix"], inputs["voice_1"], inputs["voice_2"],
        inputs["instrumental"], inputs["reference"],
    )
    report["session_id"] = previous_report["session_id"]
    report["created_at"] = previous_report.get("created_at", "")
    report["last_analyzed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    report["files"] = previous_report.get("files", [])
    write_json_atomic(session_dir / "report.json", report)
    return report


def make_report(mix_file, voice_1_files, voice_2_files, instrumental_file, reference_file) -> dict:
    mix, mix_sr, loudness, loudness_warning = read_audio(mix_file, measure_loudness=True)
    report = {
        "mix": {
            "name": mix_file.name,
            "duration_seconds": mix.shape[1] / mix_sr,
            "sample_rate": mix_sr,
            "channels": mix.shape[0],
            "metrics": {**mix_metrics(mix), **loudness},
        },
        "voice_comparisons": {},
        "warnings": [],
        "reference": None,
    }
    if loudness_warning:
        report["warnings"].append(loudness_warning)
    if mix.shape[0] != 2:
        report["warnings"].append(
            f"The current mix has {mix.shape[0]} channel(s), not stereo. Check the export before using stereo-width measurements."
        )

    voice_groups = {"Voice 1": voice_1_files, "Voice 2": voice_2_files}
    has_any_voice = any(voice_groups.values())
    if has_any_voice and instrumental_file is None:
        report["warnings"].append("Upload the instrumental stem to compare the vocal groups.")
    elif has_any_voice and instrumental_file is not None:
        music, music_sr = read_audio(instrumental_file)
        for voice_name, uploads in voice_groups.items():
            if not uploads:
                continue
            vocal, vocal_sr, combine_notes = combine_vocal_uploads(uploads)
            result, stem_notes = inspect_stems(vocal, vocal_sr, music, music_sr)
            report["voice_comparisons"][voice_name] = result
            report["warnings"].extend(f"{voice_name}: {note}" for note in combine_notes + stem_notes)

    if reference_file is not None:
        reference, reference_sr = read_audio(reference_file)
        report["reference"] = {
            "name": reference_file.name,
            "duration_seconds": reference.shape[1] / reference_sr,
            "sample_rate": reference_sr,
            "channels": reference.shape[0],
        }
    return report


def render_one_voice_comparison(voice_name: str, comparison: dict) -> None:
    st.subheader(f"{voice_name} and instrumental comparison")
    st.caption("These relative levels are descriptive. They are not target values and do not establish whether a vocal sounds right.")
    a, b, c, d = st.columns(4)
    median = comparison.get("median_delta")
    p10 = comparison.get("p10_delta")
    p90 = comparison.get("p90_delta")
    dynamics = comparison.get("vocal_dynamic_spread")
    a.metric("Median vocal-to-instrumental RMS", "—" if median is None else f"{median:.1f} dB")
    b.metric("Quieter comparable moments", "—" if p10 is None else f"{p10:.1f} dB")
    c.metric("Louder comparable moments", "—" if p90 is None else f"{p90:.1f} dB")
    d.metric("Vocal level spread (voice only)", "—" if dynamics is None else f"{dynamics:.1f} dB")

    st.markdown("**Relative energy by frequency range** — only sections with the accompaniment present")
    if median is None:
        st.info("Frequency comparisons are omitted because no reliable accompaniment sections were detected.")
    elif comparison.get("bands"):
        metric_table(comparison["bands"])

    if comparison.get("balance_method"):
        st.markdown("**Balance vocal · mismo cálculo que A/B · tramos de 5 segundos**")
        st.caption("Referencia fija del audio analizado. La comparación con A/B requiere los mismos archivos y el mismo intervalo.")
        st.dataframe(comparison["balance_rows"], hide_index=True, width="stretch")
    else:
        st.info("Informe histórico con cálculo anterior. Vuelve a analizar los originales para obtener el balance compartido con A/B.")
    if comparison.get("sections"):
        st.markdown("**Actividad y variación vocal · secciones de aproximadamente 15 segundos**")
        st.dataframe(comparison["sections"], hide_index=True, width="stretch")
        st.caption(
            "Detectable instrumental frames only means signal is above a low activity floor; 100% does not mean "
            "the accompaniment is at its usual level. Instrumental level vs typical compares the section's "
            "active level with the track's typical active level. Reduced / solo sections are excluded from EQ "
            "and vocal-to-instrumental balance suggestions; voice-only level spread still includes them."
        )

    raw_eq_insights = comparison.get("eq_insights")
    if raw_eq_insights is None:
        st.info("Re-analyze this saved session to generate the detailed EQ screen for these bands.")
    else:
        eq_insights = []
        for insight in raw_eq_insights:
            band_name = insight.get("Band", "")
            band_info = next((item for item in EQ_BANDS if item[0] == band_name), None)
            if band_info is not None and band_info[3] < 120:
                continue
            eq_insights.append({
                "Time": insight.get("Time", ""),
                "Band": band_name,
                "Relative offset vs section median (dB)": insight.get("Relative offset vs section median (dB)", ""),
                "Audition this move": insight.get("Audition this move", ""),
            })
        if eq_insights:
            st.markdown("**EQ starting points by section**")
            st.caption(
                "The offset shows how this band's voice-to-instrument balance differs from the section median across bands; "
                "it is not an EQ amount. Bands below 120 Hz are not used to suggest vocal boosts because low vocal energy there is normal. "
                "Treat each move as a small audition starting point."
            )
            st.dataframe(eq_insights, hide_index=True, width="stretch")
        else:
            st.caption("No strong section-specific EQ imbalance crossed the current 4 dB screening threshold.")

    compression_insights = comparison.get("compression_insights")
    st.markdown("**Compression / level-control candidates**")
    st.caption(
        "These candidates use the voice's own active level spread, including solo sections. "
        "They do not mean the guitar dropout itself needs compression, and the app does not simulate compressor gain reduction."
    )
    if compression_insights is None:
        st.info("Re-analyze this saved session to generate section-specific level-control candidates.")
    elif compression_insights:
        st.dataframe(compression_insights, hide_index=True, width="stretch")
    else:
        st.caption("No section crossed the current level-spread threshold for a compression suggestion.")


def invalidate_logic_capture() -> None:
    """Invalidate audio and derived results before recording/clearing a capture."""
    st.session_state.pop("logic_master_capture", None)
    st.session_state["logic_capture_request"] = uuid.uuid4().hex
    st.session_state["use_logic_capture_as_mix"] = False
    source = st.session_state.get("mix_moss_upload")
    if source and source.get("source_type") == "analysis_capture":
        invalidate_uploaded_moss_source()


def invalidate_capture_upload_report() -> None:
    capture = st.session_state.get("logic_master_capture")
    if capture is not None:
        capture.pop("upload_report", None)
        capture["upload_request_id"] = uuid.uuid4().hex


def current_logic_capture() -> dict | None:
    capture = st.session_state.get("logic_master_capture")
    if capture is None:
        return None
    if not capture.get("capture_id"):
        # A session opened before this change has no verifiable result provenance.
        capture["capture_id"] = uuid.uuid4().hex
    capture_id = capture["capture_id"]
    for key in ("report", "moss", "upload_report"):
        result = capture.get(key)
        if result is not None and result.get("capture_id") != capture_id:
            capture.pop(key, None)
    return capture


def is_current_logic_capture(capture: dict) -> bool:
    current = st.session_state.get("logic_master_capture")
    return current is not None and current.get("capture_id") == capture["capture_id"]


def analyze_capture_with_uploads(capture: dict, voice_1_files, voice_2_files,
                                 instrumental_file, reference_file) -> dict:
    """Use the current in-memory stereo capture in the existing upload pipeline."""
    if not is_current_logic_capture(capture) or capture["audio"].shape[0] != 2:
        raise ValueError("Selecciona una captura estéreo vigente antes de analizar.")
    request_id = capture.get("upload_request_id")
    mix = BytesIO()
    sf.write(mix, capture["audio"].T, capture["sample_rate"], format="WAV", subtype="FLOAT")
    mix.name = f"Capture-{capture['capture_id']}.wav"
    report = make_report(mix, voice_1_files, voice_2_files, instrumental_file, reference_file)
    if not is_current_logic_capture(capture) or capture.get("upload_request_id") != request_id:
        raise ValueError("La captura o los archivos opcionales cambiaron durante el análisis. Analiza de nuevo la selección vigente.")
    report.update(capture_id=capture["capture_id"], transient=True,
                  session_id=f"capture-{capture['capture_id']}", upload_request_id=request_id)
    report["files"] = [describe_upload(mix, "mix")]
    for label, uploads in (("Voice 1", voice_1_files), ("Voice 2", voice_2_files)):
        report["files"].extend(describe_upload(upload, label, i) for i, upload in enumerate(uploads or [], 1))
    for label, upload in (("instrumental", instrumental_file), ("reference", reference_file)):
        if upload is not None:
            report["files"].append(describe_upload(upload, label))
    return report


def render_moss_memory_manager() -> None:
    with st.expander("Memoria local de MOSS · notas aprobadas"):
        st.caption(f"Solo notas escritas y guardadas por ti. Archivo local fuera del proyecto: {MEMORY_PATH}")
        st.multiselect("Temas de memoria pertinentes para las próximas descripciones", TOPICS,
                       default=list(TOPICS), key="moss_memory_topics")
        st.caption("Se incluyen hasta las 8 notas más recientes de los temas seleccionados. Las preferencias guían la presentación; las correcciones son ejemplos de otras grabaciones, no hechos sobre la mezcla actual.")
        try:
            notes = load_notes()
        except ValueError as exc:
            st.error(str(exc))
            return
        if not notes:
            st.info("No hay notas guardadas. Escribir una corrección no la guarda: debes pulsar «Guardar» después de una descripción.")
        for note in notes:
            st.text(f"{note['kind']} · {note['topic']}\n{note['text']}")
            if st.button("Borrar nota", key=f"delete_moss_note_{note['id']}"):
                try:
                    delete_note(note["id"])
                    st.rerun()
                except (OSError, ValueError) as exc:
                    st.error(f"No se pudo borrar la nota: {exc}")
        if notes and st.button("Borrar todas las notas de MOSS", key="delete_all_moss_notes"):
            try:
                delete_note(None)
                st.rerun()
            except (OSError, ValueError) as exc:
                st.error(f"No se pudo borrar la memoria: {exc}")


def render_moss_note_form(interpretation: dict, panel_key: str) -> None:
    st.caption("Nota para futuras descripciones: escribe solo tu corrección o preferencia, sin títulos, rutas, nombres de archivos ni respuestas completas del modelo. No se guarda nada hasta pulsar «Guardar».")
    with st.form(f"moss_note_{panel_key}_{interpretation['id']}", clear_on_submit=True):
        kind = st.selectbox("Tipo de nota", KINDS)
        topic = st.selectbox("Tema de la nota", TOPICS)
        text = st.text_area("Corrección o preferencia aprobada", max_chars=500)
        save = st.form_submit_button("Guardar")
    if save:
        labels = [interpretation.get("source_name", "")]
        for key in ("new_mix", "new_voice_1", "new_voice_2", "new_instrumental", "new_reference"):
            value = st.session_state.get(key)
            for upload in value if isinstance(value, list) else [value]:
                if upload is not None:
                    labels.append(upload.name)
        try:
            add_note(kind, topic, text, private_labels=labels, model_response=interpretation["description"])
            st.session_state["_moss_note_saved_notice"] = True
            st.rerun()
        except (ValueError, OSError) as exc:
            st.error(f"No se guardó la nota: {exc}")


def invalidate_uploaded_moss_source() -> None:
    st.session_state.pop("mix_moss_upload", None)


def invalidate_direct_moss_source() -> None:
    st.session_state.pop("direct_moss_source", None)


def measure_direct_moss_upload(source: dict) -> None:
    """Compute existing file metrics on demand; retain numeric context only in RAM."""
    if source.get("source_type") != "direct_upload" or not is_current_moss_source(source):
        raise ValueError("El archivo seleccionado ya no está vigente.")
    for key in ("moss", "selection", "measurement_snapshot", "measurement_origin"):
        source.pop(key, None)
    source["analysis_id"] = f"direct-{source['capture_id'][:8]}"
    # Reuse the numerical pipeline, without save_analysis, stems or inference.
    report = make_report(source["upload"], [], [], None, None)
    if not is_current_moss_source(source):
        raise ValueError("El archivo cambió durante la medición. Selecciona de nuevo el archivo vigente.")
    report["session_id"] = "direct-"+uuid.uuid4().hex
    source.update(analysis_id=report["session_id"], measurement_origin="current_upload",
                  measurement_snapshot=measurement_snapshot(report))


def render_direct_moss_upload() -> None:
    with st.expander("Analizar un archivo directamente con MOSS · sin BlackHole", expanded=True):
        st.caption("Carga un archivo para una descripción musical experimental, sin capturar audio ni completar el análisis de stems. El archivo y la respuesta permanecen solo en memoria.")
        upload = st.file_uploader(
            "Archivo de música para MOSS", key="direct_moss_file",
            help=f"Formatos admitidos por el decodificador local: {FORMAT_LABEL}. Hasta 200 MB.",
            on_change=invalidate_direct_moss_source,
        )
        if upload is None:
            invalidate_direct_moss_source()
            return
        st.text(f"Archivo seleccionado para MOSS: {upload.name}")
        if Path(upload.name).suffix.lower().lstrip(".") not in SUPPORTED_FORMATS:
            invalidate_direct_moss_source()
            st.error(f"Formato no compatible. Selecciona uno de estos formatos: {FORMAT_LABEL}.")
            return
        if not upload.getvalue():
            invalidate_direct_moss_source()
            st.error("El archivo está vacío. Selecciona un archivo de audio con contenido.")
            return
        audio_sha256 = hashlib.sha256(upload.getvalue()).hexdigest()
        token = audio_sha256 + ":" + upload.name
        source = st.session_state.get("direct_moss_source")
        if source is None or source.get("token") != token:
            source_id = uuid.uuid4().hex
            source = {"source_type": "direct_upload", "capture_id": source_id,
                      "analysis_id": f"direct-{source_id[:8]}", "token": token,
                      "upload": upload, "source_name": upload.name, "device_name": upload.name,
                      "audio_sha256": audio_sha256}
            st.session_state["direct_moss_source"] = source
        source["audio_sha256"] = audio_sha256
        sessions = load_saved_sessions()  # Report JSON only; no saved audio is read.
        reports = {directory.name: report for directory, report in sessions}
        selected_report = st.selectbox(
            "Tablas existentes para MOSS (opcional)", [None, *reports],
            format_func=lambda key: "Sin tablas del historial" if key is None else (
                f"{reports[key].get('mix', {}).get('name', 'Mix')} · "
                f"{reports[key].get('last_analyzed_at', reports[key].get('created_at', ''))} · "
                f"{reports[key]['session_id']}"),
            key=f"moss_history_{source['capture_id']}",
        )
        try:
            if selected_report is not None:
                source.pop("measurement_origin", None)
            changed = (False if selected_report is None and source.get("measurement_origin") == "current_upload"
                       else bind_history_tables(source, reports.get(selected_report)))
        except ValueError as exc:
            st.session_state.pop(f"moss_task_{source['capture_id']}", None)
            st.error(str(exc))
            st.info("La descripción queda bloqueada para esta selección. Elige un informe verificable del mismo audio o «Sin tablas del historial» para una descripción sin mediciones.")
            return
        if changed:
            st.session_state.pop(f"moss_task_{source['capture_id']}", None)
        if selected_report is not None:
            st.success("Audio e informe coinciden exactamente por SHA-256. Se reutilizan las tablas guardadas, sin recalcular ni copiar audio.")
            st.caption(f"Informe: {source['history_report_id']} · Archivo del informe: {source['history_report_name']} · Audio seleccionado: {source['source_name']}")
        else:
            st.caption("Si este archivo no tiene historial, puedes calcular sus tablas aquí, solo en memoria. Se mide el archivo completo; no se comparan voces y stems separados.")
            if st.button("Calcular tablas de este archivo (solo en memoria)", key=f"moss_measure_{source['capture_id']}"):
                try:
                    with st.spinner("Calculando las mediciones locales del archivo…"):
                        measure_direct_moss_upload(source)
                    st.session_state.pop(f"moss_task_{source['capture_id']}", None)
                except Exception as exc:
                    st.error(f"No se pudieron calcular las tablas: {exc}")
            if source.get("measurement_origin") == "current_upload" and source.get("measurement_snapshot"):
                st.success("Tablas del archivo actual disponibles solo en memoria. Ya puedes elegir «Explicar las tablas del análisis». No se ha ejecutado MOSS.")
        render_moss_description(source, panel_key="direct")


def invalidate_analysis_inputs() -> None:
    invalidate_capture_upload_report()
    invalidate_uploaded_moss_source()


def selected_analysis_moss_source(mix_file, use_capture: bool, capture) -> dict | None:
    """Resolve only this analysis's explicit stereo selection, never a fallback capture."""
    current = st.session_state.get("mix_moss_upload")
    if use_capture:
        if capture is None or not is_current_logic_capture(capture) or capture["audio"].shape[0] != 2:
            invalidate_uploaded_moss_source()
            return None
        token = f"capture:{capture['capture_id']}"
        values = {"source_type": "analysis_capture", "audio": capture["audio"],
                  "sample_rate": capture["sample_rate"], "source_capture_id": capture["capture_id"],
                  "source_name": f"Captura {capture['device_name']} · {capture['capture_id'][:8]}"}
    elif mix_file is not None:
        token = "upload:" + hashlib.sha256(mix_file.getvalue()).hexdigest() + ":" + mix_file.name
        values = {"source_type": "upload", "upload": mix_file, "source_name": mix_file.name}
    else:
        # Upload widgets are cleared after saving measurements. Keep only the
        # explicitly bound analysis source in RAM so its manual MOSS button works.
        if current and current.get("completed_analysis") and current.get("source_type") == "upload":
            return current
        invalidate_uploaded_moss_source()
        return None
    if current is None or current.get("token") != token:
        source_id = uuid.uuid4().hex
        current = {**values, "token": token, "capture_id": source_id,
                   "analysis_id": f"draft-{source_id[:8]}",
                   "device_name": values["source_name"]}
        st.session_state["mix_moss_upload"] = current
    return current


def bind_moss_to_analysis(source, report: dict) -> None:
    if source is not None and is_current_moss_source(source):
        source.pop("moss", None)
        source["analysis_id"] = report["session_id"]
        source["measurement_snapshot"] = measurement_snapshot(report)
        source["completed_analysis"] = True


def is_current_moss_source(source: dict) -> bool:
    if source.get("source_type") == "direct_upload":
        return st.session_state.get("direct_moss_source") is source
    if source.get("source_type") in ("upload", "analysis_capture"):
        current = st.session_state.get("mix_moss_upload")
        if current is not source:
            return False
        if source["source_type"] == "analysis_capture":
            capture = st.session_state.get("logic_master_capture")
            return (bool(st.session_state.get("use_logic_capture_as_mix")) and capture is not None
                    and capture.get("capture_id") == source.get("source_capture_id"))
        return True
    # A raw BlackHole buffer is not an analysis source. It must first be selected
    # explicitly through Current stereo mix and get its own analysis context.
    return False


def describe_capture_with_moss(audio: np.ndarray, sample_rate: int, memory_notes=None, description_task="general", measurement_context=None, return_envelope=False) -> str | dict:
    """Send samples over a local pipe to the isolated, network-denied worker."""
    isolated = Path(__file__).resolve().parent / "isolated_mlx_audio"
    python = isolated / "venv/bin/python"
    worker = isolated / "moss_worker.py"
    model = isolated / "models/MOSS-Music-8B-Thinking-8bit"
    sandbox = Path("/usr/bin/sandbox-exec")
    if not all(path.exists() for path in (python, worker, model / "config.json", sandbox)):
        raise ValueError("MOSS requiere el entorno y modelo locales ya instalados y el bloqueo de red de macOS. Las mediciones siguen disponibles.")
    samples = BytesIO()
    np.save(samples, audio, allow_pickle=False)
    if description_task not in DESCRIPTION_TASKS:
        raise ValueError("El enfoque de descripción seleccionado no es válido.")
    if measurement_context is not None:
        validate_context(measurement_context)
    elif description_task == "measurements":
        raise ValueError("Primero genera las tablas del análisis vigente.")
    header = json.dumps({"sample_rate": sample_rate, "memory_notes": memory_notes or [],
                         "measurement_context": measurement_context,
                         "description_task": description_task}).encode("utf-8")
    if len(header) > 64000:
        raise ValueError("El contexto de tablas y memoria supera el límite local.")
    payload = header + b"\n" + samples.getvalue()
    env = os.environ.copy()
    env.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HOME": str(isolated / "cache/huggingface"),
        "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1",
    })
    try:
        result = subprocess.run(
            [str(sandbox), "-p", "(version 1)(allow default)(deny network*)",
             str(python), "-I", "-B", str(worker)],
            input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, timeout=300, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("MOSS superó cinco minutos. La captura y las mediciones siguen disponibles.") from exc
    except OSError as exc:
        raise ValueError("No se pudo iniciar el proceso local de MOSS.") from exc
    if result.returncode != 0:
        raise ValueError("MOSS no pudo completar la descripción offline. Comprueba Metal, el modelo local y el bloqueo de red de macOS. No se intentará una conexión ni una descarga alternativa.")
    try:
        response = json.loads(result.stdout)
        text = response["description"]
        if not isinstance(text, str) or not text.strip() or response.get("truncated"):
            raise ValueError("Respuesta vacía o incompleta")
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("MOSS devolvió una descripción vacía, incompleta o inválida. No se presenta como resultado válido.") from exc
    if return_envelope:
        envelope = {"description": text.strip(), "row_explanations": None, "link_error": None,
                    "generation_strategy": response.get("generation_strategy"),
                    "row_generation": response.get("row_generation", [])}
        try:
            parsed = check_worker_response(response, measurement_context)
            envelope.update(row_explanations=parsed["explanations"], missing_row_ids=parsed["missing_row_ids"])
        except ValueError as exc:
            envelope["link_error"] = str(exc)
        return envelope
    return text.strip()


def render_moss_description(capture: dict, panel_key: str = "mix") -> None:
    if not is_current_moss_source(capture):
        return
    capture_id = capture["capture_id"]
    st.subheader("Descripción opcional de MOSS · experimental")
    st.warning("Interpretación tentativa, sujeta a revisión. MOSS puede inventar u omitir instrumentos y voces. No es una medición ni una evaluación fiable de la mezcla.")
    st.caption("Se ejecuta solo con este botón, en el Mac y con red bloqueada. Usa una copia en memoria en mono a 16 kHz; no evalúa imagen estéreo ni recomienda o aplica EQ, compresión o plugins.")
    st.caption(f"Fuente seleccionada {capture_id[:8]} · {capture['device_name']}")
    if capture.get("analysis_id"):
        label = "Archivo para MOSS" if capture.get("source_type") == "direct_upload" else "Fuente de Current stereo mix"
        st.caption(f"Análisis: {capture['analysis_id']} · {label}: {capture['source_name']}")
    snapshot = capture.get("measurement_snapshot")
    analysis_source = capture.get("source_type") in ("upload", "analysis_capture")
    if analysis_source and not snapshot:
        st.info("Primero analiza Current stereo mix con sus archivos opcionales para generar las tablas vigentes. Después podrás pedir su explicación manual a MOSS.")
        return
    if not analysis_source and not snapshot:
        st.caption("Este archivo directo no tiene tablas asociadas. Puedes calcular sus tablas con el botón de arriba o seleccionar un informe verificable del historial.")
    start = 0.0
    length = 20
    tasks = list(DESCRIPTION_TASKS) if snapshot else [key for key in DESCRIPTION_TASKS if key != "measurements"]
    task = st.selectbox("Qué quieres explorar en este fragmento", tasks,
                        format_func=DESCRIPTION_TASKS.get, key=f"moss_task_{capture_id}")
    if capture.get("source_type") in ("upload", "direct_upload"):
        start = st.number_input("Inicio del fragmento para MOSS (segundos)", min_value=0.0,
                                value=0.0, key=f"moss_start_{capture_id}")
        length = st.selectbox("Duración del fragmento para MOSS", [15, 20, 30], index=1,
                              key=f"moss_length_{capture_id}")
    measurement_context = None
    if snapshot:
        duration = (capture["audio"].shape[1] / capture["sample_rate"]
                    if capture.get("source_type") == "analysis_capture" else length)
        try:
            measurement_context = context_for_clip(snapshot, capture["analysis_id"], start, duration)
        except ValueError as exc:
            capture.pop("moss", None)
            st.error(str(exc))
            return
        st.caption("MOSS recibe las tablas de este análisis. Los valores globales describen el archivo completo; solo se incluyen secciones que se solapan con el fragmento. Voz y stem conservan sus fuentes separadas.")
        if st.checkbox("Ver mediciones exactas que recibirá MOSS", key=f"moss_tables_{capture_id}"):
            st.dataframe(measurement_context["rows"], hide_index=True, width="stretch")
        st.caption("Las tablas son los datos calculados; la explicación del modelo sigue sin verificar y podría contradecirlos. No equivale a una recomendación de procesamiento.")
    fingerprint = context_fingerprint(measurement_context) if measurement_context else None
    selection = (start, length, task, fingerprint)
    if capture.get("selection") != selection:
        capture.pop("moss", None)
        capture["selection"] = selection
    try:
        notes = select_notes(load_notes(), st.session_state.get("moss_memory_topics", TOPICS))
        memory_error = None
    except ValueError as exc:
        notes, memory_error = [], str(exc)
        st.error(memory_error)
    st.caption(f"Contexto local: {len(notes)} notas aprobadas de los temas seleccionados. No se usan para cambiar mediciones ni recomendar procesamiento.")
    if st.button(f"Describir {capture['source_name']} con MOSS (experimental)", key=f"describe_logic_capture_moss_{panel_key}_{capture_id}", disabled=memory_error is not None):
        capture.pop("moss", None)
        try:
            with st.spinner("Generando descripción local con MOSS…"):
                if capture.get("source_type") in ("upload", "direct_upload"):
                    try:
                        audio, rate = read_audio(capture["upload"])
                    except ValueError as exc:
                        raise ValueError(f"No se pudo decodificar «{capture['source_name']}» con el backend local. Puede estar dañado o usar un códec no compatible. Detalle: {exc}") from exc
                    first, last = round(start * rate), round((start + length) * rate)
                    if last > audio.shape[1]:
                        raise ValueError("El fragmento elegido queda fuera del archivo. Elige otro inicio o duración.")
                    audio = audio[:, first:last]
                else:
                    audio, rate = capture["audio"], capture["sample_rate"]
                description = describe_capture_with_moss(audio, rate, notes, description_task=task,
                                                         measurement_context=measurement_context,
                                                         return_envelope=(task == "measurements"))
            if (not is_current_moss_source(capture)
                    or capture.get("selection") != selection):
                return
            linked_result = description if isinstance(description, dict) else None
            if linked_result is not None:
                description = linked_result["description"]
            capture["moss"] = {"description": description, "id": uuid.uuid4().hex,
                               "capture_id": capture_id,
                               "analysis_id": capture.get("analysis_id"),
                               "source_name": capture.get("source_name", capture["device_name"]),
                               "description_task": task, "measurement_fingerprint": fingerprint,
                               "review": "Sin revisar", "correction": ""}
            if linked_result is not None:
                capture["moss"].update({key: linked_result.get(key) for key in
                                       ("row_explanations", "missing_row_ids", "link_error", "generation_strategy", "row_generation")})
        except ValueError as exc:
            st.error(str(exc))
    interpretation = capture.get("moss")
    if (not interpretation or interpretation.get("capture_id") != capture_id
            or interpretation.get("analysis_id") != capture.get("analysis_id")):
        return
    if interpretation.get("measurement_fingerprint") != fingerprint:
        capture.pop("moss", None)
        st.warning("La descripción pertenece a otro contexto de tablas. Genera una nueva descripción para las mediciones vigentes.")
        return
    st.caption(f"Audio analizado: {interpretation.get('source_name', capture['device_name'])}")
    st.caption(f"Enfoque: {DESCRIPTION_TASKS.get(interpretation.get('description_task'), 'Descripción musical')}. Las afirmaciones del modelo requieren tu revisión; no son mediciones.")
    audit = None
    linked_rows = []
    if measurement_context is not None and interpretation.get("description_task") == "measurements":
        if interpretation.get("row_explanations"):
            try:
                linked_rows = resolve_explanations(interpretation["row_explanations"], measurement_context, fingerprint)
            except ValueError as exc:
                st.error(str(exc))
        if not linked_rows:
            st.error("La respuesta no tiene explicaciones vinculadas válidas. No se muestran como verificadas: "
                     + (interpretation.get("link_error") or "Genera una nueva respuesta estructurada con el botón manual."))
    if measurement_context is not None:
        audit = (audit_linked_rows(linked_rows, measurement_context, fingerprint) if linked_rows else
                 audit_description(interpretation["description"], measurement_context,
                                   interpretation.get("measurement_fingerprint")))
        interval = measurement_context["audio_scope"]
        st.info(f"Alcance: las tablas globales describen la fuente completa (archivo o captura); los stems describen sus propias fuentes y las filas por sección conservan su intervalo. El fragmento enviado a MOSS es {interval['start_seconds']:g}–{interval['end_seconds']:g} s. Los valores globales no se han recalculado para ese fragmento.")
        st.caption("Límites: pico de muestras y muestras próximas al techo no descartan clipping previo ni picos entre muestras. Sin referencia no se califica la dinámica. La energía lateral/central no determina por sí sola la anchura percibida.")
        if audit["findings"]:
            st.error("La explicación contiene afirmaciones que requieren revisión. No se presenta como una interpretación validada de las tablas. Consulta los avisos debajo; el original se conserva sin corregirlo automáticamente.")
    if linked_rows:
        st.markdown("**Explicaciones vinculadas · valores originales de la app**")
        st.caption("Valores, unidades, fuentes e intervalos proceden de las tablas. Solo la explicación es de MOSS y sigue sujeta a revisión.")
        for row in linked_rows:
            st.dataframe([{key: value for key, value in row.items() if key != "explanation"}],
                         hide_index=True, width="stretch")
            st.text("Explicación de MOSS: " + row["explanation"])
        missing = interpretation.get("missing_row_ids") or []
        if missing:
            st.warning(f"MOSS no explicó {len(missing)} fila(s): " + ", ".join(missing))
    if interpretation.get("generation_strategy") in ("fixed_row_prefill_v1", "fixed_row_text_v2"):
        st.caption("Salida vinculada del worker: la app fija cada referencia y un inicio neutro de frase; MOSS genera la continuación de la explicación. Los valores no se envían al modelo; solo recibe la etiqueta y el estado de disponibilidad/signo.")
        if st.checkbox("Ver continuaciones originales de MOSS por fila", key=f"moss_raw_rows_{interpretation['id']}"):
            for attempt in interpretation.get("row_generation", []):
                st.text(f"Fila {attempt['row_id']} · intento {attempt['attempt']} · aceptado: {attempt.get('accepted', False)}")
                st.text("Prefijo fijado por la app: " + attempt.get("prefill", ""))
                st.text(attempt.get("raw_output", ""))
        st.caption("JSON vinculado ensamblado por el worker, con explicaciones del modelo sin reescribir:")
    else:
        st.caption("Respuesta original de MOSS sobre las tablas — pendiente de revisión:" if fingerprint else "Respuesta original del modelo — no verificada:")
    # Plain text prevents generated Markdown/URLs from loading remote images.
    st.text(interpretation["description"])
    if audit is not None:
        st.markdown("**Comprobación local de la explicación**")
        st.caption("Revisión por reglas, sin otra IA ni inferencia. Comprueba algunas cifras y expresiones reconocidas; puede omitir errores o señalar frases que requieren revisión. No valida la calidad musical ni demuestra ausencia de contradicciones.")
        if audit["findings"]:
            st.warning(f"{len(audit['findings'])} aviso(s) para revisar. La respuesta original no se ha modificado.")
            for finding in audit["findings"]:
                st.text(f"{finding['code']}: {finding['explanation']}")
                if finding["evidence"]:
                    st.text("Texto señalado: "+finding["evidence"])
        else:
            st.info("No se detectaron avisos con estas reglas. La explicación sigue sin verificar por completo.")
        if audit["numeric_checks"]:
            st.caption("Valores mostrados desde las tablas; la explicación sigue sin validación semántica completa:" if linked_rows else "Cifras vinculadas a una medición y coincidentes en valor y unidad:")
            st.dataframe(audit["numeric_checks"], hide_index=True, width="stretch")
    with st.form(f"moss_review_{panel_key}_{interpretation['id']}"):
        review = st.radio("¿Cómo valoras esta descripción?", ["Sin revisar", "Confirmar", "Corregir", "Descartar"])
        correction = st.text_area("Tu corrección (obligatoria si eliges Corregir)")
        submitted = st.form_submit_button("Guardar revisión en esta sesión")
    if submitted and is_current_moss_source(capture):
        if review == "Corregir" and not correction.strip():
            st.error("Escribe qué debe corregirse antes de guardar la revisión.")
        else:
            interpretation["review"] = review
            interpretation["correction"] = correction.strip() if review == "Corregir" else ""
            st.success("Revisión conservada solo en memoria. No cambia las mediciones ni genera recomendaciones.")
    st.write(f"Revisión del usuario: {interpretation['review']}")
    if interpretation["correction"]:
        st.text(interpretation["correction"])
    render_moss_note_form(interpretation, panel_key)


def render_logic_capture() -> None:
    with st.expander("Capture a short sample from Logic (local, experimental)"):
        st.caption(
            "This records a short sample from a Mac audio input so you can inspect and replay Logic's Stereo Out. "
            "The capture stays in this app session's memory and is not written to a file or analysis history."
        )
        inputs, device_error = available_audio_inputs()
        if device_error:
            st.warning(
                "I couldn't list audio inputs. Install the app requirements, restart it, and check macOS microphone privacy settings. "
                f"Details: {device_error[:400]}"
            )
            return
        if not inputs:
            st.info(
                "No audio input is available. To capture Logic's output, install a virtual loopback input such as "
                "[BlackHole](https://existential.audio/blackhole/), then route Logic's output to it."
            )
            return

        blackhole_inputs = [item for item in inputs if "blackhole" in item["name"].casefold()]
        if not blackhole_inputs:
            st.info(
                "Inputs are available, but I don't see BlackHole. A built-in microphone records the room; "
                "to capture Logic's direct output, select a loopback device such as BlackHole and route Logic to it. "
                "[BlackHole setup help](https://existential.audio/blackhole/support/)"
            )

        input_by_id = {item["index"]: item for item in inputs}
        default_input = blackhole_inputs[0]["index"] if blackhole_inputs else inputs[0]["index"]
        device_index = st.selectbox(
            "Audio input",
            options=list(input_by_id),
            index=list(input_by_id).index(default_input),
            format_func=lambda index: (
                f"{input_by_id[index]['name']} · {input_by_id[index]['channels']} ch · "
                f"{input_by_id[index]['sample_rate']:,} Hz"
            ),
            key="logic_capture_device",
        )
        seconds = st.selectbox("Capture length", [5, 10, 15, 20, 30], index=2, key="logic_capture_seconds")
        st.caption("Start playback in Logic, then click Capture. The app listens for the selected number of seconds.")
        if st.button("Capture Logic output", key="capture_logic_output", on_click=invalidate_logic_capture):
            request_id = st.session_state["logic_capture_request"]
            try:
                with st.spinner(f"Capturing {seconds} seconds from the selected input…"):
                    audio, sample_rate = capture_audio_input(device_index, seconds)
                if st.session_state.get("logic_capture_request") == request_id:
                    st.session_state["logic_master_capture"] = {
                        "capture_id": request_id,
                        "audio": audio,
                        "sample_rate": sample_rate,
                        "device_name": input_by_id[device_index]["name"],
                    }
            except ValueError as exc:
                st.error(str(exc))

        capture = current_logic_capture()
        if capture:
            capture_id = capture["capture_id"]
            audio = capture["audio"]
            st.success(f"Captured from {capture['device_name']} · {audio.shape[1] / capture['sample_rate']:.1f} seconds")
            st.caption(f"Captura vigente: {capture_id[:8]} · {audio.shape[0]} canales · {capture['sample_rate']:,} Hz. Reproductor, mediciones y MOSS usan estas mismas muestras.")
            # Encode explicitly to avoid Streamlit's array-to-WAV conversion.
            playback_wav = BytesIO()
            sf.write(playback_wav, audio.T, capture["sample_rate"], format="WAV", subtype="PCM_16")
            st.audio(playback_wav.getvalue(), format="audio/wav")
            st.subheader("Mediciones reproducibles del fragmento")
            st.caption("Cálculos sobre las muestras capturadas, independientes de MOSS. Las mediciones no determinan por sí solas cómo debe sonar la mezcla.")
            if st.button("Analizar captura", key=f"analyze_logic_capture_{capture_id}"):
                capture.pop("report", None)
                try:
                    with st.spinner("Analizando el fragmento capturado…"):
                        # Float WAV preserves the captured levels, including peaks above full scale.
                        analysis_wav = BytesIO()
                        sf.write(analysis_wav, audio.T, capture["sample_rate"], format="WAV", subtype="FLOAT")
                        analysis_wav.name = "Logic capture.wav"
                        report = make_report(analysis_wav, [], [], None, None)
                    if is_current_logic_capture(capture):
                        report["capture_id"] = capture_id
                        capture["report"] = report
                except Exception as exc:
                    st.error(f"No se pudo analizar la captura: {exc}")
            if not is_current_logic_capture(capture):
                return
            if "report" in capture and capture["report"].get("capture_id") == capture_id:
                render_mix_summary(capture["report"])
                render_capture_recommendations(capture["report"])
            else:
                st.info("Esta captura aún no está analizada. Pulsa «Analizar captura» para calcular sus mediciones.")
            st.caption("Para describir una captura con MOSS, selecciónala explícitamente como Current stereo mix en «Upload a new analysis». El botón MOSS pertenece a ese análisis.")
            if audio.shape[0] != 2:
                st.warning("This input recorded mono. Select a two-channel loopback device to inspect stereo output.")
            st.button("Clear captured sample from memory", key="clear_logic_capture", on_click=invalidate_logic_capture)


def render_capture_recommendations(report: dict) -> None:
    st.subheader("Avisos por reglas sobre las mediciones")
    st.caption(
        "Reglas de medición, no una IA que haya escuchado el audio. Los resultados describen solo "
        "este fragmento; sus LUFS/LRA no representan necesariamente la canción completa. "
        "El audio y este análisis permanecen en memoria hasta borrar la captura o terminar la sesión."
    )
    metrics = report["mix"]["metrics"]
    if metrics["Average RMS (dBFS)"] <= -80:
        st.info("El fragmento es prácticamente silencioso. Comprueba que Logic estaba reproduciendo el tramo deseado y vuelve a capturar antes de valorar la mezcla.")
    elif metrics["Peak (dBFS)"] >= 0:
        st.warning("Hay picos a 0 dBFS o superiores. Revisa el margen de nivel de la mezcla antes de exportar a PCM; esta medición no prueba por sí sola que haya distorsión audible.")
    elif metrics["Samples at/above -0.1 dBFS (%)"] > 0:
        st.info("Hay muestras cerca de 0 dBFS. Revisa el margen de picos antes de exportar; esto no es una medición de true peak ni demuestra clipping.")
    else:
        st.info("No se detectan silencio casi total ni muestras próximas a 0 dBFS. Estas comprobaciones no evalúan la calidad ni el balance de la mezcla.")
    st.info(
        "Para reutilizar las recomendaciones actuales de EQ, compresión y plugins hacen falta pistas "
        "de voz e instrumental separadas y alineadas, que puedes cargar en el análisis de archivos. "
        "La captura del máster no permite aislarlas: la app no tiene separación de fuentes. "
        "La descripción experimental de MOSS no se usa para recomendar ajustes ni plugins."
    )


def render_mix_summary(report: dict) -> None:
    mix = report["mix"]
    st.subheader("Stereo mix")
    st.write(f"Duration: {mix['duration_seconds']:.1f} s · Sample rate: {mix['sample_rate']:,} Hz · Channels: {mix['channels']}")
    metric_table(mix["metrics"])
    if "Integrated loudness (LUFS)" in mix.get("metrics", {}):
        st.caption("Integrated LUFS and LRA summarize the whole mix using EBU R128. They are measurements only; the app does not normalize or alter the audio.")
    for warning in report.get("warnings", []):
        st.warning(warning)


def render_report(report: dict) -> None:
    render_mix_summary(report)
    render_change_comparison(report)

    # Current reports compare each voice group separately. Older saved reports used one combined vocal group.
    voice_comparisons = report.get("voice_comparisons") or {}
    for voice_name, comparison in voice_comparisons.items():
        render_one_voice_comparison(voice_name, comparison)
    legacy_comparison = report.get("vocal_comparison")
    if legacy_comparison is not None:
        render_one_voice_comparison("Combined vocals", legacy_comparison)

    reference = report.get("reference")
    if reference is not None:
        st.subheader("Reference track")
        st.write(f"{reference['name']} · {reference['duration_seconds']:.1f} s · {reference['sample_rate']:,} Hz · {reference['channels']} channels")
        st.caption("Reference loudness and vocal balance are not directly compared yet. Match playback levels when making an A/B listening comparison.")

    with st.expander("Audio files used with this analysis"):
        files = report.get("files", [])
        session_dir = SESSIONS_DIR / report["session_id"]
        if report.get("transient"):
            st.caption("Captura y análisis solo en memoria; no se guardaron audios ni este informe en el historial.")
        elif saved_audio_available(session_dir, report):
            st.caption(f"Audio is stored in `saved_sessions/{report['session_id']}`.")
        else:
            st.caption("New analyses keep only the report. Uploaded audio is cleared after analysis and is not saved here.")
        for index, saved_file in enumerate(files):
            st.write(f"{saved_file['label']}: {saved_file['name']}")
            saved_name = saved_file.get("saved_name")
            saved_path = session_dir / saved_name if saved_name else None
            if saved_path is not None and saved_path.is_file():
                st.download_button(
                    f"Download {saved_file['name']}",
                    data=lambda path=saved_path: path.read_bytes(),
                    file_name=saved_file["name"],
                    key=f"download_{report['session_id']}_{index}",
                )

    render_plugin_recommendations(report, load_plugin_profile())


st.title("Vocal Mix Check")
st.caption("A first-pass measurement assistant for vocal placement. It reports clues to listen for; it does not replace a mix engineer.")
if st.session_state.pop("_moss_note_saved_notice", False):
    st.success("Solo tu nota aprobada se guardó en la memoria local. No se guardaron audio, nombres ni respuesta del modelo.")
render_moss_memory_manager()
render_direct_moss_upload()
render_logic_capture()

if st.session_state.pop("_clear_upload_fields", False):
    for upload_key in ("new_mix", "new_voice_1", "new_voice_2", "new_instrumental", "new_reference"):
        st.session_state.pop(upload_key, None)
if st.session_state.pop("_analysis_saved_notice", False):
    st.success("Analysis report saved. Uploaded audio has been cleared and was not retained.")

plugin_profile = load_plugin_profile()
with st.expander("Mis plugins y memoria de recomendaciones"):
    st.caption("Lista inicial tomada de tus capturas. Edita lo que falte o quita lo que no tengas. Se guarda localmente; no se suben tus audios.")
    with st.form("plugin_profile_form"):
        profile_inputs = {}
        for category, names in plugin_profile.items():
            profile_inputs[category] = st.text_area(
                category,
                value="\n".join(names),
                help="Escribe un plugin por línea.",
                key=f"plugin_profile_{category}",
            )
        profile_submitted = st.form_submit_button("Guardar mi lista de plugins")
    if profile_submitted:
        plugin_profile = {
            category: [line.strip() for line in text.splitlines() if line.strip()]
            for category, text in profile_inputs.items()
        }
        save_plugin_profile(plugin_profile)
        st.success("Guardé tu lista de plugins en este Mac.")

saved_sessions = load_saved_sessions()
if saved_sessions:
    st.subheader("Saved analysis history")
    labels = [
        f"{report.get('last_analyzed_at', report.get('created_at', 'Unknown date'))} — {report.get('mix', {}).get('name', 'Mix')}"
        for _, report in saved_sessions
    ]
    selected_index = st.selectbox(
        "Choose an analysis to reopen",
        options=range(len(saved_sessions)),
        format_func=lambda index: labels[index],
    )
    selected_dir, selected_report = saved_sessions[selected_index]
    if saved_audio_available(selected_dir, selected_report):
        if st.button("Reopen and analyze these saved files", key=f"reanalyze_{selected_report['session_id']}"):
            try:
                reopened_report = reanalyze_saved_session(selected_dir, selected_report)
                st.success("The saved files were reopened and analyzed again.")
                render_report(reopened_report)
            except Exception as exc:
                st.error(f"Could not reopen the saved files: {exc}")
        else:
            render_report(selected_report)
    else:
        st.caption("This report is saved without its audio. To analyze it again, upload the original files.")
        render_report(selected_report)
else:
    st.info("No saved reports yet. After analysis, this app saves the measurements but does not retain the uploaded audio.")

st.subheader("Upload a new analysis")
upload_capture = current_logic_capture()
capture_is_stereo = upload_capture is not None and upload_capture["audio"].shape[0] == 2
if not capture_is_stereo:
    st.session_state["use_logic_capture_as_mix"] = False
use_capture_as_mix = st.checkbox(
    "Usar la captura estéreo actual como Current stereo mix (required)",
    key="use_logic_capture_as_mix", disabled=not capture_is_stereo,
    on_change=invalidate_analysis_inputs,
)
if use_capture_as_mix:
    st.caption(f"Mezcla seleccionada: captura {upload_capture['capture_id'][:8]} · {upload_capture['device_name']} · {upload_capture['audio'].shape[1] / upload_capture['sample_rate']:.1f} s. Este análisis permanece solo en memoria.")
    st.info("Voice 1, Voice 2 e Instrumental stem deben corresponder al mismo fragmento y empezar en el mismo punto que la captura. No se alinean automáticamente. Reference track conserva su análisis básico independiente.")
elif not capture_is_stereo:
    st.caption("No hay una captura estéreo vigente. Carga el archivo de mezcla o realiza una captura estéreo para habilitar esta opción.")
left, right = st.columns(2)
with left:
    mix_file = st.file_uploader(
        "Current stereo mix (required)",
        type=SUPPORTED_FORMATS,
        help=f"Supported formats: {FORMAT_LABEL}. Files up to 200 MB.",
        key="new_mix",
        disabled=use_capture_as_mix,
        on_change=invalidate_uploaded_moss_source,
    )
    voice_1_files = st.file_uploader(
        "Voice 1 (optional; select one or more tracks)",
        type=SUPPORTED_FORMATS,
        accept_multiple_files=True,
        help="Select the tracks that belong to Voice 1. They should share the same start point.",
        key="new_voice_1",
        on_change=invalidate_analysis_inputs,
    )
with right:
    voice_2_files = st.file_uploader(
        "Voice 2 (optional; select one or more tracks)",
        type=SUPPORTED_FORMATS,
        accept_multiple_files=True,
        help="Select the tracks that belong to Voice 2. They should share the same start point.",
        key="new_voice_2",
        on_change=invalidate_analysis_inputs,
    )
    instrumental_file = st.file_uploader("Instrumental stem (optional)", type=SUPPORTED_FORMATS, key="new_instrumental", on_change=invalidate_analysis_inputs)
    reference_file = st.file_uploader("Reference track (optional)", type=SUPPORTED_FORMATS, key="new_reference", on_change=invalidate_analysis_inputs)

st.info(
    "Supported formats: WAV, FLAC, AIFF/AIF, MP3, M4A/AAC, OGG/Opus, WMA, and CAF, including 44.1 kHz and 48 kHz audio. "
    "Voice 1 and Voice 2 are analyzed separately against the same instrumental. You may add several aligned tracks within either voice group. "
    "Include vocal processing and vocal effects in each voice export; leave master-bus processing off the stems. "
    + ("Este análisis de captura y sus archivos opcionales permanece en memoria; no se guardan audios ni informe."
       if use_capture_as_mix else
       "Analysis reports stay in `saved_sessions`; uploaded audio is cleared after analysis and is not saved.")
)

render_vocal_balance({"Voice 1": voice_1_files, "Voice 2": voice_2_files}, instrumental_file, read_audio)

analysis_moss_source = selected_analysis_moss_source(mix_file, use_capture_as_mix, upload_capture)

if st.button("Analizar captura y archivos opcionales" if use_capture_as_mix else "Analyze and save report", type="primary"):
    if analysis_moss_source is not None:
        for key in ("moss", "measurement_snapshot", "completed_analysis"):
            analysis_moss_source.pop(key, None)
    if use_capture_as_mix:
        invalidate_capture_upload_report()
        try:
            with st.spinner("Analizando la captura con los archivos opcionales, solo en memoria…"):
                report = analyze_capture_with_uploads(upload_capture, voice_1_files, voice_2_files,
                                                     instrumental_file, reference_file)
            upload_capture["upload_report"] = report
            bind_moss_to_analysis(analysis_moss_source, report)
        except Exception as exc:
            st.error(f"No se pudo analizar la captura con los archivos opcionales: {exc}")
    elif mix_file is None:
        st.error("The app did not receive the required stereo mix. Refresh the page and select it again.")
    else:
        try:
            with st.spinner("Analyzing the audio and saving the report. Long tracks can take a few minutes; keep this page open."):
                report = make_report(mix_file, voice_1_files, voice_2_files, instrumental_file, reference_file)
                identity = track_identity(mix_file)
                report["track_identity"] = identity
                previous_match, matched_by = find_previous_track_report(identity, saved_sessions)
                if previous_match is not None:
                    _, previous_report = previous_match
                    report["change_comparison"] = {
                        "session_id": previous_report.get("session_id"),
                        "created_at": previous_report.get("last_analyzed_at", previous_report.get("created_at", "Unknown date")),
                        "mix_name": previous_report.get("mix", {}).get("name", "Mix"),
                        "matched_by": matched_by,
                        "rows": comparison_rows(report, previous_report),
                    }
                save_analysis(report, mix_file, voice_1_files, voice_2_files, instrumental_file, reference_file)
                bind_moss_to_analysis(analysis_moss_source, report)
        except Exception as exc:
            st.error(f"Could not analyze the uploaded audio: {exc}")
        else:
            st.session_state["_clear_upload_fields"] = True
            st.session_state["_analysis_saved_notice"] = True
            st.rerun()

if use_capture_as_mix and is_current_logic_capture(upload_capture):
    capture_upload_report = upload_capture.get("upload_report")
    if (capture_upload_report and capture_upload_report.get("capture_id") == upload_capture["capture_id"]
            and capture_upload_report.get("upload_request_id") == upload_capture.get("upload_request_id")):
        render_report(capture_upload_report)

with st.expander("MOSS de la mezcla seleccionada · descripción manual y memoria local"):
    st.caption("MOSS usa únicamente Current stereo mix: no combina ni sustituye Voice 1, Voice 2, Instrumental stem o Reference track, ni cambia sus mediciones.")
    if analysis_moss_source is not None and is_current_moss_source(analysis_moss_source):
        render_moss_description(analysis_moss_source, panel_key="mix")
        if (analysis_moss_source.get("completed_analysis") and mix_file is None
                and analysis_moss_source.get("source_type") == "upload"):
            st.caption("El archivo de este análisis sigue disponible solo en memoria para la descripción manual.")
            st.button("Liberar archivo y descripción de este análisis", key="release_analysis_moss_source",
                      on_click=invalidate_uploaded_moss_source)
    else:
        st.info("Selecciona un archivo Current stereo mix o la captura estéreo vigente. MOSS no se ejecuta al analizar las mediciones.")

st.divider()
st.caption("Prototype v1.6.1 · Local Logic loopback capture, EBU R128 loudness, plugin suggestions, and repeat-track comparisons. Reports are saved; captured/uploaded audio is not retained.")
