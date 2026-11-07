"""Deterministic vocal balance measurements. All audio stays in RAM."""
from dataclasses import dataclass
from math import gcd
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

RATE = 48000
FRAME_SECONDS = .05
HOP_SECONDS = .025


@dataclass
class Clip:
    samples: np.ndarray  # samples x channels, stereo; mono duplicated for audition.
    name: str
    source_rate: int
    source_channels: int
    source_duration: float
    start: float
    duration: float


def load_clip(path, start, duration):
    """Read only the requested original interval; no temporary file or download."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise ValueError("No existe el archivo seleccionado.")
    if not np.isfinite([start, duration]).all() or start < 0 or not 1 <= duration <= 120:
        raise ValueError("El inicio debe ser positivo o cero; el fragmento debe durar entre 1 y 120 s.")
    before = path.stat()
    try:
        with sf.SoundFile(str(path)) as source:
            rate, channels = source.samplerate, source.channels
            if channels not in (1, 2):
                raise ValueError("Solo se admiten stems mono o estéreo; no se hace downmix multicanal.")
            source_duration = len(source) / rate
            begin, count = round(start * rate), round(duration * rate)
            if begin + count > len(source):
                raise ValueError("El archivo no contiene el intervalo completo solicitado.")
            source.seek(begin)
            data = source.read(count, dtype="float32", always_2d=True)
    except sf.LibsndfileError as exc:
        raise ValueError("Formato no compatible con este lector local. Usa WAV, AIFF o FLAC original; MP3/OGG dependen del soporte instalado.") from exc
    if data.shape[0] != count or not np.isfinite(data).all():
        raise ValueError("Lectura incompleta o muestras no válidas.")
    if rate != RATE:
        factor = gcd(rate, RATE)
        data = resample_poly(data, RATE // factor, rate // factor, axis=0).astype("float32")
    expected = round(duration * RATE)
    if abs(len(data) - expected) > 1 or len(data) < expected:
        raise ValueError("El remuestreo no conserva el intervalo completo.")
    data = data[:expected]
    if channels == 1:
        data = np.repeat(data, 2, axis=1)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("El original cambió durante la lectura; vuelve a cargarlo.")
    return Clip(data, path.name, rate, channels, source_duration, start, duration)


def level(rms):
    return 20 * np.log10(np.maximum(rms, 1e-12))


def envelopes(samples):
    """RMS across samples AND channels: no mono fold-down/cancellation in metrics."""
    frame, hop = round(FRAME_SECONDS * RATE), round(HOP_SECONDS * RATE)
    if len(samples) < frame:
        raise ValueError("Fragmento insuficiente.")
    starts = np.arange(0, len(samples) - frame + 1, hop)
    energy = np.mean(np.square(samples, dtype=np.float64), axis=1)
    cumulative = np.concatenate(([0.], np.cumsum(energy)))
    rms = np.sqrt(np.maximum((cumulative[starts + frame] - cumulative[starts]) / frame, 0))
    return (starts + frame / 2) / RATE, rms


def measure(voice, instrumental, section_seconds=5., delta_threshold=3.):
    if voice.samples.shape != instrumental.samples.shape:
        raise ValueError("Los fragmentos deben tener exactamente la misma longitud.")
    if voice.start != instrumental.start or voice.duration != instrumental.duration:
        raise ValueError("Los fragmentos deben usar el mismo inicio y duración.")
    if not np.isfinite([section_seconds, delta_threshold]).all() or section_seconds < 1 or delta_threshold <= 0:
        raise ValueError("Parámetros de comparación no válidos.")
    times, vr = envelopes(voice.samples)
    _, mr = envelopes(instrumental.samples)
    # Relative activity gates + absolute floor. These are configurable engineering
    # heuristics, not speech detection, masking, intelligibility or quality scores.
    vg = max(10 ** (-60 / 20), float(vr.max()) * 10 ** (-35 / 20))
    mg = max(10 ** (-60 / 20), float(mr.max()) * 10 ** (-40 / 20))
    v_active, m_active = vr > vg, mr > mg
    comparable = v_active & m_active
    ratio = level(vr) - level(mr)
    baseline = float(np.median(ratio[comparable])) if comparable.sum() >= 20 else None
    rows = []
    for start in np.arange(0., voice.duration, section_seconds):
        end = min(float(start + section_seconds), voice.duration)
        mask = (times >= start) & (times < end)
        if not mask.any():
            continue
        eligible = mask & comparable
        coverage = float(eligible.sum() / mask.sum())
        valid = eligible.sum() >= 20 and coverage >= .25
        median = float(np.median(ratio[eligible])) if valid else None
        difference = median - baseline if median is not None and baseline is not None else None
        rows.append({
            "start": voice.start + float(start), "end": voice.start + end,
            "voice_active_pct": float(np.mean(v_active[mask]) * 100),
            "instrumental_active_pct": float(np.mean(m_active[mask]) * 100),
            "comparable_pct": coverage * 100,
            "ratio_db": median, "difference_db": difference,
            "p10_db": float(np.percentile(ratio[eligible], 10)) if valid else None,
            "p90_db": float(np.percentile(ratio[eligible], 90)) if valid else None,
            "review_change": difference is not None and coverage >= .5 and abs(difference) >= delta_threshold,
        })
    return {
        "rows": rows, "baseline_db": baseline, "baseline_scope": "fragmento cargado",
        "p10_db": float(np.percentile(ratio[comparable], 10)) if comparable.sum() >= 20 else None,
        "p90_db": float(np.percentile(ratio[comparable], 90)) if comparable.sum() >= 20 else None,
        "voice_gate_dbfs": float(level(vg)), "instrumental_gate_dbfs": float(level(mg)),
        "delta_threshold_db": delta_threshold,
        "note": "Relación de amplitud RMS entre stems exportados; no mide inteligibilidad, enmascaramiento, LUFS ni calidad de mezcla.",
    }


def playback_sources(voice, instrumental, mix=None):
    """One common audition attenuation for every source and both A/B sections."""
    sources = {"Suma de stems": voice.samples + instrumental.samples,
               "Solo voz": voice.samples, "Solo instrumental": instrumental.samples}
    if mix is not None:
        if mix.samples.shape != voice.samples.shape or mix.start != voice.start:
            raise ValueError("La mezcla no cubre el mismo intervalo.")
        sources["Mezcla original"] = mix.samples
    peak = max(float(np.abs(data).max()) for data in sources.values())
    gain = min(1., .89 / peak) if peak else 1.
    return sources, gain


def audition_slice(samples, clip_start, start, end, gain):
    a, b = round((start - clip_start) * RATE), round((end - clip_start) * RATE)
    if not 0 <= a < b <= len(samples) or not 0 < gain <= 1:
        raise ValueError("Intervalo o atenuación de escucha no válido.")
    return np.ascontiguousarray(samples[a:b] * gain, dtype=np.float32)
