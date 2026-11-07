"""Local suggestions for manual listening approval."""
import numpy as np
if __package__:
    from .core import measure, RATE
else:
    from core import measure, RATE


def suggest(voice, music, accepted=()):
    """Keep the original target fixed; measure accepted gain ramps before proposing.

    Accepted intervals are locked until explicitly undone. Pending suggestions
    never become the reference for another suggestion.
    """
    from dataclasses import replace
    original = measure(voice, music)
    target = original["baseline_db"]
    if target is None:
        return []
    accepted = list(accepted)
    adjusted = render(voice.samples, np.zeros_like(music.samples), voice.start, accepted)
    rows = measure(replace(voice, samples=adjusted), music)["rows"]
    proposals = []
    for current in rows:
        if current["ratio_db"] is None or current["comparable_pct"] < 50:
            continue
        if any(current["start"] < item["end"] and current["end"] > item["start"]
               for item in accepted):
            continue
        change = current["ratio_db"] - target
        if abs(change) < 1.5:
            continue
        proposals.append({
            "start": current["start"], "end": current["end"],
            "target_db": target, "ratio_db": current["ratio_db"],
            "change_db": change,
            "gain_db": round(float(np.clip(-change / 2, -1, 1)), 1),
        })
    return proposals


def render(voice, music, clip_start, adjustments, rate=RATE):
    if voice.shape != music.shape:
        raise ValueError("Los stems deben compartir longitud.")
    gain_db = np.zeros(len(voice), dtype=np.float64)
    last_end = 0
    for adjustment in sorted(adjustments, key=lambda item: item["start"]):
        start = round((adjustment["start"] - clip_start) * rate)
        end = round((adjustment["end"] - clip_start) * rate)
        value = adjustment["gain_db"]
        if not 0 <= start < end <= len(voice) or start < last_end or not np.isfinite(value) or abs(value) > 6:
            raise ValueError("Ajustes superpuestos, fuera del fragmento o mayores de 6 dB.")
        fade = min(round(.25 * rate), (end - start) // 2)
        shape = np.ones(end-start)
        ramp = .5 - .5*np.cos(np.linspace(0, np.pi, fade))
        shape[:fade] = ramp
        shape[-fade:] = ramp[::-1]
        gain_db[start:end] = shape * value
        last_end = end
    return (voice*np.power(10, gain_db[:, None]/20)+music).astype("float32")
