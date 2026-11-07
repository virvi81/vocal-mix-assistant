"""Manual vocal balance audition, held only in the Streamlit session."""
from dataclasses import replace
from math import gcd
import hashlib
from io import BytesIO
import soundfile as sf
import numpy as np
import streamlit as st
from scipy.signal import resample_poly
from isolated_vocal_balance.core import Clip, RATE
from isolated_vocal_balance.proposals import suggest, render


def clip_upload(upload, start, duration, read_audio):
    audio, rate = read_audio(upload)
    if audio.shape[0] not in (1, 2):
        raise ValueError("Selecciona un archivo mono o estéreo.")
    begin, end = round(start * rate), round((start + duration) * rate)
    if end > audio.shape[1]:
        raise ValueError(f"{upload.name} no contiene el intervalo completo.")
    samples = audio[:, begin:end].T
    if rate != RATE:
        factor = gcd(rate, RATE)
        samples = resample_poly(samples, RATE // factor, rate // factor, axis=0)
    samples = samples[:round(duration * RATE)]
    if samples.shape[1] == 1:
        samples = np.repeat(samples, 2, axis=1)
    return Clip(samples.astype("float32"), upload.name, rate, audio.shape[0],
                audio.shape[1] / rate, start, duration)



def prepare_full(uploads, read_audio):
    decoded = [read_audio(upload) for upload in uploads]
    rate = decoded[0][1]
    channels = max(audio.shape[0] for audio, _ in decoded)
    if channels not in (1, 2):
        raise ValueError("Selecciona archivos mono o estéreo.")
    tracks = []
    for audio, source_rate in decoded:
        if audio.shape[0] not in (1, 2):
            raise ValueError("Selecciona archivos mono o estéreo.")
        samples = audio.T
        if source_rate != rate:
            factor = gcd(source_rate, rate)
            samples = resample_poly(samples, rate // factor, source_rate // factor, axis=0)
        if samples.shape[1] == 1 and channels == 2:
            samples = np.repeat(samples, 2, axis=1)
        tracks.append(samples)
    if len({len(track) for track in tracks}) != 1:
        raise ValueError("Las voces del grupo deben compartir duración e inicio. Revisa los exports.")
    native = np.sum(tracks, axis=0, dtype=np.float32)
    samples = native
    if rate != RATE:
        factor = gcd(rate, RATE)
        samples = resample_poly(samples, RATE // factor, rate // factor, axis=0)
    if channels == 1:
        samples = np.repeat(samples, 2, axis=1)
    duration = len(native) / rate
    clip = Clip(samples.astype("float32"), uploads[0].name, rate, channels, duration, 0, duration)
    return clip, native, rate


def full_playback_export(state):
    voice, music = state["voice"], state["music"]
    accepted = list(state["accepted"].values())
    st.subheader("Canción completa · A/B")
    original = voice.samples + music.samples
    adjusted = render(voice.samples, music.samples, 0, accepted)
    peak = max(float(np.abs(original).max()), float(np.abs(adjusted).max()))
    common = min(1.0, .89 / peak) if peak else 1.0
    st.caption("Escucha de la suma de voz e instrumental. A conserva el balance original. B incluye todos los ajustes aceptados, con la misma atenuación de escucha.")
    left, right = st.columns(2)
    with left:
        st.write("A · Canción original")
        st.audio((original * common).T, sample_rate=RATE)
    with right:
        st.write("B · Ajustes aceptados")
        st.audio((adjusted * common).T, sample_rate=RATE)
    st.caption("Exportación WAV de la voz o suma del grupo vocal, con su inicio, duración, canales y frecuencia originales del primer archivo. Formato de 32 bits flotantes.")
    signature = tuple((p["start"], p["end"], p["gain_db"]) for p in accepted)
    if state.get("export_signature") != signature:
        state.pop("export_wav", None)
    if st.button("Preparar exportación de voz ajustada", disabled=not accepted):
        native = state["native_voice"]
        result = render(native, np.zeros_like(native), 0, accepted, rate=state["native_rate"])
        buffer = BytesIO()
        sf.write(buffer, result, state["native_rate"], format="WAV", subtype="FLOAT")
        state["export_wav"] = buffer.getvalue()
        state["export_signature"] = signature
        if np.abs(result).max() >= 1:
            st.warning("La voz ajustada supera 0 dBFS. El WAV flotante conserva esos valores. Revisa la ganancia antes de convertir a PCM.")
    if "export_wav" in state:
        st.download_button("Guardar WAV de voz ajustada", state["export_wav"],
                           file_name="voz_ajustada.wav", mime="audio/wav")


def render_vocal_balance(voice_groups, instrumental, read_audio):
    with st.expander("Balance vocal · probar ajustes A/B"):
        st.caption("Compara ajustes de voz por tramos de 5 segundos. Las aceptaciones y el audio permanecen en memoria durante esta sesión.")
        available = [name for name, files in voice_groups.items() if files]
        if not available or instrumental is None:
            st.session_state.pop("vocal_balance", None)
            st.info("Carga Voice 1 o Voice 2 y el archivo Instrumental stem para probar el balance.")
            return
        group = st.selectbox("Grupo de voz", available, key="balance_group")
        voices = voice_groups[group]
        st.caption("Voz · " + ", ".join(v.name for v in voices) + " · Instrumental · " + instrumental.name)
        full = st.checkbox("Analizar canción completa", value=True, key="balance_full")
        start = st.number_input("Inicio del fragmento (s)", min_value=0.0, value=45.0, key="balance_start", disabled=full)
        duration = st.number_input("Duración (s)", min_value=1.0, max_value=120.0, value=120.0, key="balance_duration", disabled=full)
        if full:
            start, duration = 0.0, None
        aligned = st.checkbox("Confirmo que voz e instrumental corresponden a la misma versión y comparten el mismo inicio.", key="balance_aligned")
        identity = (group, start, duration, aligned, full, tuple(
            hashlib.sha256(upload.getvalue()).hexdigest() for upload in [*voices, instrumental]))
        state = st.session_state.get("vocal_balance")
        if state and state["identity"] != identity:
            st.session_state.pop("vocal_balance", None)
            state = None
        if st.button("Preparar balance y escucha", disabled=not aligned):
            st.session_state.pop("vocal_balance", None)
            state = None
            try:
                native_voice = None
                native_rate = None
                if full:
                    voice, native_voice, native_rate = prepare_full(voices, read_audio)
                    music, _, _ = prepare_full([instrumental], read_audio)
                    if voice.samples.shape != music.samples.shape:
                        raise ValueError("Voz e instrumental deben compartir la duración completa. Revisa los exports.")
                else:
                    clips = [clip_upload(upload, start, duration, read_audio) for upload in voices]
                    voice = replace(clips[0], samples=np.sum([c.samples for c in clips], axis=0))
                    music = clip_upload(instrumental, start, duration, read_audio)
                state = dict(identity=identity, voice=voice, music=music, accepted={},
                             native_voice=native_voice, native_rate=native_rate)
                st.session_state["vocal_balance"] = state
            except Exception as exc:
                st.error(str(exc))
        if state is None:
            return
        if st.button("Liberar audio y ajustes de balance"):
            st.session_state.pop("vocal_balance", None)
            st.rerun()
        accepted = state["accepted"]
        proposals = suggest(state["voice"], state["music"], accepted.values())
        rows = sorted(proposals + list(accepted.values()), key=lambda row: row["start"])
        st.caption("Referencia fija · mediana del balance original del fragmento. Propuestas desde 1,5 dB de desvío, con actividad comparable ≥50%. Ajuste sugerido hasta ±1 dB.")
        if full:
            full_playback_export(state)
        if not rows:
            st.info("Este fragmento no presenta propuestas con el criterio actual.")
            return
        st.dataframe([{"Inicio (s)": p["start"], "Fin (s)": p["end"],
                       "Balance medido al proponer (dB)": p.get("ratio_db"),
                       "Referencia (dB)": p.get("target_db"),
                       "Desvío al proponer (dB)": p.get("change_db"),
                       "Ajuste vocal (dB)": p["gain_db"],
                       "Estado": "Aceptado" if p["start"] in accepted else "Pendiente"} for p in rows],
                     hide_index=True)
        selected = st.selectbox("Tramo para escuchar", [p["start"] for p in rows],
                                format_func=lambda value: f"{value:g}–{next(p['end'] for p in rows if p['start'] == value):g} s")
        proposal = next(p for p in rows if p["start"] == selected)
        amount = st.slider("Ajuste de voz en B (dB)", -6.0, 6.0, float(proposal["gain_db"]), .1,
                           key=f"balance_gain_{hashlib.sha256(repr(identity).encode()).hexdigest()[:12]}_{selected}")
        candidate = dict(accepted)
        candidate[selected] = dict(proposal, gain_db=amount)
        a = render(state["voice"].samples, state["music"].samples, start, accepted.values())
        b = render(state["voice"].samples, state["music"].samples, start, candidate.values())
        begin, end = round((selected-start)*RATE), round((proposal["end"]-start)*RATE)
        a, b = a[begin:end], b[begin:end]
        peak = max(float(np.abs(a).max()), float(np.abs(b).max()))
        common = min(1.0, .89 / peak) if peak else 1.0
        st.caption("A incluye los ajustes aceptados. B añade el ajuste elegido. Ambos reproducen el mismo tramo con una atenuación común.")
        left, right = st.columns(2)
        with left:
            st.write("A · Base aceptada")
            st.audio((a * common).T, sample_rate=RATE)
        with right:
            st.write("B · Ajuste propuesto")
            st.audio((b * common).T, sample_rate=RATE)
        if st.button("Aceptar ajuste de este tramo"):
            accepted[selected] = dict(proposal, gain_db=amount)
            st.rerun()
        if st.button("Deshacer aceptación", disabled=selected not in accepted):
            accepted.pop(selected, None)
            st.rerun()
