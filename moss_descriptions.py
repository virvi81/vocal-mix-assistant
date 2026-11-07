"""Description tasks shared by the app and local worker; no model execution."""

DESCRIPTION_TASKS = {
    "measurements": "Explicar las tablas del análisis",
    "general": "Descripción musical",
    "sources": "Instrumentos y voces",
    "development": "Ritmo, textura y cambios",
}

_TASKS = {
    "measurements": "Explain only the supplied numerical tables in Spanish, using measured values, units, sources and scopes. In this task do not describe the audible clip or infer instrument identities.",
    "general": "Describe the prominent sound families, vocal character, rhythm, texture and any changes within this clip. Keep style and mood as tentative interpretations.",
    "sources": "Focus on distinctly audible sound sources and voices. Separate clear sound families from ambiguous timbres. Do not infer a separate bass instrument from low notes or drums from rhythmic attacks alone. Do not infer piano from chordal accompaniment alone. State a voice count only when distinct simultaneous voices are distinguishable; do not confuse doubling or reverb with different singers.",
    "development": "Focus on audible rhythmic patterns, texture, entrances, exits and changes within this clip. Separate rhythm carried by a melodic instrument from a distinct percussion source. Do not infer verse/chorus labels or the structure of an entire song from this fragment.",
}


def description_prompt(task="general", measurement_context=None):
    if task not in DESCRIPTION_TASKS:
        raise ValueError("Unknown musical description task")
    if measurement_context is not None:
        from moss_measurements import measurement_instructions
        from moss_linked import linked_prompt_contract
        return ("Explain the current analysis using the supplied measured data. "
                + _TASKS[task] + " Do not identify songs/artists or transcribe lyrics. "
                "Any audio observations are unverified predictions about this clip only. "
                + measurement_instructions(measurement_context)
                + (linked_prompt_contract()
                   if task == "measurements" else ""))
    if task == "measurements":
        raise ValueError("This task requires the current analysis tables")
    return (
        "Analyze only the audio in the supplied clip. " + _TASKS[task] + "\n"
        "Use three concise sections: AUDIBLE OBSERVATIONS (model predictions with brief "
        "audible supporting cues); UNCERTAIN INTERPRETATIONS (ambiguous identities, style "
        "or mood); CANNOT DETERMINE (details unsupported by this clip).\n"
        "Do not assume the usual band lineup or invent missing accompaniment. "
        "Prefer a broad sound family or 'uncertain' over an unsupported instrument name. "
        "Absence of a confidently identified source does not prove its absence. "
        "Do not identify songs/artists, transcribe lyrics, invent exact BPM, key or timestamps, "
        "assign numeric confidence, assess mix quality, or recommend EQ, compression, "
        "gain changes, effects, mastering or plugins. This mono 16 kHz input cannot "
        "establish the original stereo image or full-band fidelity. "
        "All observations remain unverified predictions for the user to check."
    )
