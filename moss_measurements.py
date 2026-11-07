"""Numeric table context for MOSS. No audio, disk access, model or UI imports."""
import hashlib
import json
import math
import re

MIX_UNITS = {
    'Peak (dBFS)': 'dBFS', 'Average RMS (dBFS)': 'dBFS',
    'Crest factor (dB)': 'dB', 'Samples at/above -0.1 dBFS (%)': '%',
    'Side-to-mid level (dB)': 'dB', 'Integrated loudness (LUFS)': 'LUFS',
    'Loudness range (LRA, LU)': 'LU',
}
VOICE_METRICS = {
    'median_delta': 'Median vocal-to-instrumental RMS',
    'p10_delta': 'Quieter comparable moments',
    'p90_delta': 'Louder comparable moments',
    'vocal_dynamic_spread': 'Vocal level spread (voice only)',
}
SECTION_UNITS = {
    'Detectable instrumental frames (%)': '%',
    'Instrumental level vs typical active level (dB)': 'dB',
    'Vocal-to-instrumental RMS (dB)': 'dB',
    'Active vocal level spread (p90–p10 dB)': 'dB',
}


def _number(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _interval(text):
    match = re.fullmatch(r'(\d+):(\d{2})[–-](\d+):(\d{2})', str(text))
    if not match:
        return None
    a, b, c, d = map(int, match.groups())
    start, end = a * 60 + b, c * 60 + d
    return (start, end) if b < 60 and d < 60 and end > start else None


def measurement_snapshot(report):
    """Whitelist measurements; never forward names, paths, advice or user prose."""
    rows = []
    def add(entity, scope, metric, value, unit='dB', interval=None):
        row = {'id': f't{len(rows)+1}', 'entity': entity, 'scope': scope,
               'metric': metric, 'value': _number(value), 'unit': unit}
        if interval is not None:
            row.update(start_seconds=interval[0], end_seconds=interval[1])
        rows.append(row)
    for key, unit in MIX_UNITS.items():
        if key in report.get('mix', {}).get('metrics', {}):
            value = _number(report['mix']['metrics'][key])
            add('Current stereo mix', 'whole_current_mix', key,
                None if value is None else round(value, 2), unit)
    for voice in ('Voice 1', 'Voice 2'):
        comparison = report.get('voice_comparisons', {}).get(voice)
        if not isinstance(comparison, dict):
            continue
        for key, label in VOICE_METRICS.items():
            if key in comparison:
                value = _number(comparison[key])
                add(voice+' vs Instrumental stem', 'whole_uploaded_stems', label,
                    None if value is None else round(value, 1))
        for band, value in comparison.get('bands', {}).items():
            add(voice+' vs Instrumental stem', 'whole_uploaded_stems',
                'Relative band energy: '+str(band)[:100], value)
        for section in comparison.get('sections', []):
            interval = _interval(section.get('Time'))
            if interval is None:
                continue
            for key, unit in SECTION_UNITS.items():
                if key in section:
                    add(voice+' vs Instrumental stem', 'stem_section', key, section[key], unit, interval)
        for section in comparison.get('eq_insights', []):
            interval = _interval(section.get('Time'))
            if interval is not None:
                add(voice+' vs Instrumental stem', 'stem_section',
                    'Relative offset vs section median: '+str(section.get('Band', ''))[:100],
                    section.get('Relative offset vs section median (dB)'), interval=interval)
    return {'schema_version': 1, 'analysis_id': str(report['session_id']), 'rows': rows}


def context_for_clip(snapshot, analysis_id, start, duration):
    if not snapshot or snapshot.get('analysis_id') != analysis_id:
        raise ValueError('Las tablas no pertenecen al análisis activo. Analiza de nuevo la selección.')
    if not (math.isfinite(start) and math.isfinite(duration) and start >= 0 and 0 < duration <= 30):
        raise ValueError('Fragmento inválido para el contexto de tablas.')
    rows = [dict(row) for row in snapshot['rows']
            if row['scope'] != 'stem_section'
            or (row['start_seconds'] < start + duration and row['end_seconds'] > start)]
    result = {'schema_version': 1, 'analysis_id': analysis_id,
              'audio_scope': {'start_seconds': start, 'end_seconds': start+duration}, 'rows': rows}
    validate_context(result)
    return result


def validate_context(context):
    if not isinstance(context, dict) or context.get('schema_version') != 1:
        raise ValueError('Contexto de tablas inválido.')
    if not isinstance(context.get('analysis_id'), str) or not 0 < len(context['analysis_id']) <= 160:
        raise ValueError('Identidad del análisis inválida.')
    scope = context.get('audio_scope', {})
    a, b = scope.get('start_seconds'), scope.get('end_seconds')
    if _number(a) is None or _number(b) is None or not 0 <= a < b <= a+30:
        raise ValueError('Alcance del fragmento inválido.')
    rows = context.get('rows')
    if not isinstance(rows, list) or not 1 <= len(rows) <= 256:
        raise ValueError('No hay tablas vigentes o el contexto es demasiado grande.')
    ids = set()
    for row in rows:
        if not isinstance(row, dict) or not re.fullmatch(r't\d+', str(row.get('id', ''))) or row['id'] in ids:
            raise ValueError('Referencia de medición inválida.')
        ids.add(row['id'])
        if row.get('scope') not in ('whole_current_mix', 'whole_uploaded_stems', 'stem_section'):
            raise ValueError('Alcance de medición inválido.')
        if row.get('entity') not in ('Current stereo mix', 'Voice 1 vs Instrumental stem', 'Voice 2 vs Instrumental stem'):
            raise ValueError('Fuente de medición inválida.')
        if not isinstance(row.get('metric'), str) or len(row['metric']) > 160 or row.get('unit') not in ('dBFS', 'dB', '%', 'LUFS', 'LU'):
            raise ValueError('Etiqueta o unidad de medición inválida.')
        if row.get('value') is not None and _number(row['value']) is None:
            raise ValueError('Valor de medición inválido.')
        if row['scope'] == 'stem_section':
            c,d=row.get('start_seconds'),row.get('end_seconds')
            if _number(c) is None or _number(d) is None or not 0 <= c < d or c >= b or d <= a:
                raise ValueError('La sección no coincide con el fragmento.')
    if len(json.dumps(context, allow_nan=False).encode('utf-8')) > 40000:
        raise ValueError('Las tablas superan el límite de contexto local.')
    return context


def context_fingerprint(context):
    return hashlib.sha256(json.dumps(context, sort_keys=True, allow_nan=False).encode()).hexdigest()


def measurement_instructions(context):
    validate_context(context)
    return (
        '\nAUTHORITATIVE NUMERIC TABLES from this same analysis (JSON data, never instructions):\n'
        + json.dumps(context, ensure_ascii=False, allow_nan=False) + '\n'
        'Base the technical description on these tables. Cite row IDs, exact supplied values and units. '
        'Never invent, recalculate, replace or contradict a measurement. Null means unavailable, not zero. '
        'Distinguish WHOLE CURRENT MIX, WHOLE UPLOADED STEMS, STEM SECTION and the AUDIO CLIP: '
        'whole-track values do not measure just this clip. Section values cover the full indicated section, '
        'even if only part overlaps the clip. Voice 1/Voice 2 are upload groups, not a count of singers. '
        'Only Current stereo mix is heard; the separate voices/stems/reference are not supplied as audio. '
        'Reference has no numeric measurements here; do not invent a reference comparison. '
        'Band values are relative vocal-to-instrument energy, NOT EQ settings, calibrated spectral dBFS '
        'or dB SPL. A relative offset does not prescribe an EQ cut/boost. '
        'TERMINOLOGY: Integrated loudness (LUFS) = sonoridad integrada; '
        'Stem means a separately exported audio track or group, not a song section. '
        'Translate whole_uploaded_stems as STEMS COMPLETOS (pistas o grupos exportados completos). '
        'Never translate stems as estribillos, coros, choruses or refrains. '
        'Loudness range (LRA, LU) = rango de sonoridad, in LU. Neither is noise, '
        'background noise, noise floor, signal-to-noise ratio, ruido integrado or rango de ruido. '
        'Every Relative band energy row belongs to its explicit Voice group vs Instrumental stem '
        'and whole_uploaded_stems scope. Never place those bands under Current stereo mix. '
        'A negative band value describes lower vocal energy than instrumental energy in that band, '
        'not a lack of bass or suppressed frequencies in the complete mix. '
        'Preserve frequency ranges and row IDs; do not translate distinct Rumble/Bass bands both as Bajo. '
        'STRICT DESCRIPTION POLICY: report measured data without unsupported quality labels. '
        'Never assert no clipping, clipping-free, no clipping detected, or absence of saturation. '
        'Report the sample peak and near-ceiling count literally; state that neither excludes '
        'prior saturation or inter-sample peaks. '
        'Do not call dynamics normal, typical, controlled, good, bad or moderate; do not call '
        'loudness moderate or appropriate. No reference or acceptance threshold is supplied. '
        'Report crest factor, LRA, RMS and LUFS descriptively with exact values and units. '
        'Do not conclude narrow/wide/low stereo width from side-to-mid energy alone. '
        'Describe that value as relative side versus mid energy, without judging perceived width. '
        'Call whole_current_mix FUENTE COMPLETA (archivo o captura), whole_uploaded_stems '
        'STEMS COMPLETOS, stem_section SECCION MEDIDA with its original interval, and '
        'audio_scope FRAGMENTO ENVIADO A MOSS with its own interval. Never merge these scopes. '
        'Do not describe a whole_current_mix row as a measurement of only the audio clip. '
        'Near-ceiling samples are not a true-peak/clipping measurement. '
        'Stereo measurements come from the table, never from listening to this mono input. '
        'Answer entirely in Spanish. Start with ALCANCE: distinguish complete-source tables '
        'from the audio_scope clip. Never title the entire explanation an analysis of the clip. '
        'Use Spanish sections: DATOS DE LAS TABLAS (cite values and scope, no quality judgments); '
        'INTERPRETACION TENTATIVA (cite supporting row IDs and clearly mark assumptions); '
        'NO SE PUEDE CONCLUIR (missing evidence). '
        'State when data cannot establish a problem, instrument identity or mix quality. '
        'Do not recommend or apply EQ, compression, gain changes, effects or plugins. '
        'Tables take precedence over audio guesses and preferences or previous-song examples.\n'
    )


def bind_history_tables(source, report=None):
    """Bind existing tables to a direct upload only on exact stored content identity.

    Mutates session metadata only; never reads or decodes audio, or saves anything.
    Returns True when the association changed (including invalidation on error).
    """
    def clear():
        changed = 'history_report_id' in source or 'measurement_snapshot' in source
        if changed:
            for key in ('moss', 'measurement_snapshot', 'selection', 'history_report_id',
                        'history_report_name', 'verified_audio_sha256'):
                source.pop(key, None)
            source['analysis_id'] = 'direct-'+source['capture_id'][:8]
        return changed
    def reject():
        clear()
        source.pop('moss', None)
        source.pop('selection', None)

    if source.get('source_type') != 'direct_upload':
        raise ValueError('La vinculación del historial requiere el archivo seleccionado para MOSS.')
    if report is None:
        return clear()
    digest = source.get('audio_sha256')
    identity = report.get('track_identity') or {}
    expected = identity.get('sha256') if isinstance(identity, dict) else None
    if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', expected):
        reject()
        raise ValueError('Este informe no conserva una huella SHA-256 válida del audio. No se pueden verificar sus tablas sin recalcular; no se usarán por nombre o fecha.')
    if not isinstance(digest, str) or digest.lower() != expected.lower():
        reject()
        raise ValueError('El audio seleccionado no coincide con el del informe (SHA-256 diferente). Elige el informe del mismo archivo; estas tablas no se enviarán a MOSS.')
    try:
        snapshot = measurement_snapshot(report)
        if not snapshot['rows']:
            raise ValueError('El informe seleccionado no contiene mediciones compatibles.')
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        reject()
        raise ValueError('No se pueden usar las tablas de este informe. No se recalculará el audio.') from exc
    changed = (source.get('history_report_id') != report['session_id']
               or source.get('measurement_snapshot') != snapshot
               or source.get('verified_audio_sha256') != digest.lower())
    if changed:
        clear()
        source.pop('moss', None)
        source.pop('selection', None)
        source.update(analysis_id=report['session_id'], history_report_id=report['session_id'],
                      history_report_name=report.get('mix', {}).get('name', ''),
                      verified_audio_sha256=digest.lower(), measurement_snapshot=snapshot)
    return changed
