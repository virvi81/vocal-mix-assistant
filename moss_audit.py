"""Conservative, deterministic review of MOSS prose against supplied tables.

No audio/model execution, persistence or network. Warnings are review aids,
not a proof of correctness or a complete semantic/numeric validator.
"""
import math
import re
import unicodedata

from moss_measurements import context_fingerprint, validate_context

# Recognize common metric names; other phrasing remains explicitly unverified.
METRICS = [
    ('Median vocal-to-instrumental RMS', r'\b(?:median vocal[- ]to[- ]instrumental rms|mediana (?:vocal frente a instrumental|(?:de )?voz/instrumental)(?: rms)?)\b'),
    ('Vocal-to-instrumental RMS (dB)', r'\b(?:vocal[- ]to[- ]instrumental rms|vocal frente a instrumental rms|rms voz/instrumental)\b'),
    ('Peak (dBFS)', r'\b(?:peak(?: level)?|pico(?: de muestra)?)\b'),
    ('Average RMS (dBFS)', r'\b(?:average rms(?: level)?|rms (?:medio|promedio)|rms(?=\s*(?:is|es|:|=)))\b'),
    ('Crest factor (dB)', r'\b(?:crest factor|factor de cresta)\b'),
    ('Integrated loudness (LUFS)', r'\b(?:integrated loudness|loudness integrado|sonoridad integrada|cifra de ruido integrado)\b'),
    ('Loudness range (LRA, LU)', r'\b(?:loudness range|lra|rango de sonoridad|rango de ruido)\b'),
    ('Side-to-mid level (dB)', r'\b(?:side[- ]to[- ]mid(?: level)?|relacion lateral/central|nivel lateral/central|nivel lateral frente a medio)\b'),
    ('Quieter comparable moments', r'\b(?:quieter comparable moments|momentos comparables mas bajos)\b'),
    ('Louder comparable moments', r'\b(?:louder comparable moments|momentos comparables mas altos)\b'),
    ('Vocal level spread (voice only)', r'\b(?:vocal level spread(?! \(p90)|desplazamiento del nivel vocal(?! activo))\b'),
    ('Active vocal level spread (p90–p10 dB)', r'\b(?:active vocal level spread|desplazamiento del nivel vocal activo)\b'),
    ('Detectable instrumental frames (%)', r'\b(?:detectable instrumental frames|marcos instrumentales detectables)\b'),
    ('Instrumental level vs typical active level (dB)', r'\b(?:instrumental level vs typical active level|nivel instrumental frente al nivel activo tipico)\b'),
]

NUMBER = r'[+\-]?\d+(?:[.,]\d+)?'
QUANTITY = re.compile(r'(?<![\w.])('+NUMBER+r')\s*(dBFS|LUFS|dB|LU|%)(?!\w)', re.I)
UNIT = re.compile(r'\b(dBFS|LUFS|dB|LU)\b|%', re.I)
CITATION = re.compile(r'\bt\d+\b', re.I)


def _normalize(text):
    text = unicodedata.normalize('NFKD',text).casefold()
    return ''.join(c for c in text if not unicodedata.combining(c)).replace('−','-').replace('**','')


def audit_description(text, context, result_fingerprint):
    validate_context(context)
    findings=[];checks=[]
    def flag(code, evidence, explanation):
        item={'code':code,'evidence':evidence[:500],'explanation':explanation}
        if item not in findings:findings.append(item)
    expected=context_fingerprint(context)
    if result_fingerprint != expected:
        flag('context_mismatch','', 'La respuesta no tiene la misma huella de tablas que el contexto vigente. No se pueden validar sus cifras.')
        return {'context_matches':False,'findings':findings,'numeric_checks':[],
                'status':'context_mismatch'}
    rows={r['id'].lower():r for r in context['rows']}
    global_mix=any(r['scope']=='whole_current_mix' for r in rows.values())
    # An unrelated disclaimer must not excuse an unsafe claim in the same paragraph.
    lines=[part.strip() for line in text.splitlines()
           for part in re.split(r'(?<=[.!?;])\s+|\s+(?:but|however|pero|sin embargo)\s+',line,flags=re.I)
           if part.strip()]
    current_scope = None
    current_interval = None
    for line in lines:
        normalized=_normalize(line)
        # A chorus can be a real musical section; flag only its use as a stem label
        # or explicit translation, not every mention of an estribillo.
        stem_term = re.search(r'\bstems?\b|whole_uploaded_stems', normalized)
        chorus_term = re.search(r'\bestribillos?\b|\bcoros?\b|\bchoruses?\b|\brefrains?\b', normalized)
        correction = re.search(r'\bno (?:son|es|significa|se traduce|debe)|\bnot\b|\bnunca\b|\bnever\b', normalized)
        if stem_term and chorus_term and not correction and (
                re.search(r'^\s*(?:#+\s*|\*+\s*)?(?:estribillos?|coros?|choruses?|refrains?)\s+completos?\b', normalized)
                or re.search(r'stems?.{0,45}(?:significa|traduce|equivale|means|=|:).{0,20}(?:estribill|coro|chorus|refrain)', normalized)
                or re.search(r'whole_uploaded_stems.{0,30}(?:estribill|coro|chorus|refrain)', normalized)):
            flag('stem_mistranslation',line,'Stems completos son pistas o grupos exportados completos, no estribillos ni coros. La respuesta original se conserva; revisa esa etiqueta.')
        heading_scope = next((scope for scope in ('whole_current_mix','whole_uploaded_stems','stem_section')
                              if re.search(r'\b'+scope+r'\b',normalized)), None)
        if heading_scope:
            current_scope = heading_scope
            current_interval = None
        times = re.findall(r'(\d+):(\d{2})', normalized)
        if len(times)==2:
            bounds=[int(a)*60+int(b) for a,b in times]
            if bounds[0]<bounds[1]:current_interval=tuple(bounds)
        if re.search(r'(?:ruido|noise).{0,35}(?:integrado|integrated|lra|lufs)|(?:cifra|rango) de ruido',normalized) and not re.search(r'no (?:es|mide)|not (?:noise|a noise)',normalized):
            flag('loudness_as_noise',line,'LUFS mide sonoridad integrada y LRA su variación en LU. No son ruido de fondo ni relación señal/ruido.')
        if current_scope=='whole_current_mix' and re.search(r'energia por banda|relative band energy',normalized):
            flag('band_wrong_source',line,'Estas bandas pertenecen a la comparación voz/instrumental de stems completos; no son energía de la mezcla completa.')
        refs=CITATION.findall(normalized)
        for ref in refs:
            if ref not in rows:flag('unknown_reference',line,f'La referencia {ref} no existe en las tablas enviadas.')
        # Per-metric spans avoid accepting a peak/RMS swap merely because both
        # numbers occur somewhere in the tables. Repeated/ambiguous labels are reviewed.
        matches=[]
        for metric,pattern in METRICS:
            for match in re.finditer(pattern,normalized):matches.append((match.start(),match.end(),metric))
        # Prefer the longest, most specific match. A relative vocal RMS must
        # never also be interpreted as Average RMS merely because it ends in RMS.
        selected=[]
        for item in sorted(matches,key=lambda m: (-(m[1]-m[0]),m[0])):
            if not any(item[0]<old[1] and item[1]>old[0] for old in selected):selected.append(item)
        matches=sorted(selected)
        chunks=[]
        if matches:
            for i,(start,end,metric) in enumerate(matches):
                chunks.append((normalized[start:matches[i+1][0] if i+1<len(matches) else len(normalized)],metric))
        else:chunks=[(normalized,None)]
        for chunk,metric in chunks:
            quantities=[]
            for m in QUANTITY.finditer(chunk):
                prefix=chunk[max(0,m.start()-45):m.start()]
                if re.search(r'(?:below|above|under|over|reaching|threshold|umbral|inferior a|superior a|menor que|mayor que|por debajo de|por encima de|[<>])\s*$',prefix):
                    continue  # A threshold/comparison is not an asserted measured value.
                quantities.append((float(m.group(1).replace(',','.')),m.group(2).lower()))
            if not quantities and metric:
                # E.g. "Peak (dBFS): -3.78"; don't mistake an ID for a value.
                match=re.search(r'[:=]\s*('+NUMBER+r')(?![\d.,])',chunk)
                units=UNIT.findall(chunk)
                if match and units:quantities=[(float(match.group(1).replace(',','.')),units[0].lower())]
            chunk_refs=CITATION.findall(chunk) or (refs if len(matches)==1 else [])
            candidates=[rows[r] for r in chunk_refs if r in rows]
            if metric:
                named=[r for r in context['rows'] if r['metric']==metric]
                candidates=([r for r in candidates if r['metric']==metric] if candidates else named)
            if current_interval and not chunk_refs and any(r['scope']=='stem_section' for r in candidates):
                candidates=[r for r in candidates if r.get('start_seconds')==current_interval[0] and r.get('end_seconds')==current_interval[1]]
            for value,unit in quantities:
                if not candidates:
                    flag('unverified_number',line,'No se puede vincular esta cifra a una medición identificada. Revisa fuente, sección y unidad.')
                    continue
                same_units=[r for r in candidates if r['unit'].lower()==unit]
                exact=[r for r in same_units if r['value'] is not None and math.isclose(value,r['value'],rel_tol=0,abs_tol=1e-8)]
                if not exact:
                    measured='; '.join(f"{r['id']}: {r['value']} {r['unit']} ({r['entity']}, {r['scope']})" for r in candidates)
                    flag('numeric_mismatch',line,'La cifra o unidad no coincide con la medición identificada. Tabla: '+measured)
                elif len(exact)>1 or (not chunk_refs and len(candidates)>1):
                    flag('ambiguous_measurement',line,'Hay varias fuentes/secciones posibles. La cifra coincide con alguna, pero falta una referencia inequívoca.')
                else:
                    if current_scope and exact[0]['scope']!=current_scope:
                        flag('measurement_scope_mismatch',line,f"La cifra coincide con {exact[0]['id']}, pero pertenece a {exact[0]['entity']} / {exact[0]['scope']}, no a {current_scope}.")
                    checks.append({'row_id':exact[0]['id'],'value':value,'unit':exact[0]['unit']})
        safe = re.search(r'cannot|can.t conclude|does not (?:prove|establish|rule out)|not enough|no (?:demuestra|permite|descarta|se puede)|no basta',normalized)
        if not safe:
            if re.search(r'no clipping|sin (?:clipping|saturacion)|does not (?:exhibit|have|show) clipping|clipping[- ]free|ausencia de (?:clipping|saturacion)|no (?:hay|presenta|se detecta) (?:clipping|saturacion)',normalized):
                flag('unsupported_no_clipping',line,'El pico de muestras y el porcentaje cercano al techo no descartan saturación previa ni picos entre muestras. No hay medición true-peak.')
            if re.search(r'(?:normal|typical|standard|tipic[oa]|normal)\s+(?:\w+\s+){0,5}(?:dynamic|dynamics|range|music|track|loudness|dinamica|musica|sonoridad)|(?:dynamic range|dynamics|loudness|dinamica|sonoridad).{0,55}(?:typical|normal|standard|tipic)',normalized):
                flag('unsupported_norm',line,'Las tablas no incluyen un criterio de normalidad ni una referencia que justifique ese juicio.')
            if re.search(r'moderate (?:perceived )?loudness|controlled dynamic range|sonoridad moderada|dinamica controlada',normalized):
                flag('judgment_without_reference',line,'La medición no incluye un umbral o referencia para calificar la sonoridad como moderada o la dinámica como controlada.')
            if re.search(r'narrow|low stereo width|wide stereo|estrech|imagen ancha',normalized) and re.search(r'stereo|estereo|image|imagen',normalized):
                flag('unsupported_stereo_width',line,'La relación lateral/central describe energía relativa. Por sí sola no determina toda la anchura estéreo percibida.')
            if re.search(r'low frequenc|graves',normalized) and re.search(r'low in energy|lack|insuficien|falta|poc[oa]',normalized) and not re.search(r'vocal.to.instrument|voz.*instrument|relative to|respecto',normalized):
                flag('relative_band_as_mix',line,'Estas bandas comparan voz e instrumental; no prueban falta de graves en la mezcla completa.')
            if re.search(r'vocals? (?:are|is).{0,25}prominent|voz.{0,25}prominente',normalized):
                flag('unsupported_prominence',line,'La diferencia RMS voz/instrumental no determina por sí sola prominencia o inteligibilidad perceptiva.')
        if global_mix and re.search(r'audio clip|fragmento',normalized) and (matches or re.search(r'characteristics|caracteristicas|analysis|analisis',normalized)) and not re.search(r'whole|complet[oa]|global|not|no ',normalized):
            flag('scope_review',line,'Las mediciones globales corresponden al archivo completo, no solo al fragmento que escucha MOSS.')
    if not checks:flag('no_verified_numeric_claims','', 'No se han podido comprobar cifras con las formas reconocidas. Esto no valida la descripción.')
    return {'context_matches':True,'findings':findings,'numeric_checks':checks,
            'status':'review_required' if findings else 'no_flags_detected'}
