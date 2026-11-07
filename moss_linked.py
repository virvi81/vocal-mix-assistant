"""Strict row-linked explanations. Values always come from the app's context."""
import json
from contextlib import contextmanager
import re
from moss_measurements import validate_context,context_fingerprint


def _strip_label_constants(text,metric):
    remaining=text.replace('−','-')
    label=metric.replace('−','-')
    constants=re.findall(r'(?<![\w.])[+-]?\d+(?:\.\d+)?(?:[–-]\d+(?:\.\d+)?)?\s*(?:dBFS|kHz|Hz)\b',label,re.I)
    for constant in sorted(constants,key=len,reverse=True):
        remaining=re.sub(re.escape(constant),'',remaining,flags=re.I)
    return remaining


def parse_explanations(text, context):
    validate_context(context)
    raw=text.strip()
    if raw.startswith('```'):
        match=re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```',raw,re.S)
        if not match:raise ValueError('El modelo no devolvió un objeto JSON completo.')
        raw=match.group(1)
    def unique_pairs(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Claves JSON duplicadas.')
            result[key]=value
        return result
    try:obj=json.loads(raw,object_pairs_hook=unique_pairs)
    except (ValueError,TypeError) as exc:raise ValueError('La salida no es JSON válido.') from exc
    if not isinstance(obj,dict) or set(obj)!={'explanations'}:
        raise ValueError('La salida debe contener solo explanations; no valores generados por el modelo.')
    entries=obj['explanations'];row_map={r['id']:r for r in context['rows']};known=set(row_map);seen=set();clean=[]
    if not isinstance(entries,list) or not 1<=len(entries)<=len(known):
        raise ValueError('La lista de explicaciones está vacía o no corresponde a las tablas.')
    for entry in entries:
        if not isinstance(entry,dict) or set(entry)!={'row_id','explanation'}:
            raise ValueError('Cada explicación requiere solo row_id y explanation; el valor y la unidad los aporta la app.')
        row_id=entry['row_id'];explanation=entry['explanation']
        if not isinstance(row_id,str) or row_id not in known or row_id in seen:
            raise ValueError('Referencia de fila inexistente o duplicada.')
        if not isinstance(explanation,str) or not 1<=len(explanation.strip())<=600:
            raise ValueError('Explicación vacía o demasiado larga.')
        # A fixed label threshold (e.g. -0.1 dBFS) is metadata, not a measured
        # value. Allow only exact, unit-bearing constants from this same label.
        remaining=_strip_label_constants(explanation,row_map[row_id]['metric'])
        if re.search(r'\d',remaining):
            raise ValueError('La explicación contiene cifras libres: solo se permiten constantes exactas de la etiqueta de esa fila; los valores medidos los aporta la app.')
        seen.add(row_id);clean.append({'row_id':row_id,'explanation':explanation.strip()})
    return {'explanations':clean,'missing_row_ids':[r['id'] for r in context['rows'] if r['id'] not in seen]}


def resolve_explanations(entries,context,fingerprint):
    validate_context(context)
    if fingerprint!=context_fingerprint(context):raise ValueError('La explicación pertenece a otro contexto de tablas.')
    # Validate again at the rendering boundary, not just at worker receipt.
    parsed=parse_explanations(json.dumps({'explanations':entries}),context)
    rows={r['id']:r for r in context['rows']}
    return [{**rows[e['row_id']],'explanation':e['explanation']} for e in parsed['explanations']]


def check_worker_response(response,context):
    validate_context(context)
    if (response.get('analysis_id')!=context['analysis_id'] or
            response.get('measurement_fingerprint')!=context_fingerprint(context)):
        raise ValueError('El worker devolvió otro análisis o una huella de tablas distinta.')
    if response.get('truncated'):raise ValueError('Respuesta incompleta del modelo.')
    return parse_explanations(response.get('description',''),context)


def linked_prompt_contract():
    return (
        '\nTABLE MODE OUTPUT CONTRACT: Return ONLY a JSON object, no introduction, headings, '
        'tables or Markdown. Schema: {"explanations":[{"row_id":"t1",'
        '"explanation":"Explicación breve en español, sin cifras."}]}. '
        'Use one JSON explanation per cited measurement. Cover every supplied row once if possible. '
        'Every explanation must reference exactly one existing row_id. '
        'NEVER include value, unit, metric, entity, scope or other fields. The app supplies these. '
        'No digits, timestamps, row IDs or copied measurements inside explanation text. '
        'Explain the meaning of that row only, using words such as the measured value. '
        'Do not describe song structure, instruments, quality or make production recommendations. '
        'LUFS is sonoridad integrada, LRA is rango de sonoridad, neither measures ruido. '
        'Stems are pistas o grupos exportados completos, never estribillos. '
        'A side/mid energy measurement alone cannot establish perceived width; sample peaks '
        'cannot exclude prior distortion or inter-sample peaks. No normal/moderate/typical labels. '
        'The table value, unit, source, scope and interval will be inserted by the app, not you. '
        'Do not add any prose outside the JSON object.'
    )


def table_request_prompt(instructions):
    """Use the installed processor's supported chat format; no fabricated reasoning."""
    return ('<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
            '<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>\n'+instructions+
            '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')


def audit_linked_rows(linked_rows,context,fingerprint):
    """Audit model prose; table values are trusted app data, not model claims."""
    from moss_audit import audit_description
    entries=[{'row_id':r['id'],'explanation':r['explanation']} for r in linked_rows]
    canonical=resolve_explanations(entries,context,fingerprint)
    prose='\n'.join(_strip_label_constants(r['explanation'],r['metric']) for r in canonical)
    audit=audit_description(prose,context,fingerprint)
    # No generated numeric claims are expected here; their absence is correct.
    audit['findings']=[f for f in audit['findings'] if f['code']!='no_verified_numeric_claims']
    from moss_technical import review_explanation
    audit['technical_reviews']=[]
    for row in canonical:
        review=review_explanation(row['metric'],row['explanation'])
        audit['technical_reviews'].append({'row_id':row['id'],**review})
        for finding in review['findings']:
            audit['findings'].append({'code':finding['code'],'row_id':row['id'],
                'classification':finding['classification'],'evidence':row['explanation'],
                'explanation':finding['explanation']+' Definición de referencia de la app (no de MOSS): '+finding['reference_definition']})
    audit['numeric_checks']=[{'row_id':r['id'],'value':r['value'],'unit':r['unit'],'origin':'app'} for r in canonical]
    audit['status']='review_required' if audit['findings'] else 'no_flags_detected'
    return audit


class LinkedGenerationError(ValueError):
    """No fabricated fallback: a row still failed strict validation."""
    def __init__(self,message,attempts):
        super().__init__(message)
        self.attempts=attempts


def generate_linked_rows(generate,audio,context,memory_text='',max_attempts=2):
    """Generate one fixed-reference continuation per row. Validation is unchanged.

    Numeric measurements and time bounds stay in the app. The model gets only
    label, unit, source, scope and a deterministic sign/availability description.
    Its original continuations are kept in RAM for review, including retries.
    """
    validate_context(context)
    if max_attempts not in (1,2):raise ValueError('Invalid bounded retry count')
    explanations=[];attempts=[]
    for row in context['rows']:
        value=row['value']
        state=('unavailable' if value is None else 'zero' if value==0 else 'positive' if value>0 else 'negative')
        names={
            'Peak (dBFS)':'pico de muestras',
            'Average RMS (dBFS)':'amplitud eficaz RMS',
            'Crest factor (dB)':'factor de cresta',
            'Samples at/above -0.1 dBFS (%)':'porcentaje de muestras que alcanza o supera el umbral indicado en la tabla',
            'Side-to-mid level (dB)':'diferencia de niveles RMS entre las componentes Side y Mid',
            'Integrated loudness (LUFS)':'sonoridad integrada con ponderación en frecuencia y compuertas de selección de tramos',
            'Loudness range (LRA, LU)':'rango de sonoridad',
        }
        # Fixed label constants remain in the app's table, not in model prose.
        label=names.get(row['metric'],re.sub(r'\d+(?:[.,]\d+)?','',row['metric']))
        scopes={'whole_current_mix':'fuente completa',
                'whole_uploaded_stems':'pistas o grupos exportados completos',
                'stem_section':'sección medida de las pistas exportadas'}
        states={'negative':'negativo','positive':'positivo','zero':'cero',
                'unavailable':'no disponible'}
        from moss_technical import REFERENCES, review_explanation
        metadata={'medicion':label,'unidad_correcta':row['unit'],'alcance':scopes[row['scope']],
                  'fuente':'mezcla' if row['entity']=='Current stereo mix' else 'grupo vocal frente a instrumental',
                  'estado_del_valor':states[state]}
        if row['metric'] in REFERENCES:
            metadata['definicion_tecnica_de_referencia']=REFERENCES[row['metric']]
        instructions=(
            'Completa UNA sola oración muy breve EN ESPAÑOL sobre el significado de esta medición. '
            'No traduzcas tu respuesta al inglés. Los valores y referencias los coloca la aplicación. '
            'Datos de contexto: '+json.dumps(metadata,ensure_ascii=False)+'. '
            'PROHIBIDO escribir dígitos, cifras, cantidades, tiempos u otras referencias. '
            'Tampoco escribas con cifras el nivel de referencia digital. Di máximo digital, '
            'escala digital completa o umbral indicado en la tabla, sin cantidades. '
            'Explica la definición de esta medición, sin estimar valores del audio. '
            'Si no está disponible, indica que no se puede evaluar, sin reemplazarlo por nulo. '
            'No des consejos de EQ, compresión o producción ni describas instrumentos. '
            'Usa solo la definición de referencia de esta fila, conservando su magnitud y unidad. '
            'No introduzcas definiciones o unidades de otras mediciones. '
            'No califiques anchura percibida, calidad, normalidad o compresión. '
            'Un pico no permite descartar saturación previa ni picos entre muestras. '
            'Continúa únicamente la oración ya iniciada. La app coloca la unidad y la referencia; '
            'no las repitas. Devuelve solo texto natural, sin JSON, llaves, listas, comillas de '
            'envoltura ni campos. No escribas otras oraciones ni añadas datos.'
        )
        if row['metric']=='Loudness range (LRA, LU)':
            instructions+=' Esta fila resume variación temporal de sonoridad; la única unidad que puede atribuirse a esta medición es LU. '
        elif row['metric']=='Average RMS (dBFS)':
            instructions+=' Esta fila describe amplitud eficaz: conserva el cálculo de raíz de la media de amplitudes al cuadrado. No infieras cómo se escucha. '
        elif row['metric']=='Integrated loudness (LUFS)':
            instructions+=' Incluye explícitamente integración a lo largo de la fuente, ponderación en frecuencia y compuertas de selección de tramos. Describe el método, no solo percepción general. '
        elif row['metric']=='Side-to-mid level (dB)':
            instructions+=' Incluye explícitamente diferencia entre niveles RMS y la derivación de componentes: Mid es semisuma y Side es semidiferencia de izquierdo y derecho. No son zonas de un canal. RMS designa amplitud eficaz, no potencia media cuadrática. Es diferencia de NIVELES en decibelios, no resta de amplitudes lineales; di explícitamente niveles o decibelios. '

        if memory_text:
            instructions+=' Preferencias citadas de otras sesiones, nunca evidencia de esta medición: '+memory_text
        # Row ID and a neutral Spanish sentence starter are syntax scaffolding;
        # the model continuation itself is retained untouched for inspection.
        prefix='Esta medición describe '
        one_context={**context,'rows':[row]}
        error=None
        for attempt in range(max_attempts):
            feedback='' if attempt==0 else ('\nLa continuación anterior fue rechazada por: '+str(error)+'. Usa exclusivamente la definición y unidad de esta fila; una sola frase breve, sin cifras, JSON, listas ni campos.')
            prompt=table_request_prompt(instructions+feedback)+prefix
            output=generate(audio,prompt=prompt,temperature=0.0,max_tokens=384,strip_thinking=True)
            raw=output.text
            detail={'row_id':row['id'],'attempt':attempt+1,'prefill':prefix,'raw_output':raw,
                    'generation_tokens':int(output.generation_tokens),
                    'truncated':output.generation_tokens>=384}
            try:
                if detail['truncated']:raise ValueError('Row generation truncated')
                if not isinstance(raw,str) or not raw.strip() or re.search(r'[{}\[\]]|row_id|explanations',raw):
                    raise ValueError('Se esperaba una oración, no estructura JSON o referencias')
                if len(re.findall(r'[^\W\d_]+',raw,re.U))<3:
                    raise ValueError('Continuación vacía o insuficiente')
                if row['metric']=='Loudness range (LRA, LU)' and re.search(r'\bLUFS\b',raw,re.I):
                    raise ValueError('La explicación de LRA no puede atribuirle la unidad LUFS')
                candidate={'explanations':[{'row_id':row['id'],'explanation':prefix+raw.strip()}]}
                parsed=parse_explanations(json.dumps(candidate,ensure_ascii=False),one_context)
                technical=review_explanation(row['metric'],candidate['explanations'][0]['explanation'])
                strict_metric=row['metric'] in ('Integrated loudness (LUFS)','Side-to-mid level (dB)')
                errors=[f for f in technical['findings'] if strict_metric or f['classification']=='technical_confusion']
                if errors:
                    raise ValueError('Confusión técnica: '+'; '.join(f['explanation'] for f in errors))
                if len(parsed['explanations'])!=1 or parsed['missing_row_ids']:
                    raise ValueError('Expected exactly the requested row')
                explanations.extend(parsed['explanations']);detail['accepted']=True
                attempts.append(detail);error=None;break
            except (ValueError,TypeError) as exc:
                error=str(exc);detail.update(accepted=False,error=error);attempts.append(detail)
        if error is not None:
            raise LinkedGenerationError('No se pudo generar una explicación válida para '+row['id']+': '+error,attempts)
    result={'explanations':explanations}
    # Complete aggregate is validated again against the original full context.
    parsed=parse_explanations(json.dumps(result,ensure_ascii=False),context)
    if parsed['missing_row_ids']:raise LinkedGenerationError('Faltan explicaciones',attempts)
    return {'description':json.dumps(result,ensure_ascii=False),'truncated':False,
            'generation_strategy':'fixed_row_text_v2','row_generation':attempts}


def numeric_token_ids(vocabulary):
    """Constrain common digit-bearing vocabulary tokens; final schema stays strict."""
    return sorted({int(token_id) for token,token_id in vocabulary.items() if re.search(r'\d',token)})


def numeric_logits_factory(original,token_ids):
    def guarded(*args,**kwargs):
        bias=dict((args[0] if args else kwargs.get('logit_bias')) or {})
        bias.update({token_id:float('-inf') for token_id in token_ids})
        if args:
            return original(bias,*args[1:],**kwargs)
        return original(**{**kwargs,'logit_bias':bias})
    return guarded


@contextmanager
def block_numeric_output_tokens(model):
    """Worker-local constrained decoding; no package files or weights are edited."""
    from mlx_audio.lm import sample_utils
    vocabulary=model._ensure_processor().tokenizer.get_vocab()
    ids=numeric_token_ids(vocabulary)
    if not ids:raise ValueError('No se pudo establecer la restricción de tokens numéricos.')
    original=sample_utils.make_logits_processors
    sample_utils.make_logits_processors=numeric_logits_factory(original,ids)
    try:
        yield len(ids)
    finally:
        sample_utils.make_logits_processors=original
