"""Local technical references and conservative row-aware prose review.

These are application rules, not new MOSS output or a learned evaluator.
"""
import re
import unicodedata

REFERENCES = {
    'Peak (dBFS)': 'Pico de muestras de la fuente, referido a la escala digital completa. Un nivel negativo en dBFS está por debajo de esa referencia; no es un estado del audio ni descarta saturación previa o picos entre muestras.',
    'Average RMS (dBFS)': 'Nivel de amplitud eficaz: se calcula la raíz de la media de las amplitudes de muestra al cuadrado y se expresa en dBFS. Describe la amplitud de la señal mediante ese cálculo.',
    'Crest factor (dB)': 'Diferencia entre el nivel del pico de muestras y el nivel RMS de la misma fuente, expresada en dB. Compara pico y amplitud eficaz; no mide por sí sola todo el rango dinámico ni la compresión aplicada.',
    'Samples at/above -0.1 dBFS (%)': 'Porcentaje de muestras que alcanza o supera el umbral de la tabla. No demuestra por sí solo presencia o ausencia de clipping previo ni mide picos entre muestras.',
    'Side-to-mid level (dB)': 'Diferencia entre el nivel RMS de Side y el nivel RMS de Mid, expresada en dB. Mid es la semisuma de los canales izquierdo y derecho; Side es su semidiferencia. Un valor negativo indica menor nivel Side que Mid. Esta relación no determina por sí sola la anchura percibida.',
    'Integrated loudness (LUFS)': 'Sonoridad integrada a lo largo de la fuente, calculada con ponderación en frecuencia y compuertas de selección de tramos según el método de medición aplicado. Se expresa en LUFS.',
    'Loudness range (LRA, LU)': 'Variación estadística de la sonoridad a lo largo del tiempo, expresada en LU. Resume la dispersión de la distribución temporal de sonoridad.',
}


def _normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKD',text.casefold()) if not unicodedata.combining(c))


def review_explanation(metric,text):
    """Return a narrow rule-based classification, never a full correctness proof."""
    if metric not in REFERENCES:
        return {'classification':'not_evaluated','findings':[],
                'reference_definition':'','limitation':'Esta medición no tiene reglas técnicas específicas en este módulo.'}
    t=_normalized(text);findings=[]
    def add(kind,code,note):
        findings.append({'classification':kind,'code':code,'explanation':note,
                         'reference_definition':REFERENCES.get(metric,'')})
    if metric in ('Average RMS (dBFS)','Side-to-mid level (dB)'):
        power=re.search(r'(?:potencia\s+media|(?:nivel\s+)?promedio\s+de\s+potencia|mean\s+power|average\s+power)',t)
        distinction=re.search(r'no (?:es|equivale|debe equipararse)|not (?:equal|mean power)|se relaciona|sin (?:distinguir|especificar)|amplitud.*(?:distinta|diferente).*potencia',t)
        if power and not distinction:
            add('technical_confusion','rms_power_confusion','La frase equipara amplitud eficaz y potencia media sin distinguirlas. El nivel RMS en dB se relaciona con potencia; RMS se calcula como raíz de la media de amplitudes al cuadrado.')
        if re.search(r'(?:intensidad|sonoridad) percibida|perceived loudness',t) and not re.search(r'no (?:es|mide|representa|equivale).{0,30}(?:percibida|loudness)|does not.{0,25}loudness',t):
            add('technical_confusion','rms_as_perceived_loudness','RMS mide amplitud eficaz; no es una medida perceptual de sonoridad como LUFS.')
    if metric in ('Peak (dBFS)','Side-to-mid level (dB)') and re.search(r'estado (?:es )?negativo|estado negativo',t):
        add('imprecise','ambiguous_negative_state','Un signo negativo pertenece al nivel relativo en dBFS o dB, no a un estado del audio ni a energía negativa. Explica la referencia correspondiente.')
    if metric=='Side-to-mid level (dB)':
        equating=re.search(r'(?:side|lateral).{0,20}(?:son|es|equivale a|representa) (?:los )?canales (?:izquierd[oa] y derech[oa]|left and right)',t)
        if equating and not re.search(r'no (?:equivale|son|es)|not (?:the |equal)',t):
            add('technical_confusion','side_as_left_right','Side no son los canales izquierdo y derecho: es su semidiferencia; Mid es su semisuma.')
        if re.search(r'(?:parte|zona) lateral.{0,110}(?:parte|zona) central',t):
            add('imprecise','side_as_spatial_parts','Mid/Side son componentes calculadas a partir de canales, no zonas físicas de un único canal.')
        if not (re.search(r'\brms\b|eficaz',t) and re.search(r'side|lateral',t) and re.search(r'mid|central',t)):
            add('incomplete','side_level_definition_missing','Especifica que se comparan los niveles RMS de las componentes Side y Mid.')
        if not re.search(r'nivel|decibel|\bdb\b',t):
            add('imprecise','side_levels_not_linear_amplitude','La medición resta niveles RMS en dB, no amplitudes lineales de las componentes.')
        if not (re.search(r'semisuma|suma',t) and re.search(r'semidiferencia|diferencia',t) and re.search(r'izquierd|derech|canales',t)):
            add('incomplete','side_components_definition_missing','Falta identificar Mid como semisuma y Side como semidiferencia de los canales izquierdo y derecho.')
    if metric=='Crest factor (dB)':
        if re.search(r'(?:es|mide|equivale a) (?:todo |el )?rango dinamico',t) and not re.search(r'no (?:es|mide|equivale)',t):
            add('technical_confusion','crest_as_dynamic_range','El factor de cresta compara pico y RMS; no equivale al rango dinámico completo.')
        elif not (re.search(r'pico|peak',t) and re.search(r'\brms\b|eficaz',t)):
            add('incomplete','crest_definition_missing','Nombrar el factor de cresta o decir que es positivo no explica su comparación entre pico y RMS.')
    if metric=='Integrated loudness (LUFS)':
        if not re.search(r'percept|percibid|ponder|a lo largo|duracion|temporal|seleccion de tramos|gating',t):
            add('incomplete','integrated_loudness_definition_missing','La frase repite sonoridad integrada sin explicar su integración temporal o su relación con la sonoridad percibida.')
        temporal=re.search(r'integrad|integraci|a lo largo|tiempo|temporal',t)
        frequency=re.search(r'ponder.{0,35}frecuenc|frecuenc.{0,35}ponder|k[- ]weight',t)
        selection=re.search(r'compuerta|seleccion.{0,25}(?:tramos|bloques)|gating',t)
        if not (temporal and frequency and selection):
            add('incomplete','loudness_method_missing','Para explicar el método hacen falta integración temporal, ponderación en frecuencia y compuertas o selección de tramos.')
        if re.search(r'percibe.{0,35}ruido y (?:el )?silencio|percepcion.{0,35}ruido.{0,25}silencio',t):
            add('imprecise','loudness_vague_noise_silence','LUFS no se define como percepción de ruido y silencio: especifica ponderación en frecuencia e integración con selección de tramos.')
    if metric=='Loudness range (LRA, LU)':
        negated_unit=re.search(r'(?:no|nunca)\s+(?:(?:se\s+)?(?:expresa|mide|usa|utiliza)\s+)?(?:en\s+|es\s+)?lufs\b|not\s+(?:in\s+)?lufs\b|\blufs\b\s+no\s+es\s+(?:la\s+)?unidad',t)
        if re.search(r'\blufs\b',t) and not negated_unit:
            add('technical_confusion','lra_wrong_unit','LRA se expresa en LU, no en LUFS.')
        maximum=re.search(r'maxim[oa].{0,35}minim[oa]|maxim[oa] menos minim[oa]|maximum.{0,30}minimum',t)
        if maximum and not re.search(r'no (?:es|equivale|mide)|not |no es simplemente',t):
            add('technical_confusion','lra_as_extrema','LRA resume una distribución temporal de sonoridad; no es simplemente máximo menos mínimo.')
        elif not re.search(r'variaci|dispersi|distribu|estadistic|temporal|a lo largo',t):
            add('incomplete','lra_definition_missing','Decir rango de sonoridad no explica que resume variación estadística a lo largo del tiempo.')
    order={'technical_confusion':3,'imprecise':2,'incomplete':1}
    classification=max((f['classification'] for f in findings),key=lambda c:order[c],default='correct_basic')
    return {'classification':classification,'findings':findings,
            'reference_definition':REFERENCES.get(metric,''),
            'limitation':'Reglas de revisión limitadas; correct_basic no certifica toda la frase.'}
