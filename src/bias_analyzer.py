"""Motor híbrido de análisis de sesgos: inferencia ML + detección y resaltado de frases."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import html
import json
import math
import re
from typing import Any, Dict, List, Tuple

import joblib

from src.config import (
    ALL_CATEGORIES,
    BEST_MODEL_META_PATH,
    BIAS_CATEGORIES,
    CATEGORY_INFO,
    METRICS_JSON_PATH,
    MODEL_DISPLAY_NAMES,
    MODELS_DIR,
    RF_MODEL_PATH,
)
from src.text_utils import SPANISH_MINIMAL_STOPWORDS, fold_accents_1to1, normalize_for_ml


@dataclass
class DetectedSpan:
    start: int
    end: int
    text: str
    category: str
    explanation: str
    suggestion: str


@dataclass
class MLInfluentialTerm:
    term: str
    tfidf_weight: float
    influence_score: float
    source: str


@dataclass
class AnalysisResult:
    model_key: str
    model_name: str
    predicted_class: str
    probabilities: Dict[str, float]
    bias_probabilities: Dict[str, float]
    inclusivity_score: float
    risk_level: str
    spans: List[DetectedSpan]
    highlighted_html: str
    detected_categories: List[str]
    technical_terms: List[str]
    raw_ml_category: str = "Neutro"
    ml_confidence: float = 0.0
    ml_margin: float = 0.0
    ml_entropy: float = 0.0
    vocabulary_coverage: float = 0.0
    recognized_tokens: int = 0
    total_tokens: int = 0
    matched_features: int = 0
    ml_reliability: str = "alta"
    ml_reliability_label: str = "Alta"
    decision_mode: str = "neutral_confirmed"
    decision_explanation: str = ""
    influential_terms: List[MLInfluentialTerm] = field(default_factory=list)
    explainability_method: str = ""


# Contexto inmediato inclusivo/anti-discriminación dentro de la misma oración:
# evita falsos positivos al mencionar palabras sensibles dentro de cláusulas de inclusión
# (ej. "no discriminamos por embarazo ni expresión de género", "abierto a personas no binarias").
_INCLUSIVE_CONTEXT_RE = re.compile(
    r"\b(?:sin\s+distincion|sin\s+importar|sin\s+discriminacion|libre\s+de\s+discriminacion|"
    r"cero\s+discriminacion|no\s+discrimin\w*|prohibe\s+cualquier\s+discriminacion|"
    r"independientemente\s+de|igualdad\s+de\s+oportunidades|sin\s+restriccion(?:es)?\s+(?:de|por)|"
    r"respet\w*\s+(?:plenamente\s+)?(?:la|el|su|todas|toda)|incluyendo\s+personas|"
    r"bienvenid[oa]s\s+personas|abiert[oa]\s+a\s+[^.;\n]{0,40}|proceso\s+inclusivo|"
    r"entorno\s+(?:laboral\s+)?diverso)\b",
    flags=re.IGNORECASE,
)

# Marcadores intrínsecamente excluyentes que nunca deben ser ignorados por contexto inclusivo
_ALWAYS_BIASED_RE = re.compile(
    r"\b(?:abstenerse|excluy|descart|excluimos|solo\s+|unicamente|exclusivamente|"
    r"no\s+apto|no\s+se\s+admit|no\s+se\s+permit|no\s+se\s+recib|obligatori|"
    r"perfil(?:es)?\s+tradicional(?:es)?|valores\s+familiares\s+convencionales|"
    r"valores\s+tradicionales\s+de\s+familia|recepcionista\s+mujer|ingenier[oa]\s+hombre|"
    r"sin\s+hijos)\b",
    flags=re.IGNORECASE,
)

# Patrones en texto normalizado sin tildes (1-a-1) ordenados por especificidad
# Formato: (regex_sin_tildes, categoria, explicacion, sugerencia)
BIAS_PATTERNS: List[Tuple[str, str, str, str]] = [
    # ==================== EXCLUYENTE ====================
    (
        r"\brequisito excluyente\s*:\s*sexo\s+femenino\b",
        "Excluyente",
        "Establece el sexo femenino como filtro excluyente para acceder al empleo.",
        "Eliminar el filtro de sexo y especificar competencias objetivas del puesto.",
    ),
    (
        r"\b(?:solo|solamente|unicamente|exclusivamente)\s+(?:mujeres|damas|personal femenino|candidatas mujeres|candidatas femeninas|postulantes de sexo femenino|personal del sexo femenino)\b",
        "Excluyente",
        "Restringe la postulación exclusivamente a mujeres.",
        "Convocatoria abierta a todas las personas que cumplan con los requisitos técnicos.",
    ),
    (
        r"\babstenerse\s+(?:mujeres|hombres|varones|damas|caballeros|senoritas|madres|candidatas|postulantes del sexo opuesto|personas\s+con\s+hijos)(?:\s+de\s+postularse|\s+de\s+postular|\s+de\s+enviar\s+su\s+hoja\s+de\s+vida)?\b",
        "Excluyente",
        "Excluye explícitamente a personas por su género o maternidad.",
        "Eliminar la cláusula de exclusión.",
    ),
    (
        r"\b(?:no apto|no apta|no aptos)\s+para\s+(?:mujeres\s+embarazadas|personas\s+en\s+estado\s+de\s+gestacion)\b",
        "Excluyente",
        "Discrimina por embarazo o estado de gestación, vulnerando derechos laborales fundamentales.",
        "Eliminar esta restricción y detallar condiciones ergonómicas/seguridad objetivas.",
    ),
    (
        r"\b(?:se descartan|excluimos|no se recibiran|no seran tomadas en cuenta)\s+(?:automaticamente\s+)?(?:postulantes|candidatas|postulaciones|candidaturas)[^.;,\n]{0,75}(?:hijos|familiares|sexo|femenino|masculino|mujeres|varones|fertil|maternidad|embaraz\w*|gestacion)\b",
        "Excluyente",
        "Descarta candidaturas por sexo, maternidad, embarazo o responsabilidades familiares.",
        "Evaluar únicamente experiencia, habilidades y disponibilidad acordada del puesto.",
    ),
    (
        r"\b(?:preferiblemente|preferentemente)\s+(?:hombres|varones|mujeres|personal\s+masculino|personal\s+femenino)\s+sin\s+(?:hijos|responsabilidades|cargas|obligaciones)[^.;,\n]{0,40}\b",
        "Excluyente",
        "Combina preferencia de género con discriminación por hijos o responsabilidades familiares.",
        "Indicar claramente la jornada o requerimientos del puesto sin referencias de género ni familia.",
    ),
    (
        r"\b(?:indispensable\s+)?no\s+tener\s+hijos(?:\s+ni\s+planes\s+de\s+(?:maternidad|embarazo)(?:\s+a\s+corto\s+plazo)?)?\b",
        "Excluyente",
        "Condiciona la contratación a la ausencia de maternidad, embarazo o hijos.",
        "Eliminar cualquier referencia a maternidad, paternidad o planes familiares.",
    ),
    (
        r"\b(?:no\s+estar\s+en\s+|personas?\s+en\s+|candidatas?\s+en\s+|mujeres?\s+en\s+|en\s+)?estado\s+de\s+gestacion\b",
        "Excluyente",
        "Introduce discriminación laboral por embarazo o estado de gestación.",
        "Eliminar cualquier requisito o restricción relacionada con embarazo o gestación.",
    ),
    (
        r"\b(?:planes\s+de\s+|prueba\s+de\s+|test\s+de\s+|sin\s+|mujeres?\s+)?embaraz(?:o|ada|adas)\b",
        "Excluyente",
        "Discrimina o filtra candidaturas por embarazo.",
        "Suprimir toda referencia al embarazo en la oferta laboral.",
    ),
    (
        r"\b(?:preferentemente\s+|preferiblemente\s+|indispensable\s+|requisito\s+)?sin\s+hijos(?:\s+pequenos|\s+a\s+cargo|\s+ni\s+(?:planes\s+de\s+(?:maternidad|embarazo)|compromisos\s+conyugales|cargas\s+familiares))?\b",
        "Excluyente",
        "Excluye o desincentiva la postulación de personas con hijos o tareas de cuidado.",
        "Eliminar la condición 'sin hijos' y describir objetivamente la jornada del puesto.",
    ),
    (
        r"\bsin\s+(?:cargas|responsabilidades|obligaciones|impedimentos)\s+(?:familiares|maternales|domesticas)\b",
        "Excluyente",
        "Penaliza a personas con tareas de cuidado o responsabilidades familiares.",
        "Especificar objetivamente los horarios o disponibilidad requerida por el puesto.",
    ),
    # Títulos con género femenino exclusivo ("recepcionista mujer", "(Mujer)", "- Mujer", "MUJERES")
    (
        r"\b(?:recepcionista|ingenier[oa]|abogad[oa]|asistente|secretari[oa]|vendedor[a]?|ejecutiv[oa]|gerent[ea]|supervisor[a]?|coordinador[a]?|analista|desarrollador[a]?|programador[a]?|tecnic[oa]|operari[oa]|auxiliar|contador[a]?|arquitect[oa]|auditor[a]?|chofer|conductor[a]?|jefe|jefa|encargad[oa]|especialista|profesional|administrativ[oa]|cajer[oa]|meser[oa]|camarer[oa]|personal|candidat[oa]|postulante|disenador[a]?|medico|medica|enfermer[oa]|docente|profesor[a]?)(?:\s+de\s+\w+){0,2}\s+(?:mujer|mujeres|femenin[oa]|dama|damas)\b",
        "Excluyente",
        "Asocia explícitamente el título del puesto exclusivamente a mujeres (p. ej., 'recepcionista mujer' o 'personal femenino').",
        "Utilizar denominación neutra del cargo sin restringir por sexo.",
    ),
    (
        r"\b(?:mujer|dama)\s+(?:recepcionista|ingenier[oa]|abogad[oa]|asistente|secretari[oa]|vendedor[a]?|ejecutiv[oa]|gerent[ea]|supervisor[a]?|coordinador[a]?|analista|desarrollador[a]?|programador[a]?|tecnic[oa]|operari[oa]|auxiliar|contador[a]?|arquitect[oa]|auditor[a]?|chofer|conductor[a]?|jefe|jefa|encargad[oa]|administrativ[oa]|cajer[oa]|disenador[a]?)\b",
        "Excluyente",
        "Vincula el cargo exclusivamente a mujeres en la denominación del puesto.",
        "Redactar el título del cargo en formato neutro e inclusivo.",
    ),
    (
        r"\(\s*(?:solo\s+)?(?:mujer(?:es)?|damas?|femenino)\s*\)",
        "Excluyente",
        "Restringe el puesto exclusivamente a mujeres mediante etiqueta entre paréntesis (p. ej., '(Mujer)').",
        "Eliminar la restricción de sexo entre paréntesis del título o descripción.",
    ),
    (
        r"(?:[-–—/|:]\s*)\b(?:mujer(?:es)?|damas?)\b(?!\s*(?:y|o|\/)\s*(?:varon(?:es)?|hombre(?:s)?|caballeros?))",
        "Excluyente",
        "Restringe la vacante exclusivamente a mujeres en el título o encabezado (p. ej., '- Mujer').",
        "Suprimir la especificación de sexo en el título de la oferta.",
    ),
    (
        r"\b(?:operari[oa]s?|personal|auxiliar(?:es)?|ayudantes?|reponedor(?:es)?|almaceneros?|mozos?|vendedor(?:es|as)?|promotor(?:es|as)?|cajer[oa]s?|limpieza|produccion|seguridad|choferes?|conductores?|estibadores?|tecnic[oa]s?)(?:\s+de\s+\w+){0,2}\s+(?:mujeres|mujer|damas|femenino|femenina)\b",
        "Excluyente",
        "Asocia el puesto operativo o comercial exclusivamente a mujeres.",
        "Redactar el puesto de forma neutra sin restringir por sexo.",
    ),
    (
        r"\b(?:buscamos|requerimos|solicitamos|necesitamos|convocamos|contratamos|se\s+requieren?|se\s+necesitan?)\s+(?:a\s+)?(?:mujeres|damas)\b|(?<!de\s)(?<!para\s)\bdamas\s+(?:universitarias|jovenes|de\s+buena|con\s+experiencia|para\s+trabajar|proactivas|dinamicas)\b|(?<!y\s)(?<!o\s)(?<!ni\s)(?<!hombres,\s)(?<!varones,\s)\bmujeres\b(?!\s*(?:,|y|o|\/)\s*(?:hombres|varones))(?=\s*(?:[-–—/|.\n]|$))",
        "Excluyente",
        "Restringe la convocatoria específicamente a mujeres o damas.",
        "Abrir la convocatoria a todas las personas que cumplan los requisitos del puesto.",
    ),
    (
        r"\b(?:reservada|destinada)\s+(?:unicamente|exclusivamente)\s+para\s+(?:mujeres|personal femenino)\b",
        "Excluyente",
        "Reserva la vacante exclusivamente a un solo género.",
        "Indicar que la vacante está abierta a todas las personas calificadas.",
    ),
    (
        r"\bsexo\s+femenino\s+excluyente\b|\bpertenecer\s+al\s+sexo\s+femenino\b",
        "Excluyente",
        "Impone el sexo femenino como criterio obligatorio de selección.",
        "Suprimir el criterio de sexo.",
    ),
    (
        r"\ben\s+edad\s+fertil\b|\bpermisos\s+por\s+maternidad(?:\s+o\s+paternidad)?\b",
        "Excluyente",
        "Introduce sesgo excluyente vinculado a la maternidad o fertilidad.",
        "Eliminar referencias a edad fértil o permisos de maternidad.",
    ),
    (
        r"\bsolo\s+(?:se aceptaran|se evaluaran|seran contactados|se consideraran)[^.;,\n]{0,65}(?:varones|hombres|mujeres|candidatas|masculino|femenino)\b",
        "Excluyente",
        "Limita la evaluación de candidaturas a un único sexo.",
        "Evaluar a todas las personas postulantes bajo los mismos criterios técnicos.",
    ),
    # ==================== LGBTQ+ ====================
    (
        r"\bvalores\s+familiares\s+(?:convencionales|tradicionales|clasicos)\b|\bvalores\s+tradicionales\s+de\s+familia(?:\s+conformada\s+unicamente\s+por\s+hombre\s+y\s+mujer)?\b|\bprincipios\s+familiares\s+tradicionales\b|\bmodelo\s+de\s+matrimonio\s+clasico\b",
        "LGBTQ+",
        "Impone un modelo familiar heteronormativo o convencional excluyente hacia personas LGBTQ+.",
        "Sustituir por 'valores de respeto profesional, integridad y convivencia'.",
    ),
    (
        r"\bperfil(?:es)?\s+tradicional(?:es)?\b",
        "LGBTQ+",
        "La exigencia de un 'perfil tradicional' suele operar como filtro cisheteronormativo o excluyente de identidades diversas.",
        "Sustituir por competencias profesionales y culturales objetivas (p. ej., 'enfoque ético y profesional').",
    ),
    (
        r"\b(?:aspecto|identidad|imagen|presentacion(?:\s+personal)?)\s+(?:estrictamente\s+)?acorde\s+a(?:l|\s+su)\s+sexo\s+(?:biologico|de\s+nacimiento|registral)\b",
        "LGBTQ+",
        "Exige conformidad cisnormativa respecto al sexo asignado al nacer.",
        "Sustituir por 'presentación profesional acorde al entorno laboral'.",
    ),
    (
        r"\bvestimenta\s+estrictamente\s+masculina\s+o\s+femenina(?:\s+segun\s+(?:el\s+)?dni)?\b|\bcodigo\s+(?:de\s+vestimenta|tradicional|de\s+imagen)\s+(?:estrictamente\s+)?(?:binario|masculino\s+o\s+femenino)\b",
        "LGBTQ+",
        "Impone un código de vestimenta estrictamente binario que discrimina a personas trans y no binarias.",
        "Reemplazar por 'código de vestimenta profesional acorde al entorno laboral'.",
    ),
    (
        r"\babstenerse\s+personas\s+(?:con\s+expresion\s+de\s+genero\s+no\s+convencional|trans(?:\s*,\s*|\s+o\s+)(?:de\s+identidad\s+)?no\s+binari[oa]s)\b",
        "LGBTQ+",
        "Excluye explícitamente a personas trans, no binarias o con expresión de género diversa.",
        "Eliminar esta cláusula discriminatoria.",
    ),
    (
        r"\bexpresion(?:es)?\s+de\s+genero(?:\s+(?:no\s+convencional(?:es)?|no\s+binari[oa]s?|divers[oa]s?|convencional(?:es)?))?\b",
        "LGBTQ+",
        "Condiciona o restringe la expresión de género de las personas postulantes.",
        "Eliminar restricciones sobre expresión de género y promover un entorno respetuoso.",
    ),
    (
        r"\b(?:personas?\s+|identidad(?:es)?\s+|denominaciones?\s+|esteticas?\s+)?no\s+binari[oa]s?\b",
        "LGBTQ+",
        "Introduce una restricción o señalamiento discriminatorio hacia personas no binarias.",
        "Garantizar un entorno inclusivo abierto a todas las identidades de género.",
    ),
    (
        r"\bsin\s+(?:rasgos\s+androginos|ambiguedades\s+de\s+genero)\b|\besteticas\s+diversas\s+o\s+no\s+binarias\b",
        "LGBTQ+",
        "Estigmatiza expresiones de género diversas o no normativas.",
        "Eliminar restricciones sobre expresión de género.",
    ),
    (
        r"\bnombre\s+legal\s+obligatorio\s+(?:debe\s+coincidir|coincidente)\s+con\s+(?:el\s+)?sexo\s+de\s+nacimiento\b|\bcoincidencia\s+entre\s+identidad\s+expresada\s+y\s+sexo\s+de\s+nacimiento\b",
        "LGBTQ+",
        "Niega el derecho al nombre elegido e identidad de género de personas trans.",
        "Respetar el nombre y pronombres con los que se identifica cada persona.",
    ),
    (
        r"\bno\s+se\s+(?:permiten?|admiten?)\s+(?:el\s+uso\s+de\s+pronombres\s+neutros|cambios\s+de\s+nombre\s+social|expresiones\s+de\s+genero\s+diversas)\b",
        "LGBTQ+",
        "Restringe el respeto a la identidad de género y los pronombres personales.",
        "Promover un ambiente respetuoso de la identidad y expresión de cada persona.",
    ),
    (
        r"\b(?:entorno|ambiente|cultura)(?:\s+laboral|\s+corporativa|\s+conservadora)?\s+(?:estrictamente\s+)?(?:tradicional\s+(?:y\s+)?)?heteronormativ[oa]\b",
        "LGBTQ+",
        "Declara un entorno laboral heteronormativo excluyente para diversidades sexogenéricas.",
        "Sustituir por 'entorno laboral respetuoso e inclusivo'.",
    ),
    (
        r"\b(?:hombres\s+o\s+mujeres\s+biologicos?|identidad\s+cisgenero|sexo\s+biologico|sexo\s+de\s+nacimiento|sexo\s+registral|orientacion\s+tradicional)\b",
        "LGBTQ+",
        "Introduce criterios cisnormativos o heteronormativos irrelevantes para el trabajo.",
        "Eliminar referencias al sexo biológico, identidad cisgénero u orientación.",
    ),
    (
        r"\bcoherencia\s+total\s+entre\s+voz[^.;\n]{0,50}sexo\s+registral\b|\bnormas\s+(?:estrictas\s+de\s+presentacion\s+binaria|tradicionales\s+de\s+genero)\b|\bimagen\s+binaria\s+tradicional\b",
        "LGBTQ+",
        "Impone normas binarias de género ajenas al desempeño profesional.",
        "Enfocar los criterios en profesionalismo y competencias técnicas.",
    ),
    # ==================== DEMOGRÁFICO ====================
    (
        r"\bedad(?:\s+excluyente|\s+comprendida|\s+referencial)?\s*:?\s*(?:entre|de)?\s*\d{2}\s*(?:y|a|-)\s*\d{2}\s+anos\b|\bedad(?:\s+maxima|\s+referencial|\s+limite)\s*:?\s*(?:de\s+|hasta\s+)?\d{2}(?:\s+anos)?(?:\s+excluyente)?\b",
        "Demográfico",
        "Establece un filtro discriminatorio por rango de edad (edadismo).",
        "Eliminar límites de edad y especificar años de experiencia si son necesarios.",
    ),
    (
        r"\b(?:personas\s+)?mayores\s+de\s+\d{2}\s+anos\s+(?:por\s+favor\s+)?abstenerse(?:\s+de\s+postular)?\b|\babstenerse\s+(?:personas\s+)?mayores\s+de\s+\d{2}\s+anos\b",
        "Demográfico",
        "Excluye explícitamente a personas candidatas por su edad.",
        "Eliminar la exclusión por edad.",
    ),
    (
        r"\b(?:muy\s+)?joven(?:es)?\s*\(\s*(?:edad\s+)?(?:de\s+)?(?:entre\s+)?\d{2}\s+(?:y|a)\s+\d{2}\s+anos\s*\)",
        "Demográfico",
        "Restringe la postulación a personas jóvenes dentro de un rango etario específico (edadismo).",
        "Sustituir por 'con competencias acordes al puesto'.",
    ),
    (
        r"\b(?:joven(?:es)?\s+(?:recien\s+egresad[oa]s?\s+)?|profesional\s+|talento\s+)?menor(?:es)?\s+de\s+\d{2}\s+anos\b|\b(?:de\s+)?entre\s+\d{2}\s+(?:y|a)\s+\d{2}\s+anos\b(?!\s+de\s+experiencia)|\bde\s+\d{2}\s+a\s+\d{2}\s+anos\b(?!\s+de\s+experiencia)",
        "Demográfico",
        "Limita la postulación a un rango etario específico.",
        "Sustituir por 'con competencias acordes al puesto' o el nivel de seniority requerido.",
    ),
    (
        r"\b(?:excelente\s+)?presencia\s+fisica(?:\s+impecable)?(?:\s*,\s*adjuntar\s+foto\s+de\s+cuerpo\s+entero(?:\s+en\s+el\s+cv)?)?\b|\badjuntar\s+foto\s+de\s+cuerpo\s+entero(?:\s+en\s+el\s+cv)?\b|\bfotografia\s+reciente\s+actualizada\b|\bfoto\s+de\s+cuerpo\s+entero\b|\bfoto\s+actualizada\b",
        "Demográfico",
        "Exige apariencia física o fotografía corporal, promoviendo discriminación estética.",
        "Eliminar el requisito de fotografía o apariencia física.",
    ),
    (
        r"\b(?:estatura\s+acorde\s+al\s+puesto|complexion\s+esbelta|buena\s+pinta|apariencia\s+fisica\s+impecable|presencia\s+estetica)\b",
        "Demográfico",
        "Condiciona la selección a características físicas o estéticas ajenas al rol.",
        "Eliminar referencias a la apariencia o complexión física.",
    ),
    (
        r"\b(?:estado\s+civil\s+|perfil\s+)?solter[oa]s?(?:\s+sin\s+hijos(?:\s+ni\s+compromisos\s+conyugales)?)?\b|\bsin\s+(?:compromisos\s+conyugales|dependientes\s+economicos)\b",
        "Demográfico",
        "Filtra por estado civil o situación personal.",
        "Eliminar referencias al estado civil.",
    ),
    (
        r"\bnacionalidad\s+local\s+excluyente\b|\b(?:unicamente|solo\s+se\s+aceptaran)\s+ciudadanos\s+nativos\s+del\s+pais\b|\bnativo\s+del\s+pais\b|\babstenerse\s+(?:postulantes\s+)?extranjeros(?:\s+o\s+migrantes)?\b|\bpersonas\s+extranjeras\b",
        "Demográfico",
        "Discrimina por nacionalidad u origen migratorio.",
        "Sustituir por 'contar con autorización legal vigente para trabajar en el país'.",
    ),
    (
        r"\b(?:residir|residente|domicilio\s+comprobable)\s+en\s+(?:zona\s+norte|barrios\s+residenciales(?:\s+exclusivos)?|distritos\s+centricos\s+acomodados)\b|\buniversidades\s+privadas\s+de\s+elite\b",
        "Demográfico",
        "Introduce sesgo socioeconómico o territorial excluyente.",
        "Indicar la ubicación de la sede o modalidad de trabajo sin filtrar por barrio de residencia.",
    ),
    (
        r"\b(?:fecha\s+de\s+nacimiento|edad\s+exacta|estado\s+civil|limite\s+de\s+edad|rango\s+etario)\b",
        "Demográfico",
        "Solicita datos demográficos personales susceptibles de sesgo en selección.",
        "Evitar solicitar edad, fecha de nacimiento o estado civil en la postulación.",
    ),
    (
        r"\b(?:energia\s+juvenil|talento\s+muy\s+joven|perfil\s+joven|joven\s+recien\s+egresad[oa])\b|\banalista\s+de\s+marketing\s+digital\s+joven\b|\b(?:muy\s+)?joven\b(?!\s+(?:empresa|startup|compania|equipo|organizacion))",
        "Demográfico",
        "Asocia la idoneidad del puesto con la juventud (sesgo etario).",
        "Reemplazar por 'con competencias acordes al puesto' o indicar el nivel de experiencia requerido.",
    ),
    # ==================== MASCULINO ====================
    # Restricciones explícitas hacia hombres/varones/caballeros/personal masculino
    (
        r"\brequisito excluyente\s*:\s*sexo\s+masculino\b|\bsexo\s+masculino\s+excluyente\b|\bvaron\s+excluyente\b|\bpertenecer\s+al\s+sexo\s+masculino\b",
        "Masculino",
        "Impone el sexo masculino como requisito obligatorio de selección.",
        "Suprimir el criterio de sexo y evaluar únicamente competencias objetivas.",
    ),
    (
        r"\b(?:solo|solamente|unicamente|exclusivamente)\s+(?:hombres|varones|caballeros|personal masculino|candidatos varones|candidatos masculinos|postulantes de sexo masculino|personal del sexo masculino)\b",
        "Masculino",
        "Restringe la postulación exclusivamente a hombres o varones.",
        "Abrir la convocatoria a todas las personas que cumplan con los requisitos técnicos.",
    ),
    (
        r"\b(?:reservada|destinada)\s+(?:unicamente|exclusivamente)\s+para\s+(?:varones|hombres|personal masculino)\b",
        "Masculino",
        "Reserva la vacante exclusivamente para hombres o personal masculino.",
        "Indicar que la vacante está abierta a todas las personas calificadas.",
    ),
    (
        r"\(\s*(?:solo\s+)?(?:varon(?:es)?|hombres?|caballeros?|masculino)\s*\)",
        "Masculino",
        "Restringe el puesto específicamente a varones mediante etiqueta entre paréntesis (p. ej., '(Varón)' o '(HOMBRE)').",
        "Eliminar la restricción de sexo entre paréntesis del título o descripción.",
    ),
    (
        r"(?:[-–—/|:]\s*)\b(?:varon(?:es)?|caballeros?|hombres?)\b(?!\s*(?:y|o|\/)\s*(?:mujer(?:es)?|damas?))",
        "Masculino",
        "Restringe la vacante a varones en el título o encabezado (p. ej., '- Varón' o '/ Varones').",
        "Suprimir la especificación de sexo en el título de la oferta.",
    ),
    (
        r"\b(?:recepcionista|ingenier[oa]|abogad[oa]|asistente|secretari[oa]|vendedor[a]?|ejecutiv[oa]|gerent[ea]|supervisor[a]?|coordinador[a]?|analista|desarrollador[a]?|programador[a]?|tecnic[oa]|operari[oa]s?|auxiliar(?:es)?|ayudantes?|reponedor(?:es)?|almaceneros?|mozos?|promotor(?:es|as)?|contador[a]?|arquitect[oa]|auditor[a]?|chofer(?:es)?|conductor(?:es|as)?|jefe|jefa|encargad[oa]|especialista|profesional|administrativ[oa]|cajer[oa]s?|meser[oa]|camarer[oa]|personal|candidat[oa]|postulante|disenador[a]?|medico|medica|enfermer[oa]|docente|profesor[a]?|limpieza|produccion|seguridad|estibadores?)(?:\s+de\s+\w+){0,2}\s+(?:hombre|hombres|varon|varones|masculino|masculinos|caballero|caballeros)\b",
        "Masculino",
        "Asocia explícitamente el puesto a personal masculino o varones (p. ej., 'personal masculino', 'operario varón' o 'ingeniero hombre').",
        "Utilizar denominación neutra del cargo sin restringir por sexo.",
    ),
    (
        r"\b(?:hombre|varon|caballero)\s+(?:recepcionista|ingenier[oa]|abogad[oa]|asistente|secretari[oa]|vendedor[a]?|ejecutiv[oa]|gerent[ea]|supervisor[a]?|coordinador[a]?|analista|desarrollador[a]?|programador[a]?|tecnic[oa]|operari[oa]|auxiliar|contador[a]?|arquitect[oa]|auditor[a]?|chofer|conductor[a]?|jefe|jefa|encargad[oa]|administrativ[oa]|cajer[oa]|disenador[a]?)\b",
        "Masculino",
        "Vincula el cargo específicamente a varones en la denominación del puesto.",
        "Redactar el título del cargo en formato neutro e inclusivo.",
    ),
    (
        r"(?<!y\s)(?<!o\s)(?<!ni\s)(?<!hombres,\s)(?<!mujeres,\s)(?<!damas\s)(?<!damas,\s)\bvaron(?:es)?\b(?!\s*(?:y|o|,|\/)\s*(?:mujer|dama|femenin|persona))",
        "Masculino",
        "Dirige la convocatoria específicamente a varones, restringiendo el acceso por género.",
        "Utilizar términos neutros como 'personas postulantes' o 'personal'.",
    ),
    (
        r"\b(?:buscamos|requerimos|solicitamos|necesitamos|convocamos|contratamos|se\s+requieren?|se\s+necesitan?)\s+(?:a\s+)?(?:caballeros|hombres)\b|(?<!de\s)(?<!para\s)(?<!y\s)(?<!o\s)(?<!damas,\s)\bcaballeros?\b(?=\s+(?:para\s+|con\s+(?:experiencia|disponibilidad|secundaria|estudios|brevete|licencia)|de\s+\d{2}|entre\s+\d{2}|jovenes|responsables|proactivos|dinamicos|mayores))",
        "Masculino",
        "Dirige la vacante específicamente a caballeros u hombres.",
        "Reemplazar por 'personas postulantes' o 'personal'.",
    ),
    (
        r"\b(?:un\s+verdadero\s+)?tiburon\s+(?:de\s+la\s+tecnologia|comercial|de\s+negocios|en\s+su\s+area|decidido)\b|\btiburon\b",
        "Masculino",
        "Metáfora hipercompetitiva asociada a estereotipos masculinos agresivos.",
        "Reemplazar por 'especialista con sólido enfoque estratégico'.",
    ),
    (
        r"\b(?:un\s+)?guerrero\s+(?:de\s+ventas|de\s+resultados|comercial|con\s+mentalidad\s+ganadora)\b|\bguerrero\b",
        "Masculino",
        "Metáfora bélica codificada como masculina que disuade a perfiles diversos.",
        "Reemplazar por 'profesional orientado/a a objetivos y alto desempeño'.",
    ),
    (
        r"\b(?:un\s+)?hombre\s+de\s+accion\b|\bhombre\s+con\s+empuje\s+arrollador\b|\bcaballero\s+(?:con\s+(?:caracter\s+fuerte|fuerte\s+personalidad)|resuelto)\b",
        "Masculino",
        "Asocia explícitamente el puesto con el género masculino.",
        "Reemplazar por 'profesional proactivo/a con iniciativa y liderazgo'.",
    ),
    (
        r"\blider(?:azgo)?\s+(?:dominante|autoritario|alfa)\b|\bdon\s+de\s+mando\b|\bimponer\s+(?:sus\s+)?decisiones(?:\s+tecnicas)?\b|\bmarcar\s+autoridad\b",
        "Masculino",
        "Promueve estereotipos de liderazgo dominante/autoritario codificados como masculinos.",
        "Reemplazar por 'liderazgo estratégico, colaborativo y orientado a objetivos'.",
    ),
    (
        r"\b(?:espiritu|perfil|agresividad)\s+(?:sumamente\s+)?agresiv[oa]\b|\bagresiv[oa]\s+comercial\b|\bperfil\s+agresivo\b",
        "Masculino",
        "El adjetivo 'agresivo' es uno de los marcadores más fuertes de sesgo masculino en ofertas laborales.",
        "Reemplazar por 'enfoque proactivo y orientado al crecimiento'.",
    ),
    (
        r"\b(?:rockstar|ninja|crack\s+incansable|campeon\s+corporativo)\b",
        "Masculino",
        "Jerga competitiva asociada a culturas laborales masculinizadas ('bro culture').",
        "Reemplazar por 'profesional con sólida experiencia técnica'.",
    ),
    (
        r"\b(?:aguerrid[oa]|mentalidad\s+ganadora|mentalidad\s+de\s+conquista|instinto\s+cazador|energia\s+viril|temple\s+de\s+acero|sangre\s+fria|hambre\s+feroz|ambiente\s+hipercompetitivo|hermandad\s+de\s+profesionales|entre\s+muchachos)\b",
        "Masculino",
        "Lenguaje altamente codificado hacia estereotipos masculinos competitivos.",
        "Utilizar descriptores profesionales neutros enfocados en competencias.",
    ),
    (
        r"\b(?:caracter\s+fuerte|duro\s+e\s+imponente|recio\s+y\s+dominante|competitivo\s+y\s+ambicioso|voraz\s+y\s+dispuesto\s+a\s+dominar|implacable\s+y\s+competitivo|ambicioso\s+con\s+mentalidad\s+ganadora|audaz\s+y\s+contundente|combativo)\b|(?<!salario\s)(?<!salario\sbase\s)(?<!sueldo\s)(?<!remuneracion\s)(?<!compensacion\s)(?<!beneficios\s)(?<!paquete\s)(?<!mercado\s)(?<!entorno\s)(?<!ritmo\s)\bcompetitiv[oa]s?\b",
        "Masculino",
        "Adjetivos agentivos o hipercompetitivos asociados a codificación de género masculina.",
        "Sustituir por competencias observables como 'orientado al logro' o 'toma de decisiones fundamentada'.",
    ),
    # ==================== FEMENINO ====================
    (
        r"\b(?:una\s+)?(?:joven\s+)?senorita\s+(?:amable|carismatica|elegante|coqueta|prolija|atenta)(?:\s+y\s+\w+)?\b|\bsenorit[ao]s?\b",
        "Femenino",
        "El trato 'señorita' infantiliza y limita el puesto a mujeres jóvenes bajo criterios tradicionales.",
        "Reemplazar por 'profesional' o 'persona especialista'.",
    ),
    (
        r"\b(?:excelente\s+)?(?:presencia|imagen|estetica)\s+femenina(?:\s+(?:impecable|distinguida|cuidada))?\b|\btoque\s+femenino\b|\bsensibilidad\s+femenina\b",
        "Femenino",
        "Asocia la idoneidad del cargo con estereotipos de feminidad y apariencia.",
        "Sustituir por 'excelente trato profesional y comunicación efectiva'.",
    ),
    (
        r"\b(?:excelente|buena)\s+presencia\b",
        "Femenino",
        "Exigir 'excelente presencia' o 'buena presencia' suele encubrir sesgos de apariencia y género en puestos de atención o administración.",
        "Reemplazar por criterios objetivos como 'excelente comunicación y trato profesional'.",
    ),
    (
        r"\b(?:trato\s+maternal|actitud\s+hospitalaria[^.;\n]{0,30}maternal|instinto\s+de\s+cuidado|vocacion\s+de\s+cuidado(?:\s*,\s*paciencia\s+y\s+sensibilidad)?|mimar\s+a\s+los\s+clientes)\b|\bmaternal(?:es)?\b",
        "Femenino",
        "Reproduce estereotipos de rol doméstico o maternal ajenos a las competencias técnicas del puesto.",
        "Reemplazar por 'trato cordial y orientación profesional al servicio'.",
    ),
    (
        r"\bvocacion\s+de\s+servicio\b",
        "Femenino",
        "La expresión 'vocación de servicio' aparece frecuentemente codificada en roles feminizados de asistencia o cuidado.",
        "Sustituir por 'orientación profesional al cliente y resolución eficiente de consultas'.",
    ),
    (
        r"\b(?:trato\s+dulce|mujer\s+delicada|dulce\s*,\s*carinosa|dulce\s+y\s+servicial|dulce\s*,\s*comprensiva|sonrisa\s+calida|sonrisa\s+permanente|voz\s+suave|personalidad\s+docil[^.;\n]{0,25}sumisa|actitud\s+servicial|dispuesta\s+a\s+complacer|complacer\s+las\s+necesidades|entrega\s+abnegada|coqueta|embellezca\s+el\s+ambiente)\b|\bsumis[oa]s?\b|\bempatic[oa]s?\b(?!\s+y\s+respetuos[oa])",
        "Femenino",
        "Utiliza atributos estereotipadamente asociados al género femenino (dulzura, sumisión o emotividad) en lugar de competencias laborales objetivas.",
        "Sustituir por competencias medibles como 'comunicación asertiva, colaboración y trato profesional'.",
    ),
    (
        r"\b(?:una\s+)?(?:asistente\s+administrativa|asistente\s+femenina|secretaria\s+ejecutiva|chica\s+ordenada|chica\s+simpatica(?:\s+y\s+atenta)?|chica\s+encantadora|dama\s+(?:elegante|de\s+trato\s+calido)|mujer\s+(?:armoniosa|simpatica|calida)|perfil\s+femenino\s+colaborador)\b|\brecepcionista\b|(?<!y\s)(?<!o\s)(?<!\/)\b(?:anfitrionas?|azafatas?)\b(?!\s*(?:y|o|\/)\s*(?:anfitrion|mozo|mesero))",
        "Femenino",
        "Femeniza la denominación del puesto o recurre a etiquetas estereotipadas (como 'asistente administrativa', 'recepcionista', 'anfitriona' o 'azafata').",
        "Usar títulos neutros como 'profesional de asistencia administrativa', 'persona especialista en recepción' o 'personal de atención y protocolo'.",
    ),
]

# Patrones para detectar y verificar preservación de requisitos técnicos
TECHNICAL_TERM_PATTERNS: List[str] = [
    r"\bPython(?:\s*\d+(?:\.\d+)*\+?)?\b",
    r"\bFastAPI\b",
    r"\bDjango\b",
    r"\bPostgreSQL\b",
    r"\bMySQL\b",
    r"\bSQL(?:\s+avanzado|\s+basico)?\b",
    r"\bDocker\b",
    r"\bKubernetes\b",
    r"\bAWS(?:\s*\([^)]+\))?\b",
    r"\bGCP\b",
    r"\bAzure\b",
    r"\bTypeScript\b",
    r"\bJavaScript\b",
    r"\bReact\b",
    r"\bNext\.js\b",
    r"\bTailwind\s+CSS\b",
    r"\bGraphQL\b",
    r"\bJest\b",
    r"\bPytest\b",
    r"\bGitHub\s+Actions\b",
    r"\bGit\b",
    r"\bCI/CD\b",
    r"\bPandas\b",
    r"\bScikit-learn\b",
    r"\bPyTorch\b",
    r"\bTensorFlow\b",
    r"\bPower\s+BI\b",
    r"\bTableau\b",
    r"\bLooker\s+Studio\b",
    r"\bCRM\s+(?:Salesforce|HubSpot)\b",
    r"\bSalesforce\b",
    r"\bHubSpot\b",
    r"\bZendesk\b",
    r"\bFreshdesk\b",
    r"\bMailchimp\b",
    r"\bKlaviyo\b",
    r"\bGoogle\s+Analytics\s*4\b",
    r"\bMeta\s+Ads\b",
    r"\bGoogle\s+Ads\b",
    r"\bSEO/SEM\b",
    r"\bMicrosoft\s+Excel(?:\s+avanzado)?\b",
    r"\bExcel(?:\s+avanzado)?\b",
    r"\bMicrosoft\s+Office\b|\bpaquete\s+Office\b|\bGoogle\s+Workspace\b",
    r"\bERP\s+SAP(?:\s+(?:WM|FI|MM))?\b",
    r"\bSAP(?:\s+(?:WM|FI|MM))?\b",
    r"\bERP\b",
    r"\bCRM\b",
    r"\bWMS\b",
    r"\bISO\s*9001\b",
    r"\bLean\s+Manufacturing\b",
    r"\bSix\s+Sigma(?:\s+Green\s+Belt)?\b",
    r"\bAutoCAD\b",
    r"\bMinitab\b",
    r"\bFigma\b",
    r"\bAdobe\s+XD\b",
    r"\bIllustrator\b",
    r"\bCanva\b",
    r"\bScrum\b",
    r"\bKanban\b",
    r"\bJira(?:\s+Service\s+Management)?\b",
    r"\bWindows\s+Server\b",
    r"\bLinux\b",
    r"\bActive\s+Directory\b",
    r"\bTCP/IP\b",
    r"\bIFRS\b",
    r"\bKPIs?(?:\s+logisticos)?\b",
    r"\bSLA\b",
    r"\bSaaS\s+B2B\b",
    r"\bETL\b",
    r"\bHIS/EHR\b",
    r"\bfacturacion\s+electronica\b",
    r"\bgestion\s+documental(?:\s+digital)?\b",
    r"\bcontrol\s+de\s+calidad\b",
    r"\bgestion\s+de\s+inventarios?\b",
    r"\bgestion\s+de\s+cadena\s+de\s+suministro\b",
    r"\bcontabilidad\s+general\b",
    r"\bconciliaciones\s+bancarias\b",
    r"\bforecasting\s+comercial\b",
    r"\bmodelado\s+financiero\b",
    r"\banalisis\s+de\s+datos\b",
    r"\bbases?\s+de\s+datos\b",
    r"\bingles(?:\s+(?:nivel\s+)?(?:B1|B2|C1|C2|avanzado|intermedio-alto|juridico)(?:\s*\(C1\))?)?\b",
    r"\b\d+\s+anos\s+de\s+experiencia\b",
]


def load_model(model_key: str = "random_forest"):
    """Carga el pipeline preentrenado comprimido desde disco (sin entrenar en tiempo de ejecución)."""
    if model_key in ("random_forest", "rf_model") and RF_MODEL_PATH.exists():
        return joblib.load(RF_MODEL_PATH)
    model_path = MODELS_DIR / f"{model_key}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo preentrenado '{RF_MODEL_PATH}'. Ejecuta 'python -m src.train_models' primero."
        )
    return joblib.load(model_path)


def load_best_model_key() -> str:
    """Devuelve la clave del mejor modelo registrado."""
    if BEST_MODEL_META_PATH.exists():
        with open(BEST_MODEL_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
            return meta.get("model_key", "random_forest")
    return "random_forest"


def load_saved_metrics() -> Dict[str, Any]:
    """Carga las métricas guardadas en JSON."""
    if not METRICS_JSON_PATH.exists():
        return {}
    with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_technical_terms(text: str) -> List[str]:
    """Extrae términos y requisitos técnicos deduplicando sub-términos de términos compuestos."""
    folded = fold_accents_1to1(text)
    # key -> (primera_posicion, texto_original)
    unique_matches: Dict[str, Tuple[int, str]] = {}
    for pattern in TECHNICAL_TERM_PATTERNS:
        for m in re.finditer(pattern, folded, flags=re.IGNORECASE):
            orig = text[m.start() : m.end()].strip()
            key = fold_accents_1to1(orig.lower())
            if key not in unique_matches or m.start() < unique_matches[key][0]:
                unique_matches[key] = (m.start(), orig)

    # Ordenar de más largo a más corto para priorizar términos compuestos
    # (ej. 'ERP SAP' elimina 'ERP' y 'SAP'; 'Microsoft Excel' elimina 'Excel')
    sorted_keys = sorted(unique_matches.keys(), key=lambda k: (-len(k), unique_matches[k][0]))
    kept_keys: List[str] = []
    for cand_key in sorted_keys:
        sub_re = re.compile(rf"(?<!\w){re.escape(cand_key)}(?!\w)")
        if any(sub_re.search(existing_key) for existing_key in kept_keys):
            continue
        kept_keys.append(cand_key)

    # Devolver en el orden natural de aparición dentro del texto
    kept_keys.sort(key=lambda k: unique_matches[k][0])
    return [unique_matches[k][1] for k in kept_keys]


def _is_in_inclusive_clause(folded_text: str, start: int, matched_folded: str) -> bool:
    """Detecta si una mención sensible aparece dentro de una cláusula inclusiva/anti-discriminación."""
    if _ALWAYS_BIASED_RE.search(matched_folded):
        return False
    # Buscar inicio de la oración actual (máximo 110 caracteres hacia atrás)
    window_start = max(0, start - 110)
    prefix = folded_text[window_start:start]
    # Recortar al último delimitador de oración para no mezclar oraciones previas
    last_sep = max(prefix.rfind("."), prefix.rfind("!"), prefix.rfind("?"), prefix.rfind("\n"))
    if last_sep != -1:
        prefix = prefix[last_sep + 1 :]
    return bool(_INCLUSIVE_CONTEXT_RE.search(prefix))


def detect_biased_spans(text: str) -> List[DetectedSpan]:
    """Detecta frases sesgadas y resuelve superposiciones priorizando coincidencias más específicas."""
    if not text.strip():
        return []
    folded = fold_accents_1to1(text)
    candidates: List[DetectedSpan] = []

    for pattern, category, explanation, suggestion in BIAS_PATTERNS:
        for match in re.finditer(pattern, folded, flags=re.IGNORECASE):
            start, end = match.start(), match.end()
            if end <= start:
                continue
            matched_folded = folded[start:end]
            if _is_in_inclusive_clause(folded, start, matched_folded):
                continue
            candidates.append(
                DetectedSpan(
                    start=start,
                    end=end,
                    text=text[start:end],
                    category=category,
                    explanation=explanation,
                    suggestion=suggestion,
                )
            )

    # Priorizar spans más largos (y luego prioridad de categoría más grave) para evitar solapamientos
    priority = {"Excluyente": 0, "LGBTQ+": 1, "Demográfico": 2, "Femenino": 3, "Masculino": 4}
    candidates.sort(key=lambda s: (-(s.end - s.start), priority.get(s.category, 5), s.start))

    selected: List[DetectedSpan] = []
    for cand in candidates:
        overlaps = any(not (cand.end <= s.start or cand.start >= s.end) for s in selected)
        if not overlaps:
            selected.append(cand)

    selected.sort(key=lambda s: s.start)
    return selected


def render_highlighted_html(text: str, spans: List[DetectedSpan]) -> str:
    """Genera HTML seguro con frases sesgadas resaltadas por categoría y etiqueta visual."""
    if not text:
        return ""
    parts: List[str] = []
    cursor = 0

    for span in spans:
        if span.start > cursor:
            parts.append(html.escape(text[cursor : span.start]).replace("\n", "<br>"))
        meta = CATEGORY_INFO[span.category]
        escaped_phrase = html.escape(span.text).replace("\n", "<br>")
        escaped_tip = html.escape(f"[{span.category}] {span.explanation} Sugerencia: {span.suggestion}")
        parts.append(
            f'<span title="{escaped_tip}" '
            f'style="background-color:{meta["bg"]}; color:{meta["text"]}; '
            f'border-bottom:2px solid {meta["color"]}; border-radius:4px; '
            f'padding:2px 5px; margin:0 1px; font-weight:600; cursor:help;">'
            f"{escaped_phrase}"
            f'<sup style="background-color:{meta["color"]}; color:#ffffff; '
            f'font-size:0.65em; padding:1px 4px; border-radius:3px; '
            f'margin-left:3px; font-weight:700; vertical-align:super;">'
            f"{html.escape(span.category)}</sup></span>"
        )
        cursor = span.end

    if cursor < len(text):
        parts.append(html.escape(text[cursor:]).replace("\n", "<br>"))

    return (
        '<div style="line-height:1.85; font-size:1.02rem; padding:16px 18px; '
        'background-color:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; '
        'color:#1E293B;">'
        + "".join(parts)
        + "</div>"
    )


# Umbrales calibrados del motor híbrido (incertidumbre ML + cobertura TF-IDF)
_MIN_VOCAB_COVERAGE_OOD = 0.50
_MIN_VOCAB_COVERAGE_HIGH = 0.55
_MIN_VOCAB_COVERAGE_DT = 0.55
_MIN_VOCAB_COVERAGE_DT_SINGLE_SPLIT = 0.60
_MIN_RECOGNIZED_TOKENS = 4
_MIN_CONFIDENCE_LOW = 0.45
_MIN_CONFIDENCE_HIGH = 0.60
_MIN_MARGIN_LOW = 0.15
_MIN_MARGIN_HIGH = 0.25
_MAX_ENTROPY_HIGH = 0.75
_MAX_ENTROPY_LOW = 0.82

_SPAN_SEVERITY_WEIGHTS: Dict[str, float] = {
    "Excluyente": 28.0,
    "LGBTQ+": 22.0,
    "Demográfico": 20.0,
    "Femenino": 15.0,
    "Masculino": 15.0,
}

_CATEGORY_SEVERITY_PRIORITY: Dict[str, int] = {
    "Excluyente": 0,
    "LGBTQ+": 1,
    "Demográfico": 2,
    "Femenino": 3,
    "Masculino": 4,
}

_TOKEN_FALLBACK_RE = re.compile(r"(?u)\b\w\w+\b")


def _extract_pipeline_components(model: Any) -> Tuple[Any | None, Any | None]:
    """Extrae de forma segura el vectorizador TF-IDF y el clasificador final de un Pipeline."""
    vectorizer = None
    classifier = None
    named_steps = getattr(model, "named_steps", None)
    if isinstance(named_steps, dict):
        vectorizer = named_steps.get("tfidf")
        classifier = named_steps.get("clf")
        steps = list(named_steps.values())
        if vectorizer is None:
            for step in steps:
                if hasattr(step, "vocabulary_") and hasattr(step, "transform"):
                    vectorizer = step
                    break
        if classifier is None and steps:
            classifier = steps[-1]
    elif hasattr(model, "steps") and isinstance(model.steps, list) and model.steps:
        for _, step in model.steps:
            if vectorizer is None and hasattr(step, "vocabulary_") and hasattr(step, "transform"):
                vectorizer = step
        classifier = model.steps[-1][1]
    else:
        classifier = model
    return vectorizer, classifier


def _extract_sparse_row(X: Any) -> Tuple[List[int], List[float]]:
    """Extrae índices y valores no nulos de la primera fila de una matriz dispersa o densa."""
    if hasattr(X, "tocsr"):
        X_csr = X.tocsr()
        start_ptr = int(X_csr.indptr[0])
        end_ptr = int(X_csr.indptr[1])
        nz_indices = [int(i) for i in X_csr.indices[start_ptr:end_ptr]]
        nz_values = [float(v) for v in X_csr.data[start_ptr:end_ptr]]
        return nz_indices, nz_values
    row0 = X[0]
    nz_indices = [idx for idx, val in enumerate(row0) if float(val) != 0.0]
    nz_values = [float(row0[idx]) for idx in nz_indices]
    return nz_indices, nz_values


def compute_prediction_uncertainty(probabilities: Dict[str, float]) -> Dict[str, float]:
    """Calcula confianza máxima, margen top1-top2 y entropía normalizada [0, 1]."""
    probs = [max(0.0, float(p)) for p in probabilities.values()]
    if not probs:
        return {
            "top1_prob": 0.0,
            "top2_prob": 0.0,
            "margin": 0.0,
            "normalized_entropy": 1.0,
        }
    total = sum(probs)
    if total > 0.0:
        probs = [p / total for p in probs]
    sorted_probs = sorted(probs, reverse=True)
    top1_prob = sorted_probs[0]
    top2_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    margin = max(0.0, top1_prob - top2_prob)

    n_classes = max(len(probs), 2)
    entropy = -sum(p * math.log(p) for p in probs if p > 0.0)
    normalized_entropy = max(0.0, min(1.0, entropy / math.log(n_classes)))

    return {
        "top1_prob": round(top1_prob, 4),
        "top2_prob": round(top2_prob, 4),
        "margin": round(margin, 4),
        "normalized_entropy": round(normalized_entropy, 4),
    }


def compute_vocabulary_coverage(text: str, vectorizer: Any) -> Dict[str, Any]:
    """Calcula la cobertura del vocabulario TF-IDF del modelo sobre los tokens del aviso."""
    if vectorizer is None or not hasattr(vectorizer, "vocabulary_"):
        return {
            "has_vectorizer": False,
            "vocabulary_coverage": 0.0,
            "recognized_tokens": 0,
            "total_tokens": 0,
            "matched_features": 0,
            "tfidf_vector": None,
        }

    vocab = vectorizer.vocabulary_ or {}
    try:
        preprocessor = vectorizer.build_preprocessor()
        preprocessed = preprocessor(text)
    except Exception:
        preprocessed = normalize_for_ml(text)

    try:
        tokenizer = vectorizer.build_tokenizer()
        raw_tokens = list(tokenizer(preprocessed))
    except Exception:
        raw_tokens = _TOKEN_FALLBACK_RE.findall(preprocessed)

    try:
        stop_words = set(vectorizer.get_stop_words() or ())
    except Exception:
        stop_words = set(SPANISH_MINIMAL_STOPWORDS)

    content_tokens = [tok for tok in raw_tokens if tok not in stop_words]
    total_tokens = len(content_tokens)
    recognized_tokens = sum(1 for tok in content_tokens if tok in vocab)
    vocabulary_coverage = (recognized_tokens / total_tokens) if total_tokens > 0 else 0.0

    tfidf_vector = None
    matched_features = 0
    if hasattr(vectorizer, "transform"):
        try:
            tfidf_vector = vectorizer.transform([text])
            nz_indices, _ = _extract_sparse_row(tfidf_vector)
            matched_features = len(nz_indices)
        except Exception:
            tfidf_vector = None
            matched_features = 0

    return {
        "has_vectorizer": True,
        "vocabulary_coverage": round(vocabulary_coverage, 4),
        "recognized_tokens": recognized_tokens,
        "total_tokens": total_tokens,
        "matched_features": matched_features,
        "tfidf_vector": tfidf_vector,
    }


def _extract_linear_class_coef(
    classifier: Any,
    classes: List[str],
    target_class: str,
    nz_indices: List[int],
) -> Dict[int, float] | None:
    """Extrae coeficientes lineales para la clase objetivo si el estimador los expone."""
    if target_class not in classes:
        return None
    target_idx = classes.index(target_class)

    estimators_to_check: List[Any] = [classifier]
    calibrated = getattr(classifier, "calibrated_classifiers_", None)
    if calibrated:
        for cc in calibrated:
            inner = getattr(cc, "estimator", getattr(cc, "base_estimator", None))
            if inner is not None:
                estimators_to_check.append(inner)

    collected_maps: List[Dict[int, float]] = []
    for est in estimators_to_check:
        try:
            coef = getattr(est, "coef_", None)
        except Exception:
            coef = None
        if coef is None:
            continue
        shape = getattr(coef, "shape", None)
        if shape is None or len(shape) != 2 or shape[0] != len(classes):
            continue
        try:
            if hasattr(coef, "tocsr"):
                row_csr = coef.tocsr()[target_idx]
                row_lookup = {int(i): float(v) for i, v in zip(row_csr.indices, row_csr.data)}
                collected_maps.append({j: row_lookup.get(j, 0.0) for j in nz_indices})
            else:
                row = coef[target_idx]
                collected_maps.append({j: float(row[j]) for j in nz_indices})
        except Exception:
            continue

    if not collected_maps:
        return None

    return {
        j: sum(m[j] for m in collected_maps) / len(collected_maps)
        for j in nz_indices
    }


def _inspect_decision_tree_path(
    classifier: Any,
    X: Any,
    nz_map: Dict[int, float],
) -> Tuple[set[int], int | None]:
    """Inspecciona si un DecisionTree alcanzó su hoja mediante rasgos presentes (> umbral)."""
    if not (hasattr(classifier, "tree_") and hasattr(classifier, "decision_path")):
        return set(), None
    try:
        node_ind = classifier.decision_path(X)
        node_ind_csr = node_ind.tocsr() if hasattr(node_ind, "tocsr") else node_ind
        start_ptr = int(node_ind_csr.indptr[0])
        end_ptr = int(node_ind_csr.indptr[1])
        node_ids = [int(nid) for nid in node_ind_csr.indices[start_ptr:end_ptr]]
        tree = classifier.tree_
        active_features: set[int] = set()
        # Excluir la hoja final (node_ids[-1]), donde tree.feature == -2
        for nid in node_ids[:-1]:
            feat_idx = int(tree.feature[nid])
            if feat_idx >= 0:
                threshold = float(tree.threshold[nid])
                if nz_map.get(feat_idx, 0.0) > threshold:
                    active_features.add(feat_idx)
        return active_features, len(active_features)
    except Exception:
        return set(), None


def _compute_svm_local_sensitivity(
    classifier: Any,
    X: Any,
    nz_indices: List[int],
    nz_values: List[float],
    target_class: str,
    base_prob: float,
    max_candidates: int = 25,
) -> Dict[int, float]:
    """Estima la contribución local de cada término en clasificadores sin coef_ (p. ej. SVM RBF)."""
    if not hasattr(classifier, "predict_proba") or not hasattr(X, "tocsr") or not nz_indices:
        return {}
    classes = [str(c) for c in getattr(classifier, "classes_", [])]
    if target_class not in classes:
        return {}
    pred_idx = classes.index(target_class)

    X_csr = X.tocsr()
    ranked_positions = sorted(
        range(len(nz_indices)),
        key=lambda p: nz_values[p],
        reverse=True,
    )[:max_candidates]

    row_data: List[float] = []
    row_indices: List[int] = []
    indptr: List[int] = [0]
    for drop_pos in ranked_positions:
        sq_norm = sum(nz_values[p] ** 2 for p in range(len(nz_indices)) if p != drop_pos)
        inv_norm = (1.0 / math.sqrt(sq_norm)) if sq_norm > 1e-12 else 1.0
        for p in range(len(nz_indices)):
            if p != drop_pos:
                row_indices.append(nz_indices[p])
                row_data.append(nz_values[p] * inv_norm)
        indptr.append(len(row_indices))

    try:
        csr_cls = type(X_csr)
        X_batch = csr_cls(
            (row_data, row_indices, indptr),
            shape=(len(ranked_positions), X_csr.shape[1]),
            dtype=getattr(X_csr, "dtype", float),
        )
        pert_probs = classifier.predict_proba(X_batch)
    except Exception:
        return {}

    contributions: Dict[int, float] = {}
    for row_i, drop_pos in enumerate(ranked_positions):
        delta = base_prob - float(pert_probs[row_i, pred_idx])
        if delta > 1e-4:
            contributions[nz_indices[drop_pos]] = delta
    return contributions


def extract_ml_influential_terms(
    text: str,
    model: Any,
    raw_ml_category: str,
    probabilities: Dict[str, float] | None = None,
    tfidf_vector: Any = None,
    top_k: int = 8,
) -> Tuple[List[MLInfluentialTerm], str, int | None]:
    """Identifica palabras o n-gramas presentes en el texto que más influyeron en la predicción ML."""
    vectorizer, classifier = _extract_pipeline_components(model)
    is_decision_tree = hasattr(classifier, "tree_") and hasattr(classifier, "decision_path")

    if vectorizer is None or not hasattr(vectorizer, "transform"):
        return [], "Sin vectorizador TF-IDF disponible", None

    try:
        X = tfidf_vector if tfidf_vector is not None else vectorizer.transform([text])
    except Exception:
        return [], "No fue posible transformar el texto con TF-IDF", None

    nz_indices, nz_values = _extract_sparse_row(X)
    if not nz_indices:
        return [], "Sin términos reconocidos en el vocabulario TF-IDF", (0 if is_decision_tree else None)

    if hasattr(vectorizer, "get_feature_names_out"):
        feature_names = vectorizer.get_feature_names_out()

        def get_feature_name(idx: int) -> str:
            return str(feature_names[idx])
    elif hasattr(vectorizer, "vocabulary_"):
        inv_vocab = {int(v): str(k) for k, v in vectorizer.vocabulary_.items()}

        def get_feature_name(idx: int) -> str:
            return inv_vocab.get(idx, f"feat_{idx}")
    else:
        return [], "Vocabulario TF-IDF no disponible", (0 if is_decision_tree else None)

    nz_map = dict(zip(nz_indices, nz_values))
    classes = [str(c) for c in getattr(classifier, "classes_", getattr(model, "classes_", []))]

    dt_active_features, dt_positive_splits = _inspect_decision_tree_path(classifier, X, nz_map)

    scored_candidates: List[Tuple[int, float, float, str]] = []
    method_label = ""

    # 1. Coeficientes lineales (p. ej. LinearSVC / SVM lineal / modelos lineales calibrados)
    linear_coefs = _extract_linear_class_coef(classifier, classes, raw_ml_category, nz_indices)
    if linear_coefs is not None:
        positive_linear = [
            (j, nz_map[j], nz_map[j] * linear_coefs[j], "coeficiente_lineal")
            for j in nz_indices
            if linear_coefs[j] > 0.0
        ]
        if positive_linear:
            positive_linear.sort(key=lambda item: item[2], reverse=True)
            scored_candidates = positive_linear[:top_k]
            method_label = "Coeficientes lineales × peso TF-IDF"

    # 2. Importancias basadas en árboles (Decision Tree / Random Forest)
    if not scored_candidates and hasattr(classifier, "feature_importances_"):
        try:
            importances = classifier.feature_importances_
            max_idx = max(nz_indices)
            if len(importances) > max_idx:
                if is_decision_tree:
                    path_items: List[Tuple[int, float, float, str]] = []
                    other_items: List[Tuple[int, float, float, str]] = []
                    for j in nz_indices:
                        w_j = nz_map[j]
                        imp_j = float(importances[j])
                        if j in dt_active_features:
                            path_items.append((j, w_j, w_j * max(imp_j, 1e-4), "ruta_arbol"))
                        elif imp_j > 0.0:
                            other_items.append((j, w_j, w_j * imp_j, "importancia_arbol"))
                    path_items.sort(key=lambda item: item[2], reverse=True)
                    other_items.sort(key=lambda item: item[2], reverse=True)
                    if path_items and other_items:
                        max_other = other_items[0][2]
                        path_items = [
                            (j, w_j, raw_s + max_other + 1e-6, src)
                            for j, w_j, raw_s, src in path_items
                        ]
                    combined = path_items + other_items
                    if combined:
                        scored_candidates = combined[:top_k]
                        method_label = (
                            "Ruta de decisión e importancia del Árbol × TF-IDF"
                            if path_items
                            else "Importancia del Árbol × peso TF-IDF"
                        )
                else:
                    rf_items = [
                        (j, nz_map[j], nz_map[j] * float(importances[j]), "importancia_bosque")
                        for j in nz_indices
                        if float(importances[j]) > 0.0
                    ]
                    if rf_items:
                        rf_items.sort(key=lambda item: item[2], reverse=True)
                        scored_candidates = rf_items[:top_k]
                        method_label = "Importancia de variables (Random Forest) × peso TF-IDF"
        except Exception:
            scored_candidates = []

    # 3. Sensibilidad local para clasificadores no lineales sin coef_ (SVM RBF calibrado)
    if not scored_candidates and probabilities is not None:
        base_prob = float(probabilities.get(raw_ml_category, 0.0))
        svm_contribs = _compute_svm_local_sensitivity(
            classifier=classifier,
            X=X,
            nz_indices=nz_indices,
            nz_values=nz_values,
            target_class=raw_ml_category,
            base_prob=base_prob,
        )
        if svm_contribs:
            svm_items = [
                (j, nz_map[j], delta, "sensibilidad_svm")
                for j, delta in svm_contribs.items()
            ]
            svm_items.sort(key=lambda item: item[2], reverse=True)
            scored_candidates = svm_items[:top_k]
            method_label = "Sensibilidad local del clasificador SVM (leave-one-out)"

    # 4. Respaldo seguro: pesos TF-IDF no nulos más altos presentes en el documento
    if not scored_candidates:
        tfidf_items = [
            (j, nz_map[j], nz_map[j], "peso_tfidf")
            for j in nz_indices
        ]
        tfidf_items.sort(key=lambda item: item[2], reverse=True)
        scored_candidates = tfidf_items[:top_k]
        method_label = "Pesos TF-IDF más altos reconocidos en el texto"

    max_raw_score = max((item[2] for item in scored_candidates), default=1.0)
    if max_raw_score <= 0.0:
        max_raw_score = 1.0

    terms = [
        MLInfluentialTerm(
            term=get_feature_name(j),
            tfidf_weight=round(w_j, 4),
            influence_score=round(raw_score / max_raw_score, 4),
            source=src,
        )
        for j, w_j, raw_score, src in scored_candidates
    ]
    return terms, method_label, dt_positive_splits


def _assess_ml_reliability(
    top1_prob: float,
    margin: float,
    normalized_entropy: float,
    vocabulary_coverage: float,
    recognized_tokens: int,
    has_vectorizer: bool,
    is_decision_tree: bool,
    dt_positive_splits: int | None,
) -> Tuple[str, str, List[str]]:
    """Evalúa la confiabilidad estadística del ML ('alta', 'media' o 'baja') y sus motivos."""
    low_reasons: List[str] = []
    is_default_leaf_collapse = is_decision_tree and (dt_positive_splits is not None and dt_positive_splits == 0)

    if has_vectorizer:
        if vocabulary_coverage < _MIN_VOCAB_COVERAGE_OOD:
            low_reasons.append(f"cobertura de vocabulario TF-IDF baja ({vocabulary_coverage * 100:.1f}%)")
        elif is_decision_tree and vocabulary_coverage < _MIN_VOCAB_COVERAGE_DT:
            low_reasons.append(
                f"cobertura TF-IDF insuficiente para Decision Tree ({vocabulary_coverage * 100:.1f}%)"
            )
        if recognized_tokens < _MIN_RECOGNIZED_TOKENS:
            low_reasons.append(f"muy pocos tokens reconocidos ({recognized_tokens})")

    if is_default_leaf_collapse:
        low_reasons.append("Decision Tree cayó en hoja por defecto sin rasgos presentes activos")
    if top1_prob < _MIN_CONFIDENCE_LOW:
        low_reasons.append(f"confianza máxima baja ({top1_prob * 100:.1f}%)")
    if margin < _MIN_MARGIN_LOW:
        low_reasons.append(f"margen estrecho entre clases ({margin * 100:.1f}%)")
    if normalized_entropy > _MAX_ENTROPY_LOW:
        low_reasons.append(f"distribución probabilística dispersa (entropía {normalized_entropy:.2f})")

    if low_reasons:
        return "baja", "Baja (Fuera de dominio / Incierto)", low_reasons

    dt_split_ok = (
        not is_decision_tree
        or dt_positive_splits is None
        or dt_positive_splits >= 2
        or (dt_positive_splits >= 1 and (not has_vectorizer or vocabulary_coverage >= _MIN_VOCAB_COVERAGE_DT_SINGLE_SPLIT))
    )
    cov_high_ok = (not has_vectorizer) or (vocabulary_coverage >= _MIN_VOCAB_COVERAGE_HIGH)

    if (
        cov_high_ok
        and top1_prob >= _MIN_CONFIDENCE_HIGH
        and margin >= _MIN_MARGIN_HIGH
        and normalized_entropy <= _MAX_ENTROPY_HIGH
        and dt_split_ok
    ):
        return "alta", "Alta", []

    media_reasons: List[str] = []
    if has_vectorizer and vocabulary_coverage < _MIN_VOCAB_COVERAGE_HIGH:
        media_reasons.append(f"cobertura de vocabulario moderada ({vocabulary_coverage * 100:.1f}%)")
    if top1_prob < _MIN_CONFIDENCE_HIGH:
        media_reasons.append(f"confianza moderada ({top1_prob * 100:.1f}%)")
    if margin < _MIN_MARGIN_HIGH:
        media_reasons.append(f"margen moderado ({margin * 100:.1f}%)")
    if normalized_entropy > _MAX_ENTROPY_HIGH:
        media_reasons.append(f"entropía moderada ({normalized_entropy:.2f})")
    if not dt_split_ok:
        media_reasons.append("respaldo limitado de nodos activos en Decision Tree")

    return "media", "Media", media_reasons


_EXPLICIT_MALE_RESTRICTION_RE = re.compile(
    r"\b(?:varon(?:es)?|personal\s+masculino|sexo\s+masculino|solo\s+hombres|solamente\s+hombres|"
    r"unicamente\s+hombres|exclusivamente\s+hombres|caballeros?|\(\s*hombres?\s*\)|"
    r"personal\s+de\s+\w+\s+masculino|masculino\s+excluyente)\b",
    flags=re.IGNORECASE,
)


def _span_weight(span: DetectedSpan) -> float:
    """Peso de severidad de un span; da prioridad alta a restricciones explícitas de varones."""
    if span.category == "Masculino":
        folded = fold_accents_1to1(span.text).lower()
        if _EXPLICIT_MALE_RESTRICTION_RE.search(folded):
            return 26.0
    return _SPAN_SEVERITY_WEIGHTS.get(span.category, 15.0)


def _select_dominant_lexical_category(spans: List[DetectedSpan]) -> str:
    """Selecciona la categoría léxica dominante ponderando severidad y frecuencia de frases."""
    weighted_totals: Counter[str] = Counter()
    for span in spans:
        weighted_totals[span.category] += _span_weight(span)
    return max(
        weighted_totals.keys(),
        key=lambda cat: (weighted_totals[cat], -_CATEGORY_SEVERITY_PRIORITY.get(cat, 5)),
    )


def analyze_job_posting(text: str, model_key: str | None = None, pipeline=None) -> AnalysisResult:
    """Ejecuta inferencia híbrida calibrada (ML + incertidumbre/OOV + motor léxico + explicabilidad)."""
    chosen_key = model_key or load_best_model_key()
    model = pipeline if pipeline is not None else load_model(chosen_key)
    model_name = MODEL_DISPLAY_NAMES.get(chosen_key, chosen_key)

    proba_arr = model.predict_proba([text])[0]
    classes = list(model.classes_)
    raw_probs = {cls: float(prob) for cls, prob in zip(classes, proba_arr)}
    probabilities = {cat: raw_probs.get(cat, 0.0) for cat in ALL_CATEGORIES}
    bias_probabilities = {cat: probabilities[cat] for cat in BIAS_CATEGORIES}
    raw_ml_category = max(probabilities, key=probabilities.get)

    vectorizer, classifier = _extract_pipeline_components(model)
    is_decision_tree = hasattr(classifier, "tree_") and hasattr(classifier, "decision_path")

    uncertainty = compute_prediction_uncertainty(probabilities)
    top1_prob = uncertainty["top1_prob"]
    margin = uncertainty["margin"]
    normalized_entropy = uncertainty["normalized_entropy"]

    coverage_info = compute_vocabulary_coverage(text, vectorizer)
    has_vectorizer = bool(coverage_info["has_vectorizer"])
    vocabulary_coverage = float(coverage_info["vocabulary_coverage"])
    recognized_tokens = int(coverage_info["recognized_tokens"])
    total_tokens = int(coverage_info["total_tokens"])
    matched_features = int(coverage_info["matched_features"])

    influential_terms, explainability_method, dt_positive_splits = extract_ml_influential_terms(
        text=text,
        model=model,
        raw_ml_category=raw_ml_category,
        probabilities=probabilities,
        tfidf_vector=coverage_info["tfidf_vector"],
    )

    ml_reliability, ml_reliability_label, reliability_reasons = _assess_ml_reliability(
        top1_prob=top1_prob,
        margin=margin,
        normalized_entropy=normalized_entropy,
        vocabulary_coverage=vocabulary_coverage,
        recognized_tokens=recognized_tokens,
        has_vectorizer=has_vectorizer,
        is_decision_tree=is_decision_tree,
        dt_positive_splits=dt_positive_splits,
    )

    spans = detect_biased_spans(text)
    detected_categories = sorted(
        {s.category for s in spans},
        key=lambda c: BIAS_CATEGORIES.index(c),
    )
    highlighted_html = render_highlighted_html(text, spans)
    technical_terms = extract_technical_terms(text)

    ml_neutro = probabilities["Neutro"]
    ml_neutro_pct = ml_neutro * 100.0
    is_default_leaf_collapse = is_decision_tree and (dt_positive_splits is not None and dt_positive_splits == 0)

    # ==================== FUSIÓN HÍBRIDA EN 3 ESCENARIOS (+ NEUTRO CONFIRMADO) ====================
    if spans:
        # Escenario 1: Evidencia léxica explícita detectada
        decision_mode = "lexical_explicit"
        dominant_lex_cat = _select_dominant_lexical_category(spans)
        if raw_ml_category in detected_categories and ml_reliability != "baja":
            predicted_class = raw_ml_category
        elif raw_ml_category != "Neutro" and ml_reliability == "alta":
            predicted_class = raw_ml_category
        else:
            predicted_class = dominant_lex_cat

        span_penalty = min(
            15.0 + sum(_span_weight(s) for s in spans),
            90.0,
        )
        lexical_score = max(0.0, 100.0 - span_penalty)
        ml_weight = 0.45 if ml_reliability == "alta" else (0.30 if ml_reliability == "media" else 0.10)
        inclusivity_score = round(
            max(0.0, min(lexical_score, (1.0 - ml_weight) * lexical_score + ml_weight * ml_neutro_pct)),
            1,
        )

        has_explicit_sex_restriction = ("Excluyente" in detected_categories) or any(
            _span_weight(s) >= 26.0 for s in spans
        )
        if (
            inclusivity_score < 45.0
            or len(spans) >= 3
            or (has_explicit_sex_restriction and (len(spans) >= 2 or inclusivity_score < 55.0))
        ):
            risk_level = "Alto"
        else:
            risk_level = "Moderado"

        cats_str = ", ".join(detected_categories)
        if predicted_class == raw_ml_category:
            decision_explanation = (
                f"Se detectaron {len(spans)} frase(s) sesgada(s) explícita(s) ({cats_str}), "
                f"corroboradas por el clasificador {model_name} "
                f"(confianza ML: {top1_prob * 100:.1f}%, cobertura TF-IDF: {vocabulary_coverage * 100:.1f}%)."
            )
        else:
            decision_explanation = (
                f"Se detectaron {len(spans)} frase(s) sesgada(s) explícita(s) ({cats_str}). "
                f"El motor híbrido priorizó la evidencia léxica sobre la predicción cruda de {model_name} "
                f"('{raw_ml_category}', {top1_prob * 100:.1f}%, confiabilidad {ml_reliability_label.lower()})."
            )

    elif raw_ml_category != "Neutro" and ml_reliability == "alta":
        # Escenario 2: 0 frases léxicas explícitas, pero ML con confianza real alta y buena cobertura
        decision_mode = "ml_implicit_bias"
        predicted_class = raw_ml_category
        strength = min(1.0, max(0.0, (top1_prob - _MIN_CONFIDENCE_HIGH) / (1.0 - _MIN_CONFIDENCE_HIGH)))
        inclusivity_score = round(
            max(50.0, min(70.0, 70.0 - 20.0 * strength + 5.0 * ml_neutro)),
            1,
        )
        risk_level = "Medio"
        decision_explanation = (
            f"No se detectaron frases sesgadas explícitas en el texto, pero {model_name} identificó un "
            f"patrón estadístico consistente con posible sesgo implícito '{raw_ml_category}' "
            f"(confianza: {top1_prob * 100:.1f}%, margen: {margin * 100:.1f}%, entropía: {normalized_entropy:.2f}, "
            f"cobertura TF-IDF: {vocabulary_coverage * 100:.1f}%). Se mantiene alerta preventiva de nivel Medio; "
            f"revisa los términos influyentes del modelo."
        )

    elif raw_ml_category == "Neutro" and ml_reliability in ("alta", "media"):
        # Caso Neutro confirmado en dominio: 0 frases léxicas y ML consistente con Neutro
        decision_mode = "neutral_confirmed"
        predicted_class = "Neutro"
        base_floor = 86.0 if ml_reliability == "alta" else 83.0
        inclusivity_score = round(
            min(100.0, max(ml_neutro_pct, base_floor + (100.0 - base_floor) * ml_neutro)),
            1,
        )
        risk_level = "Bajo (Inclusivo)"
        decision_explanation = (
            f"No se detectaron frases sesgadas explícitas y {model_name} confirmó redacción neutra e inclusiva "
            f"(probabilidad Neutro: {ml_neutro_pct:.1f}%, cobertura TF-IDF: {vocabulary_coverage * 100:.1f}%)."
        )

    else:
        # Escenario 3: 0 frases léxicas explícitas y ML incierto / disperso / fuera de dominio
        decision_mode = "calibrated_neutral"
        predicted_class = "Neutro"
        risk_level = "Bajo (Inclusivo)"

        cov_factor = min(1.0, (vocabulary_coverage / 0.75)) if has_vectorizer else 0.65
        dt_ood_collapse = is_default_leaf_collapse or (
            is_decision_tree and has_vectorizer and vocabulary_coverage < _MIN_VOCAB_COVERAGE_DT
        )
        effective_entropy = 0.85 if dt_ood_collapse else normalized_entropy

        if is_default_leaf_collapse:
            certainty_factor = 0.05
        else:
            certainty_factor = max(0.0, 1.0 - normalized_entropy) * min(1.0, margin / 0.40)
            if dt_ood_collapse:
                certainty_factor *= 0.20 * cov_factor

        effective_ml_trust = max(0.0, min(1.0, cov_factor * certainty_factor))

        if raw_ml_category == "Neutro":
            inclusivity_score = round(
                max(82.0, min(94.0, 88.0 + 6.0 * ml_neutro - 4.0 * (1.0 - cov_factor))),
                1,
            )
            reasons_txt = "; ".join(reliability_reasons) if reliability_reasons else "baja certeza estadística"
            decision_explanation = (
                f"No se detectaron frases sesgadas explícitas y la predicción cruda de {model_name} "
                f"también se inclinó hacia Neutro ({top1_prob * 100:.1f}%), pero con baja confiabilidad "
                f"estadística ({reasons_txt}). El motor híbrido calibró el Índice de Inclusividad "
                f"combinando la ausencia de sesgos léxicos con la confianza efectiva del modelo."
            )
        else:
            caution_deduction = 5.0 * (1.0 - cov_factor) + 3.5 * effective_entropy
            residual_bias_penalty = 18.0 * (1.0 - ml_neutro) * effective_ml_trust
            media_adj = (
                4.0 * max(0.0, (top1_prob - _MIN_CONFIDENCE_LOW) / (_MIN_CONFIDENCE_HIGH - _MIN_CONFIDENCE_LOW))
                if ml_reliability == "media"
                else 0.0
            )
            inclusivity_score = round(
                max(80.0, min(92.0, 94.0 - caution_deduction - residual_bias_penalty - media_adj)),
                1,
            )
            reasons_txt = "; ".join(reliability_reasons) if reliability_reasons else "baja certeza estadística"
            decision_explanation = (
                f"No se detectaron frases sesgadas explícitas en el texto. Aunque la señal cruda de "
                f"{model_name} se inclinó hacia '{raw_ml_category}' ({top1_prob * 100:.1f}%), el clasificador "
                f"presentó confiabilidad {ml_reliability_label.lower()} ({reasons_txt}). "
                f"Por ello, el motor híbrido calibró el diagnóstico final a Neutro (riesgo Bajo) "
                f"manteniendo visibles las probabilidades crudas del ML por transparencia."
            )

    return AnalysisResult(
        model_key=chosen_key,
        model_name=model_name,
        predicted_class=predicted_class,
        probabilities=probabilities,
        bias_probabilities=bias_probabilities,
        inclusivity_score=inclusivity_score,
        risk_level=risk_level,
        spans=spans,
        highlighted_html=highlighted_html,
        detected_categories=detected_categories,
        technical_terms=technical_terms,
        raw_ml_category=raw_ml_category,
        ml_confidence=top1_prob,
        ml_margin=margin,
        ml_entropy=normalized_entropy,
        vocabulary_coverage=vocabulary_coverage,
        recognized_tokens=recognized_tokens,
        total_tokens=total_tokens,
        matched_features=matched_features,
        ml_reliability=ml_reliability,
        ml_reliability_label=ml_reliability_label,
        decision_mode=decision_mode,
        decision_explanation=decision_explanation,
        influential_terms=influential_terms,
        explainability_method=explainability_method,
    )
