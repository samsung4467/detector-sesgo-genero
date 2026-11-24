"""Motor híbrido de análisis de sesgos: inferencia ML + detección y resaltado de frases."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
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
)
from src.text_utils import fold_accents_1to1


@dataclass
class DetectedSpan:
    start: int
    end: int
    text: str
    category: str
    explanation: str
    suggestion: str


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
        r"\brequisito excluyente\s*:\s*sexo\s+(?:masculino|femenino)\b",
        "Excluyente",
        "Establece el sexo como filtro excluyente para acceder al empleo.",
        "Eliminar el filtro de sexo y especificar competencias objetivas del puesto.",
    ),
    (
        r"\b(?:solo|solamente|unicamente|exclusivamente)\s+(?:hombres|varones|caballeros|mujeres|damas|personal masculino|personal femenino|candidatos varones|candidatos masculinos|candidatas mujeres|candidatas femeninas|postulantes de sexo (?:masculino|femenino)|personal del sexo (?:masculino|femenino))\b",
        "Excluyente",
        "Restringe la postulación exclusivamente a un género específico.",
        "Convocatoria abierta a todas las personas que cumplan con los requisitos técnicos.",
    ),
    (
        r"\babstenerse\s+(?:mujeres|hombres|varones|madres|candidatas|postulantes del sexo opuesto)(?:\s+de\s+postularse|\s+de\s+enviar\s+su\s+hoja\s+de\s+vida)?\b",
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
    # Títulos con género exclusivo ("recepcionista mujer", "ingeniero hombre", etc.)
    (
        r"\b(?:recepcionista|ingenier[oa]|abogad[oa]|asistente|secretari[oa]|vendedor[a]?|ejecutiv[oa]|gerent[ea]|supervisor[a]?|coordinador[a]?|analista|desarrollador[a]?|programador[a]?|tecnic[oa]|operari[oa]|auxiliar|contador[a]?|arquitect[oa]|auditor[a]?|chofer|conductor[a]?|jefe|jefa|encargad[oa]|especialista|profesional|administrativ[oa]|cajer[oa]|meser[oa]|camarer[oa]|personal|candidat[oa]|postulante|disenador[a]?|medico|medica|enfermer[oa]|docente|profesor[a]?)(?:\s+de\s+\w+)?\s+(?:mujer|mujeres|hombre|hombres|varon|varones|femenin[oa]|masculin[oa]|dama|caballero)\b",
        "Excluyente",
        "Asocia explícitamente el título del puesto a un género exclusivo (p. ej., 'recepcionista mujer' o 'ingeniero hombre').",
        "Utilizar denominación neutra del cargo sin restringir por sexo (p. ej., 'persona especialista en recepción' o 'profesional en ingeniería').",
    ),
    (
        r"\b(?:mujer|hombre|varon|dama|caballero)\s+(?:recepcionista|ingenier[oa]|abogad[oa]|asistente|secretari[oa]|vendedor[a]?|ejecutiv[oa]|gerent[ea]|supervisor[a]?|coordinador[a]?|analista|desarrollador[a]?|programador[a]?|tecnic[oa]|operari[oa]|auxiliar|contador[a]?|arquitect[oa]|auditor[a]?|chofer|conductor[a]?|jefe|jefa|encargad[oa]|administrativ[oa]|cajer[oa]|disenador[a]?)\b",
        "Excluyente",
        "Vincula el cargo a un sexo exclusivo en la denominación del puesto.",
        "Redactar el título del cargo en formato neutro e inclusivo.",
    ),
    (
        r"\b(?:reservada|destinada)\s+(?:unicamente|exclusivamente)\s+para\s+(?:varones|hombres|mujeres|personal femenino|personal masculino)\b",
        "Excluyente",
        "Reserva la vacante exclusivamente a un solo género.",
        "Indicar que la vacante está abierta a todas las personas calificadas.",
    ),
    (
        r"\bsexo\s+(?:masculino|femenino)\s+excluyente\b|\bvaron\s+excluyente\b|\bpertenecer\s+al\s+sexo\s+(?:masculino|femenino)\b",
        "Excluyente",
        "Impone el sexo como criterio obligatorio de selección.",
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
        r"\bedad(?:\s+excluyente|\s+comprendida)?\s+(?:entre|de)\s+\d{2}\s+y\s+\d{2}\s+anos\b|\bedad\s+maxima\s+(?:de\s+)?\d{2}\s+anos(?:\s+excluyente)?\b",
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
        r"\b(?:joven(?:es)?\s+(?:recien\s+egresad[oa]s?\s+)?|profesional\s+|talento\s+)?menor(?:es)?\s+de\s+\d{2}\s+anos\b|\b(?:de\s+)?entre\s+\d{2}\s+y\s+\d{2}\s+anos\b(?!\s+de\s+experiencia)|\bde\s+\d{2}\s+a\s+\d{2}\s+anos\b(?!\s+de\s+experiencia)",
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
        r"\b(?:caracter\s+fuerte|duro\s+e\s+imponente|recio\s+y\s+dominante|competitivo\s+y\s+ambicioso|voraz\s+y\s+dispuesto\s+a\s+dominar|implacable\s+y\s+competitivo|ambicioso\s+con\s+mentalidad\s+ganadora|audaz\s+y\s+contundente|combativo)\b|(?<!salario\s)(?<!salario\sbase\s)(?<!sueldo\s)(?<!remuneracion\s)(?<!compensacion\s)(?<!paquete\s)(?<!mercado\s)(?<!entorno\s)(?<!ritmo\s)\bcompetitiv[oa]s?\b",
        "Masculino",
        "Adjetivos agentivos o hipercompetitivos asociados a codificación de género masculina.",
        "Sustituir por competencias observables como 'orientado al logro' o 'toma de decisiones fundamentada'.",
    ),
    (
        r"\bun\s+(?:arquitecto|analista|desarrollador|ingeniero|programador|jefe|supervisor|vendedor|ejecutivo|gerente|coordinador|auditor|abogado|disenador|cientifico|tecnico|comprador|consultor|administrador|contador|director|lider)(?:\s+(?:de\s+(?:software|datos|sistemas|proyectos|marketing|operaciones|ventas|finanzas|producto|negocio|calidad|infraestructura|nube|cloud|soluciones|seguridad|\w+)|backend|frontend|fullstack|full\s+stack|cloud|comercial|financiero|funcional|senior|junior|semi-senior|b2b))*\b",
        "Masculino",
        "Uso de cargo en masculino genérico singular ('un ...') que puede sustituirse por redacción neutra.",
        "Usar fórmulas neutras como 'profesional de arquitectura de software' o 'una persona especialista en...'.",
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
        r"\b(?:una\s+)?(?:asistente\s+administrativa|asistente\s+femenina|secretaria\s+ejecutiva|chica\s+ordenada|chica\s+simpatica(?:\s+y\s+atenta)?|chica\s+encantadora|dama\s+(?:elegante|de\s+trato\s+calido)|mujer\s+(?:armoniosa|simpatica|calida)|perfil\s+femenino\s+colaborador)\b|\brecepcionista\b",
        "Femenino",
        "Femeniza la denominación del puesto o recurre a etiquetas estereotipadas (como 'asistente administrativa', 'recepcionista' o 'chica ordenada').",
        "Usar títulos neutros como 'profesional de asistencia administrativa' o 'persona especialista en recepción'.",
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


def load_model(model_key: str):
    """Carga un pipeline entrenado desde disco."""
    model_path = MODELS_DIR / f"{model_key}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo '{model_path}'. Ejecuta 'python -m src.train_models' primero."
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


def analyze_job_posting(text: str, model_key: str | None = None, pipeline=None) -> AnalysisResult:
    """Ejecuta predicción ML + detección de frases + resaltado HTML sobre una oferta laboral."""
    chosen_key = model_key or load_best_model_key()
    model = pipeline if pipeline is not None else load_model(chosen_key)

    proba_arr = model.predict_proba([text])[0]
    classes = list(model.classes_)
    raw_probs = {cls: float(prob) for cls, prob in zip(classes, proba_arr)}
    probabilities = {cat: raw_probs.get(cat, 0.0) for cat in ALL_CATEGORIES}
    bias_probabilities = {cat: probabilities[cat] for cat in BIAS_CATEGORIES}
    predicted_class = max(probabilities, key=probabilities.get)

    spans = detect_biased_spans(text)
    detected_categories = sorted(
        {s.category for s in spans},
        key=lambda c: BIAS_CATEGORIES.index(c),
    )
    highlighted_html = render_highlighted_html(text, spans)
    technical_terms = extract_technical_terms(text)

    # Índice de inclusividad transparente combinando probabilidad Neutro del modelo y frases detectadas
    ml_neutro_pct = probabilities["Neutro"] * 100.0
    span_penalty = min(len(spans) * 14.0, 85.0)
    if spans:
        inclusivity_score = max(0.0, round(min(ml_neutro_pct, 100.0 - span_penalty), 1))
    else:
        inclusivity_score = round(ml_neutro_pct, 1)

    if inclusivity_score >= 70.0 and not spans:
        risk_level = "Bajo (Inclusivo)"
    elif inclusivity_score >= 40.0 and len(spans) <= 2:
        risk_level = "Moderado"
    else:
        risk_level = "Alto"

    return AnalysisResult(
        model_key=chosen_key,
        model_name=MODEL_DISPLAY_NAMES.get(chosen_key, chosen_key),
        predicted_class=predicted_class,
        probabilities=probabilities,
        bias_probabilities=bias_probabilities,
        inclusivity_score=inclusivity_score,
        risk_level=risk_level,
        spans=spans,
        highlighted_html=highlighted_html,
        detected_categories=detected_categories,
        technical_terms=technical_terms,
    )
