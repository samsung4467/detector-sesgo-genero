"""Motor de reescritura inclusiva y neutra preservando los requisitos técnicos del puesto."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.bias_analyzer import _is_in_inclusive_clause, extract_technical_terms
from src.text_utils import fold_accents_1to1


@dataclass
class RewriteChange:
    original: str
    replacement: str
    category: str
    reason: str


@dataclass
class RewriteResult:
    original_text: str
    rewritten_text: str
    changes: List[RewriteChange]
    preserved_technical_terms: List[str]
    missing_technical_terms: List[str]


# Reglas ordenadas de más específicas a más generales (regex evaluado sobre versión sin tildes 1-a-1)
# Formato: (regex_sin_tildes, reemplazo_en_espanol_neutro, categoria, motivo)
REWRITE_RULES: List[Tuple[str, str, str, str]] = [
    # ==================== EXCLUYENTE ====================
    (
        r"\brequisito excluyente\s*:\s*sexo\s+(?:masculino|femenino)\s*,?\s*(?:abstenerse\s+(?:mujeres|hombres|varones)(?:\s+debido\s+a[^.;\n]+)?)?",
        "convocatoria abierta a todas las personas que cumplan con los requisitos técnicos del puesto",
        "Excluyente",
        "Eliminación de filtro excluyente por sexo.",
    ),
    (
        r"\bsolo\s+candidatos\s+varones\s+de\s+entre\s+\d{2}\s+y\s+\d{2}\s+anos\s*,?\s*(?:preferiblemente\s+solteros\s+y\s+sin\s+cargas\s+familiares)?",
        "convocatoria abierta a personas profesionales con experiencia acorde al puesto",
        "Excluyente",
        "Eliminación de restricciones por género, edad y situación familiar.",
    ),
    (
        r"\b(?:solo|solamente|unicamente|exclusivamente)\s+(?:hombres|varones|caballeros|mujeres|damas|personal masculino|personal femenino|candidatos varones|candidatos masculinos|candidatas mujeres|candidatas femeninas|postulantes de sexo (?:masculino|femenino)|personal del sexo (?:masculino|femenino))(?:\s+sin\s+(?:cargas|responsabilidades)\s+familiares)?\b",
        "todas las personas calificadas",
        "Excluyente",
        "Sustitución de restricción por sexo por convocatoria abierta.",
    ),
    (
        r"\babstenerse\s+(?:mujeres|hombres|varones|madres|candidatas|postulantes del sexo opuesto)(?:\s+y\s+personas\s+mayores\s+de\s+\d{2}\s+anos)?(?:\s+debido\s+a[^.;\n]+|\s+de\s+postularse|\s+de\s+enviar\s+su\s+hoja\s+de\s+vida)?",
        "todas las personas que cumplan el perfil técnico son bienvenidas a postularse",
        "Excluyente",
        "Eliminación de cláusula explícita de exclusión.",
    ),
    (
        r"\bse\s+descartan(?:\s+automaticamente)?\s+postulantes\s+con\s+hijos\s+pequenos\s+o\s+responsabilidades\s+familiares\s+que\s+limiten\s+la\s+disponibilidad\s+total\s*;?\s*(?:puesto\s+no\s+apto\s+para\s+mujeres\s+embarazadas)?",
        "se requiere disponibilidad acorde a la modalidad y horarios definidos para el puesto",
        "Excluyente",
        "Reemplazo de discriminación por maternidad y responsabilidades familiares por criterios horarios objetivos.",
    ),
    (
        r"\b(?:puesto\s+)?no\s+apto\s+para\s+mujeres\s+embarazadas(?:\s+ni\s+personas\s+con\s+compromisos\s+de\s+cuidado\s+familiar)?",
        "el puesto cuenta con protocolos de seguridad y salud ocupacional vigentes",
        "Excluyente",
        "Eliminación de exclusión por embarazo.",
    ),
    (
        r"\b(?:indispensable\s+)?no\s+tener\s+hijos(?:\s+ni\s+planes\s+de\s+(?:maternidad|embarazo)(?:\s+a\s+corto\s+plazo)?)?\b",
        "contar con compromiso profesional acorde a la jornada laboral",
        "Excluyente",
        "Eliminación de restricción por maternidad, embarazo o hijos.",
    ),
    (
        r"\b(?:que\s+no\s+este\s+en\s+|no\s+este\s+en\s+|no\s+estar\s+en\s+|abstenerse\s+personas\s+en\s+|personas?\s+en\s+|candidatas?\s+en\s+|mujeres?\s+en\s+|en\s+)?estado\s+de\s+gestacion(?:\s+o\s+embarazo)?\b",
        "en igualdad de condiciones laborales",
        "Excluyente",
        "Eliminación de restricción discriminatoria por estado de gestación.",
    ),
    (
        r"\b(?:no\s+estar\s+en\s+|sin\s+|planes\s+de\s+|prueba\s+de\s+)?embaraz(?:o|ada|adas)\b",
        "en igualdad de condiciones laborales",
        "Excluyente",
        "Eliminación de filtro por embarazo.",
    ),
    (
        r"\b(?:estado\s+civil\s+|perfil\s+)?solter[oa]s?\s+sin\s+hijos(?:\s+ni\s+compromisos\s+conyugales)?\b",
        "con orientación al servicio y disponibilidad para el horario establecido",
        "Demográfico",
        "Eliminación combinada de estado civil y filtro 'sin hijos'.",
    ),
    (
        r"\b(?:preferentemente\s+|preferiblemente\s+|indispensable\s+|requisito\s+)?sin\s+hijos(?:\s+pequenos|\s+a\s+cargo|\s+ni\s+cargas\s+familiares)?\b",
        "con disponibilidad para el horario establecido",
        "Excluyente",
        "Eliminación del filtro excluyente 'sin hijos'.",
    ),
    (
        r"\brecepcionista\s+(?:mujer|hombre|femenin[oa]|masculin[oa])\b",
        "persona especialista en recepción",
        "Excluyente",
        "Neutralización de título con género exclusivo.",
    ),
    (
        r"\bingenier[oa](?:\s+de\s+\w+)?\s+(?:hombre|mujer|varon|masculin[oa]|femenin[oa])\b",
        "profesional en ingeniería",
        "Excluyente",
        "Neutralización de título con género exclusivo.",
    ),
    (
        r"\b(asistente|secretari[oa]|vendedor[a]?|ejecutiv[oa]|gerent[ea]|supervisor[a]?|coordinador[a]?|analista|desarrollador[a]?|programador[a]?|tecnic[oa]|abogad[oa]|contador[a]?|auxiliar|operari[oa]|administrativ[oa])\s+(?:mujer|hombre|varon|dama|caballero)\b",
        r"profesional en el rol de \1",
        "Excluyente",
        "Neutralización de título con género exclusivo.",
    ),
    (
        r"\b(?:preferiblemente|preferentemente)\s+(?:hombres|varones|mujeres)\s+sin\s+(?:responsabilidades|cargas|obligaciones)\s+(?:familiares|domesticas|maternales)\b",
        "personas profesionales con disponibilidad para la modalidad del puesto",
        "Excluyente",
        "Neutralización de preferencia por género y situación familiar.",
    ),
    (
        r"\bsin\s+(?:cargas|responsabilidades|obligaciones|impedimentos)\s+(?:familiares|maternales|domesticas)\b",
        "con disponibilidad acorde al horario del puesto",
        "Excluyente",
        "Sustitución de filtro por responsabilidades familiares.",
    ),
    (
        r"\b(?:reservada|destinada)\s+(?:unicamente|exclusivamente)\s+para\s+(?:varones|hombres|mujeres|personal femenino|personal masculino)\b",
        "abierta a todas las personas calificadas",
        "Excluyente",
        "Apertura inclusiva de la vacante.",
    ),
    (
        r"\bsexo\s+(?:masculino|femenino)\s+excluyente\b|\bvaron\s+excluyente\b",
        "convocatoria abierta sin distinción de género",
        "Excluyente",
        "Eliminación de requisito de sexo.",
    ),
    # ==================== LGBTQ+ ====================
    (
        r"\bes\s+indispensable\s+que\s+el\s+personal\s+se\s+ajuste\s+a\s+valores\s+tradicionales\s+de\s+familia\s+y\s+mantenga\s+un\s+aspecto\s+estrictamente\s+acorde\s+a\s+su\s+sexo\s+biologico",
        "valoramos el respeto profesional, la ética laboral y una presentación acorde al entorno corporativo",
        "LGBTQ+",
        "Reemplazo de requisitos cisheteronormativos por estándares profesionales inclusivos.",
    ),
    (
        r"\bexigimos\s+vestimenta\s+estrictamente\s+masculina\s+o\s+femenina(?:\s+segun\s+(?:el\s+)?dni)?\s+y\s+abstenerse\s+personas\s+con\s+expresion\s+de\s+genero\s+no\s+convencional(?:\s+o\s+rasgos\s+androginos)?",
        "promovemos un código de vestimenta profesional respetuoso de la identidad y expresión de cada persona",
        "LGBTQ+",
        "Sustitución de código binario y exclusión de expresión de género.",
    ),
    (
        r"\bel\s+nombre\s+legal\s+obligatorio\s+debe\s+coincidir\s+con\s+el\s+sexo\s+de\s+nacimiento(?:\s+en\s+toda\s+documentacion\s+interna)?",
        "respetamos el nombre elegido y la identidad de cada integrante del equipo en la comunicación interna",
        "LGBTQ+",
        "Respeto a la identidad de género y nombre elegido.",
    ),
    (
        r"\bvalores\s+familiares\s+(?:convencionales|tradicionales|clasicos)\b|\bvalores\s+tradicionales\s+de\s+familia(?:\s+conformada\s+unicamente\s+por\s+hombre\s+y\s+mujer)?\b|\bprincipios\s+familiares\s+tradicionales\b",
        "compromiso con el respeto e integridad",
        "LGBTQ+",
        "Neutralización de referencia familiar heteronormativa o convencional.",
    ),
    (
        r"\bperfil(?:es)?\s+tradicional(?:es)?\b",
        "enfoque ético y colaborativo",
        "LGBTQ+",
        "Sustitución de exigencia ambigua/excluyente de 'perfil tradicional'.",
    ),
    (
        r"\bexpresion(?:es)?\s+de\s+genero\s+(?:no\s+convencional(?:es)?|no\s+binari[oa]s?|divers[oa]s?)\b",
        "todas las expresiones e identidades de género",
        "LGBTQ+",
        "Neutralización de restricción sobre expresión de género.",
    ),
    (
        r"\b(?:identidad(?:es)?\s+)?no\s+binari[oa]s?\b",
        "todas las identidades de género",
        "LGBTQ+",
        "Sustitución inclusiva para identidad no binaria.",
    ),
    (
        r"\b(?:aspecto|identidad|imagen|presentacion(?:\s+personal)?)\s+(?:estrictamente\s+)?acorde\s+a(?:l|\s+su)\s+sexo\s+(?:biologico|de\s+nacimiento|registral)(?:\s+y\s+sin\s+rasgos\s+androginos)?\b",
        "presentación profesional acorde al entorno laboral",
        "LGBTQ+",
        "Eliminación de requisito cisnormativo.",
    ),
    (
        r"\bvestimenta\s+estrictamente\s+masculina\s+o\s+femenina(?:\s+segun\s+(?:el\s+)?dni)?\b|\bcodigo\s+de\s+imagen\s+binario\s+estricto(?:\s+para\s+caballeros\s+y\s+damas)?\b",
        "código de vestimenta profesional acorde al entorno de trabajo",
        "LGBTQ+",
        "Neutralización de código de vestimenta binario.",
    ),
    (
        r"\babstenerse\s+personas\s+(?:con\s+expresion\s+de\s+genero\s+no\s+convencional|trans(?:\s*,\s*|\s+o\s+)no\s+binarias)(?:\s+o\s+con\s+identidades\s+de\s+genero\s+diversas)?",
        "damos la bienvenida a personas de todas las identidades y expresiones de género",
        "LGBTQ+",
        "Eliminación de exclusión a personas LGBTQ+.",
    ),
    (
        r"\b(?:entorno|ambiente|cultura)(?:\s+laboral|\s+corporativa|\s+conservadora)?\s+(?:estrictamente\s+)?(?:tradicional\s+(?:y\s+)?)?heteronormativ[oa]\b",
        "entorno laboral respetuoso e inclusivo",
        "LGBTQ+",
        "Reemplazo de referencia a entorno heteronormativo.",
    ),
    (
        r"\b(?:hombres\s+o\s+mujeres\s+biologicos?|postulantes\s+de\s+identidad\s+cisgenero|sexo\s+biologico|sexo\s+de\s+nacimiento|sexo\s+registral)\b",
        "personas profesionales calificadas",
        "LGBTQ+",
        "Eliminación de restricción cisnormativa.",
    ),
    (
        r"\bsin\s+(?:rasgos\s+androginos|ambiguedades\s+de\s+genero)\b",
        "con presentación profesional adecuada al puesto",
        "LGBTQ+",
        "Eliminación de restricción sobre expresión de género.",
    ),
    # ==================== DEMOGRÁFICO ====================
    (
        r"\bjoven\s*,?\s*con\s+edad\s+entre\s+\d{2}\s+y\s+\d{2}\s+anos\s*\(\s*personas\s+mayores\s+de\s+\d{2}\s+anos\s+abstenerse\s*\)",
        "con sólida motivación profesional y sin restricciones de edad",
        "Demográfico",
        "Eliminación de filtro etario.",
    ),
    (
        r"\bedad(?:\s+excluyente|\s+comprendida)?\s+(?:entre|de)\s+\d{2}\s+y\s+\d{2}\s+anos(?:\s*;\s*personas\s+mayores\s+de\s+\d{2}\s+anos(?:\s+por\s+favor)?\s+abstenerse(?:\s+de\s+postular)?)?\b",
        "sin restricciones de edad",
        "Demográfico",
        "Eliminación de rango etario excluyente.",
    ),
    (
        r"\b(?:personas\s+)?mayores\s+de\s+\d{2}\s+anos\s+(?:por\s+favor\s+)?abstenerse(?:\s+de\s+postular)?\b|\babstenerse\s+(?:personas\s+)?mayores\s+de\s+\d{2}\s+anos\b",
        "convocatoria abierta a personas de todas las edades",
        "Demográfico",
        "Eliminación de exclusión por edad.",
    ),
    (
        r"\b(?:muy\s+)?joven(?:es)?\s*\(\s*(?:edad\s+)?(?:de\s+)?(?:entre\s+)?\d{2}\s+(?:y|a)\s+\d{2}\s+anos\s*\)",
        "con competencias técnicas comprobadas",
        "Demográfico",
        "Eliminación de filtro etario explícito ('joven (entre X y Y años)').",
    ),
    (
        r"\(\s*(?:edad\s+)?(?:de\s+)?(?:entre\s+)?\d{2}\s+(?:y|a)\s+\d{2}\s+anos\s*\)",
        "con competencias técnicas comprobadas",
        "Demográfico",
        "Eliminación de rango etario entre paréntesis.",
    ),
    (
        r"\b(?:de\s+)?entre\s+\d{2}\s+y\s+\d{2}\s+anos\b(?!\s+de\s+experiencia)|\bde\s+\d{2}\s+a\s+\d{2}\s+anos\b(?!\s+de\s+experiencia)",
        "con competencias técnicas comprobadas",
        "Demográfico",
        "Eliminación de rango de edad explícito.",
    ),
    (
        r"\b(?:joven\s+recien\s+egresad[oa]|jovenes\s+menores\s+de\s+\d{2}\s+anos|profesional\s+menor\s+de\s+\d{2}\s+anos|menor(?:es)?\s+de\s+\d{2}\s+anos|edad\s+maxima\s+(?:de\s+)?\d{2}\s+anos(?:\s+excluyente)?)\b",
        "con competencias técnicas comprobadas",
        "Demográfico",
        "Sustitución de requisito etario por criterio de competencias.",
    ),
    (
        r"\b(?:perfil\s+muy\s+joven|talento\s+muy\s+joven|perfil\s+joven|energia\s+juvenil|muy\s+joven|joven)\b(?!\s+(?:empresa|startup|compania|equipo|organizacion))",
        "con competencias técnicas comprobadas",
        "Demográfico",
        "Eliminación de filtro etario ('joven').",
    ),
    (
        r"\bpreferimos\s+perfil\s+soltero\s+sin\s+hijos\s+y\s+de\s+nacionalidad\s+local\s+excluyente\s*,?\s*que\s+resida\s+en\s+zona\s+norte(?:\s+de\s+la\s+ciudad)?",
        "requerimos autorización legal vigente para trabajar y disponibilidad para la modalidad del puesto",
        "Demográfico",
        "Reemplazo de sesgos por estado civil, nacionalidad y zona de residencia.",
    ),
    (
        r"\brequisito\s+indispensable\s*:\s*excelente\s+presencia\s+fisica\s*,\s*adjuntar\s+foto\s+de\s+cuerpo\s+entero(?:\s+en\s+el\s+cv)?",
        "valoramos la comunicación profesional y el cumplimiento de las competencias técnicas del puesto",
        "Demográfico",
        "Eliminación de requisito de apariencia física y fotografía corporal.",
    ),
    (
        r"\b(?:excelente\s+)?presencia\s+fisica(?:\s+impecable)?(?:\s*,\s*adjuntar\s+foto\s+de\s+cuerpo\s+entero(?:\s+en\s+el\s+cv)?)?\b|\badjuntar\s+foto\s+de\s+cuerpo\s+entero(?:\s+en\s+el\s+cv)?\b",
        "sólida comunicación profesional",
        "Demográfico",
        "Eliminación de requisito de apariencia física.",
    ),
    (
        r"\b(?:estado\s+civil\s+|perfil\s+)?solter[oa]s?\b",
        "con orientación al servicio",
        "Demográfico",
        "Eliminación de referencia al estado civil.",
    ),
    (
        r"\bnacionalidad\s+local\s+excluyente\b|\bciudadanos\s+nativos\s+del\s+pais\b|\bnativo\s+del\s+pais\b",
        "autorización legal vigente para trabajar en el país",
        "Demográfico",
        "Sustitución de restricción por nacionalidad por criterio legal objetivo.",
    ),
    (
        r"\babstenerse\s+(?:postulantes\s+)?extranjeros(?:\s+o\s+migrantes)?\b",
        "contar con permiso laboral vigente en el país",
        "Demográfico",
        "Eliminación de exclusión por origen nacional.",
    ),
    (
        r"\b(?:residir|residente|domicilio\s+comprobable)\s+en\s+(?:zona\s+norte|barrios\s+residenciales(?:\s+exclusivos)?|distritos\s+centricos\s+acomodados)\b",
        "disponibilidad para asistir a la sede según la modalidad laboral",
        "Demográfico",
        "Reemplazo de filtro socioeconómico/geográfico por disponibilidad de modalidad.",
    ),
    (
        r"\b(?:fotografia\s+reciente\s+actualizada|foto\s+de\s+cuerpo\s+entero|foto\s+actualizada)\b",
        "currículum actualizado con experiencia relevante",
        "Demográfico",
        "Eliminación de solicitud de fotografía.",
    ),
    (
        r"\b(?:buena\s+pinta|estatura\s+acorde\s+al\s+puesto|complexion\s+esbelta|presencia\s+estetica)\b",
        "trato profesional",
        "Demográfico",
        "Eliminación de criterios físicos o estéticos.",
    ),
    # ==================== MASCULINO ====================
    (
        r"\bun\s+desarrollador\s+backend\s+senior\s+que\s+sea\s+un\s+verdadero\s+tiburon\s+de\s+la\s+tecnologia\s+y\s+un\s+hombre\s+de\s+accion",
        "una persona especialista en desarrollo backend senior con sólida visión tecnológica e iniciativa estratégica",
        "Masculino",
        "Neutralización de cargo masculino y metáforas androcéntricas.",
    ),
    (
        r"\bun\s+ingeniero\s+aguerrido\s*,\s*competitivo\s+y\s+con\s+liderazgo\s+dominante\s*,\s*capaz\s+de\s+imponer\s+decisiones\s+tecnicas",
        "profesional en ingeniería con orientación a resultados, liderazgo colaborativo y capacidad para fundamentar decisiones técnicas",
        "Masculino",
        "Reemplazo de adjetivos masculinos agresivos por competencias de liderazgo técnico.",
    ),
    (
        r"\bsi\s+eres\s+un\s+rockstar\s+ambicioso\s+con\s+mentalidad\s+ganadora",
        "si cuentas con sólida experiencia técnica y orientación a la mejora continua",
        "Masculino",
        "Eliminación de jerga masculinizada ('rockstar') y codificación hipercompetitiva.",
    ),
    (
        r"\bun\s+jefe\s+comercial\s+b2b\s*,\s*caballero\s+con\s+caracter\s+fuerte\s+y\s+espiritu\s+agresivo\s+para\s+conquistar\s+cuentas",
        "una persona para la jefatura comercial B2B con liderazgo estratégico y enfoque proactivo para desarrollar cuentas",
        "Masculino",
        "Neutralización de género y reemplazo de estereotipos agresivos.",
    ),
    (
        r"\bun\s+guerrero\s+de\s+ventas\s+ambicioso",
        "una persona especialista en ventas orientada a resultados",
        "Masculino",
        "Sustitución de metáfora bélica masculina.",
    ),
    (
        r"\b(?:un\s+verdadero\s+)?tiburon(?:\s+(?:de\s+la\s+tecnologia|comercial|de\s+negocios|en\s+su\s+area|decidido))?\b",
        "profesional con visión estratégica",
        "Masculino",
        "Neutralización de metáfora agresiva.",
    ),
    (
        r"\b(?:un\s+)?guerrero(?:\s+(?:de\s+ventas|de\s+resultados|comercial))?\b",
        "profesional orientado a objetivos",
        "Masculino",
        "Neutralización de metáfora bélica.",
    ),
    (
        r"\bun\s+hombre\s+de\s+accion\b",
        "un perfil con iniciativa y sólido criterio",
        "Masculino",
        "Eliminación de exclusividad masculina en la descripción del perfil.",
    ),
    (
        r"\bhombre\s+de\s+accion\b|\bcaballero\s+con\s+(?:caracter\s+fuerte|fuerte\s+personalidad)\b",
        "un perfil con iniciativa y sólido criterio",
        "Masculino",
        "Eliminación de exclusividad masculina en la descripción del perfil.",
    ),
    (
        r"\blider(?:azgo)?\s+(?:dominante|autoritario|alfa)\b",
        "liderazgo estratégico y colaborativo",
        "Masculino",
        "Reemplazo de estilo autoritario codificado como masculino.",
    ),
    (
        r"\b(?:espiritu|perfil)\s+(?:sumamente\s+)?agresiv[oa]\b|\bagresividad\s+comercial\b",
        "enfoque proactivo orientado al crecimiento",
        "Masculino",
        "Neutralización del adjetivo 'agresivo'.",
    ),
    (
        r"\b(?:rockstar|ninja|crack\s+incansable)\b",
        "profesional especialista",
        "Masculino",
        "Sustitución de jerga masculinizada por denominación profesional neutra.",
    ),
    (
        r"\b(?:muy\s+|sumamente\s+)?aguerrid[oa]s?\b",
        "con enfoque estratégico",
        "Masculino",
        "Sustitución gramaticalmente concordante del adjetivo 'aguerrido'.",
    ),
    (
        r"\bcon\s+mentalidad\s+ganadora\b",
        "con enfoque en la mejora continua",
        "Masculino",
        "Reemplazo de 'mentalidad ganadora' por competencia profesional neutra.",
    ),
    (
        r"\bmentalidad\s+ganadora\b",
        "enfoque en la mejora continua",
        "Masculino",
        "Reemplazo de 'mentalidad ganadora' por competencia profesional neutra.",
    ),
    (
        r"\bmentalidad\s+de\s+conquista\b",
        "visión estratégica de crecimiento",
        "Masculino",
        "Sustitución de metáfora competitiva masculina.",
    ),
    (
        r"\binstinto\s+cazador\b",
        "iniciativa comercial proactiva",
        "Masculino",
        "Sustitución de metáfora masculina por competencia comercial.",
    ),
    (
        r"\benergia\s+viril\b",
        "dinamismo profesional",
        "Masculino",
        "Eliminación de rasgo asociado exclusivamente al género masculino.",
    ),
    (
        r"\bcaracter\s+fuerte\b",
        "criterio sólido y asertivo",
        "Masculino",
        "Sustitución de adjetivo codificado como masculino por competencia objetiva.",
    ),
    (
        r"\b(?:temple\s+de\s+acero|sangre\s+fria)\b",
        "templanza analítica",
        "Masculino",
        "Sustitución de expresión codificada como masculina.",
    ),
    (
        r"\bcompetitiv[oa]\s+y\s+ambicios[oa]\b",
        "orientado al logro y con visión de crecimiento",
        "Masculino",
        "Neutralización de rasgos agentivos extremos.",
    ),
    (
        r"(?<!salario\s)(?<!salario\sbase\s)(?<!sueldo\s)(?<!remuneracion\s)(?<!compensacion\s)(?<!paquete\s)(?<!mercado\s)(?<!entorno\s)(?<!ritmo\s)\b(?:muy\s+|sumamente\s+|altamente\s+)?competitiv[oa]s?\b",
        "orientado al logro",
        "Masculino",
        "Sustitución de 'competitivo' por descriptor profesional neutro preservando referencias salariales.",
    ),
    (
        r"\bambicios[oa]s?\b",
        "con proyección profesional",
        "Masculino",
        "Neutralización de adjetivo codificado como masculino.",
    ),
    (
        r"\b(?:combativo|combativa|duro\s+e\s+imponente|recio\s+y\s+dominante|audaz\s+y\s+contundente)\b",
        "resolutivo y colaborativo",
        "Masculino",
        "Sustitución de adjetivos masculinos dominantes por competencias laborales.",
    ),
    (
        r"\bimponer\s+(?:sus\s+)?decisiones(?:\s+tecnicas)?\b",
        "liderar y consensuar decisiones técnicas",
        "Masculino",
        "Sustitución de verbo autoritario por competencia de liderazgo.",
    ),
    # Neutralización de títulos en masculino genérico singular
    (r"\bun\s+arquitecto\s+de\s+software\b", "un/a profesional en arquitectura de software", "Masculino", "Neutralización de cargo en masculino genérico."),
    (r"\bun\s+arquitecto\s+de\s+([a-z0-9]+)\b", r"un/a profesional en arquitectura de \1", "Masculino", "Neutralización de cargo en masculino genérico."),
    (r"\bun\s+arquitecto\s+(cloud|senior|junior)\b", r"un/a profesional en arquitectura \1", "Masculino", "Neutralización de cargo en masculino genérico."),
    (r"\bun\s+arquitecto\b", "un/a profesional en arquitectura", "Masculino", "Neutralización de cargo en masculino genérico."),
    (r"\bun\s+analista\s+de\s+([a-z0-9]+)\b", r"una persona especialista en análisis de \1", "Masculino", "Neutralización de cargo en masculino genérico."),
    (r"\bun\s+analista\b", "una persona profesional de análisis", "Masculino", "Neutralización de cargo en masculino genérico."),
    (r"\bun\s+desarrollador\s+(backend|frontend|fullstack|full\s+stack|senior|junior)\b", r"una persona especialista en desarrollo \1", "Masculino", "Neutralización de título de puesto."),
    (r"\bun\s+desarrollador\b", "una persona especialista en desarrollo", "Masculino", "Neutralización de título de puesto."),
    (r"\bun\s+programador\b", "una persona especialista en programación", "Masculino", "Neutralización de título de puesto."),
    (r"\bun\s+ingeniero\b", "profesional en ingeniería", "Masculino", "Neutralización de título de puesto."),
    (r"\bun\s+vendedor\b", "profesional en ventas", "Masculino", "Neutralización de título de puesto."),
    (r"\bun\s+jefe\b", "una persona para la jefatura", "Masculino", "Neutralización de título de puesto."),
    (r"\bun\s+supervisor\b", "profesional en supervisión", "Masculino", "Neutralización de título de puesto."),
    (r"\bvacante\s+de\s+supervisor\b", "vacante para la supervisión", "Masculino", "Neutralización de título de puesto."),
    (r"\bselecciona\s+ejecutivo\s+de\s+cuentas\b", "selecciona profesional para la gestión de cuentas", "Masculino", "Neutralización de título de puesto."),
    (r"\bun\s+ejecutivo\b", "profesional en gestión ejecutiva", "Masculino", "Neutralización de título de puesto."),
    (r"\bun\s+gerente\b", "una persona para la gerencia", "Masculino", "Neutralización de título de puesto."),
    (r"\bun\s+coordinador\b", "una persona para la coordinación", "Masculino", "Neutralización de título de puesto."),
    (r"\bun\s+director\b", "una persona para la dirección", "Masculino", "Neutralización de título de puesto."),
    (r"\bun\s+consultor\b", "profesional en consultoría", "Masculino", "Neutralización de título de puesto."),
    (r"\bun\s+administrador\b", "profesional en administración", "Masculino", "Neutralización de título de puesto."),
    (r"\bun\s+contador\b", "profesional en contabilidad", "Masculino", "Neutralización de título de puesto."),
    (r"\bun\s+lider\b", "una persona líder", "Masculino", "Neutralización de título de puesto."),
    (r"\bun\s+abogado\b", "profesional en derecho", "Masculino", "Neutralización de título de puesto."),
    (r"\bun\s+disenador\b", "profesional en diseño", "Masculino", "Neutralización de título de puesto."),
    (r"\bun\s+cientifico\s+de\s+datos\b", "una persona especialista en ciencia de datos", "Masculino", "Neutralización de título de puesto."),
    (r"\bun\s+tecnico\b", "profesional técnico", "Masculino", "Neutralización de título de puesto."),
    (r"\bun\s+auditor\b", "profesional en auditoría", "Masculino", "Neutralización de título de puesto."),
    # ==================== FEMENINO ====================
    (
        r"\buna\s+senorita\s+amable\s+y\s+carismatica\s+para\s+el\s+cargo\s+de\s+secretaria\s+ejecutiva",
        "una persona profesional con excelentes habilidades de comunicación para el cargo de asistencia ejecutiva",
        "Femenino",
        "Neutralización de 'señorita' y cargo feminizado.",
    ),
    (
        r"\buna\s+mujer\s+delicada\s*,\s*dulce\s+y\s+servicial\s*,\s*con\s+excelente\s+presencia\s+femenina\s+y\s+sonrisa\s+calida\s+para\s+brindar\s+un\s+trato\s+maternal\s+y\s+acogedor",
        "profesional con empatía, organización y orientación al servicio para brindar una atención cordial y eficiente",
        "Femenino",
        "Eliminación de estereotipos femeninos y de rol maternal.",
    ),
    (
        r"\buna\s+asistente\s+femenina\s+siempre\s+dispuesta\s+a\s+complacer\s+las\s+necesidades\s+de\s+la\s+gerencia",
        "profesional de asistencia ejecutiva con capacidad de anticipación y soporte estratégico a la dirección",
        "Femenino",
        "Neutralización de rol subordinado y marca de género femenino.",
    ),
    (
        r"\b(?:una\s+)?(?:joven\s+)?senorita(?:\s+(?:amable|carismatica|elegante|coqueta|prolija|atenta)(?:\s+y\s+\w+)?)?\b",
        "una persona profesional",
        "Femenino",
        "Reemplazo de trato estereotipado 'señorita'.",
    ),
    (
        r"\b(?:excelente\s+)?(?:presencia|imagen|estetica)\s+femenina(?:\s+(?:impecable|distinguida|cuidada))?\b",
        "excelente comunicación y trato profesional",
        "Femenino",
        "Eliminación de requisito de apariencia femenina.",
    ),
    (
        r"\bde\s+excelente\s+presencia\b",
        "con excelente comunicación y trato profesional",
        "Femenino",
        "Sustitución de criterio estético subjetivo con concordancia preposicional.",
    ),
    (
        r"\bexcelente\s+presencia\b",
        "excelente comunicación y trato profesional",
        "Femenino",
        "Sustitución de criterio estético subjetivo por competencia profesional.",
    ),
    (
        r"\bde\s+buena\s+presencia\b",
        "con comunicación clara y trato profesional",
        "Femenino",
        "Sustitución de criterio estético subjetivo con concordancia preposicional.",
    ),
    (
        r"\bbuena\s+presencia\b",
        "comunicación clara y trato profesional",
        "Femenino",
        "Sustitución de criterio estético subjetivo por competencia profesional.",
    ),
    (
        r"\bempatic[oa]s?\b(?!\s+y\s+respetuos[oa])",
        "con comunicación asertiva",
        "Femenino",
        "Sustitución por competencia interpersonal objetiva.",
    ),
    (
        r"\btrato\s+maternal\b",
        "trato empático y respetuoso",
        "Femenino",
        "Reemplazo de estereotipo maternal por trato profesional inclusivo.",
    ),
    (
        r"\b(?:instinto\s+de\s+cuidado|vocacion\s+de\s+cuidado(?:\s*,\s*paciencia\s+y\s+sensibilidad)?)\b",
        "orientación al servicio y atención cordial",
        "Femenino",
        "Reemplazo de estereotipo de cuidado por competencia de servicio.",
    ),
    (
        r"\bmaternal(?:es)?\b",
        "empático y respetuoso",
        "Femenino",
        "Eliminación de adjetivo maternal en descripción laboral.",
    ),
    (
        r"\bvocacion\s+de\s+servicio\b",
        "orientación profesional al cliente",
        "Femenino",
        "Neutralización de expresión asociada a roles feminizados.",
    ),
    (
        r"\btrato\s+dulce\b",
        "trato cordial y asertivo",
        "Femenino",
        "Neutralización de adjetivo estereotipado 'trato dulce'.",
    ),
    (
        r"\b(?:dulce\s*,\s*carinosa\s+y\s+comprensiva|dulce\s+y\s+servicial|personalidad\s+docil\s*,\s*atenta\s+y\s+sumisa|sonrisa\s+calida|sonrisa\s+permanente|toque\s+femenino)\b",
        "trato cordial y asertivo",
        "Femenino",
        "Neutralización de estereotipos femeninos de docilidad o apariencia.",
    ),
    (
        r"\bsumis[oa]s?\b",
        "con iniciativa y espíritu colaborativo",
        "Femenino",
        "Eliminación de adjetivo de sumisión.",
    ),
    (
        r"\buna\s+asistente\s+administrativa\b",
        "un/a especialista en asistencia administrativa",
        "Femenino",
        "Neutralización de título de cargo en femenino.",
    ),
    (
        r"\basistente\s+administrativa\b",
        "especialista en asistencia administrativa",
        "Femenino",
        "Neutralización de título de cargo en femenino.",
    ),
    (
        r"\bsecretaria\s+ejecutiva\b",
        "asistencia ejecutiva",
        "Femenino",
        "Neutralización de denominación de cargo.",
    ),
    (
        r"\b(?:una\s+)?chica\s+ordenada\b",
        "un perfil organizado y metódico",
        "Femenino",
        "Neutralización de etiqueta informal con sesgo de género.",
    ),
    (
        r"\b(?:una\s+)?(?:chica\s+simpatica(?:\s+y\s+atenta)?|chica\s+encantadora|dama\s+elegante|asistente\s+femenina)\b",
        "una persona profesional",
        "Femenino",
        "Neutralización de género y estereotipos.",
    ),
    (
        r"\brecepcionista\b",
        "persona especialista en recepción",
        "Femenino",
        "Neutralización de denominación del cargo de recepción.",
    ),
    (
        r"\b(?:siempre\s+)?dispuesta\s+a\s+complacer(?:\s+las\s+necesidades\s+de\s+la\s+gerencia)?\b",
        "orientada a brindar soporte estratégico y profesional",
        "Femenino",
        "Eliminación de estereotipo femenino de sumisión.",
    ),
]


def _match_case(original: str, replacement: str) -> str:
    """Preserva la mayúscula inicial si la frase original comenzaba con mayúscula."""
    if not original or not replacement:
        return replacement
    if original[0].isupper() and replacement[0].islower():
        return replacement[0].upper() + replacement[1:]
    return replacement


_SYNONYM_FALLBACKS: Dict[str, List[str]] = {
    "Masculino": [
        "con orientación a resultados",
        "con enfoque en la mejora continua",
        "orientado al logro",
        "con sólida visión estratégica",
        "con iniciativa y criterio técnico",
        "con liderazgo colaborativo",
    ],
    "Femenino": [
        "con comunicación asertiva",
        "con trato cordial y profesional",
        "con orientación profesional al cliente",
        "con capacidad de organización",
        "con actitud colaborativa",
    ],
    "Excluyente": [
        "con evaluación objetiva por competencias",
        "abierto a todas las personas calificadas",
        "con disponibilidad acorde a la modalidad del puesto",
    ],
    "Demográfico": [
        "enfocado en competencias y experiencia comprobable",
        "sin restricciones ajenas al desempeño del puesto",
        "con autorización legal vigente para trabajar",
    ],
    "LGBTQ+": [
        "en un entorno laboral respetuoso e inclusivo",
        "con estándares profesionales respetuosos de la diversidad",
        "valorando la integridad y el respeto mutuo",
    ],
}


def _get_sentence_slice(text: str, pos: int) -> str:
    """Devuelve la oración completa que contiene el índice pos."""
    left_sep = max(
        text.rfind(".", 0, pos),
        text.rfind("!", 0, pos),
        text.rfind("?", 0, pos),
        text.rfind("\n", 0, pos),
    )
    start = left_sep + 1 if left_sep != -1 else 0
    right_candidates = [idx for sep in (".", "!", "?", "\n") if (idx := text.find(sep, pos)) != -1]
    end = min(right_candidates) if right_candidates else len(text)
    return text[start:end]


def _choose_non_repeating_replacement(
    current: str, start: int, end: int, candidate: str, category: str
) -> str:
    """Evita repetir la misma frase de reemplazo dos veces dentro de una misma oración."""
    sentence_without_span = _get_sentence_slice(current[:start] + " " + current[end:], start)
    sent_norm = fold_accents_1to1(sentence_without_span.lower())
    cand_norm = fold_accents_1to1(candidate.lower())

    if len(cand_norm.split()) >= 2 and cand_norm in sent_norm:
        for alt in _SYNONYM_FALLBACKS.get(category, []):
            alt_norm = fold_accents_1to1(alt.lower())
            if alt_norm not in sent_norm:
                return alt
    return candidate


def _deduplicate_sentence_keywords(sentence: str) -> str:
    """Elimina o simplifica repeticiones intra-oración de palabras clave, adjetivos o frases."""
    s = sentence

    # 1. Colapsar sustantivos de cabecera repetidos en coordinación ("trato X y trato Y" -> "trato X y Y")
    for head in ("trato", "enfoque", "orientación", "disponibilidad", "comunicación", "liderazgo"):
        s = re.sub(
            rf"\b({head})\s+([^.;:!?,\n]{{2,45}}?)\s*(,|y)\s+\1\s+",
            r"\1 \2\3 ",
            s,
            flags=re.IGNORECASE,
        )

    # 2. Evitar repeticiones de "al puesto" / "del puesto" dentro de la misma oración
    puesto_matches = list(re.finditer(r"\bal\s+puesto\b", s, flags=re.IGNORECASE))
    if len(puesto_matches) >= 2:
        # Sustituir ocurrencias repetidas desde la segunda en adelante por "al cargo" o simplificar
        for m in reversed(puesto_matches[1:]):
            s = s[: m.start()] + "al cargo" + s[m.end() :]

    # 3. Simplificar repetición de "profesional" cuando aparece >= 2 veces en la misma oración
    prof_matches = list(re.finditer(r"\bprofesional(?:es)?\b", s, flags=re.IGNORECASE))
    if len(prof_matches) >= 2:
        # Si aparece "trato profesional" como segunda ocurrencia -> "trato de excelencia"
        s = re.sub(
            r"(\bprofesional\b[^.;:!?]+?)\btrato\s+profesional\b",
            r"\1trato de excelencia",
            s,
            count=1,
            flags=re.IGNORECASE,
        )
        s = re.sub(
            r"(\bprofesional\b[^.;:!?]+?)\bcomunicación\s+profesional\b",
            r"\1comunicación efectiva",
            s,
            count=1,
            flags=re.IGNORECASE,
        )
        s = re.sub(
            r"(\bprofesional\b[^.;:!?]+?)\borientación\s+profesional\b",
            r"\1orientación estratégica",
            s,
            count=1,
            flags=re.IGNORECASE,
        )

    # 4. Eliminar adjetivos clave duplicados en enumeraciones dentro de la misma oración
    key_adjs = [
        "cordial",
        "asertivo",
        "asertiva",
        "respetuoso",
        "respetuosa",
        "empático",
        "empática",
        "colaborativo",
        "colaborativa",
        "estratégico",
        "estratégica",
        "organizado",
        "organizada",
        "eficiente",
    ]
    for adj in key_adjs:
        matches = list(re.finditer(rf"\b{adj}\b", s, flags=re.IGNORECASE))
        if len(matches) >= 2:
            # Eliminar ocurrencias duplicadas desde la última hacia atrás cuando están en listas
            for m in reversed(matches[1:]):
                before = s[: m.start()]
                after = s[m.end() :]
                if re.search(r"(?:,|\by|\be)\s*$", before, flags=re.IGNORECASE):
                    before = re.sub(r"(?:,|\by|\be)\s*$", "", before, flags=re.IGNORECASE)
                    s = before + after
                elif re.match(r"^\s*(?:,|\by|\be)\s+", after, flags=re.IGNORECASE):
                    after = re.sub(r"^\s*(?:,|\by|\be)\s+", "", after, flags=re.IGNORECASE)
                    s = before + after

    return s


def _polish_sentence(sentence: str) -> str:
    """Pule conectores, enumeraciones, artículos y repeticiones dentro de una sola oración."""
    s = sentence.strip()
    if not s:
        return s

    # 1. Correcciones de conectores y preposiciones residuales
    s = re.sub(
        r"\bo\s+(?:con\s+)?todas\s+las\s+expresiones\b",
        "y respetando todas las expresiones",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"\bde\s+excelente\s+comunicación\b", "con excelente comunicación", s, flags=re.IGNORECASE)
    s = re.sub(r"\bde\s+comunicación\s+clara\b", "con comunicación clara", s, flags=re.IGNORECASE)
    s = re.sub(r"\bcon\s+compromiso\s+con\b", "y compromiso con", s, flags=re.IGNORECASE)

    # 2. Asegurar artículos inclusivos tras verbos de búsqueda
    s = re.sub(
        r"\b(Buscamos|Solicitamos|Requerimos|Necesitamos|Seleccionamos|Incorporamos|Contratamos)\s+profesional\b",
        r"\1 un/a profesional",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"\b(Buscamos|Solicitamos|Requerimos|Necesitamos|Seleccionamos|Incorporamos|Contratamos)\s+especialista\b",
        r"\1 un/a especialista",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"\b(Buscamos|Solicitamos|Requerimos|Necesitamos|Seleccionamos|Incorporamos|Contratamos)\s+persona\s+especialista\b",
        r"\1 un/a especialista",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"\b(Buscamos|Solicitamos|Requerimos|Necesitamos|Seleccionamos|Incorporamos|Contratamos)\s+con\s+",
        r"\1 un perfil con ",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"\bun/a\s+un/a\s+", "un/a ", s, flags=re.IGNORECASE)
    s = re.sub(r"\buna?\s+un/a\s+", "un/a ", s, flags=re.IGNORECASE)
    s = re.sub(r"\buna\s+un\s+perfil\b", "un perfil", s, flags=re.IGNORECASE)

    # 3. Simplificar repeticiones seguidas de "con A, con B y con C" -> "con A, B y C"
    for _ in range(5):
        updated = re.sub(
            r"(\bcon\s+[^.;:!?]{3,110}?),\s+con\s+",
            r"\1, ",
            s,
            flags=re.IGNORECASE,
        )
        updated = re.sub(
            r"(\bcon\s+[^.;:!?]{3,110}?)\s+y\s+con\s+",
            r"\1 y ",
            updated,
            flags=re.IGNORECASE,
        )
        if updated == s:
            break
        s = updated

    # 4. Deduplicación intra-oración de palabras clave y adjetivos
    s = _deduplicate_sentence_keywords(s)

    # 5. Convertir cadenas de enumeración "X y Y y Z" -> "X, Y y Z" (sin cruzar preposiciones)
    for _ in range(6):
        updated = re.sub(
            r"\b((?:(?!\b(?:con|para|por|sin|en|de|que)\b)[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9/+-]+))\s+y\s+"
            r"((?:(?!\b(?:con|para|por|sin|en|de|que)\b)[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9/+-]+))\s+y\s+",
            r"\1, \2 y ",
            s,
        )
        if updated == s:
            break
        s = updated

    # 6. Eufonía española ("y i..." / "y hi..." -> "e i...")
    s = re.sub(r"\by\s+(?=(?:i|í|hi(?!e))[a-záéíóúñ])", "e ", s, flags=re.IGNORECASE)

    # 7. Limpieza de puntuación fina dentro de la oración
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    s = re.sub(r",\s*,+", ",", s)
    s = re.sub(r",\s+(y|e)\s+", r" \1 ", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(y|e)\s*,", ",", s, flags=re.IGNORECASE)
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s.strip()


def _polish_corporate_spanish(text: str) -> str:
    """Post-procesamiento lingüístico empresarial aplicado siempre al texto reescrito completo."""
    # Limpieza básica global previa
    cleaned = re.sub(r"[ \t]{2,}", " ", text)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r",\s*,+", ",", cleaned)
    cleaned = re.sub(r";\s*;+", ";", cleaned)
    cleaned = re.sub(r"\.\s*\.+", ".", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    # Corregir duplicaciones directas de artículos/preposiciones
    cleaned = re.sub(r"\buna\s+una\s+", "una ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bun\s+una\s+", "una ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bpara\s+para\s+", "para ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bcon\s+con\s+", "con ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bde\s+de\s+", "de ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\ben\s+en\s+", "en ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bpor\s+por\s+", "por ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\b([a-záéíóúñ]+(?:\s+[a-záéíóúñ]+){1,4})\s*(?:,|y)\s+\1\b",
        r"\1",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Dividir preservando delimitadores de oración (. ! ? \n) y pulir cada oración
    parts = re.split(r"([.!?\n]+)", cleaned)
    rebuilt: List[str] = []
    for idx in range(0, len(parts), 2):
        seg = parts[idx]
        sep = parts[idx + 1] if idx + 1 < len(parts) else ""
        if seg.strip():
            rebuilt.append(_polish_sentence(seg) + sep)
        elif sep:
            rebuilt.append(sep)

    result = "".join(rebuilt)
    result = re.sub(r"([.!?])(?=[A-ZÁÉÍÓÚÑ])", r"\1 ", result)
    result = re.sub(r"[ \t]{2,}", " ", result).strip()
    return result


def rewrite_inclusive(text: str, append_equality_clause: bool = True) -> RewriteResult:
    """Genera una versión inclusiva y neutra preservando íntegros los requisitos técnicos."""
    if not text.strip():
        return RewriteResult(
            original_text=text,
            rewritten_text=text,
            changes=[],
            preserved_technical_terms=[],
            missing_technical_terms=[],
        )

    original_tech_terms = extract_technical_terms(text)
    current = text
    changes: List[RewriteChange] = []

    for pattern, replacement, category, reason in REWRITE_RULES:
        # Aplicar reemplazos iterativamente sobre coincidencias actuales en versión sin tildes 1-a-1
        folded = fold_accents_1to1(current)
        matches = list(re.finditer(pattern, folded, flags=re.IGNORECASE))
        if not matches:
            continue

        # Reemplazar de derecha a izquierda para no desplazar índices
        for m in reversed(matches):
            matched_folded = folded[m.start() : m.end()]
            if _is_in_inclusive_clause(folded, m.start(), matched_folded):
                continue
            orig_slice = current[m.start() : m.end()]
            expanded = m.expand(replacement) if "\\" in replacement else replacement
            deduped = _choose_non_repeating_replacement(current, m.start(), m.end(), expanded, category)
            rep = _match_case(orig_slice, deduped)
            current = current[: m.start()] + rep + current[m.end() :]
            changes.append(
                RewriteChange(
                    original=orig_slice,
                    replacement=rep,
                    category=category,
                    reason=reason,
                )
            )

    # Restaurar orden cronológico aproximado de cambios
    changes.reverse()
    rewritten = _polish_corporate_spanish(current)

    # Añadir cláusula de igualdad de oportunidades si hubo mitigaciones y no existe ya una
    equality_markers = ("igualdad de oportunidades", "entorno laboral diverso", "proceso inclusivo y diverso")
    has_equality = any(m in fold_accents_1to1(rewritten.lower()) for m in equality_markers)
    if append_equality_clause and changes and not has_equality:
        if not rewritten.endswith("."):
            rewritten += "."
        rewritten += (
            " Nuestro proceso inclusivo y diverso garantiza igualdad de oportunidades para todas las personas, "
            "evaluando únicamente competencias técnicas y experiencia sin distinción de género, edad ni orientación."
        )

    # Ejecutar pasada final ligera de pulido empresarial sobre todo el texto
    rewritten = _polish_corporate_spanish(rewritten)

    # Verificar preservación exacta de los requisitos técnicos detectados
    rewritten_folded_lower = fold_accents_1to1(rewritten.lower())
    preserved: List[str] = []
    missing: List[str] = []
    for term in original_tech_terms:
        if fold_accents_1to1(term.lower()) in rewritten_folded_lower:
            preserved.append(term)
        else:
            missing.append(term)

    return RewriteResult(
        original_text=text,
        rewritten_text=rewritten,
        changes=changes,
        preserved_technical_terms=preserved,
        missing_technical_terms=missing,
    )
