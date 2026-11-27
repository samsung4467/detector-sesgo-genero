"""Configuración global, categorías de sesgo, paleta visual y ejemplos precargados."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

DATASET_PATH = DATA_DIR / "dataset_sesgos_empleo.csv"
REAL_PERU_DATASET_PATH = DATA_DIR / "ofertas_reales_peru.csv"
CONSOLIDATED_DATASET_PATH = DATA_DIR / "dataset_consolidado_1400.csv"
CONSOLIDATED_EXCEL_PATH = DATA_DIR / "dataset_consolidado_1400.xlsx"
CONSOLIDATED_2000_PATH = DATA_DIR / "dataset_consolidado_2000.csv"
CONSOLIDATED_2000_EXCEL_PATH = DATA_DIR / "dataset_consolidado_2000.xlsx"
METRICS_JSON_PATH = REPORTS_DIR / "model_metrics.json"
METRICS_SUMMARY_CSV_PATH = REPORTS_DIR / "model_metrics_summary.csv"
PER_CLASS_CSV_PATH = REPORTS_DIR / "per_class_metrics.csv"
CONFUSION_CSV_PATH = REPORTS_DIR / "confusion_matrices.csv"
BEST_MODEL_META_PATH = MODELS_DIR / "best_model_meta.json"
RF_MODEL_PATH = MODELS_DIR / "rf_model.joblib"

BIAS_CATEGORIES = [
    "Masculino",
    "Femenino",
    "Excluyente",
    "LGBTQ+",
    "Demográfico",
]

ALL_CATEGORIES = BIAS_CATEGORIES + ["Neutro"]

CATEGORY_INFO = {
    "Masculino": {
        "color": "#2563EB",
        "bg": "#DBEAFE",
        "border": "#93C5FD",
        "text": "#1E3A8A",
        "short_desc": "Lenguaje androcéntrico o codificado con estereotipos masculinos (agresividad, dominio, títulos solo masculinos).",
    },
    "Femenino": {
        "color": "#DB2777",
        "bg": "#FCE7F3",
        "border": "#F9A8D4",
        "text": "#831843",
        "short_desc": "Estereotipos asociados a roles femeninos tradicionales (apariencia, trato maternal, sumisión o títulos exclusivamente femeninos).",
    },
    "Excluyente": {
        "color": "#DC2626",
        "bg": "#FEE2E2",
        "border": "#FCA5A5",
        "text": "#7F1D1D",
        "short_desc": "Exclusión explícita o barreras discriminatorias por sexo, género, embarazo o responsabilidades familiares.",
    },
    "LGBTQ+": {
        "color": "#7C3AED",
        "bg": "#EDE9FE",
        "border": "#C4B5FD",
        "text": "#4C1D95",
        "short_desc": "Requisitos cisheteronormativos o restricciones sobre identidad de género, expresión personal o estructura familiar.",
    },
    "Demográfico": {
        "color": "#D97706",
        "bg": "#FEF3C7",
        "border": "#FCD34D",
        "text": "#78350F",
        "short_desc": "Filtros discriminatorios por edad, estado civil, nacionalidad, zona de residencia o apariencia física.",
    },
    "Neutro": {
        "color": "#059669",
        "bg": "#D1FAE5",
        "border": "#6EE7B7",
        "text": "#065F46",
        "short_desc": "Redacción inclusiva, objetiva y centrada en competencias técnicas y profesionales.",
    },
}

MODEL_DISPLAY_NAMES = {
    "random_forest": "Random Forest",
}

PRELOADED_EXAMPLES = {
    "[Real Perú] Masculino — Asistente de Facturación (Bumeran)": (
        "ASISTENTE DE FACTURACIÓN – PERSONAL MASCULINO | TURNOS ROTATIVOS [EMPRESA], empresa dedicada a la "
        "comercialización y distribución de productos cárnicos, se encuentra en búsqueda de un Asistente de Facturación "
        "para formar parte de nuestro equipo de trabajo. Lugar de trabajo: Breña, Lima Sueldo: S/ 1,600 Turnos: "
        "Rotativos Descansos: Rotativos ¿Qué harás? Serás responsable de apoyar en el proceso de facturación y control "
        "documentario de la empresa, asegurando la correcta emisión y registro de los comprobantes de pago y la "
        "coordinación con las áreas involucradas. Principales funciones: - Emitir facturas, boletas y notas de crédito "
        "y débito. - Registrar las ventas y comprobantes en el sistema. - Verificar que la documentación de las ventas "
        "se encuentre correctamente registrada. - Coordinar con las áreas involucradas para la correcta atención de "
        "los pedidos y documentos. - Apoyar en el control y archivo de documentos. - Elaborar y actualizar reportes "
        "relacionados con facturación. - Mantener el orden y control de la documentación del área.¿Qué buscamos? "
        "- Personal masculino. - Experiencia mínima de 6 meses en facturación. - Experiencia en emisión y manejo de "
        "facturas, boletas, notas de crédito y débito. - Conocimientos básicos de Excel. - Manejo de sistemas de "
        "facturación, deseable. - Persona organizada, responsable y con atención al detalle. - Disponibilidad para "
        "trabajar en turnos rotativos.Horarios Turno 1: 1:00 a.m. – 9:00 a.m.Turno 2: 3:00 a.m. – 12:00 p.m. "
        "Turno 3: 10:00 a.m. – 7:00 p.m. Descansos rotativos Si tienes experiencia en facturación, eres organizado y "
        "buscas una oportunidad para desarrollarte en el área administrativa, te invitamos a postular."
    ),
    "[Real Perú] Femenino — Anfitriona / Azafata (Bumeran)": (
        "Anfitriona / Azafata Resumen del Puesto Estamos en la búsqueda de talentos dinámicos, proactivos y orientados "
        "al servicio al cliente para ocupar el puesto de Anfitriona / Azafata. Si posees excelentes habilidades "
        "interpersonales y disfrutas interactuando con el público, esta oportunidad es para ti. Responsabilidades "
        "- Brindar un servicio de excelencia a los clientes. - Colaborar en las operaciones diarias del área asignada. "
        "- Mantener un ambiente de trabajo ordenado y seguro. - Mantener una actitud amable y profesional en todo "
        "momento. - Adaptarse a diferentes tipos de eventos y horarios. Requisitos - Excelente presencia y trato "
        "interpersonal. - Habilidades de comunicación y capacidad para interactuar con el público. - Disponibilidad "
        "para trabajar en horarios rotativos y escalonados según la necesidad de los eventos. Ofrecemos - Incorporación "
        "a planilla con todos los beneficios de ley."
    ),
    "[Real Perú] Demográfico — Supervisor/a Mercaderista (Computrabajo)": (
        "Centro de lima: Supervisor/a mercaderista de telefonia / basico 1400 + Comisiones 1000 + Mov 500. "
        "Descripción de la oferta A convenir Contrato de Temporada Tiempo completo Estamos en la búsqueda de "
        "SUPERVISOR/A MERCADERISTA TELEFONÍA (supervisor 360) Requisitos Experiencia realizando ruta en campo. "
        "Experiencia supervisando personal de ventas o mercaderistas. Conocimiento deseable en ventas de telefonía. "
        "Disponibilidad para movilizarse por puntos asignados en Centro de Lima. Edad máxima: 45 años. Beneficios "
        "Sueldo fijo de S/ 1,400. Comisiones de hasta S/ 1,000. Movilidad de S/ 500. Ingreso a planilla completa desde "
        "el primer día. Pagos mensuales. Descanso fijo los domingos. Funciones Supervisar el desempeño y las ventas "
        "del equipo. Capacitar y reforzar conocimientos del personal a cargo. Verificar el correcto acomodo y "
        "exhibición de productos de telefonía en los puntos de venta. Realizar visitas a la ruta asignada en Centro de "
        "Lima. Horario Lunes a sábado de 10:00 a.m. a 7:00 p.m. (1 hora de refrigerio). Zona de trabajo Cobertura de "
        "locales de la marca en Centro de Lima.: Educación Secundaria años de experiencia Edad: Hasta 45 años "
        "Palabras clave: residente, encargado, supervisor, subgerente, responsable, coordinador, gestor, capitan, "
        "comerciante, merchandiser, merchant Hace 2 días (actualizada)"
    ),
    "[Real Perú] Excluyente — Damas Universitarias / Asesora Comercial (Bumeran)": (
        "DAMAS UNIVERSITARIAS PARA TRABAJAR/ATENCION AL CLIENTE/ASESORA COMERCIAL. - Perfil profesional:buenas "
        "relaciones sociales, gusto e identificación con las labores comerciales de negocios; personalidad altamente "
        "emprendedora, 100% pro activo. Carácter extrovertido, alegre, dinámico, gran disposición para el aprendizaje "
        "y la formacion ejecutiva que tenga grandes ambiciones de progreso personal y profesional alta autoestima,gran "
        "fuerza mental. Perfil laboral;Desarrollar la labor de atención personalizada y asesoria comercial a "
        "ejecutivos, Profesionales y estudiantes,prospectos de usuarios a nuestro programa \"PREMIUM ENGLISH PROGRAM\", "
        "orientar, sus ventajas competitivas, su ingenieria de servicios, valores agregados, garantias, comodidades y "
        "el portafolio de posibilidades para el desarrollo del programa que otorga nuestra empresa. "
        "Beneficios;capacitación para ocupar el cargo,relaciones publicas, marketing personal, marketing "
        "empresarial,publicidad directa, publicidad masiva,promociones, asesoria comercial,comunicacion y clima "
        "organizacional.Contratación directa,beneficios de ley y beneficios corporativos, bonos, premios,comisiones,"
        "incentivos, basico, 1800 soles mensuales. Requisitos;full time, edad 18 a 30 años"
    ),
    "[Real Perú] Caso Mixto (Múltiples Sesgos) — Ayudante de Confección Textil (Computrabajo)": (
        "Señorita ayudante de confección textil - Con o sin Experiencia. Descripción de la oferta S/. 1.400,00 "
        "(Mensual) + Comisiones Contrato a Plazo Indeterminado Tiempo completo Empresa de confección textil ubicada "
        "en La Victoria busca señorita agraciada, dinámicas, sociable, alegre y con muchas ganas de aprender y "
        "desarrollarse en el sector textil. ¿Qué buscamos? SEÑORITA AYUDANTE DE CONFECCIÓN TEXTIL Empresa de "
        "confección textil ubicada en La Victoria – Gamarra, dedicada a la elaboración de prendas de vestir para dama "
        "y caballero (polos, poleras, polerones, casacas y moda en general), requiere señoritas con: - Actitud "
        "proactiva e iniciativa. - Dinamismo y responsabilidad. - Ganas de aprender y desarrollarse. - Interés por el "
        "rubro de confección textil. No es indispensable contar con experiencia. Brindamos inducción y capacitación "
        "para aprender las diferentes actividades del área de confección. No es indispensable contar con experiencia. "
        "Se brindará capacitación y acompañamiento. Ubicación: La Victoria, Gamarra: Educación Secundaria años de "
        "experiencia Edad: entre 19 y 25 años Conocimientos: Dinamismo Disponibilidad de viajar: Si Aptitudes "
        "asociadas a esta ofertaNuevo Dinamismo Palabras clave: asistente, auxiliar, aux, auxiliary, assistant, helper, "
        "aide 9 de noviembre (actualizada)"
    ),
    "[Real Perú] Neutro — Asistente Administrativo SAP (Computrabajo)": (
        "Asistente Administrativo SAP. Descripción de la oferta S/. 1.999,00 (Mensual) Contrato de Suplencia Tiempo "
        "completo Presencial y remoto ¡Estamos en búsqueda de un/a ASISTENTE ADMINISTRATIVO TEMPORAL! ¿Cuentas con "
        "experiencia en gestión administrativa y conocimientos de SAP? ¡Esta oportunidad es para ti! Nos encontramos "
        "en búsqueda de un/a Asistente Administrativo para brindar soporte a las áreas de Mantenimiento y Proyectos. "
        "¿Cuáles serán tus principales funciones? - Gestionar la generación, provisión y anulación de SOLPED y órdenes "
        "de compra (OC) para las áreas de Mantenimiento y Proyectos. - Realizar la generación y gestión de CAPEX para "
        "las áreas de Mantenimiento y Proyectos. - Coordinar y elaborar agendas de reuniones para el área de Proyectos. "
        "- Brindar soporte en las actividades administrativas de ambas áreas. ¿Qué buscamos? - Egresado/a de carrera "
        "técnica o universitaria de Administración. - Experiencia mínima de 1 año en funciones administrativas "
        "similares. - Conocimientos de SAP. - Manejo de Excel a nivel intermedio. - Organización, planificación y "
        "atención al detalle. ¿Qué ofrecemos? - Remuneración mensual: S/ 2,000. - Contrato temporal: 3 meses. "
        "- Modalidad híbrida: 3 días presenciales y 2 días remotos. - Horario: - Lunes a jueves: 8:00 a. m. a "
        "6:30 p. m. - Viernes: 8:00 a. m. a 4:30 p. m. - Lugar de trabajo: San Isidro ¡Postula y forma parte de esta "
        "oportunidad laboral! Si cumples con el perfil, esperamos tu postulación.: Técnico años de experiencia "
        "Palabras clave: ayudante, asistente, auxiliar, aux, auxiliary, assistant, helper, aide, administrative Hace "
        "12 minutos (actualizada)"
    ),
    "[Caso de prueba] Sesgo LGBTQ+ — Profesional en Auditoría Interna": (
        "Equipo corporativo del área de auditoría selecciona profesional para desempeñarse como profesional en "
        "auditoría interna. Requisitos excluyentes: dominio técnico comprobable y disponibilidad horaria para la "
        "modalidad del puesto. Es importante contar con residencia habilitada para trabajar en el país y disponibilidad "
        "inmediata. Requisitos técnicos: normas IIA, análisis de datos en SQL, SAP y reportes ejecutivos. Funciones "
        "principales: ejecutar pruebas de control interno, detectar riesgos y emitir recomendaciones. Para este rol "
        "buscamos preferencia por un perfil tradicional alineado a estándares clásicos de género. Formación técnica o "
        "universitaria vinculada al puesto y experiencia práctica."
    ),
}
