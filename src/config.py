"""Configuración global, categorías de sesgo, paleta visual y ejemplos precargados."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

DATASET_PATH = DATA_DIR / "dataset_sesgos_empleo.csv"
METRICS_JSON_PATH = REPORTS_DIR / "model_metrics.json"
METRICS_SUMMARY_CSV_PATH = REPORTS_DIR / "model_metrics_summary.csv"
PER_CLASS_CSV_PATH = REPORTS_DIR / "per_class_metrics.csv"
CONFUSION_CSV_PATH = REPORTS_DIR / "confusion_matrices.csv"
BEST_MODEL_META_PATH = MODELS_DIR / "best_model_meta.json"

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
    "svm": "SVM",
    "decision_tree": "Decision Tree",
}

PRELOADED_EXAMPLES = {
    "Sesgo Masculino — Desarrollador Backend Senior": (
        "Buscamos un desarrollador backend senior que sea un verdadero tiburón de la tecnología y "
        "un hombre de acción para liderar nuestra arquitectura cloud. Necesitamos un ingeniero aguerrido, "
        "competitivo y con liderazgo dominante, capaz de imponer decisiones técnicas bajo presión. "
        "Requisitos técnicos: 4 años de experiencia con Python 3.11+, FastAPI, PostgreSQL, Docker, Kubernetes y AWS (ECS/Lambda). "
        "Conocimiento sólido de arquitectura de microservicios, CI/CD con GitHub Actions y pruebas automatizadas con Pytest. "
        "Si eres un rockstar ambicioso con mentalidad ganadora, postúlate hoy mismo."
    ),
    "Sesgo Femenino — Asistente Ejecutiva y Atención": (
        "Importante firma consultora solicita una señorita amable y carismática para el cargo de secretaria ejecutiva. "
        "Buscamos una mujer delicada, dulce y servicial, con excelente presencia femenina y sonrisa cálida "
        "para brindar un trato maternal y acogedor a nuestros directivos y clientes VIP. "
        "Requisitos técnicos: manejo avanzado de Microsoft Excel (tablas dinámicas y macros), ERP SAP módulo administrativo, "
        "CRM Salesforce, gestión de agendas corporativas e inglés nivel B2 comprobable. "
        "Se valora una asistente femenina siempre dispuesta a complacer las necesidades de la gerencia."
    ),
    "Sesgo Excluyente — Supervisor de Operaciones Logísticas": (
        "Empresa industrial requiere cubrir vacante de supervisor de planta. Requisito excluyente: sexo masculino, "
        "abstenerse mujeres debido a las exigencias operativas del turno nocturno. "
        "Se descartan postulantes con hijos pequeños o responsabilidades familiares que limiten la disponibilidad total; "
        "puesto no apto para mujeres embarazadas. "
        "Requisitos técnicos: ingeniería industrial o afín, 3 años en gestión de cadena de suministro, dominio de SAP WM, "
        "Power BI, normas ISO 9001, Lean Manufacturing y control de KPIs logísticos. "
        "Ofrecemos contrato indefinido y bono por cumplimiento de metas."
    ),
    "Sesgo LGBTQ+ — Ejecutivo de Cuentas Corporativas": (
        "Firma financiera tradicional selecciona ejecutivo de cuentas corporativas para nuestra casa matriz. "
        "Es indispensable que el personal se ajuste a valores tradicionales de familia y mantenga un aspecto "
        "estrictamente acorde a su sexo biológico. Exigimos vestimenta estrictamente masculina o femenina según DNI "
        "y abstenerse personas con expresión de género no convencional o rasgos andróginos. "
        "Requisitos técnicos: licenciatura en Finanzas o Administración, experiencia comprobable en banca corporativa, "
        "manejo de CRM HubSpot, análisis de riesgo crediticio, modelado financiero en Excel y nivel de inglés C1. "
        "El nombre legal obligatorio debe coincidir con el sexo de nacimiento en toda documentación interna."
    ),
    "Sesgo Demográfico — Analista de Marketing Digital": (
        "Agencia digital en crecimiento busca analista de marketing digital joven, con edad entre 21 y 28 años "
        "(personas mayores de 30 años abstenerse). Preferimos perfil soltero sin hijos y de nacionalidad local excluyente, "
        "que resida en zona norte de la ciudad. Requisito indispensable: excelente presencia física, adjuntar foto de cuerpo entero en el CV. "
        "Requisitos técnicos: dominio de Google Analytics 4, Meta Ads, Google Ads, SQL básico, Looker Studio, "
        "estrategias SEO/SEM y automatización de campañas con Mailchimp o Klaviyo. "
        "Ofrecemos modalidad híbrida y capacitaciones certificadas."
    ),
    "Caso Mixto (Múltiples Sesgos) — Jefe Comercial B2B": (
        "Buscamos un jefe comercial B2B, caballero con carácter fuerte y espíritu agresivo para conquistar cuentas enterprise. "
        "Solo candidatos varones de entre 28 y 38 años, preferiblemente solteros y sin cargas familiares. "
        "Necesitamos un guerrero de ventas ambicioso con excelente presencia física para representar a la marca; "
        "abstenerse mujeres y personas mayores de 40 años. "
        "Requisitos técnicos: 5 años de experiencia en ventas SaaS B2B, gestión de pipeline en Salesforce y HubSpot, "
        "elaboración de forecasting comercial, negociación de contratos SLA e inglés avanzado (C1). "
        "Salario base competitivo más esquema de comisiones sin tope."
    ),
    "Oferta Neutra e Inclusiva — Especialista en Ciencia de Datos": (
        "Nuestro equipo de analítica busca una persona especialista en Ciencia de Datos para diseñar modelos predictivos "
        "orientados a la toma de decisiones de negocio. Quien ocupe el puesto colaborará con equipos multidisciplinarios "
        "de ingeniería y producto en la construcción de pipelines reproducibles. "
        "Requisitos técnicos: experiencia comprobable de al menos 3 años en Python (Pandas, Scikit-learn, PyTorch), "
        "SQL avanzado, visualización de datos en Power BI o Tableau, control de versiones con Git y despliegue en AWS o GCP. "
        "Promovemos un entorno laboral diverso e inclusivo con igualdad de oportunidades para todas las personas."
    ),
}
