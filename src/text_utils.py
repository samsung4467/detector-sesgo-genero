"""Utilidades de normalización de texto en español preservando señales de sesgo."""

import re

# Tabla 1-a-1 para quitar tildes preservando exactamente las posiciones de los caracteres
_ACCENT_TRANS = str.maketrans(
    "áéíóúüñÁÉÍÓÚÜÑ",
    "aeiouunAEIOUUN",
)

# Stopwords funcionales mínimas en español:
# Intencionalmente NO eliminamos artículos/pronombres con marca de género ni negaciones
# ni palabras de exclusividad ('solo', 'sin', 'no', 'ni', 'unicamente', 'hombre', 'mujer', etc.)
SPANISH_MINIMAL_STOPWORDS = [
    "de",
    "en",
    "y",
    "a",
    "que",
    "por",
    "para",
    "del",
    "al",
    "se",
    "es",
    "son",
    "su",
    "sus",
    "o",
    "u",
    "este",
    "esta",
    "estos",
    "estas",
    "ese",
    "esa",
    "como",
    "pero",
    "donde",
    "cuando",
    "cual",
    "cuales",
    "tambien",
    "sobre",
    "desde",
    "hasta",
    "mediante",
    "segun",
    "durante",
    "hacia",
    "tras",
    "con",
    "muy",
    "mas",
    "menos",
    "aqui",
    "alli",
    "asi",
    "cada",
    "entre",
]


def fold_accents_1to1(text: str) -> str:
    """Elimina tildes y diéresis en español manteniendo una correspondencia 1-a-1 de índices."""
    return text.translate(_ACCENT_TRANS)


def normalize_for_ml(text: str) -> str:
    """Normaliza texto en español para vectorización TF-IDF."""
    if not isinstance(text, str):
        return ""
    text = fold_accents_1to1(text.lower())
    text = re.sub(r"[^a-z0-9+/%\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
