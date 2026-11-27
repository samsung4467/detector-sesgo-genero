"""Limpieza, anonimización y evaluación de 40 avisos reales de Bumeran Perú.

Genera sin modificar el dataset sintético (data/dataset_sesgos_empleo.csv):
- data/bumeran_reales.csv (columnas: id, busqueda, puesto, fuente, texto)
- data/validacion_real_resultados.csv (predicciones de Random Forest + motor híbrido)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.bias_analyzer import analyze_job_posting, load_model
from src.config import ALL_CATEGORIES, DATA_DIR

BUMERAN_OUTPUTS_DIR = Path(
    r"C:\Users\verde\.gemini\antigravity\brain\b00ec03b-26a8-4e1e-b542-3c5b15cece43\browser_sandbox\outputs"
)

INPUT_BATCHES: List[Tuple[str, str]] = [
    ("step_22.json", "senoritas"),
    ("step_24.json", "buenaPresencia"),
    ("step_28.json", "results"),
    ("step_30.json", "results"),
    ("step_32.json", "results"),
    ("step_34.json", "results"),
    ("step_47.json", "results"),
    ("step_57.json", "results"),
    ("step_59.json", "results"),
    ("step_61.json", "results"),
    ("step_63.json", "results"),
]

# Alias y nombres comerciales conocidos que pueden aparecer abreviados dentro del cuerpo del aviso
EXTRA_BRAND_ALIASES = [
    "ManpowerGroup Perú",
    "ManpowerGroup",
    "Manpower",
    "Grupo RPP",
    "GRUPORPP",
    "RPP",
    "VIETTEL PERU",
    "Viettel",
    "Bitel",
    "GS1 PERU",
    "GS1 Perú",
    "GS1",
    "TIVIT Perú",
    "TIVIT",
    "Grupo Tawa",
    "Tawa",
    "FARENET",
    "Believing Talents",
    "ISP Consulting",
    "MEKA INMOBILIARIA",
    "Meka",
    "Software Enterprise Services",
    "Samara Spa",
    "Daga & Jimenez",
    "Daga y Jimenez",
    "Inversiones Ayme",
    "TVS Muebles",
    "CISTERNA DEL MAR",
    "GRUPO FORMART",
    "Formart",
    "Seleccion RH",
    "Selección RH",
    "THINK-E",
    "CONTROL SAC",
    "Panimiant",
    "COLD IMPORT",
    "instituto IFE",
    "IFE",
    "ISSAMODA",
    "PRECOTEX",
    "ARCH LATAM",
    "DECORACIONES CESR",
    "CESR",
    "Vascusalud",
    "AOM SUMINISTROS INDUSTRIALES",
    "AOM",
    "TDP CORP",
    "Nernet Automation Systems",
    "Nernet",
    "IDOM CONSULTING, ENGINEERING, ARCHITECTURE",
    "IDOM",
    "ESTUDIO S&M CASTILLO ABOGADOS",
    "S&M CASTILLO",
    "Castillo Abogados",
    "Andrea Ruiz",
    "Oscar Benavides",
]

LEGAL_SUFFIX_RE = re.compile(
    r"\b(?:S\.?A\.?C\.?|S\.?A\.?A\.?|S\.?A\.?|S\.?R\.?L\.?|E\.?I\.?R\.?L\.?|SUCURSAL\s+DEL\s+PER[UÚ]|PER[UÚ])\b",
    flags=re.IGNORECASE,
)


def _company_variants(empresa: str) -> List[str]:
    """Genera variantes de un nombre de empresa para anonimizarlo completamente."""
    variants: List[str] = []
    cleaned = re.sub(r"\s+", " ", (empresa or "")).strip()
    if not cleaned:
        return variants
    variants.append(cleaned)
    base = LEGAL_SUFFIX_RE.sub("", cleaned)
    base = re.sub(r"[,.;:\-]+$", "", base).strip()
    base = re.sub(r"\s+", " ", base).strip()
    if base and len(base) >= 3 and base.lower() != cleaned.lower():
        variants.append(base)
    return variants


def anonymize_and_clean(text: str, empresa: str, all_company_names: List[str]) -> str:
    """Limpia formato/ruido de scraping y anonimiza nombres de empresas y datos de contacto."""
    if not text:
        return ""

    # Normalizar saltos de línea, viñetas y espacios
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[\u200b\ufeff\xa0]", " ", s)
    s = re.sub(r"[•●▪■►▶✔✓➤➢❖⭐🔹🔸📌📍💼📩📞]+", " ", s)

    # Eliminar posibles cabeceras/botones residuales de UI del portal
    ui_noise = [
        r"^\s*Descripci[oó]n\s+del\s+puesto\s*:?\s*",
        r"\bPostularme\b",
        r"\bDenunciar\s+aviso\b",
        r"\bGuardar\s+aviso\b",
        r"\bCompartir\s+aviso\b",
    ]
    for pat in ui_noise:
        s = re.sub(pat, " ", s, flags=re.IGNORECASE)

    # Anonimizar correos electrónicos, URLs y teléfonos peruanos
    s = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "[CORREO]", s)
    s = re.sub(r"https?://\S+|www\.\S+", "[WEB]", s, flags=re.IGNORECASE)
    s = re.sub(r"(?:\+?51[\s-]?)?\b9\d{2}[\s-]?\d{3}[\s-]?\d{3}\b", "[TELEFONO]", s)

    # Anonimizar nombre de la empresa del propio aviso + todas las empresas/alias del lote
    targets = set(EXTRA_BRAND_ALIASES)
    for name in _company_variants(empresa):
        targets.add(name)
    for name in all_company_names:
        for v in _company_variants(name):
            targets.add(v)

    # Ordenar de más largo a más corto para reemplazar nombres completos antes que cortos
    for target in sorted(targets, key=len, reverse=True):
        if len(target) < 3:
            continue
        # Evitar falsos reemplazos en palabras cortas comunes
        if target.lower() in {"sac", "srl", "eirl", "peru", "perú", "spa", "cyl"}:
            if target.lower() == "cyl" and empresa.strip().lower() == "cyl":
                s = re.sub(r"\bCYL\b", "[EMPRESA]", s)
            continue
        s = re.sub(rf"(?<!\w){re.escape(target)}(?!\w)", "[EMPRESA]", s, flags=re.IGNORECASE)

    # Anonimizar cualquier otra razón social peruana restante ("Nombre Propio S.A.C.")
    s = re.sub(
        r"\b(?:[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ0-9&.\-]{1,25}\s+){1,4}"
        r"(?:S\.?A\.?C\.?|S\.?A\.?A\.?|S\.?R\.?L\.?|E\.?I\.?R\.?L\.?|S\.A\.)\b",
        "[EMPRESA]",
        s,
    )

    # Limpiar posibles repeticiones de "[EMPRESA]" y espacios múltiples
    s = re.sub(r"(?:\[EMPRESA\]\s*(?:S\.?A\.?C\.?|S\.?A\.?|S\.?R\.?L\.?|E\.?I\.?R\.?L\.?|PER[UÚ])?\s*)+", "[EMPRESA] ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*\n+\s*", " ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    return s.strip()


def load_raw_postings() -> List[Dict[str, str]]:
    """Carga y deduplica los 40 avisos reales desde los archivos JSON del navegador."""
    postings: List[Dict[str, str]] = []
    seen_urls = set()

    for filename, key in INPUT_BATCHES:
        path = BUMERAN_OUTPUTS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el archivo extraído: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload["value"][key]
        for item in items:
            url = (item.get("url") or "").strip()
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            postings.append(item)

    return postings


def main() -> None:
    raw_postings = load_raw_postings()
    all_companies = [p.get("empresa", "") for p in raw_postings if p.get("empresa")]

    bumeran_rows: List[Dict[str, str]] = []
    for idx, item in enumerate(raw_postings, start=1):
        empresa = item.get("empresa", "")
        busqueda = (item.get("busqueda") or "").strip()
        puesto_clean = anonymize_and_clean(item.get("puesto", ""), empresa, all_companies)
        desc_clean = anonymize_and_clean(item.get("descripcion", ""), empresa, all_companies)

        # Incluir el título al inicio del texto si no aparece ya al comienzo de la descripción
        if puesto_clean and not desc_clean.lower().startswith(puesto_clean.lower()[:25]):
            full_text = f"{puesto_clean}. {desc_clean}"
        else:
            full_text = desc_clean

        bumeran_rows.append(
            {
                "id": f"BUM-{idx:03d}",
                "busqueda": busqueda,
                "puesto": puesto_clean,
                "fuente": "Bumeran Perú",
                "texto": full_text,
            }
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    reales_path = DATA_DIR / "bumeran_reales.csv"
    df_reales = pd.DataFrame(bumeran_rows)[["id", "busqueda", "puesto", "fuente", "texto"]]
    df_reales.to_csv(reales_path, index=False, encoding="utf-8-sig")

    # Evaluar los 40 avisos con nuestro modelo Random Forest + detector híbrido
    rf_pipeline = load_model("random_forest")
    eval_rows: List[Dict[str, object]] = []

    for row in bumeran_rows:
        res = analyze_job_posting(row["texto"], model_key="random_forest", pipeline=rf_pipeline)
        eval_rows.append(
            {
                "id": row["id"],
                "busqueda": row["busqueda"],
                "puesto": row["puesto"],
                "fuente": row["fuente"],
                "categoria_final": res.predicted_class,
                "categoria_ml_cruda": res.raw_ml_category,
                "modo_decision": res.decision_mode,
                "confiabilidad_ml": res.ml_reliability_label,
                "cobertura_vocabulario_pct": round(res.vocabulary_coverage * 100.0, 1),
                "entropia_ml": round(res.ml_entropy, 3),
                "prob_masculino": round(res.probabilities["Masculino"], 4),
                "prob_femenino": round(res.probabilities["Femenino"], 4),
                "prob_excluyente": round(res.probabilities["Excluyente"], 4),
                "prob_lgbtq": round(res.probabilities["LGBTQ+"], 4),
                "prob_demografico": round(res.probabilities["Demográfico"], 4),
                "prob_neutro": round(res.probabilities["Neutro"], 4),
                "indice_inclusividad": res.inclusivity_score,
                "nivel_riesgo": res.risk_level,
                "num_frases_sesgadas": len(res.spans),
                "categorias_lexicas": ", ".join(res.detected_categories),
                "frases_sesgadas_detectadas": " | ".join(
                    f"[{s.category}] {s.text}" for s in res.spans
                ),
                "requisitos_tecnicos": ", ".join(res.technical_terms),
            }
        )

    results_path = DATA_DIR / "validacion_real_resultados.csv"
    df_eval = pd.DataFrame(eval_rows)
    df_eval.to_csv(results_path, index=False, encoding="utf-8-sig")

    print(f"Avisos reales guardados en: {reales_path} (total={len(df_reales)})")
    print(f"Resultados de validación guardados en: {results_path} (total={len(df_eval)})")
    print("\n=== DISTRIBUCIÓN POR BÚSQUEDA ===")
    print(df_reales["busqueda"].value_counts().to_string())
    print("\n=== DISTRIBUCIÓN FINAL (DETECTOR HÍBRIDO + RANDOM FOREST) ===")
    final_counts = df_eval["categoria_final"].value_counts().reindex(ALL_CATEGORIES, fill_value=0)
    print(final_counts.to_string())
    print("\n=== DISTRIBUCIÓN PREDICCIÓN ML CRUDA (RANDOM FOREST) ===")
    ml_counts = df_eval["categoria_ml_cruda"].value_counts().reindex(ALL_CATEGORIES, fill_value=0)
    print(ml_counts.to_string())
    print("\n=== DISTRIBUCIÓN NIVEL DE RIESGO ===")
    print(df_eval["nivel_riesgo"].value_counts().to_string())
    print("\n=== DISTRIBUCIÓN MODO DE DECISIÓN ===")
    print(df_eval["modo_decision"].value_counts().to_string())
    print("\n=== DETALLE POR AVISO ===")
    for _, r in df_eval.iterrows():
        print(
            f"{r['id']} | [{r['busqueda']:19s}] | Final={r['categoria_final']:11s} "
            f"| ML={r['categoria_ml_cruda']:11s} | Riesgo={r['nivel_riesgo']:16s} "
            f"| Spans={r['num_frases_sesgadas']} | {r['puesto'][:38]} | {r['frases_sesgadas_detectadas'][:90]}"
        )


if __name__ == "__main__":
    main()
