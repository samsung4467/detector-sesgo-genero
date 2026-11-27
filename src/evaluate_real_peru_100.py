"""Consolidación, limpieza, anonimización y evaluación de 100 avisos reales de empleo en Perú.

Combina:
- 40 avisos originales de Bumeran Perú (lote 1)
- 35 avisos de Computrabajo Perú
- 25 avisos adicionales de Bumeran Perú (lote 2)
Total exacto: 100 avisos reales únicos, sin modificar data/dataset_sesgos_empleo.csv (1,200 avisos).

Genera:
- data/ofertas_reales_peru.csv (columnas: id, portal, busqueda, puesto, texto)
- data/validacion_real_resultados.csv (predicciones de Random Forest + motor híbrido)
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.bias_analyzer import analyze_job_posting, load_model
from src.config import ALL_CATEGORIES, DATA_DIR

BUMERAN_DIR = Path(
    r"C:\Users\verde\.gemini\antigravity\brain\b00ec03b-26a8-4e1e-b542-3c5b15cece43\browser_sandbox\outputs"
)
COMPUTRABAJO_DIR = Path(
    r"C:\Users\verde\.gemini\antigravity\brain\4efe456c-9544-409f-ae2c-9545ae11ad51\browser_sandbox\outputs"
)

BUMERAN_BATCH_1: List[Tuple[str, str]] = [
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

COMPUTRABAJO_BATCHES: List[str] = [
    "step_12.json",
    "step_14.json",
    "step_16.json",
    "step_18.json",
    "step_20.json",
    "step_22.json",
    "step_24.json",
    "step_26.json",
    "step_28.json",
    "step_30.json",
]

BUMERAN_BATCH_2: List[str] = [
    "step_75.json",
    "step_77.json",
    "step_79.json",
    "step_81.json",
    "step_83.json",
    "step_85.json",
    "step_87.json",
    "step_89.json",
]

# Alias y marcas comerciales conocidas de los 100 avisos para asegurar anonimización total
EXTRA_BRAND_ALIASES = [
    # Lote 1 Bumeran
    "ManpowerGroup Perú", "ManpowerGroup", "Manpower",
    "Grupo RPP", "GRUPORPP", "RPP",
    "VIETTEL PERU", "Viettel", "Bitel",
    "GS1 PERU", "GS1 Perú", "GS1",
    "TIVIT Perú", "TIVIT",
    "Grupo Tawa", "Tawa",
    "FARENET", "Believing Talents", "ISP Consulting",
    "MEKA INMOBILIARIA", "Meka", "Software Enterprise Services",
    "Samara Spa", "Daga & Jimenez", "Daga y Jimenez",
    "Inversiones Ayme", "TVS Muebles", "CISTERNA DEL MAR",
    "GRUPO FORMART", "Formart", "Seleccion RH", "Selección RH",
    "THINK-E", "CONTROL SAC", "Panimiant", "COLD IMPORT",
    "instituto IFE", "IFE", "ISSAMODA", "PRECOTEX",
    "ARCH LATAM", "DECORACIONES CESR", "CESR", "Vascusalud",
    "AOM SUMINISTROS INDUSTRIALES", "AOM", "TDP CORP",
    "Nernet Automation Systems", "Nernet",
    "IDOM CONSULTING, ENGINEERING, ARCHITECTURE", "IDOM",
    "ESTUDIO S&M CASTILLO ABOGADOS", "S&M CASTILLO", "Castillo Abogados",
    "Andrea Ruiz", "Oscar Benavides",
    # Lote 2 Bumeran
    "2DAY", "ADN RETAIL", "AZERTA", "CREPIER",
    "Caja de Pensiones Militar Policial", "Chifa Nachito",
    "DISTRIBUIDORA LEXBOOKS", "Lexbooks",
    "ESCUELA SUDAMERICANA DE EXTENSION CULTURAL PARA PERU",
    "EY (Ernst & Young)", "Ernst & Young",
    "Exact", "HIRING 360", "KUELAP LOGISTICA EXPRESS", "Kuelap",
    "La Chinita Peru", "La Chinita",
    "Laboratorios Biomont", "Biomont", "Latin Glow Agency", "Latin Glow",
    "MACGA", "NEXIA STAFF", "Progelato", "SYNTHEC SOLUTIONS", "Synthec",
    "TELCOLEVANTE", "Telecenter",
    "Universidad Privada del Norte - Docentes", "Universidad Privada del Norte", "UPN",
    "WINNER TRADING",
    # Lote Computrabajo
    "Restaurante Cabaña Vista Alegre", "Cabaña Vista Alegre",
    "Adecco Perú", "Adecco", "BSG Institute",
    "CREDIALIANZA PERU", "Credialianza", "Concentrix",
    "ECA Arquitectos", "El Tridente",
    "Fonbienes Peru EAFC", "Fonbienes",
    "GRUPO FENCRI", "Fencri", "GRUPO NATIONAL", "Grupo Expro",
    "Lavoro Perú", "Lavoro", "LÓGICA CONSULTORÍA", "Logica Consultoria",
    "NORFELINDA ESPINOZA MORALES", "Overall Strategy", "Overall",
    "PETROLCENTER", "Partner Service", "Prosegur",
    "RECSA PERÚ", "RECSA", "SSP ventas del Peru", "SSP ventas",
    "TRINY RENTAL", "Total Talent Human Resources", "Total Talent",
    "Tándem Eje de soluciones Integradas", "Tandem Eje de soluciones Integradas",
    "YONAYA JACKETS", "Yonaya",
    # Marcas adicionales detectadas en el cuerpo de los avisos
    "Provefabrica", "PERU PACIFIC FOOD", "Bipay",
    "CASINO CONTTENTO HUÁNUCO", "CASINO CONTTENTO", "CONTTENTO",
    "El Tridente Sabor Divino", "El Tridente  Sabor Divino", "Sabor Divino",
    "JOHN HOLDEN", "Consultora AKTIVA", "AKTIVA", "VISTAMODA",
    "PARTNER SERVICE CLARO", "PARTNER SERVICE  CLARO",
]

NON_COMPANY_BADGES = {"empleo destacado", "empresa confidencial", "confidencial"}

LEGAL_SUFFIX_RE = re.compile(
    r"\b(?:S\.?A\.?C\.?|S\.?A\.?A\.?|S\.?A\.?|S\.?R\.?L\.?|E\.?I\.?R\.?L\.?|EAFC|"
    r"SUCURSAL\s+DEL\s+PER[UÚ]|PER[UÚ])\b",
    flags=re.IGNORECASE,
)


def _company_variants(empresa: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", (empresa or "")).strip()
    if not cleaned or cleaned.lower() in NON_COMPANY_BADGES:
        return []
    variants = [cleaned]
    base = LEGAL_SUFFIX_RE.sub("", cleaned)
    base = re.sub(r"[,.;:\-]+$", "", base).strip()
    base = re.sub(r"\s+", " ", base).strip()
    if base and len(base) >= 3 and base.lower() != cleaned.lower():
        variants.append(base)
    return variants


def anonymize_and_clean(text: str, empresa: str, all_company_names: List[str]) -> str:
    """Limpia HTML/ruido de portal y anonimiza empresas y datos de contacto."""
    if not text:
        return ""

    s = html.unescape(text)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[\u200b\ufeff\xa0]", " ", s)
    s = re.sub(r"[•●▪■►▶✔✓➤➢❖⭐🔹🔸📌📍💼📩📞🌟✨🔥📢]+", " ", s)

    # Limpiar ruido de interfaz de Bumeran / Computrabajo
    ui_noise = [
        r"^\s*Descripci[oó]n\s+del\s+puesto\s*:?\s*",
        r"\bPostularme\b",
        r"\bDenunciar\s+aviso\b",
        r"\bGuardar\s+aviso\b",
        r"\bCompartir\s+aviso\b",
        r"\bRequerimientos\s+Educaci[oó]n\s+m[ií]nima\b",
    ]
    for pat in ui_noise:
        s = re.sub(pat, " ", s, flags=re.IGNORECASE)

    # Anonimizar correos, URLs y teléfonos
    s = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "[CORREO]", s)
    s = re.sub(r"https?://\S+|www\.\S+", "[WEB]", s, flags=re.IGNORECASE)
    s = re.sub(r"(?:\+?51[\s-]?)?\b9\d{2}[\s-]?\d{3}[\s-]?\d{3}\b", "[TELEFONO]", s)

    targets = set(EXTRA_BRAND_ALIASES)
    for name in _company_variants(empresa):
        targets.add(name)
    for name in all_company_names:
        for v in _company_variants(name):
            targets.add(v)

    skip_short = {"sac", "srl", "eirl", "peru", "perú", "spa", "cyl", "sip", "ey", "exact", "control"}
    for target in sorted(targets, key=len, reverse=True):
        if len(target) < 3:
            continue
        low = target.lower()
        if low in skip_short:
            continue
        s = re.sub(rf"(?<!\w){re.escape(target)}(?!\w)", "[EMPRESA]", s, flags=re.IGNORECASE)

    # Casos cortos sensibles a mayúsculas (solo cuando coincide el aviso o siglas exactas)
    if empresa.strip().lower() == "cyl":
        s = re.sub(r"\bCYL\b", "[EMPRESA]", s)
    if empresa.strip().lower() == "sip":
        s = re.sub(r"\bSip\b|\bSIP\b", "[EMPRESA]", s)
    if "ey" in empresa.lower():
        s = re.sub(r"\bEY\b", "[EMPRESA]", s)
    if empresa.strip().lower().startswith("exact"):
        s = re.sub(r"\bExact\b", "[EMPRESA]", s)
    if "caja de pensiones" in empresa.lower():
        s = re.sub(r"\bCPMP\b", "[EMPRESA]", s)
    s = re.sub(r"\[EMPRESA\]\s+CLARO\b", "[EMPRESA]", s)

    # Anonimizar razones sociales peruanas restantes
    s = re.sub(
        r"\b(?:[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ0-9&.\-]{1,25}\s+){1,4}"
        r"(?:S\.?A\.?C\.?|S\.?A\.?A\.?|S\.?R\.?L\.?|E\.?I\.?R\.?L\.?|S\.A\.)\b",
        "[EMPRESA]",
        s,
    )

    # Limpiar repeticiones de [EMPRESA] y sufijos societarios sueltos sin comerse palabras siguientes
    s = re.sub(
        r"(?:\[EMPRESA\](?:\s*(?:-\s*\[EMPRESA\]|(?:EAFC\s+)?(?:S\.?A\.?C\.?|S\.?A\.?A\.?|S\.?R\.?L\.?|E\.?I\.?R\.?L\.?|S\.A\.?|EAFC|PER[UÚ])\b\.?))*\s*)+",
        "[EMPRESA] ",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"\s*\n+\s*", " ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    return s.strip()


def load_100_real_postings() -> List[Dict[str, str]]:
    """Carga exactamente 100 avisos únicos: 40 Bumeran (lote 1) + 35 Computrabajo + 25 Bumeran (lote 2)."""
    seen_urls = set()
    consolidated: List[Dict[str, str]] = []

    def _add(item: Dict[str, str], default_portal: str) -> bool:
        url = (item.get("url") or "").strip()
        if url and url in seen_urls:
            return False
        if url:
            seen_urls.add(url)
        rec = dict(item)
        rec["portal"] = rec.get("portal") or default_portal
        consolidated.append(rec)
        return True

    # 1. Los 40 avisos originales de Bumeran Perú
    for fn, key in BUMERAN_BATCH_1:
        payload = json.loads((BUMERAN_DIR / fn).read_text(encoding="utf-8"))
        for it in payload["value"][key]:
            _add(it, "Bumeran Perú")

    # 2. Los 35 avisos de Computrabajo Perú
    for fn in COMPUTRABAJO_BATCHES:
        payload = json.loads((COMPUTRABAJO_DIR / fn).read_text(encoding="utf-8"))
        for it in payload["value"]["results"]:
            _add(it, "Computrabajo Perú")

    # 3. 25 avisos adicionales de Bumeran Perú para llegar exactamente a 100
    for fn in BUMERAN_BATCH_2:
        if len(consolidated) >= 100:
            break
        payload = json.loads((BUMERAN_DIR / fn).read_text(encoding="utf-8"))
        for it in payload["value"]["results"]:
            if len(consolidated) >= 100:
                break
            _add(it, "Bumeran Perú")

    return consolidated[:100]


def main() -> None:
    raw = load_100_real_postings()
    assert len(raw) == 100, f"Se esperaban 100 avisos únicos, se obtuvieron {len(raw)}"

    all_companies = [
        p.get("empresa", "")
        for p in raw
        if p.get("empresa") and p.get("empresa", "").strip().lower() not in NON_COMPANY_BADGES
    ]

    ofertas_rows: List[Dict[str, str]] = []
    for idx, item in enumerate(raw, start=1):
        empresa = item.get("empresa", "")
        portal = item.get("portal", "Bumeran Perú")
        busqueda = (item.get("busqueda") or "").strip()
        puesto_clean = anonymize_and_clean(item.get("puesto", ""), empresa, all_companies)
        desc_clean = anonymize_and_clean(item.get("descripcion", ""), empresa, all_companies)

        if puesto_clean and not desc_clean.lower().startswith(puesto_clean.lower()[:25]):
            full_text = f"{puesto_clean}. {desc_clean}"
        else:
            full_text = desc_clean

        ofertas_rows.append(
            {
                "id": f"PERU-{idx:03d}",
                "portal": portal,
                "busqueda": busqueda,
                "puesto": puesto_clean,
                "texto": full_text,
            }
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ofertas_path = DATA_DIR / "ofertas_reales_peru.csv"
    df_ofertas = pd.DataFrame(ofertas_rows)[["id", "portal", "busqueda", "puesto", "texto"]]
    df_ofertas.to_csv(ofertas_path, index=False, encoding="utf-8-sig")

    # Evaluar los 100 avisos con Random Forest + motor híbrido
    rf_pipeline = load_model("random_forest")
    eval_rows: List[Dict[str, object]] = []

    for row in ofertas_rows:
        res = analyze_job_posting(row["texto"], model_key="random_forest", pipeline=rf_pipeline)
        eval_rows.append(
            {
                "id": row["id"],
                "portal": row["portal"],
                "busqueda": row["busqueda"],
                "puesto": row["puesto"],
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

    # Verificar que dataset sintético sigue intacto
    df_syn = pd.read_csv(DATA_DIR / "dataset_sesgos_empleo.csv")
    print(f"=== CONSOLIDADO GUARDADO: {ofertas_path} ({len(df_ofertas)} filas) ===")
    print(f"=== RESULTADOS GUARDADOS: {results_path} ({len(df_eval)} filas) ===")
    print(f"=== DATASET SINTETICO INTACTO: {len(df_syn)} filas ===")

    print("\n=== DISTRIBUCIÓN POR PORTAL ===")
    print(df_ofertas["portal"].value_counts().to_string())

    print("\n=== DISTRIBUCIÓN FINAL POR CATEGORÍA (HÍBRIDO + RANDOM FOREST) ===")
    counts_final = df_eval["categoria_final"].value_counts().reindex(ALL_CATEGORIES, fill_value=0)
    for cat, cnt in counts_final.items():
        print(f"  {cat:12s}: {cnt:3d} ({cnt/len(df_eval)*100:5.1f}%)")

    print("\n=== DISTRIBUCIÓN POR CATEGORÍA (ML CRUDO RANDOM FOREST) ===")
    counts_ml = df_eval["categoria_ml_cruda"].value_counts().reindex(ALL_CATEGORIES, fill_value=0)
    for cat, cnt in counts_ml.items():
        print(f"  {cat:12s}: {cnt:3d} ({cnt/len(df_eval)*100:5.1f}%)")

    print("\n=== DISTRIBUCIÓN NIVEL DE RIESGO ===")
    print(df_eval["nivel_riesgo"].value_counts().to_string())

    print("\n=== DISTRIBUCIÓN MODO DE DECISIÓN ===")
    print(df_eval["modo_decision"].value_counts().to_string())

    # Imprimir 2 ejemplos reales textuales (anonimizados) por cada categoría detectada
    print("\n=== 2 EJEMPLOS REALES POR CATEGORÍA DETECTADA ===")
    merged = df_eval.merge(df_ofertas[["id", "texto"]], on="id")
    for cat in ALL_CATEGORIES:
        sub = merged[merged["categoria_final"] == cat]
        if sub.empty:
            continue
        # Priorizar ejemplos representativos (con frases detectadas claras y buen largo)
        sub_sorted = sub.sort_values(by=["num_frases_sesgadas", "indice_inclusividad"], ascending=[False, True])
        if cat == "Neutro":
            sub_sorted = sub.sort_values(by=["indice_inclusividad"], ascending=[False])
        print(f"\n--- CATEGORÍA: {cat} (total={len(sub)}) ---")
        for _, r in sub_sorted.head(2).iterrows():
            excerpt = r["texto"][:460].strip()
            if len(r["texto"]) > 460:
                excerpt += "..."
            print(
                f"  [{r['id']}] ({r['portal']} | búsqueda: '{r['busqueda']}' | riesgo: {r['nivel_riesgo']})\n"
                f"    Puesto: {r['puesto']}\n"
                f"    Frases detectadas: {r['frases_sesgadas_detectadas'] or '(Ninguna)'}\n"
                f"    Texto anonimizado: {excerpt}\n"
            )


if __name__ == "__main__":
    main()
