"""Consolidación de 800 avisos reales de Perú (300 Computrabajo + 300 Bumeran + 200 LinkedIn),
fusión a 2,000 registros (1,200 sintéticos + 800 reales), exportación Excel y reentrenamiento comprimido.

Conserva intactos:
- data/dataset_sesgos_empleo.csv (1,200 sintéticos)
- data/ofertas_reales_peru_200_backup.csv (respaldo íntegro de los 200 reales previos)
- Los 200 avisos reales previos (PERU-001..PERU-200) como base de data/ofertas_reales_peru.csv (800)

Genera:
- data/ofertas_reales_peru.csv (800 avisos reales únicos: 300 Computrabajo + 300 Bumeran + 200 LinkedIn)
- data/validacion_real_resultados.csv y data/validacion_real_resultados.xlsx (800 avisos evaluados)
- data/dataset_consolidado_2000.csv y data/dataset_consolidado_2000.xlsx (1,200 sintéticos + 800 reales)
- models/rf_model.joblib (comprimido compress=3, < 15 MB) y reports/*
"""

from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd

from src.bias_analyzer import (
    _select_dominant_lexical_category,
    _span_weight,
    analyze_job_posting,
    detect_biased_spans,
    load_model,
)
from src.config import (
    ALL_CATEGORIES,
    CONSOLIDATED_2000_EXCEL_PATH,
    CONSOLIDATED_2000_PATH,
    DATA_DIR,
    DATASET_PATH,
    REAL_PERU_DATASET_PATH,
    RF_MODEL_PATH,
)
from src.evaluate_real_peru_100 import NON_COMPANY_BADGES, _company_variants
from src.evaluate_real_peru_200 import NEW_BRAND_ALIASES, SKIP_COMMON_WORDS
from src.train_models import train_and_evaluate

RAW_600_DIR = Path(
    r"C:\Users\verde\.gemini\antigravity\brain\8c6c8e80-06b2-44e5-a8d8-62ebb1c9ed98\scratch\raw_600"
)
BACKUP_200_PATH = DATA_DIR / "ofertas_reales_peru_200_backup.csv"

ORIGEN_SINTETICO = "Sintético (1200)"
ORIGEN_REAL_PERU = "Real Perú (800)"

SYNTH_RISK_MAP = {
    "Alto": "Alto",
    "Moderado": "Moderado",
    "Limítrofe": "Moderado",
    "Ninguno": "Bajo (Inclusivo)",
}

EXTRA_BRANDS_800: List[str] = list(NEW_BRAND_ALIASES) + [
    "Medifarma", "VINIFAN", "BairesDev", "BCP", "Interbank", "BBVA", "Scotiabank",
    "Rimac", "Rímac", "Alicorp", "Gloria", "Backus", "Belcorp", "Yanbal", "Natura",
    "Falabella", "Saga Falabella", "Ripley", "Oechsle", "Plaza Vea", "Tottus", "Metro",
    "Wong", "Makro", "Sodimac", "Promart", "Maestro", "Tambo", "Oxxo", "Entel",
    "Movistar", "Telefónica", "Claro", "Win", "Latam Airlines", "SKY Airline",
    "Rappi", "PedidosYa", "Yape", "Plin", "Niubiz", "Izipay", "Culqi", "Kavak",
    "Globant", "NTT DATA", "Accenture", "Deloitte", "PwC", "KPMG", "Stefanini",
    "Indra", "Minsait", "IBM", "Oracle", "SAP", "Teleperformance", "Konecta",
    "Webhelp", "Majorel", "Atento", "Dynamicall", "Fortel", "MDY", "GSS",
    "Sodexo", "Securitas", "Liderman", "Hermes", "Eulen", "ISS", "Sapia",
    "Ferreyros", "Ferreycorp", "Unacem", "Cementos Pacasmayo", "Antamina",
    "Southern Peru", "Buenaventura", "Cerro Verde", "Las Bambas", "Minsur",
    "Química Suiza", "Inkafarma", "Mifarma", "Auna", "Clínica Internacional",
    "Clínica Ricardo Palma", "Clínica San Pablo", "Sanna", "Pacífico Seguros",
    "Mapfre", "La Positiva", "Credicorp", "MiBanco", "Caja Arequipa", "Caja Huancayo",
    "Caja Piura", "Compartamos", "BanBif", "Banco Pichincha", "Banco GNB",
    "UTP", "UPC", "UCV", "USIL", "ESAN", "PUCP", "Certus", "Toulouse Lautrec",
    "Idat", "Zegel", "Cibertec", "Senati", "Tecsup", "Británico", "ICPNA",
    "Dinet", "Ransa", "Talma", "APM Terminals", "DP World", "Scharff", "Olva Courier",
    "Shalom", "Cruz del Sur", "Oltursa", "Civa", "Supermercados Peruanos",
    "Cencosud", "InRetail", "Intercorp", "Grupo Romero", "Grupo Breca", "Grupo Gloria",
    "Cineplanet", "Cinemark", "Bembos", "Pardos Chicken", "Norky's", "Roky's",
    "Popeyes", "Papa John's", "Pizza Hut", "KFC", "Burger King", "Starbucks",
    "Chili's", "Tanta", "Madam Tusan", "La Lucha", "Don Belisario",
    "Footloose", "Bata", "Platanitos", "Topitop", "Renzo Costa", "Marathon",
    "Coolbox", "Hiraoka", "La Curacao", "Carsa", "Elektra", "Aruma",
    "La Leña", "Leña y Carbón", "Valtx", "Corporación Jeruth", "Anddes",
    "Progestión Perú", "Grupo Progestión", "Apparka", "Arca Continental",
    "Coca-Cola", "Veolia", "San Fernando", "Corporación Rey", "KO Asian Kitchen",
    "Primos Chicken Bar", "Gladcon", "Innova Schools", "SMU Perú",
    "Farmacias Peruanas", "Maersk", "HELPNET GROUP PERÚ", "HELPNET GROUP",
    "GrupoExpro", "Montalvo Institute", "Santander", "Bancom", "Grupo GCA",
    "TeamRECSA", "RECSA PERÚ", "Lavoro Perú", "Caja Metropolitana",
    "Manpower Perú", "ManpowerGroup", "Manpower", "Megapack Group",
    "SUN PHARMA", "Clínica Delgado", "Grupo Doktuz", "Adecco Perú", "Adecco",
    "Super Food Holding", "plazaVea", "Vivanda", "Oslo Logistics", "Merkao",
    "Agora", "GS1 Perú", "Universidad Continental", "familia Continental",
    "Grupo Prestamype", "Prestamype", "Grupo Flesan", "Flesan",
    "Gyu Grill House", "Kimberly-Clark", "Kleenex", "Andrex", "Cottonelle",
    "WypAll", "GoSEK", "San Miguel Industrias", "Corporación La Vitrina",
    "Factio Qualitas", "Fima Industrial", "Tomocorp", "Recreativos Fargo",
    "Inmobiliaria Vivaldi", "Vivaldi", "Grupo Cónica", "ROCK DRILL",
    "DITERSA", "Amazon Group", "EMASA PERÚ", "VámonosYa", "BPO Plus",
    "Qúma Soluciones Gastronomicas", "Bitel", "CL Selection", "Volcan",
    "Bureau Veritas Perú", "Bureau Veritas", "Pacasmayo", "TP Perú",
    "Grupo Valle Norte", "SIGNIA", "Arte Kfe", "Ambaar",
]

EXTENDED_SKIP_WORDS: Set[str] = set(SKIP_COMMON_WORDS) | {
    "claro", "win", "metro", "sap", "iss", "bata", "tanta", "plaza", "papa",
    "peru", "lima", "surco", "miraflores", "san isidro", "ate", "callao",
    "confidencial", "empresa confidencial", "importante empresa", "empleo destacado",
    "retail", "alimentos", "salud", "educacion", "educación", "tecnologia",
    "tecnología", "finanzas", "banca", "construccion", "construcción", "mineria",
    "minería", "energia", "energía", "telecomunicaciones", "transporte",
    "industria", "comercio", "consultora", "estudio", "asociacion", "asociación",
    "fundacion", "fundación", "instituto", "colegio", "universidad", "hospital",
    "farmacia", "botica", "supermercado", "tienda", "agencia", "banco", "caja",
    "seguros", "club", "casa", "cafe", "café", "bar", "polleria", "pollería",
    "cevicheria", "cevichería", "panaderia", "panadería", "pasteleria", "pastelería",
    "ferreteria", "ferretería", "textil", "moda", "calzado", "automotriz",
    "inmobiliaria", "constructora", "ingenieria", "ingeniería", "arquitectura",
    "sistemas", "software", "digital", "marketing", "publicidad", "eventos",
    "produccion", "producción", "operaciones", "almacen", "almacén", "ventas",
}


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()


def _norm_url(u: str) -> str:
    u = (u or "").strip().lower()
    u = u.split("?")[0].split("#")[0].rstrip("/")
    return u


def _word_shingles(norm_s: str, k: int = 5) -> Set[str]:
    words = norm_s.split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(0, min(len(words) - k + 1, 160))}


def _is_near_duplicate(
    shingles: Set[str],
    accepted_shingles: List[Set[str]],
    threshold: float = 0.78,
) -> bool:
    if not shingles or len(shingles) < 8:
        return False
    s_len = len(shingles)
    for prev in accepted_shingles:
        p_len = len(prev)
        # Rápido filtro de longitud para evitar comparaciones innecesarias
        if min(s_len, p_len) / max(s_len, p_len) < 0.55:
            continue
        inter = len(shingles & prev)
        if inter == 0:
            continue
        jaccard = inter / (s_len + p_len - inter)
        if jaccard >= threshold:
            return True
    return False


def anonymize_and_clean_800(text: str, empresa: str, all_company_names: List[str]) -> str:
    """Limpia HTML/ruido de portales y reemplaza empresas, correos, URLs y teléfonos por [EMPRESA]."""
    if not text:
        return ""

    s = html.unescape(text)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[\u200b\ufeff\xa0]", " ", s)
    s = re.sub(r"[•●▪■►▶✔✓✅➤➢❖⭐🔹🔸📌📍💼📩📞🌟✨🔥📢📊🕒🔎📦📲🚀🎯🏆💡🤝]+", " ", s)

    ui_noise = [
        r"^\s*Descripci[oó]n\s+del\s+puesto\s*:?\s*",
        r"\bPostularme\b",
        r"\bDenunciar\s+aviso\b",
        r"\bGuardar\s+aviso\b",
        r"\bCompartir\s+aviso\b",
        r"\bRequerimientos\s+Educaci[oó]n\s+m[ií]nima\b",
        r"\bMostrar\s+m[aá]s\b",
        r"\bMostrar\s+menos\b",
    ]
    for pat in ui_noise:
        s = re.sub(pat, " ", s, flags=re.IGNORECASE)

    # Reemplazar correos (normales, enmascarados con *, dominios @dom.com y ofuscados), URLs y teléfonos
    s = re.sub(r"[\w.*+-]+@[\w-]+\.[\w.-]+\b", "[EMPRESA]", s)
    s = re.sub(r"(?<!\w)@[\w-]+\.[a-z]{2,}\b", "[EMPRESA]", s, flags=re.IGNORECASE)
    s = re.sub(
        r"\b[\w.+-]+\s*\(\s*arroba\s*\)\s*[\w.-]+\s*\.\s*[a-z]{2,}\b",
        "[EMPRESA]",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"https?://\S+|www\.\S+", "[EMPRESA]", s, flags=re.IGNORECASE)
    s = re.sub(r"(?:\+?51[\s-]?)?\b9(?:\s*\d){8}\b", "[EMPRESA]", s)

    # Construir conjunto seguro de nombres de empresa a anonimizar
    targets: Set[str] = set()
    for name in _company_variants(empresa):
        targets.add(name)
    for brand in EXTRA_BRANDS_800:
        targets.add(brand)

    emp_low = (empresa or "").strip().lower()
    for target in sorted(targets, key=len, reverse=True):
        if len(target) < 4:
            continue
        low = target.lower().strip()
        if low in EXTENDED_SKIP_WORDS:
            continue
        # Para palabras únicas cortas (<=6), solo reemplazar si coincide con la empresa del aviso
        if " " not in low and len(low) <= 6 and low not in emp_low:
            continue
        s = re.sub(rf"(?<!\w){re.escape(target)}(?!\w)", "[EMPRESA]", s, flags=re.IGNORECASE)

    # Casos propios sensibles a mayúsculas (siglas/marcas cortas de Perú)
    s = re.sub(
        r"\b(?:BCP|BBVA|UTP|UPC|UCV|USIL|ESAN|PUCP|ICPNA|KORI|GEA|LUKERS|CPMP|MDY|GSS|IBM|EY|"
        r"GCA|SMI|ZICSA|UNICON|RECSA|MODASA|SSAYS|BRECA|MBC|LLAMK|AUNA|INPAS|BARFI|GEOMIX|SPSA|CRBC|JRC)\b",
        "[EMPRESA]",
        s,
    )
    if emp_low in ("claro", "win", "metro", "bata", "tanta", "indigo"):
        s = re.sub(rf"\b{re.escape(empresa.strip())}\b", "[EMPRESA]", s)

    # Anonimizar razones sociales peruanas restantes (ej. "ALGO PERU S.A.C.")
    s = re.sub(
        r"\b(?:[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ0-9&.\-]{1,25}\s+){1,4}"
        r"(?:S\.?A\.?C\.?|S\.?A\.?A\.?|S\.?R\.?L\.?|E\.?I\.?R\.?L\.?|S\.A\.)\b",
        "[EMPRESA]",
        s,
    )

    # Limpiar repeticiones de [EMPRESA] y sufijos societarios sueltos
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


def _load_folder_steps(folder: Path, default_portal: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if not folder.exists():
        return items
    for f in sorted(folder.glob("step_*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        val = d.get("value")
        candidates = []
        if isinstance(val, dict):
            for v in val.values():
                if isinstance(v, list):
                    candidates.extend(v)
        elif isinstance(val, list):
            candidates.extend(val)
        for it in candidates:
            if not isinstance(it, dict):
                continue
            desc = str(it.get("descripcion") or it.get("texto") or "").strip()
            if len(desc) < 120:
                continue
            rec = dict(it)
            rec["portal"] = rec.get("portal") or default_portal
            rec["descripcion"] = desc
            items.append(rec)
    return items


def load_all_new_candidates() -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Carga todos los candidatos nuevos de Computrabajo, Bumeran y LinkedIn."""
    comp_items = (
        _load_folder_steps(RAW_600_DIR / "comp1", "Computrabajo Perú")
        + _load_folder_steps(RAW_600_DIR / "comp3", "Computrabajo Perú")
        + _load_folder_steps(RAW_600_DIR / "comp2", "Computrabajo Perú")
        + _load_folder_steps(RAW_600_DIR / "comp4", "Computrabajo Perú")
    )
    bum_items = (
        _load_folder_steps(RAW_600_DIR / "bum1", "Bumeran Perú")
        + _load_folder_steps(RAW_600_DIR / "bum3", "Bumeran Perú")
        + _load_folder_steps(RAW_600_DIR / "bum2", "Bumeran Perú")
        + _load_folder_steps(RAW_600_DIR / "bum4", "Bumeran Perú")
    )
    li_path = RAW_600_DIR / "linkedin_lima.json"
    li_items: List[Dict[str, str]] = []
    if li_path.exists():
        raw_li = json.loads(li_path.read_text(encoding="utf-8"))
        for it in raw_li:
            if isinstance(it, dict) and len(str(it.get("descripcion") or "")) >= 150:
                rec = dict(it)
                rec["portal"] = "LinkedIn Perú"
                li_items.append(rec)
    return comp_items, bum_items, li_items


def select_unique_portal_batch(
    candidates: List[Dict[str, str]],
    target_count: int,
    portal_name: str,
    seen_urls: Set[str],
    seen_sigs: Set[str],
    accepted_shingles: List[Set[str]],
    all_company_names: List[str],
) -> List[Dict[str, str]]:
    """Selecciona `target_count` avisos únicos verificando URL, título y similitud de texto."""
    selected: List[Dict[str, str]] = []
    seen_title_body: Set[str] = set()

    for item in candidates:
        if len(selected) >= target_count:
            break
        u_norm = _norm_url(item.get("url", ""))
        if u_norm and u_norm in seen_urls:
            continue

        empresa = str(item.get("empresa") or "").strip()
        puesto_raw = str(item.get("puesto") or "").strip()
        desc_raw = str(item.get("descripcion") or "").strip()

        puesto_clean = anonymize_and_clean_800(puesto_raw, empresa, all_company_names)
        desc_clean = anonymize_and_clean_800(desc_raw, empresa, all_company_names)
        if not puesto_clean:
            puesto_clean = desc_clean.split(".")[0][:80].strip() or "Puesto en Lima"
        if len(desc_clean) < 120:
            continue

        if not desc_clean.lower().startswith(puesto_clean[:25].lower()):
            texto_final = f"{puesto_clean}. {desc_clean}"
        else:
            texto_final = desc_clean

        norm_full = _norm_text(texto_final)
        sig_head = norm_full[:180]
        sig_mid = norm_full[90:270]
        title_key = _norm_text(puesto_clean) + "::" + norm_full[:90]

        if sig_head in seen_sigs or (len(sig_mid) > 80 and sig_mid in seen_sigs) or title_key in seen_title_body:
            continue

        shingles = _word_shingles(norm_full, k=5)
        if _is_near_duplicate(shingles, accepted_shingles, threshold=0.78):
            continue

        if u_norm:
            seen_urls.add(u_norm)
        seen_sigs.add(sig_head)
        if len(sig_mid) > 80:
            seen_sigs.add(sig_mid)
        seen_title_body.add(title_key)
        accepted_shingles.append(shingles)

        selected.append(
            {
                "portal": portal_name,
                "busqueda": str(item.get("busqueda") or "").strip() or "general",
                "puesto": puesto_clean,
                "texto": texto_final,
            }
        )

    return selected


def _classify_real_lexical_clean(texto: str) -> Dict[str, str]:
    """Asigna etiqueta limpia y determinista usando las reglas léxicas actualizadas de bias_analyzer.py."""
    spans = detect_biased_spans(texto)
    if not spans:
        return {
            "categoria": "Neutro",
            "nivel_riesgo": "Bajo (Inclusivo)",
            "frases_detectadas": "Ninguna",
        }

    categoria = _select_dominant_lexical_category(spans)
    detected_cats = {s.category for s in spans}
    span_penalty = min(15.0 + sum(_span_weight(s) for s in spans), 90.0)
    lexical_score = max(0.0, 100.0 - span_penalty)
    has_explicit_sex = ("Excluyente" in detected_cats) or any(_span_weight(s) >= 26.0 for s in spans)

    if (
        lexical_score < 45.0
        or len(spans) >= 3
        or (has_explicit_sex and (len(spans) >= 2 or lexical_score < 55.0))
    ):
        nivel_riesgo = "Alto"
    else:
        nivel_riesgo = "Moderado"

    frases = " | ".join(f"[{s.category}] {s.text}" for s in spans)
    return {
        "categoria": categoria,
        "nivel_riesgo": nivel_riesgo,
        "frases_detectadas": frases,
    }


def _make_excel_styles():
    thin_side = Side(border_style="thin", color="CBD5E1")
    med_navy = Side(border_style="medium", color="1E3A8A")
    return {
        "navy_fill": PatternFill("solid", fgColor="1E3A8A"),
        "sub_fill": PatternFill("solid", fgColor="DBEAFE"),
        "zebra_fill": PatternFill("solid", fgColor="F8FAFC"),
        "total_fill": PatternFill("solid", fgColor="EFF6FF"),
        "white_font": Font(name="Calibri", size=11, bold=True, color="FFFFFF"),
        "title_font": Font(name="Calibri", size=14, bold=True, color="0F172A"),
        "section_font": Font(name="Calibri", size=11, bold=True, color="1E3A8A"),
        "bold_font": Font(name="Calibri", size=10, bold=True, color="0F172A"),
        "body_font": Font(name="Calibri", size=10, color="0F172A"),
        "cell_border": Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side),
        "total_border": Border(left=thin_side, right=thin_side, top=med_navy, bottom=med_navy),
        "cat_fills": {
            "Masculino": PatternFill("solid", fgColor="DBEAFE"),
            "Femenino": PatternFill("solid", fgColor="FCE7F3"),
            "Excluyente": PatternFill("solid", fgColor="FEE2E2"),
            "LGBTQ+": PatternFill("solid", fgColor="EDE9FE"),
            "Demográfico": PatternFill("solid", fgColor="FEF3C7"),
            "Neutro": PatternFill("solid", fgColor="DCFCE7"),
        },
        "risk_fills": {
            "Alto": PatternFill("solid", fgColor="FEE2E2"),
            "Moderado": PatternFill("solid", fgColor="FEF3C7"),
            "Bajo (Inclusivo)": PatternFill("solid", fgColor="DCFCE7"),
        },
    }


def export_real_800_excel(df_res: pd.DataFrame, excel_path: Path) -> None:
    """Exporta libro Excel formateado de los 800 avisos reales (3 portales)."""
    wb = openpyxl.Workbook()
    st = _make_excel_styles()

    ws1 = wb.active
    ws1.title = "Resumen 3 Portales (800)"
    ws1["A1"] = "Validación Real de Ofertas Laborales en Perú (n = 800) — Computrabajo vs. Bumeran vs. LinkedIn"
    ws1["A1"].font = st["title_font"]
    ws1.merge_cells("A1:I1")

    ws1["A2"] = "Muestra real consolidada en Perú: Computrabajo (n=300) · Bumeran (n=300) · LinkedIn (n=200)"
    ws1["A2"].font = Font(name="Calibri", size=10, italic=True, color="475569")
    ws1.merge_cells("A2:I2")

    ws1["A4"] = "1. Distribución por Categoría y Portal (Motor Híbrido + Random Forest 2,000)"
    ws1["A4"].font = st["section_font"]

    headers = [
        "Categoría",
        "Computrabajo (n)",
        "Computrabajo (%)",
        "Bumeran (n)",
        "Bumeran (%)",
        "LinkedIn (n)",
        "LinkedIn (%)",
        "Total (n)",
        "Total (%)",
    ]
    for ci, h in enumerate(headers, start=1):
        c = ws1.cell(row=5, column=ci, value=h)
        c.fill = st["navy_fill"]
        c.font = st["white_font"]
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = st["cell_border"]
    ws1.row_dimensions[5].height = 24

    df_c = df_res[df_res["portal"] == "Computrabajo Perú"]
    df_b = df_res[df_res["portal"] == "Bumeran Perú"]
    df_l = df_res[df_res["portal"] == "LinkedIn Perú"]

    r_ptr = 6
    for cat in ALL_CATEGORIES:
        cn = int((df_c["categoria_final"] == cat).sum())
        bn = int((df_b["categoria_final"] == cat).sum())
        ln = int((df_l["categoria_final"] == cat).sum())
        tn = cn + bn + ln
        vals = [cat, cn, cn / len(df_c), bn, bn / len(df_b), ln, ln / len(df_l), tn, tn / len(df_res)]
        for ci, v in enumerate(vals, start=1):
            cell = ws1.cell(row=r_ptr, column=ci, value=v)
            cell.font = st["bold_font"] if ci in (1, 8, 9) else st["body_font"]
            cell.border = st["cell_border"]
            if ci == 1:
                cell.fill = st["cat_fills"].get(cat, st["zebra_fill"])
                cell.alignment = Alignment(horizontal="left", vertical="center")
            else:
                if r_ptr % 2 == 1:
                    cell.fill = st["zebra_fill"]
                cell.alignment = Alignment(horizontal="center", vertical="center")
            if ci in (3, 5, 7, 9):
                cell.number_format = "0.0%"
        ws1.row_dimensions[r_ptr].height = 20
        r_ptr += 1

    tvals = ["Total", len(df_c), 1.0, len(df_b), 1.0, len(df_l), 1.0, len(df_res), 1.0]
    for ci, v in enumerate(tvals, start=1):
        cell = ws1.cell(row=r_ptr, column=ci, value=v)
        cell.font = st["bold_font"]
        cell.fill = st["total_fill"]
        cell.border = st["total_border"]
        cell.alignment = Alignment(horizontal="left" if ci == 1 else "center", vertical="center")
        if ci in (3, 5, 7, 9):
            cell.number_format = "0.0%"

    for ci, w in {1: 20, 2: 17, 3: 17, 4: 14, 5: 14, 6: 14, 7: 14, 8: 13, 9: 13}.items():
        ws1.column_dimensions[get_column_letter(ci)].width = w

    # Hoja 2: Resultados 800 avisos
    ws2 = wb.create_sheet("Resultados 800 Avisos")
    ws2.freeze_panes = "A2"
    specs = [
        ("id", "ID", 12),
        ("portal", "Portal", 19),
        ("busqueda", "Búsqueda", 20),
        ("puesto", "Puesto", 36),
        ("categoria_final", "Categoría", 15),
        ("nivel_riesgo", "Riesgo", 17),
        ("indice_inclusividad", "Inclusividad", 13),
        ("num_frases_sesgadas", "N° Frases", 11),
        ("frases_sesgadas_detectadas", "Frases Detectadas", 44),
        ("texto", "Texto Completo Anonimizado", 90),
    ]
    for ci, (_, title, w) in enumerate(specs, start=1):
        c = ws2.cell(row=1, column=ci, value=title)
        c.fill = st["navy_fill"]
        c.font = st["white_font"]
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border = st["cell_border"]
        ws2.column_dimensions[get_column_letter(ci)].width = w
    ws2.row_dimensions[1].height = 24

    for r_i, (_, row) in enumerate(df_res.iterrows(), start=2):
        for ci, (k, _, _) in enumerate(specs, start=1):
            val = "" if pd.isna(row[k]) else row[k]
            cell = ws2.cell(row=r_i, column=ci, value=val)
            cell.font = st["body_font"]
            cell.border = st["cell_border"]
            if k == "categoria_final":
                cell.fill = st["cat_fills"].get(str(val), st["zebra_fill"])
                cell.font = st["bold_font"]
                cell.alignment = Alignment(horizontal="center", vertical="top")
            elif k == "nivel_riesgo":
                cell.fill = st["risk_fills"].get(str(val), st["zebra_fill"])
                cell.font = st["bold_font"]
                cell.alignment = Alignment(horizontal="center", vertical="top")
            elif k in ("id", "indice_inclusividad", "num_frases_sesgadas"):
                cell.alignment = Alignment(horizontal="center", vertical="top")
            else:
                cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
        ws2.row_dimensions[r_i].height = 34

    ws2.auto_filter.ref = f"A1:{get_column_letter(len(specs))}{len(df_res) + 1}"
    wb.save(excel_path)


def export_consolidated_2000_excel(df: pd.DataFrame, metrics_payload: dict, excel_path: Path) -> None:
    """Exporta respaldo Excel formateado del dataset maestro de 2,000 registros."""
    wb = openpyxl.Workbook()
    st = _make_excel_styles()

    ws1 = wb.active
    ws1.title = "Resumen Maestro (2000)"
    ws1["A1"] = "Dataset Maestro Consolidado (2,000 avisos) — Sintético (1,200) + Real Perú (800)"
    ws1["A1"].font = st["title_font"]
    ws1.merge_cells("A1:G1")

    ws1["A2"] = "Random Forest ligero comprimido (models/rf_model.joblib) · Split estratificado 80/20 (1,600 train / 400 test)"
    ws1["A2"].font = Font(name="Calibri", size=10, italic=True, color="475569")
    ws1.merge_cells("A2:G2")

    ws1["A4"] = "1. Distribución por Categoría y Origen (n = 2,000)"
    ws1["A4"].font = st["section_font"]

    h1 = ["Categoría", "Sintético (1200)", "Sintético (%)", "Real Perú (800)", "Real Perú (%)", "Total (2000)", "Total (%)"]
    for ci, text in enumerate(h1, start=1):
        c = ws1.cell(row=5, column=ci, value=text)
        c.fill = st["navy_fill"]
        c.font = st["white_font"]
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border = st["cell_border"]
    ws1.row_dimensions[5].height = 24

    df_s = df[df["origen"] == ORIGEN_SINTETICO]
    df_r = df[df["origen"] == ORIGEN_REAL_PERU]

    r_i = 6
    for cat in ALL_CATEGORIES:
        sn = int((df_s["categoria"] == cat).sum())
        rn = int((df_r["categoria"] == cat).sum())
        tn = sn + rn
        vals = [cat, sn, sn / len(df_s), rn, rn / len(df_r), tn, tn / len(df)]
        for ci, v in enumerate(vals, start=1):
            cell = ws1.cell(row=r_i, column=ci, value=v)
            cell.font = st["bold_font"] if ci in (1, 6, 7) else st["body_font"]
            cell.border = st["cell_border"]
            if ci == 1:
                cell.fill = st["cat_fills"].get(cat, st["zebra_fill"])
                cell.alignment = Alignment(horizontal="left", vertical="center")
            else:
                if r_i % 2 == 1:
                    cell.fill = st["zebra_fill"]
                cell.alignment = Alignment(horizontal="center", vertical="center")
            if ci in (3, 5, 7):
                cell.number_format = "0.0%"
        r_i += 1

    for ci, v in enumerate(["Total", len(df_s), 1.0, len(df_r), 1.0, len(df), 1.0], start=1):
        cell = ws1.cell(row=r_i, column=ci, value=v)
        cell.font = st["bold_font"]
        cell.fill = st["total_fill"]
        cell.border = st["total_border"]
        cell.alignment = Alignment(horizontal="left" if ci == 1 else "center", vertical="center")
        if ci in (3, 5, 7):
            cell.number_format = "0.0%"

    rf_info = metrics_payload["models"]["random_forest"]
    tm = rf_info["test_metrics"]
    cv = rf_info["cv_5fold_train"]
    r_i += 2
    ws1.cell(row=r_i, column=1, value="2. Métricas Random Forest Ligero (Test 20% = 400 avisos)").font = st["section_font"]
    r_i += 1
    for ci, text in enumerate(["Métrica", "Valor", "Detalle"], start=1):
        c = ws1.cell(row=r_i, column=ci, value=text)
        c.fill = st["sub_fill"]
        c.font = st["bold_font"]
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border = st["cell_border"]
    r_i += 1
    for label, val, note in [
        ("Accuracy (Test 20%)", tm["accuracy"], "Split estratificado 80/20 (1,600 train / 400 test)"),
        ("Precision (Macro)", tm["precision_macro"], "Promedio macro en 6 clases"),
        ("Recall (Macro)", tm["recall_macro"], "Promedio macro en 6 clases"),
        ("F1-Score (Macro)", tm["f1_macro"], "Métrica balanceada principal"),
        ("F1-Score (Weighted)", tm["f1_weighted"], "Ponderado por soporte"),
        ("CV-5 Accuracy (Train)", cv["accuracy_mean"], f"±{cv['accuracy_std']:.4f}"),
        ("CV-5 F1-Macro (Train)", cv["f1_macro_mean"], f"±{cv['f1_macro_std']:.4f}"),
        ("Tamaño rf_model.joblib (MB)", rf_info["model_size_mb"], "compress=3, n_estimators=100, max_depth=25"),
    ]:
        c1 = ws1.cell(row=r_i, column=1, value=label)
        c2 = ws1.cell(row=r_i, column=2, value=val)
        c3 = ws1.cell(row=r_i, column=3, value=note)
        c1.font = c2.font = st["bold_font"]
        c3.font = st["body_font"]
        c1.border = c2.border = c3.border = st["cell_border"]
        c2.alignment = Alignment(horizontal="center", vertical="center")
        if "MB" not in label:
            c2.number_format = "0.00%"
        else:
            c2.number_format = "0.000"
        r_i += 1

    for ci, w in {1: 28, 2: 18, 3: 16, 4: 18, 5: 16, 6: 16, 7: 14}.items():
        ws1.column_dimensions[get_column_letter(ci)].width = w

    ws2 = wb.create_sheet("Dataset Consolidado (2000)")
    ws2.freeze_panes = "A2"
    specs = [
        ("id", "ID", 13),
        ("origen", "Origen", 19),
        ("portal", "Portal", 19),
        ("puesto", "Puesto", 36),
        ("categoria", "Categoría", 15),
        ("nivel_riesgo", "Nivel de Riesgo", 17),
        ("frases_detectadas", "Frases Detectadas", 44),
        ("texto", "Texto Completo", 90),
    ]
    for ci, (_, title, w) in enumerate(specs, start=1):
        c = ws2.cell(row=1, column=ci, value=title)
        c.fill = st["navy_fill"]
        c.font = st["white_font"]
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border = st["cell_border"]
        ws2.column_dimensions[get_column_letter(ci)].width = w
    ws2.row_dimensions[1].height = 24

    for r_idx, (_, row) in enumerate(df.iterrows(), start=2):
        for ci, (k, _, _) in enumerate(specs, start=1):
            val = row[k]
            cell = ws2.cell(row=r_idx, column=ci, value=val)
            cell.font = st["body_font"]
            cell.border = st["cell_border"]
            if k == "categoria":
                cell.fill = st["cat_fills"].get(str(val), st["zebra_fill"])
                cell.font = st["bold_font"]
                cell.alignment = Alignment(horizontal="center", vertical="top")
            elif k == "nivel_riesgo":
                cell.fill = st["risk_fills"].get(str(val), st["zebra_fill"])
                cell.font = st["bold_font"]
                cell.alignment = Alignment(horizontal="center", vertical="top")
            elif k in ("id", "origen", "portal"):
                cell.alignment = Alignment(horizontal="center", vertical="top")
            else:
                cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
        ws2.row_dimensions[r_idx].height = 32

    ws2.auto_filter.ref = f"A1:{get_column_letter(len(specs))}{len(df) + 1}"
    wb.save(excel_path)


def main() -> None:
    assert DATASET_PATH.exists(), f"Falta respaldo sintético: {DATASET_PATH}"
    assert REAL_PERU_DATASET_PATH.exists(), f"Falta base real: {REAL_PERU_DATASET_PATH}"

    # 1. Preservar respaldo inmutable de los 200 avisos reales originales
    df_real_base = pd.read_csv(REAL_PERU_DATASET_PATH)
    if len(df_real_base) == 200 and not BACKUP_200_PATH.exists():
        shutil.copy2(REAL_PERU_DATASET_PATH, BACKUP_200_PATH)
    elif len(df_real_base) > 200 and BACKUP_200_PATH.exists():
        df_real_base = pd.read_csv(BACKUP_200_PATH)
    else:
        df_real_base = df_real_base.iloc[:200].copy()

    assert len(df_real_base) == 200, f"Base real esperada 200, encontrada {len(df_real_base)}"
    assert int((df_real_base["portal"] == "Computrabajo Perú").sum()) == 100
    assert int((df_real_base["portal"] == "Bumeran Perú").sum()) == 100

    # Inicializar estructuras de deduplicación con los 200 avisos reales base
    seen_urls: Set[str] = set()
    seen_sigs: Set[str] = set()
    accepted_shingles: List[Set[str]] = []
    for t in df_real_base["texto"].tolist():
        nt = _norm_text(str(t))
        seen_sigs.add(nt[:180])
        if len(nt) > 170:
            seen_sigs.add(nt[90:270])
        accepted_shingles.append(_word_shingles(nt, k=5))

    # 2. Cargar candidatos nuevos (+200 Computrabajo, +200 Bumeran, +200 LinkedIn)
    comp_raw, bum_raw, li_raw = load_all_new_candidates()
    all_companies = sorted(
        {
            str(it.get("empresa") or "").strip()
            for it in (comp_raw + bum_raw + li_raw)
            if str(it.get("empresa") or "").strip()
            and str(it.get("empresa") or "").strip().lower() not in NON_COMPANY_BADGES
        }
    )

    new_comp = select_unique_portal_batch(
        comp_raw, 200, "Computrabajo Perú", seen_urls, seen_sigs, accepted_shingles, all_companies
    )
    new_bum = select_unique_portal_batch(
        bum_raw, 200, "Bumeran Perú", seen_urls, seen_sigs, accepted_shingles, all_companies
    )
    new_li = select_unique_portal_batch(
        li_raw, 200, "LinkedIn Perú", seen_urls, seen_sigs, accepted_shingles, all_companies
    )

    print(
        f"Candidatos únicos nuevos seleccionados -> Computrabajo: {len(new_comp)}/200 | "
        f"Bumeran: {len(new_bum)}/200 | LinkedIn: {len(new_li)}/200"
    )
    assert len(new_comp) == 200, f"Faltan avisos nuevos de Computrabajo: {len(new_comp)}/200"
    assert len(new_bum) == 200, f"Faltan avisos nuevos de Bumeran: {len(new_bum)}/200"
    assert len(new_li) == 200, f"Faltan avisos nuevos de LinkedIn: {len(new_li)}/200"

    # 3. Consolidar exactamente 800 avisos reales únicos en data/ofertas_reales_peru.csv
    real_rows: List[Dict[str, str]] = []
    for rec_base in df_real_base[["id", "portal", "busqueda", "puesto", "texto"]].to_dict(orient="records"):
        # Asegurar higiene de teléfonos espaciados / correos residuales sin alterar contenido ni respaldo de 200
        t_clean = re.sub(r"(?:\+?51[\s-]?)?\b9(?:\s*\d){8}\b", "[EMPRESA]", str(rec_base["texto"]))
        t_clean = re.sub(r"[\w.*+-]+@[\w-]+\.[\w.-]+\b", "[EMPRESA]", t_clean)
        rec_base["texto"] = t_clean
        real_rows.append(rec_base)
    next_id = 201
    for rec in new_comp + new_bum + new_li:
        real_rows.append(
            {
                "id": f"PERU-{next_id:03d}",
                "portal": rec["portal"],
                "busqueda": rec["busqueda"],
                "puesto": rec["puesto"],
                "texto": rec["texto"],
            }
        )
        next_id += 1

    df_real_800 = pd.DataFrame(real_rows, columns=["id", "portal", "busqueda", "puesto", "texto"])
    assert len(df_real_800) == 800
    assert int(df_real_800.isna().sum().sum()) == 0
    assert df_real_800["texto"].nunique() == 800
    assert df_real_800["portal"].value_counts().to_dict() == {
        "Computrabajo Perú": 300,
        "Bumeran Perú": 300,
        "LinkedIn Perú": 200,
    }
    df_real_800.to_csv(REAL_PERU_DATASET_PATH, index=False, encoding="utf-8-sig")

    # 4. Homologar y fusionar 1,200 sintéticos + 800 reales = 2,000 (etiquetas léxicas limpias cero fuga)
    df_synth = pd.read_csv(DATASET_PATH)
    assert len(df_synth) == 1200

    synth_rows_2000: List[Dict[str, str]] = []
    for _, row in df_synth.iterrows():
        texto = str(row["texto"]).strip()
        spans = detect_biased_spans(texto)
        frases = " | ".join(f"[{s.category}] {s.text}" for s in spans) if spans else "Ninguna"
        raw_risk = str(row.get("nivel_sesgo", "Moderado")).strip()
        nivel_riesgo = "Bajo (Inclusivo)" if str(row["categoria"]).strip() == "Neutro" else SYNTH_RISK_MAP.get(raw_risk, "Moderado")
        synth_rows_2000.append(
            {
                "id": str(row["id"]).strip(),
                "origen": ORIGEN_SINTETICO,
                "portal": "Sintético",
                "puesto": str(row["puesto"]).strip(),
                "categoria": str(row["categoria"]).strip(),
                "nivel_riesgo": nivel_riesgo,
                "frases_detectadas": frases,
                "texto": texto,
            }
        )

    real_rows_2000: List[Dict[str, str]] = []
    for _, row in df_real_800.iterrows():
        texto = str(row["texto"]).strip()
        clean_lbl = _classify_real_lexical_clean(texto)
        real_rows_2000.append(
            {
                "id": str(row["id"]).strip(),
                "origen": ORIGEN_REAL_PERU,
                "portal": str(row["portal"]).strip(),
                "puesto": str(row["puesto"]).strip(),
                "categoria": clean_lbl["categoria"],
                "nivel_riesgo": clean_lbl["nivel_riesgo"],
                "frases_detectadas": clean_lbl["frases_detectadas"],
                "texto": texto,
            }
        )

    cols_2000 = ["id", "origen", "portal", "puesto", "categoria", "nivel_riesgo", "frases_detectadas", "texto"]
    df_2000 = pd.DataFrame(synth_rows_2000 + real_rows_2000, columns=cols_2000)
    assert len(df_2000) == 2000
    assert int(df_2000.isna().sum().sum()) == 0
    assert df_2000["id"].nunique() == 2000
    assert df_2000["texto"].nunique() == 2000
    df_2000.to_csv(CONSOLIDATED_2000_PATH, index=False, encoding="utf-8-sig")

    # 5. Reentrenar Random Forest ligero comprimido (80/20 estratificado) sobre los 2,000 avisos
    metrics = train_and_evaluate(regenerate_dataset=False)
    export_consolidated_2000_excel(df_2000, metrics, CONSOLIDATED_2000_EXCEL_PATH)

    # 6. Evaluar los 800 avisos reales con el nuevo modelo RF + detector híbrido y guardar CSV/XLSX
    rf_model = load_model("random_forest")
    eval_rows = []
    for _, row in df_real_800.iterrows():
        res = analyze_job_posting(row["texto"], model_key="random_forest", pipeline=rf_model)
        frases = " | ".join(f"[{s.category}] {s.text}" for s in res.spans)
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
                "cobertura_vocabulario_pct": round(res.vocabulary_coverage * 100, 1),
                "entropia_ml": round(res.ml_entropy, 3),
                "prob_masculino": round(res.probabilities.get("Masculino", 0.0) * 100, 1),
                "prob_femenino": round(res.probabilities.get("Femenino", 0.0) * 100, 1),
                "prob_excluyente": round(res.probabilities.get("Excluyente", 0.0) * 100, 1),
                "prob_lgbtq": round(res.probabilities.get("LGBTQ+", 0.0) * 100, 1),
                "prob_demografico": round(res.probabilities.get("Demográfico", 0.0) * 100, 1),
                "prob_neutro": round(res.probabilities.get("Neutro", 0.0) * 100, 1),
                "indice_inclusividad": res.inclusivity_score,
                "nivel_riesgo": res.risk_level,
                "num_frases_sesgadas": len(res.spans),
                "categorias_lexicas": ", ".join(res.detected_categories),
                "frases_sesgadas_detectadas": frases,
                "requisitos_tecnicos": ", ".join(res.technical_terms),
                "texto": row["texto"],
            }
        )

    df_res_800 = pd.DataFrame(eval_rows)
    res_csv_path = DATA_DIR / "validacion_real_resultados.csv"
    res_xlsx_path = DATA_DIR / "validacion_real_resultados.xlsx"
    df_res_800.to_csv(res_csv_path, index=False, encoding="utf-8-sig")
    export_real_800_excel(df_res_800, res_xlsx_path)

    # Confirmar integridad final
    assert len(pd.read_csv(DATASET_PATH)) == 1200
    assert len(pd.read_csv(BACKUP_200_PATH)) == 200
    size_mb = RF_MODEL_PATH.stat().st_size / (1024 * 1024)
    assert size_mb < 15.0

    print("\n=== CONSOLIDACIÓN 800 REALES + 2,000 MAESTRO COMPLETADA ===")
    print(f"  ofertas_reales_peru.csv       : {len(df_real_800)} filas (0 nulos, 800 únicos)")
    print(f"  dataset_consolidado_2000.csv  : {len(df_2000)} filas (1200 sintéticos + 800 reales)")
    print(f"  models/rf_model.joblib        : {size_mb:.3f} MB (< 15 MB)")

    print("\n=== TABLA COMPARATIVA 800 AVISOS REALES (3 PORTALES) ===")
    df_c = df_res_800[df_res_800["portal"] == "Computrabajo Perú"]
    df_b = df_res_800[df_res_800["portal"] == "Bumeran Perú"]
    df_l = df_res_800[df_res_800["portal"] == "LinkedIn Perú"]
    for cat in ALL_CATEGORIES:
        cn = int((df_c["categoria_final"] == cat).sum())
        bn = int((df_b["categoria_final"] == cat).sum())
        ln = int((df_l["categoria_final"] == cat).sum())
        tn = cn + bn + ln
        print(
            f"  {cat:<12}: Comp={cn:3d} ({100*cn/len(df_c):5.1f}%) | "
            f"Bum={bn:3d} ({100*bn/len(df_b):5.1f}%) | "
            f"LinkedIn={ln:3d} ({100*ln/len(df_l):5.1f}%) | "
            f"Total={tn:3d} ({100*tn/len(df_res_800):5.1f}%)"
        )

    print("\n=== DISTRIBUCIÓN MAESTRA 2,000 (SINTÉTICO 1200 vs REAL PERÚ 800) ===")
    df_s2 = df_2000[df_2000["origen"] == ORIGEN_SINTETICO]
    df_r2 = df_2000[df_2000["origen"] == ORIGEN_REAL_PERU]
    for cat in ALL_CATEGORIES:
        sn = int((df_s2["categoria"] == cat).sum())
        rn = int((df_r2["categoria"] == cat).sum())
        tn = sn + rn
        print(
            f"  {cat:<12}: Sintético={sn:4d} ({100*sn/len(df_s2):5.1f}%) | "
            f"Real Perú={rn:3d} ({100*rn/len(df_r2):5.1f}%) | "
            f"Total={tn:4d} ({100*tn/len(df_2000):5.1f}%)"
        )

    rf = metrics["models"]["random_forest"]
    tm = rf["test_metrics"]
    cv = rf["cv_5fold_train"]
    print("\n=== MÉTRICAS RANDOM FOREST COMPRIMIDO EN 2,000 (TEST 20% = 400 AVISOS) ===")
    print(f"  Accuracy          : {tm['accuracy']:.4f} ({tm['accuracy']*100:.2f}%)")
    print(f"  Precision (Macro) : {tm['precision_macro']:.4f} ({tm['precision_macro']*100:.2f}%)")
    print(f"  Recall (Macro)    : {tm['recall_macro']:.4f} ({tm['recall_macro']*100:.2f}%)")
    print(f"  F1-Score (Macro)  : {tm['f1_macro']:.4f} ({tm['f1_macro']*100:.2f}%)")
    print(f"  F1-Score (Weight) : {tm['f1_weighted']:.4f} ({tm['f1_weighted']*100:.2f}%)")
    print(f"  CV-5 Accuracy     : {cv['accuracy_mean']:.4f} (±{cv['accuracy_std']:.4f})")
    print(f"  CV-5 F1-Macro     : {cv['f1_macro_mean']:.4f} (±{cv['f1_macro_std']:.4f})")
    print("\n=== MÉTRICAS POR CLASE (TEST 20% = 400) ===")
    for cat in ALL_CATEGORIES:
        pc = rf["per_class"][cat]
        print(f"  {cat:<12}: Prec={pc['precision']:.4f} | Rec={pc['recall']:.4f} | F1={pc['f1_score']:.4f} | Support={pc['support']}")


if __name__ == "__main__":
    main()
