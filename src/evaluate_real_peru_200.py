"""Consolidación, limpieza, anonimización, evaluación y exportación Excel de 200 avisos reales de Perú.

Combina:
- 100 avisos reales ya guardados en data/ofertas_reales_peru.csv (65 Bumeran Perú + 35 Computrabajo Perú)
- 35 avisos reales NUEVOS de Lima desde Bumeran Perú (lote 3: steps 122-140)
- 65 avisos reales NUEVOS de Lima desde Computrabajo Perú (lote 2: steps 36-93)
Total exacto: 200 avisos reales únicos (100 Bumeran Perú + 100 Computrabajo Perú),
sin modificar data/dataset_sesgos_empleo.csv (1,200 avisos sintéticos) ni reentrenar modelos ML.

Genera:
- data/ofertas_reales_peru.csv (200 filas: id, portal, busqueda, puesto, texto)
- data/validacion_real_resultados.csv (200 filas con predicciones RF + motor híbrido)
- data/validacion_real_resultados.xlsx (libro Excel formateado con 3 hojas)
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Dict, List, Set

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd

from src.bias_analyzer import analyze_job_posting
from src.config import ALL_CATEGORIES, DATA_DIR
from src.evaluate_real_peru_100 import (
    EXTRA_BRAND_ALIASES as BASE_BRAND_ALIASES,
    NON_COMPANY_BADGES,
    _company_variants,
)

BUMERAN_DIR = Path(
    r"C:\Users\verde\.gemini\antigravity\brain\b00ec03b-26a8-4e1e-b542-3c5b15cece43\browser_sandbox\outputs"
)
COMPUTRABAJO_DIR = Path(
    r"C:\Users\verde\.gemini\antigravity\brain\4efe456c-9544-409f-ae2c-9545ae11ad51\browser_sandbox\outputs"
)

# Directorio local opcional en scratch (si los JSON fueron copiados ahí)
SCRATCH_RAW_DIR = Path(
    r"C:\Users\verde\.gemini\antigravity\brain\8c6c8e80-06b2-44e5-a8d8-62ebb1c9ed98\scratch\raw_new_100"
)

BUMERAN_NEW_FILES: List[str] = [
    "step_122.json",
    "step_124.json",
    "step_126.json",
    "step_128.json",
    "step_130.json",
    "step_132.json",
    "step_134.json",
    "step_136.json",
    "step_138.json",
    "step_140.json",
]

COMPUTRABAJO_NEW_FILES: List[str] = [
    "step_36.json",
    "step_38.json",
    "step_55.json",
    "step_57.json",
    "step_59.json",
    "step_63.json",
    "step_65.json",
    "step_67.json",
    "step_69.json",
    "step_71.json",
    "step_75.json",
    "step_77.json",
    "step_81.json",
    "step_83.json",
    "step_87.json",
    "step_89.json",
    "step_91.json",
    "step_93.json",
]

NEW_BRAND_ALIASES: List[str] = list(BASE_BRAND_ALIASES) + [
    # Nuevos lotes Bumeran Lima
    "ESSEGURIDAD", "Esseguridad", "Grupo Villarruel", "Gizasi",
    "DM HOTEL ASIA", "DM HOTELES", "RRHH CAMBRIDGE",
    "AGENCIA PUBLICIDAD ALQUIMIA", "Clínica San Juan de Dios", "CLINICA SAN JUAN DE DIOS",
    "VANADIO SAC", "Vanadio", "A&RC Consultores", "Andrea Garcés",
    "QUIASMO", "Restaurantes Amoramar", "Amoramar", "Café de Li",
    "DeliBakery", "INVERSIONES Y NEGOCIOS MARINOS ESPINOZA", "CORPORACIÓN ESPINOZA",
    "Corporación Espinoza", "Punto Marisko", "La Playa",
    "PERUDELI", "PIHUICHO", "GRUPO LATIN GAMES", "Latin Games", "Sanduco",
    "PETRAMAS", "Petramás", "INNOVA SMILES DESING", "SEFAVET",
    "ESTUDIOS JURIDICOS OROPEZA", "Endutech", "KonectaCorp", "Claro Empresas",
    "Ceratell group", "Ceratell", "Lucha Partners", "LUCHA PARTNERS",
    "Bradken", "Lance Gráfico", "QUINTOP", "Distribuidora Navarrete",
    "MUNDOCUADRO.PE", "MUNDO CUADRO", "ASOCIACIÓN EDUCATIVA CASUARINAS",
    "Feria Sise", "feriaSISE",
    # Marcas adicionales detectadas en Computrabajo Lima
    "Villa Chicken", "SOFT Perú", "LUKERS", "Primatex", "Panorama Outsourcing",
    "Clínica Delgado", "McDonalds", "McDonald's", "Grupo AJE", "Sam Facility",
    "Gina & Michell Alimentos", "Gina & Michell", "Corporación Aramburú",
]

# Palabras comunes que NUNCA deben anonimizarse solas
SKIP_COMMON_WORDS: Set[str] = {
    "sac", "srl", "eirl", "peru", "perú", "spa", "cyl", "sip", "ey", "exact",
    "control", "indigo", "bar tematico", "bar temático", "personal", "empresa",
    "corporacion", "corporación", "inversiones", "servicios", "soluciones",
    "comercial", "industrial", "distribuidora", "importadora", "grupo", "lima",
    "seguridad", "limpieza", "logistica", "logística", "consultores", "consultoria",
    "consultoría", "restaurante", "restaurantes", "clinica", "clínica", "hotel",
    "hoteles", "tiendas", "tienda", "alimentos", "salud", "global", "express",
    "select", "talento", "talent", "capital", "human", "partner", "partners",
    "service", "services", "center", "centro", "nacional", "internacional",
    "moderna", "integral", "integrales", "general", "generales", "america", "américa",
    "sur", "norte", "este", "oeste", "del", "los", "las", "con", "por", "para",
    "uno", "dos", "tres", "la playa",
}


def _resolve_json(base_dir: Path, subdir_name: str, filename: str) -> Path:
    scratch_path = SCRATCH_RAW_DIR / subdir_name / filename
    if scratch_path.exists():
        return scratch_path
    return base_dir / filename


def _norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (text or "").lower())).strip()


def anonymize_and_clean_v2(text: str, empresa: str, all_company_names: List[str]) -> str:
    """Limpia HTML/ruido de portal y anonimiza empresas y datos de contacto sin sobre-anonimizar."""
    if not text:
        return ""

    s = html.unescape(text)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[\u200b\ufeff\xa0]", " ", s)
    s = re.sub(r"[•●▪■►▶✔✓✅➤➢❖⭐🔹🔸📌📍💼📩📞🌟✨🔥📢📊🕒🔎📦📲]+", " ", s)

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

    # Anonimizar correos (normales y ofuscados con "(arroba)"), URLs y teléfonos
    s = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "[CORREO]", s)
    s = re.sub(
        r"\b[\w.+-]+\s*\(\s*arroba\s*\)\s*[\w.-]+\s*\.\s*[a-z]{2,}\b",
        "[CORREO]",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"https?://\S+|www\.\S+", "[WEB]", s, flags=re.IGNORECASE)
    s = re.sub(r"(?:\+?51[\s-]?)?\b9\d{2}[\s-]?\d{3}[\s-]?\d{3}\b", "[TELEFONO]", s)

    targets: Set[str] = set(NEW_BRAND_ALIASES)
    for name in _company_variants(empresa):
        targets.add(name)
    for name in all_company_names:
        for v in _company_variants(name):
            targets.add(v)

    for target in sorted(targets, key=len, reverse=True):
        if len(target) < 4:
            continue
        low = target.lower().strip()
        if low in SKIP_COMMON_WORDS:
            continue
        # Evitar reemplazar palabras comunes de una sola palabra si no coincide con la empresa del aviso
        if " " not in low and len(low) <= 6 and low not in (empresa or "").lower():
            continue
        s = re.sub(rf"(?<!\w){re.escape(target)}(?!\w)", "[EMPRESA]", s, flags=re.IGNORECASE)

    # Casos cortos / propios sensibles a mayúsculas
    emp_low = (empresa or "").strip().lower()
    if emp_low == "indigo":
        s = re.sub(r"\bIndigo\b|\bINDIGO\b", "[EMPRESA]", s)
    if "espinoza" in emp_low:
        s = re.sub(r"\bLa\s+Playa\b", "[EMPRESA]", s)
    if "caja de pensiones" in emp_low:
        s = re.sub(r"\bCPMP\b", "[EMPRESA]", s)
    s = re.sub(r"\b(?:KORI|GEA|LUKERS|BCP)\b", "[EMPRESA]", s)
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


def load_new_raw_batches() -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Carga los 40 avisos nuevos de Bumeran y 70 avisos nuevos de Computrabajo."""
    bum_items: List[Dict[str, str]] = []
    for fn in BUMERAN_NEW_FILES:
        p = _resolve_json(BUMERAN_DIR, "bumeran", fn)
        payload = json.loads(p.read_text(encoding="utf-8"))
        for it in payload["value"]["results"]:
            rec = dict(it)
            rec["portal"] = "Bumeran Perú"
            bum_items.append(rec)

    comp_items: List[Dict[str, str]] = []
    for fn in COMPUTRABAJO_NEW_FILES:
        p = _resolve_json(COMPUTRABAJO_DIR, "computrabajo", fn)
        payload = json.loads(p.read_text(encoding="utf-8"))
        for it in payload["value"]["results"]:
            rec = dict(it)
            rec["portal"] = "Computrabajo Perú"
            comp_items.append(rec)

    return bum_items, comp_items


def select_unique_new(
    candidates: List[Dict[str, str]],
    target_count: int,
    existing_keys: Set[str],
    all_company_names: List[str],
) -> List[Dict[str, str]]:
    """Selecciona exactamente `target_count` avisos nuevos únicos, limpiando y anonimizando cada uno."""
    selected: List[Dict[str, str]] = []
    seen_urls: Set[str] = set()
    seen_body_sig: Set[str] = set()

    for item in candidates:
        if len(selected) >= target_count:
            break
        url = (item.get("url") or "").strip()
        if url and url in seen_urls:
            continue

        empresa = (item.get("empresa") or "").strip()
        puesto_raw = (item.get("puesto") or "").strip()
        desc_raw = (item.get("descripcion") or "").strip()

        puesto_clean = anonymize_and_clean_v2(puesto_raw, empresa, all_company_names)
        desc_clean = anonymize_and_clean_v2(desc_raw, empresa, all_company_names)

        if puesto_clean and not desc_clean.lower().startswith(puesto_clean[:25].lower()):
            texto_final = f"{puesto_clean}. {desc_clean}"
        else:
            texto_final = desc_clean

        # Dedup contra los 100 existentes y contra duplicados casi idénticos en la misma tanda
        puesto_norm = _norm_key(puesto_clean)
        body_sig = _norm_key(texto_final)[:180]
        if body_sig in existing_keys or body_sig in seen_body_sig:
            continue
        # Evitar duplicados exactos de plantilla del mismo empleador dentro de los nuevos (p. ej. 2 avisos idénticos)
        body_mid_sig = _norm_key(texto_final)[80:260]
        if len(body_mid_sig) > 80 and body_mid_sig in seen_body_sig:
            continue

        if url:
            seen_urls.add(url)
        seen_body_sig.add(body_sig)
        if len(body_mid_sig) > 80:
            seen_body_sig.add(body_mid_sig)
        existing_keys.add(body_sig)

        selected.append(
            {
                "portal": item["portal"],
                "busqueda": (item.get("busqueda") or "").strip(),
                "puesto": puesto_clean,
                "texto": texto_final,
                "empresa_raw": empresa,
                "url": url,
            }
        )

    return selected


def export_formatted_excel(
    df_ofertas: pd.DataFrame,
    df_res: pd.DataFrame,
    excel_path: Path,
) -> None:
    """Exporta un libro Excel profesional con 3 hojas: Resumen Comparativo, Resultados 200 Avisos y Textos Completos."""
    wb = openpyxl.Workbook()

    # Estilos reutilizables
    navy_fill = PatternFill("solid", fgColor="1E3A8A")
    sub_fill = PatternFill("solid", fgColor="DBEAFE")
    zebra_fill = PatternFill("solid", fgColor="F8FAFC")
    total_fill = PatternFill("solid", fgColor="EFF6FF")
    white_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    title_font = Font(name="Calibri", size=14, bold=True, color="0F172A")
    section_font = Font(name="Calibri", size=11, bold=True, color="1E3A8A")
    bold_font = Font(name="Calibri", size=10, bold=True, color="0F172A")
    body_font = Font(name="Calibri", size=10, color="0F172A")

    thin_side = Side(border_style="thin", color="CBD5E1")
    med_navy = Side(border_style="medium", color="1E3A8A")
    cell_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)
    total_border = Border(left=thin_side, right=thin_side, top=med_navy, bottom=med_navy)

    cat_fills = {
        "Excluyente": PatternFill("solid", fgColor="FEE2E2"),
        "Masculino": PatternFill("solid", fgColor="DBEAFE"),
        "Femenino": PatternFill("solid", fgColor="FCE7F3"),
        "Demográfico": PatternFill("solid", fgColor="FEF3C7"),
        "LGBTQ+": PatternFill("solid", fgColor="EDE9FE"),
        "Neutro": PatternFill("solid", fgColor="DCFCE7"),
    }
    risk_fills = {
        "Alto": PatternFill("solid", fgColor="FEE2E2"),
        "Moderado": PatternFill("solid", fgColor="FEF3C7"),
        "Bajo (Inclusivo)": PatternFill("solid", fgColor="DCFCE7"),
    }

    # ==================== HOJA 1: RESUMEN COMPARATIVO ====================
    ws1 = wb.active
    ws1.title = "Resumen Comparativo"
    ws1.views.sheetView[0].showGridLines = True

    ws1["A1"] = "Validación Real de Ofertas Laborales en Perú (n = 200) — Random Forest + Motor Híbrido"
    ws1["A1"].font = title_font
    ws1.merge_cells("A1:H1")

    ws1["A2"] = "Portales evaluados: Computrabajo Perú (n=100) y Bumeran Perú (n=100) | Dataset sintético de entrenamiento intacto (n=1,200)"
    ws1["A2"].font = Font(name="Calibri", size=10, italic=True, color="475569")
    ws1.merge_cells("A2:H2")

    # Tabla 1: Distribución por Categoría Final (Híbrido + RF)
    ws1["A4"] = "1. Distribución por Categoría Final (Motor Híbrido + Random Forest)"
    ws1["A4"].font = section_font

    headers_t1 = [
        "Categoría",
        "Computrabajo (n)",
        "Computrabajo (%)",
        "Bumeran (n)",
        "Bumeran (%)",
        "Total (n)",
        "Total (%)",
        "ML Crudo RF (n)",
    ]
    for col_idx, h in enumerate(headers_t1, start=1):
        c = ws1.cell(row=5, column=col_idx, value=h)
        c.fill = navy_fill
        c.font = white_font
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = cell_border
    ws1.row_dimensions[5].height = 24

    df_comp = df_res[df_res["portal"] == "Computrabajo Perú"]
    df_bum = df_res[df_res["portal"] == "Bumeran Perú"]
    n_comp = len(df_comp)
    n_bum = len(df_bum)
    n_tot = len(df_res)

    row_ptr = 6
    for cat in ALL_CATEGORIES:
        c_n = int((df_comp["categoria_final"] == cat).sum())
        b_n = int((df_bum["categoria_final"] == cat).sum())
        t_n = int((df_res["categoria_final"] == cat).sum())
        ml_n = int((df_res["categoria_ml_cruda"] == cat).sum())
        vals = [
            cat,
            c_n,
            c_n / n_comp if n_comp else 0.0,
            b_n,
            b_n / n_bum if n_bum else 0.0,
            t_n,
            t_n / n_tot if n_tot else 0.0,
            ml_n,
        ]
        for col_idx, v in enumerate(vals, start=1):
            cell = ws1.cell(row=row_ptr, column=col_idx, value=v)
            cell.font = bold_font if col_idx in (1, 6, 7) else body_font
            cell.border = cell_border
            if col_idx == 1:
                cell.fill = cat_fills.get(cat, zebra_fill)
                cell.alignment = Alignment(horizontal="left", vertical="center")
            else:
                if row_ptr % 2 == 1:
                    cell.fill = zebra_fill
                cell.alignment = Alignment(horizontal="center", vertical="center")
            if col_idx in (3, 5, 7):
                cell.number_format = "0.0%"
        ws1.row_dimensions[row_ptr].height = 20
        row_ptr += 1

    # Fila Total
    tot_vals = ["Total", n_comp, 1.0, n_bum, 1.0, n_tot, 1.0, n_tot]
    for col_idx, v in enumerate(tot_vals, start=1):
        cell = ws1.cell(row=row_ptr, column=col_idx, value=v)
        cell.font = bold_font
        cell.fill = total_fill
        cell.border = total_border
        cell.alignment = Alignment(horizontal="left" if col_idx == 1 else "center", vertical="center")
        if col_idx in (3, 5, 7):
            cell.number_format = "0.0%"
    ws1.row_dimensions[row_ptr].height = 22

    # Tabla 2: Distribución por Nivel de Riesgo
    row_ptr += 2
    ws1.cell(row=row_ptr, column=1, value="2. Distribución por Nivel de Riesgo").font = section_font
    row_ptr += 1
    headers_t2 = ["Nivel de Riesgo", "Computrabajo (n)", "Computrabajo (%)", "Bumeran (n)", "Bumeran (%)", "Total (n)", "Total (%)"]
    for col_idx, h in enumerate(headers_t2, start=1):
        c = ws1.cell(row=row_ptr, column=col_idx, value=h)
        c.fill = navy_fill
        c.font = white_font
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = cell_border
    ws1.row_dimensions[row_ptr].height = 22
    row_ptr += 1

    for risk in ["Alto", "Moderado", "Bajo (Inclusivo)"]:
        c_n = int((df_comp["nivel_riesgo"] == risk).sum())
        b_n = int((df_bum["nivel_riesgo"] == risk).sum())
        t_n = int((df_res["nivel_riesgo"] == risk).sum())
        vals = [risk, c_n, c_n / n_comp, b_n, b_n / n_bum, t_n, t_n / n_tot]
        for col_idx, v in enumerate(vals, start=1):
            cell = ws1.cell(row=row_ptr, column=col_idx, value=v)
            cell.font = bold_font if col_idx in (1, 6, 7) else body_font
            cell.border = cell_border
            if col_idx == 1:
                cell.fill = risk_fills.get(risk, zebra_fill)
                cell.alignment = Alignment(horizontal="left", vertical="center")
            else:
                cell.alignment = Alignment(horizontal="center", vertical="center")
            if col_idx in (3, 5, 7):
                cell.number_format = "0.0%"
        row_ptr += 1

    # Tabla 3: KPIs clave por portal
    row_ptr += 2
    ws1.cell(row=row_ptr, column=1, value="3. Indicadores Promedio por Portal").font = section_font
    row_ptr += 1
    headers_t3 = ["Portal", "Avisos", "% Con Sesgo", "Inclusividad Promedio", "Frases Sesgadas / Aviso", "Cobertura Vocab. ML (%)"]
    for col_idx, h in enumerate(headers_t3, start=1):
        c = ws1.cell(row=row_ptr, column=col_idx, value=h)
        c.fill = sub_fill
        c.font = bold_font
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = cell_border
    row_ptr += 1

    for label, sub in [("Computrabajo Perú", df_comp), ("Bumeran Perú", df_bum), ("Total Consolidado", df_res)]:
        bias_pct = float((sub["categoria_final"] != "Neutro").mean())
        incl_avg = float(sub["indice_inclusividad"].mean())
        spans_avg = float(sub["num_frases_sesgadas"].mean())
        cov_avg = float(sub["cobertura_vocabulario_pct"].mean())
        vals = [label, len(sub), bias_pct, incl_avg, spans_avg, cov_avg]
        for col_idx, v in enumerate(vals, start=1):
            cell = ws1.cell(row=row_ptr, column=col_idx, value=v)
            cell.font = bold_font if label == "Total Consolidado" else body_font
            if label == "Total Consolidado":
                cell.fill = total_fill
                cell.border = total_border
            else:
                cell.border = cell_border
            cell.alignment = Alignment(horizontal="left" if col_idx == 1 else "center", vertical="center")
            if col_idx == 3:
                cell.number_format = "0.0%"
            elif col_idx in (4, 6):
                cell.number_format = "0.0"
            elif col_idx == 5:
                cell.number_format = "0.00"
        row_ptr += 1

    ws1_widths = {1: 24, 2: 18, 3: 18, 4: 16, 5: 16, 6: 14, 7: 14, 8: 18}
    for col_idx, w in ws1_widths.items():
        ws1.column_dimensions[get_column_letter(col_idx)].width = w

    # ==================== HOJA 2: RESULTADOS 200 AVISOS ====================
    ws2 = wb.create_sheet("Resultados 200 Avisos")
    ws2.freeze_panes = "A2"

    res_cols = [
        ("id", "ID", 12),
        ("portal", "Portal", 19),
        ("busqueda", "Búsqueda", 20),
        ("puesto", "Puesto", 38),
        ("categoria_final", "Categoría Final", 16),
        ("nivel_riesgo", "Nivel de Riesgo", 17),
        ("indice_inclusividad", "Inclusividad", 13),
        ("num_frases_sesgadas", "N° Frases", 11),
        ("frases_sesgadas_detectadas", "Frases Sesgadas Detectadas", 46),
        ("categorias_lexicas", "Categorías Léxicas", 22),
        ("categoria_ml_cruda", "ML Crudo (RF)", 15),
        ("modo_decision", "Modo Decisión", 18),
        ("confiabilidad_ml", "Confiabilidad ML", 16),
        ("cobertura_vocabulario_pct", "Cobertura Vocab. (%)", 18),
        ("requisitos_tecnicos", "Requisitos Técnicos", 28),
    ]

    for col_idx, (_, title, width) in enumerate(res_cols, start=1):
        c = ws2.cell(row=1, column=col_idx, value=title)
        c.fill = navy_fill
        c.font = white_font
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = cell_border
        ws2.column_dimensions[get_column_letter(col_idx)].width = width
    ws2.row_dimensions[1].height = 26

    for r_idx, (_, row) in enumerate(df_res.iterrows(), start=2):
        for col_idx, (col_key, _, _) in enumerate(res_cols, start=1):
            val = row[col_key]
            if pd.isna(val):
                val = ""
            cell = ws2.cell(row=r_idx, column=col_idx, value=val)
            cell.font = body_font
            cell.border = cell_border
            if col_key == "categoria_final":
                cell.fill = cat_fills.get(str(val), zebra_fill)
                cell.font = bold_font
                cell.alignment = Alignment(horizontal="center", vertical="center")
            elif col_key == "nivel_riesgo":
                cell.fill = risk_fills.get(str(val), zebra_fill)
                cell.font = bold_font
                cell.alignment = Alignment(horizontal="center", vertical="center")
            elif col_key in ("id", "indice_inclusividad", "num_frases_sesgadas", "confiabilidad_ml", "cobertura_vocabulario_pct"):
                cell.alignment = Alignment(horizontal="center", vertical="center")
                if col_key in ("indice_inclusividad", "cobertura_vocabulario_pct"):
                    cell.number_format = "0.0"
                if r_idx % 2 == 1:
                    cell.fill = zebra_fill
            else:
                cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
                if r_idx % 2 == 1:
                    cell.fill = zebra_fill
        ws2.row_dimensions[r_idx].height = 30

    ws2.auto_filter.ref = f"A1:{get_column_letter(len(res_cols))}{len(df_res) + 1}"

    # ==================== HOJA 3: AVISOS COMPLETOS (TEXTO) ====================
    ws3 = wb.create_sheet("Avisos Completos")
    ws3.freeze_panes = "A2"

    merged = df_ofertas.merge(
        df_res[["id", "categoria_final", "nivel_riesgo", "indice_inclusividad", "frases_sesgadas_detectadas"]],
        on="id",
        how="left",
    )
    text_cols = [
        ("id", "ID", 12),
        ("portal", "Portal", 19),
        ("busqueda", "Búsqueda", 20),
        ("puesto", "Puesto", 36),
        ("categoria_final", "Categoría Final", 16),
        ("nivel_riesgo", "Nivel de Riesgo", 17),
        ("frases_sesgadas_detectadas", "Frases Detectadas", 42),
        ("texto", "Texto Completo Anonimizado", 95),
    ]
    for col_idx, (_, title, width) in enumerate(text_cols, start=1):
        c = ws3.cell(row=1, column=col_idx, value=title)
        c.fill = navy_fill
        c.font = white_font
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = cell_border
        ws3.column_dimensions[get_column_letter(col_idx)].width = width
    ws3.row_dimensions[1].height = 24

    for r_idx, (_, row) in enumerate(merged.iterrows(), start=2):
        for col_idx, (col_key, _, _) in enumerate(text_cols, start=1):
            val = row[col_key]
            if pd.isna(val):
                val = ""
            cell = ws3.cell(row=r_idx, column=col_idx, value=val)
            cell.font = body_font
            cell.border = cell_border
            if col_key == "categoria_final":
                cell.fill = cat_fills.get(str(val), zebra_fill)
                cell.font = bold_font
                cell.alignment = Alignment(horizontal="center", vertical="top")
            elif col_key == "nivel_riesgo":
                cell.fill = risk_fills.get(str(val), zebra_fill)
                cell.font = bold_font
                cell.alignment = Alignment(horizontal="center", vertical="top")
            else:
                cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
                if r_idx % 2 == 1:
                    cell.fill = zebra_fill
        ws3.row_dimensions[r_idx].height = 48

    ws3.auto_filter.ref = f"A1:{get_column_letter(len(text_cols))}{len(merged) + 1}"

    wb.save(excel_path)


def main() -> None:
    ofertas_path = DATA_DIR / "ofertas_reales_peru.csv"
    resultados_path = DATA_DIR / "validacion_real_resultados.csv"
    excel_path = DATA_DIR / "validacion_real_resultados.xlsx"

    # 1. Cargar los 100 avisos existentes (65 Bumeran + 35 Computrabajo)
    df_existing = pd.read_csv(ofertas_path)
    if len(df_existing) == 200:
        # Idempotencia: conservar siempre los primeros 100 originales (PERU-001 a PERU-100)
        df_existing = df_existing.iloc[:100].copy()

    existing_bum = int((df_existing["portal"] == "Bumeran Perú").sum())
    existing_comp = int((df_existing["portal"] == "Computrabajo Perú").sum())
    assert len(df_existing) == 100 and existing_bum == 65 and existing_comp == 35, (
        f"Se esperaban 100 avisos base (65 Bumeran + 35 Computrabajo), encontrados: "
        f"{len(df_existing)} ({existing_bum} Bumeran + {existing_comp} Computrabajo)"
    )

    existing_keys: Set[str] = {
        _norm_key(str(t))[:180] for t in df_existing["texto"].tolist()
    }

    # 2. Cargar lotes nuevos (40 Bumeran + 70 Computrabajo)
    bum_raw, comp_raw = load_new_raw_batches()
    all_company_names = sorted(
        {
            (it.get("empresa") or "").strip()
            for it in (bum_raw + comp_raw)
            if (it.get("empresa") or "").strip()
            and (it.get("empresa") or "").strip().lower() not in NON_COMPANY_BADGES
        }
    )

    # Seleccionar 35 nuevos de Bumeran y 65 nuevos de Computrabajo
    new_bum = select_unique_new(bum_raw, 35, existing_keys, all_company_names)
    new_comp = select_unique_new(comp_raw, 65, existing_keys, all_company_names)
    assert len(new_bum) == 35, f"Solo se obtuvieron {len(new_bum)} avisos nuevos únicos de Bumeran (se requieren 35)"
    assert len(new_comp) == 65, f"Solo se obtuvieron {len(new_comp)} avisos nuevos únicos de Computrabajo (se requieren 65)"

    # 3. Consolidar exactamente 200 avisos (PERU-001..PERU-100 base + PERU-101..PERU-200 nuevos)
    rows_200: List[Dict[str, str]] = df_existing[["id", "portal", "busqueda", "puesto", "texto"]].to_dict(orient="records")
    next_id = 101
    for item in new_bum + new_comp:
        rows_200.append(
            {
                "id": f"PERU-{next_id:03d}",
                "portal": item["portal"],
                "busqueda": item["busqueda"],
                "puesto": item["puesto"],
                "texto": item["texto"],
            }
        )
        next_id += 1

    df_ofertas = pd.DataFrame(rows_200, columns=["id", "portal", "busqueda", "puesto", "texto"])
    assert len(df_ofertas) == 200
    assert int((df_ofertas["portal"] == "Bumeran Perú").sum()) == 100
    assert int((df_ofertas["portal"] == "Computrabajo Perú").sum()) == 100
    df_ofertas.to_csv(ofertas_path, index=False, encoding="utf-8-sig")

    # 4. Evaluar los 200 avisos con Random Forest + detector híbrido
    eval_rows = []
    for _, row in df_ofertas.iterrows():
        res = analyze_job_posting(row["texto"], model_key="random_forest")
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

    df_res = pd.DataFrame(eval_rows)
    df_res.to_csv(resultados_path, index=False, encoding="utf-8-sig")

    # 5. Exportar Excel formateado
    export_formatted_excel(df_ofertas, df_res, excel_path)

    # Verificar que el dataset sintético sigue intacto
    synth_df = pd.read_csv(DATA_DIR / "dataset_sesgos_empleo.csv")
    assert len(synth_df) == 1200

    print(f"=== CONSOLIDADO GUARDADO: {ofertas_path} ({len(df_ofertas)} filas) ===")
    print(f"=== RESULTADOS CSV GUARDADOS: {resultados_path} ({len(df_res)} filas) ===")
    print(f"=== EXCEL FORMATEADO GUARDADO: {excel_path} ===")
    print(f"=== DATASET SINTETICO INTACTO: {len(synth_df)} filas ===")
    print("\n=== TABLA COMPARATIVA POR PORTAL (HÍBRIDO + RANDOM FOREST) ===")
    ct = pd.crosstab(df_res["categoria_final"], df_res["portal"], margins=True)
    for cat in ALL_CATEGORIES:
        c_n = int(((df_res["portal"] == "Computrabajo Perú") & (df_res["categoria_final"] == cat)).sum())
        b_n = int(((df_res["portal"] == "Bumeran Perú") & (df_res["categoria_final"] == cat)).sum())
        t_n = c_n + b_n
        print(f"  {cat:<12}: Computrabajo={c_n:3d} ({c_n:5.1f}%) | Bumeran={b_n:3d} ({b_n:5.1f}%) | Total={t_n:3d} ({t_n/2:5.1f}%)")

    print("\n=== NIVEL DE RIESGO POR PORTAL ===")
    for risk in ["Alto", "Moderado", "Bajo (Inclusivo)"]:
        c_n = int(((df_res["portal"] == "Computrabajo Perú") & (df_res["nivel_riesgo"] == risk)).sum())
        b_n = int(((df_res["portal"] == "Bumeran Perú") & (df_res["nivel_riesgo"] == risk)).sum())
        t_n = c_n + b_n
        print(f"  {risk:<18}: Computrabajo={c_n:3d} ({c_n:5.1f}%) | Bumeran={b_n:3d} ({b_n:5.1f}%) | Total={t_n:3d} ({t_n/2:5.1f}%)")

    print("\n=== EMPRESAS NUEVAS EXTRAÍDAS (PARA AUDITORÍA DE ANONIMIZACIÓN) ===")
    print(", ".join(all_company_names))


if __name__ == "__main__":
    main()
