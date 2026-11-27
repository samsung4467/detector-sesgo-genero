"""Fusión del dataset sintético (1,200) y real de Perú (200) en un corpus consolidado de 1,400 avisos,
exportación a CSV/Excel y reentrenamiento estratificado (80/20) de Random Forest.

Reglas estrictas:
- Conserva intactos data/dataset_sesgos_empleo.csv (1,200) y data/ofertas_reales_peru.csv (200).
- Recalcula las etiquetas de los 200 avisos reales con las reglas limpias de src/bias_analyzer.py.
- Homologa columnas sin nulos ni duplicados y guarda data/dataset_consolidado_1400.csv y .xlsx.
- Reentrena Random Forest con split estratificado 80/20 (stratify=categoria, random_state=42).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd

from src.bias_analyzer import (
    _select_dominant_lexical_category,
    _span_weight,
    detect_biased_spans,
)
from src.config import (
    ALL_CATEGORIES,
    CONSOLIDATED_DATASET_PATH,
    CONSOLIDATED_EXCEL_PATH,
    DATASET_PATH,
    REAL_PERU_DATASET_PATH,
)
from src.train_models import train_and_evaluate

ORIGEN_SINTETICO = "Sintético (1200)"
ORIGEN_REAL_PERU = "Real Perú (200)"

SYNTH_RISK_MAP = {
    "Alto": "Alto",
    "Moderado": "Moderado",
    "Limítrofe": "Moderado",
    "Ninguno": "Bajo (Inclusivo)",
}


def _classify_real_posting_clean(texto: str) -> Dict[str, str]:
    """Recalcula categoría, nivel de riesgo y frases detectadas con las reglas léxicas limpias."""
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


def build_consolidated_1400() -> pd.DataFrame:
    """Homologa y fusiona los 1,200 sintéticos + 200 reales en un DataFrame limpio de 1,400 filas."""
    assert DATASET_PATH.exists(), f"Falta respaldo sintético: {DATASET_PATH}"
    assert REAL_PERU_DATASET_PATH.exists(), f"Falta respaldo real Perú: {REAL_PERU_DATASET_PATH}"

    df_synth = pd.read_csv(DATASET_PATH)
    df_real = pd.read_csv(REAL_PERU_DATASET_PATH)

    assert len(df_synth) == 1200, f"Se esperaban 1200 avisos sintéticos, hay {len(df_synth)}"
    assert len(df_real) == 200, f"Se esperaban 200 avisos reales, hay {len(df_real)}"

    # 1. Homologar bloque sintético (1,200)
    synth_rows: List[Dict[str, str]] = []
    for _, row in df_synth.iterrows():
        texto = str(row["texto"]).strip()
        spans = detect_biased_spans(texto)
        frases = " | ".join(f"[{s.category}] {s.text}" for s in spans) if spans else "Ninguna"
        raw_risk = str(row.get("nivel_sesgo", "Moderado")).strip()
        nivel_riesgo = SYNTH_RISK_MAP.get(raw_risk, "Moderado")
        if str(row["categoria"]).strip() == "Neutro":
            nivel_riesgo = "Bajo (Inclusivo)"

        synth_rows.append(
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

    # 2. Recalcular etiquetas limpias y homologar bloque real de Perú (200)
    real_rows: List[Dict[str, str]] = []
    for _, row in df_real.iterrows():
        texto = str(row["texto"]).strip()
        clean_lbl = _classify_real_posting_clean(texto)
        real_rows.append(
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

    cols = [
        "id",
        "origen",
        "portal",
        "puesto",
        "categoria",
        "nivel_riesgo",
        "frases_detectadas",
        "texto",
    ]
    df_all = pd.DataFrame(synth_rows + real_rows, columns=cols)

    # 3. Verificaciones estrictas de integridad: cero duplicados y cero nulos
    assert len(df_all) == 1400, f"Total esperado 1400, obtenido {len(df_all)}"
    assert int(df_all.isna().sum().sum()) == 0, f"Existen valores nulos:\n{df_all.isna().sum()}"
    assert (df_all.astype(str).apply(lambda s: s.str.strip() == "").sum().sum()) == 0, (
        "Existen cadenas vacías en el dataset consolidado."
    )
    assert df_all["id"].nunique() == 1400, "Existen IDs duplicados."
    assert df_all["texto"].nunique() == 1400, (
        f"Existen textos duplicados: {1400 - df_all['texto'].nunique()}"
    )
    assert set(df_all["categoria"].unique()).issubset(set(ALL_CATEGORIES)), (
        f"Categorías inválidas: {df_all['categoria'].unique()}"
    )

    df_all.to_csv(CONSOLIDATED_DATASET_PATH, index=False, encoding="utf-8-sig")
    return df_all


def export_consolidated_excel(df: pd.DataFrame, metrics_payload: dict, excel_path: Path) -> None:
    """Exporta respaldo Excel formateado del dataset consolidado (1,400) y resumen de métricas."""
    wb = openpyxl.Workbook()

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
        "Masculino": PatternFill("solid", fgColor="DBEAFE"),
        "Femenino": PatternFill("solid", fgColor="FCE7F3"),
        "Excluyente": PatternFill("solid", fgColor="FEE2E2"),
        "LGBTQ+": PatternFill("solid", fgColor="EDE9FE"),
        "Demográfico": PatternFill("solid", fgColor="FEF3C7"),
        "Neutro": PatternFill("solid", fgColor="DCFCE7"),
    }
    risk_fills = {
        "Alto": PatternFill("solid", fgColor="FEE2E2"),
        "Moderado": PatternFill("solid", fgColor="FEF3C7"),
        "Bajo (Inclusivo)": PatternFill("solid", fgColor="DCFCE7"),
    }

    # ==================== HOJA 1: RESUMEN 1400 + RANDOM FOREST ====================
    ws1 = wb.active
    ws1.title = "Resumen 1400"

    ws1["A1"] = "Dataset Consolidado (1,400 avisos) — Sintético (1,200) + Real Perú (200)"
    ws1["A1"].font = title_font
    ws1.merge_cells("A1:G1")

    ws1["A2"] = "Modelo entrenado: Random Forest (TF-IDF + split estratificado 80% entrenamiento / 20% prueba, random_state=42)"
    ws1["A2"].font = Font(name="Calibri", size=10, italic=True, color="475569")
    ws1.merge_cells("A2:G2")

    ws1["A4"] = "1. Distribución Consolidada por Categoría y Origen"
    ws1["A4"].font = section_font

    h1 = ["Categoría", "Sintético (1200)", "Sintético (%)", "Real Perú (200)", "Real Perú (%)", "Total (1400)", "Total (%)"]
    for idx, text in enumerate(h1, start=1):
        c = ws1.cell(row=5, column=idx, value=text)
        c.fill = navy_fill
        c.font = white_font
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border = cell_border
    ws1.row_dimensions[5].height = 24

    df_s = df[df["origen"] == ORIGEN_SINTETICO]
    df_r = df[df["origen"] == ORIGEN_REAL_PERU]

    row_i = 6
    for cat in ALL_CATEGORIES:
        sn = int((df_s["categoria"] == cat).sum())
        rn = int((df_r["categoria"] == cat).sum())
        tn = sn + rn
        vals = [cat, sn, sn / len(df_s), rn, rn / len(df_r), tn, tn / len(df)]
        for ci, v in enumerate(vals, start=1):
            cell = ws1.cell(row=row_i, column=ci, value=v)
            cell.font = bold_font if ci in (1, 6, 7) else body_font
            cell.border = cell_border
            if ci == 1:
                cell.fill = cat_fills.get(cat, zebra_fill)
                cell.alignment = Alignment(horizontal="left", vertical="center")
            else:
                if row_i % 2 == 1:
                    cell.fill = zebra_fill
                cell.alignment = Alignment(horizontal="center", vertical="center")
            if ci in (3, 5, 7):
                cell.number_format = "0.0%"
        ws1.row_dimensions[row_i].height = 20
        row_i += 1

    tot_vals = ["Total", len(df_s), 1.0, len(df_r), 1.0, len(df), 1.0]
    for ci, v in enumerate(tot_vals, start=1):
        cell = ws1.cell(row=row_i, column=ci, value=v)
        cell.font = bold_font
        cell.fill = total_fill
        cell.border = total_border
        cell.alignment = Alignment(horizontal="left" if ci == 1 else "center", vertical="center")
        if ci in (3, 5, 7):
            cell.number_format = "0.0%"
    ws1.row_dimensions[row_i].height = 22

    # Métricas Random Forest (Test 20%)
    rf_info = metrics_payload["models"]["random_forest"]
    tm = rf_info["test_metrics"]
    cv = rf_info["cv_5fold_train"]
    row_i += 2
    ws1.cell(row=row_i, column=1, value="2. Métricas de Random Forest (Test Estratificado 20% = 280 avisos)").font = section_font
    row_i += 1

    h2 = ["Métrica", "Valor Test (20%)", "Notas"]
    for ci, text in enumerate(h2, start=1):
        c = ws1.cell(row=row_i, column=ci, value=text)
        c.fill = sub_fill
        c.font = bold_font
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border = cell_border
    row_i += 1

    rf_rows = [
        ("Accuracy (Test 20%)", tm["accuracy"], "División estratificada 80/20 (1,120 train / 280 test)"),
        ("Precision (Macro)", tm["precision_macro"], "Promedio no ponderado en 6 categorías"),
        ("Recall (Macro)", tm["recall_macro"], "Promedio no ponderado en 6 categorías"),
        ("F1-Score (Macro)", tm["f1_macro"], "Métrica principal balanceada"),
        ("F1-Score (Weighted)", tm["f1_weighted"], "Ponderado por soporte de cada clase"),
        ("CV-5 Accuracy (Train)", cv["accuracy_mean"], f"Desv. Est.: ±{cv['accuracy_std']:.4f}"),
        ("CV-5 F1-Macro (Train)", cv["f1_macro_mean"], f"Desv. Est.: ±{cv['f1_macro_std']:.4f}"),
    ]
    for label, val, note in rf_rows:
        c1 = ws1.cell(row=row_i, column=1, value=label)
        c2 = ws1.cell(row=row_i, column=2, value=val)
        c3 = ws1.cell(row=row_i, column=3, value=note)
        c1.font = bold_font
        c2.font = bold_font
        c3.font = body_font
        c1.border = c2.border = c3.border = cell_border
        c2.number_format = "0.00%"
        c2.alignment = Alignment(horizontal="center", vertical="center")
        row_i += 1

    for ci, w in {1: 26, 2: 18, 3: 16, 4: 18, 5: 16, 6: 16, 7: 14}.items():
        ws1.column_dimensions[get_column_letter(ci)].width = w

    # ==================== HOJA 2: DATASET CONSOLIDADO (1,400) ====================
    ws2 = wb.create_sheet("Dataset Consolidado (1400)")
    ws2.freeze_panes = "A2"

    col_specs = [
        ("id", "ID", 13),
        ("origen", "Origen", 19),
        ("portal", "Portal", 19),
        ("puesto", "Puesto", 36),
        ("categoria", "Categoría", 15),
        ("nivel_riesgo", "Nivel de Riesgo", 17),
        ("frases_detectadas", "Frases Detectadas", 44),
        ("texto", "Texto Completo", 92),
    ]
    for ci, (_, title, w) in enumerate(col_specs, start=1):
        c = ws2.cell(row=1, column=ci, value=title)
        c.fill = navy_fill
        c.font = white_font
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border = cell_border
        ws2.column_dimensions[get_column_letter(ci)].width = w
    ws2.row_dimensions[1].height = 24

    for r_idx, (_, row) in enumerate(df.iterrows(), start=2):
        for ci, (key, _, _) in enumerate(col_specs, start=1):
            val = row[key]
            cell = ws2.cell(row=r_idx, column=ci, value=val)
            cell.font = body_font
            cell.border = cell_border
            if key == "categoria":
                cell.fill = cat_fills.get(str(val), zebra_fill)
                cell.font = bold_font
                cell.alignment = Alignment(horizontal="center", vertical="top")
            elif key == "nivel_riesgo":
                cell.fill = risk_fills.get(str(val), zebra_fill)
                cell.font = bold_font
                cell.alignment = Alignment(horizontal="center", vertical="top")
            elif key in ("id", "origen", "portal"):
                cell.alignment = Alignment(horizontal="center", vertical="top")
                if r_idx % 2 == 1:
                    cell.fill = zebra_fill
            else:
                cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
                if r_idx % 2 == 1:
                    cell.fill = zebra_fill
        ws2.row_dimensions[r_idx].height = 36

    ws2.auto_filter.ref = f"A1:{get_column_letter(len(col_specs))}{len(df) + 1}"
    wb.save(excel_path)


def main() -> None:
    df_1400 = build_consolidated_1400()
    metrics = train_and_evaluate(regenerate_dataset=False)
    export_consolidated_excel(df_1400, metrics, CONSOLIDATED_EXCEL_PATH)

    # Confirmar respaldos intactos
    assert len(pd.read_csv(DATASET_PATH)) == 1200
    assert len(pd.read_csv(REAL_PERU_DATASET_PATH)) == 200

    print(f"=== DATASET CONSOLIDADO CSV: {CONSOLIDATED_DATASET_PATH} ({len(df_1400)} filas) ===")
    print(f"=== DATASET CONSOLIDADO XLSX: {CONSOLIDATED_EXCEL_PATH} ===")
    print("=== RESPALDOS INTACTOS: dataset_sesgos_empleo.csv (1200) | ofertas_reales_peru.csv (200) ===")

    print("\n=== DISTRIBUCIÓN FINAL DE LOS 1,400 AVISOS POR CATEGORÍA ===")
    df_s = df_1400[df_1400["origen"] == ORIGEN_SINTETICO]
    df_r = df_1400[df_1400["origen"] == ORIGEN_REAL_PERU]
    for cat in ALL_CATEGORIES:
        sn = int((df_s["categoria"] == cat).sum())
        rn = int((df_r["categoria"] == cat).sum())
        tn = sn + rn
        print(
            f"  {cat:<12}: Sintético={sn:4d} ({100*sn/len(df_s):5.1f}%) | "
            f"Real Perú={rn:3d} ({100*rn/len(df_r):5.1f}%) | "
            f"Total={tn:4d} ({100*tn/len(df_1400):5.1f}%)"
        )

    print("\n=== DETALLE 200 REALES POR PORTAL (POST-LIMPIEZA) ===")
    print(pd.crosstab(df_r["categoria"], df_r["portal"], margins=True))

    rf = metrics["models"]["random_forest"]
    tm = rf["test_metrics"]
    cv = rf["cv_5fold_train"]
    print("\n=== MÉTRICAS RANDOM FOREST EN DATASET CONSOLIDADO 1,400 (TEST 20% = 280 AVISOS) ===")
    print(f"  Accuracy          : {tm['accuracy']:.4f} ({tm['accuracy']*100:.2f}%)")
    print(f"  Precision (Macro) : {tm['precision_macro']:.4f} ({tm['precision_macro']*100:.2f}%)")
    print(f"  Recall (Macro)    : {tm['recall_macro']:.4f} ({tm['recall_macro']*100:.2f}%)")
    print(f"  F1-Score (Macro)  : {tm['f1_macro']:.4f} ({tm['f1_macro']*100:.2f}%)")
    print(f"  F1-Score (Weight) : {tm['f1_weighted']:.4f} ({tm['f1_weighted']*100:.2f}%)")
    print(f"  CV-5 Accuracy     : {cv['accuracy_mean']:.4f} (±{cv['accuracy_std']:.4f})")
    print(f"  CV-5 F1-Macro     : {cv['f1_macro_mean']:.4f} (±{cv['f1_macro_std']:.4f})")
    print("\n=== MÉTRICAS POR CLASE (TEST 20%) ===")
    for cat in ALL_CATEGORIES:
        pc = rf["per_class"][cat]
        print(
            f"  {cat:<12}: Prec={pc['precision']:.4f} | Rec={pc['recall']:.4f} | "
            f"F1={pc['f1_score']:.4f} | Support={pc['support']}"
        )


if __name__ == "__main__":
    main()
