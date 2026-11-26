"""Plataforma Web Interactiva para Detección y Mitigación de Sesgos de Género en Avisos de Empleo."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.bias_analyzer import (
    analyze_job_posting,
    load_best_model_key,
    load_model,
    load_saved_metrics,
)
from src.config import (
    ALL_CATEGORIES,
    BIAS_CATEGORIES,
    CATEGORY_INFO,
    CONFUSION_CSV_PATH,
    DATASET_PATH,
    METRICS_JSON_PATH,
    METRICS_SUMMARY_CSV_PATH,
    MODEL_DISPLAY_NAMES,
    MODELS_DIR,
    PER_CLASS_CSV_PATH,
    PRELOADED_EXAMPLES,
)
from src.rewriter import rewrite_inclusive
from src.train_models import train_and_evaluate

st.set_page_config(
    page_title="EquiEmpleo AI — Detector de Sesgos en Avisos de Empleo",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- ESTILOS PERSONALIZADOS ----------
st.markdown(
    """
    <style>
        .main-header {
            background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 60%, #4338CA 100%);
            padding: 1.6rem 2rem;
            border-radius: 14px;
            color: #FFFFFF;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.14);
        }
        .main-header h1 {
            margin: 0;
            font-size: 1.85rem;
            font-weight: 700;
            color: #FFFFFF;
        }
        .main-header p {
            margin: 0.45rem 0 0 0;
            font-size: 0.98rem;
            color: #E2E8F0;
        }
        .legend-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 10px;
        }
        .legend-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .rewritten-box {
            line-height: 1.8;
            font-size: 1.02rem;
            padding: 16px 18px;
            background-color: #F0FDF4;
            border: 1px solid #86EFAC;
            border-left: 5px solid #059669;
            border-radius: 10px;
            color: #064E3B;
        }
        .tech-pill {
            display: inline-block;
            background-color: #EFF6FF;
            color: #1D4ED8;
            border: 1px solid #BFDBFE;
            border-radius: 6px;
            padding: 2px 9px;
            margin: 3px 4px 3px 0;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .cat-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.92rem;
        }
        /* Ajuste tipográfico de tarjetas KPI y st.metric para evitar recortes */
        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 12px 14px;
            min-height: 92px;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.78rem !important;
            color: #475569 !important;
            white-space: normal !important;
            overflow: visible !important;
        }
        [data-testid="stMetricLabel"] p {
            font-size: 0.78rem !important;
            line-height: 1.25 !important;
        }
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] > div {
            font-size: 1.32rem !important;
            line-height: 1.25 !important;
            font-weight: 700 !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            word-break: break-word !important;
        }
        [data-testid="stMetricDelta"],
        [data-testid="stMetricDelta"] > div {
            font-size: 0.76rem !important;
            white-space: normal !important;
        }
        section[data-testid="stSidebar"] [data-testid="stMetric"] {
            padding: 8px 10px;
            min-height: auto;
        }
        section[data-testid="stSidebar"] [data-testid="stMetricValue"],
        section[data-testid="stSidebar"] [data-testid="stMetricValue"] > div {
            font-size: 1.12rem !important;
        }
        .kpi-card {
            padding: 12px 14px;
            border-radius: 10px;
            min-height: 92px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .kpi-card-label {
            font-size: 0.74rem;
            color: #475569;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .kpi-card-value {
            font-size: 1.20rem;
            font-weight: 800;
            margin-top: 3px;
            line-height: 1.2;
            white-space: normal;
            word-break: break-word;
        }
        .kpi-card-sub {
            font-size: 0.76rem;
            color: #334155;
            margin-top: 2px;
        }
        .kpi-card-multi {
            font-size: 0.74rem;
            color: #991B1B;
            background-color: rgba(255, 255, 255, 0.78);
            border-left: 3px solid #DC2626;
            border-radius: 4px;
            padding: 4px 7px;
            margin-top: 6px;
            line-height: 1.3;
            white-space: normal;
            word-break: break-word;
        }
        /* Tabla HTML de hallazgos y transformaciones con ajuste de línea sin cortes ni desbordes */
        .findings-table-wrap {
            width: 100%;
            overflow-x: auto;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            background: #FFFFFF;
            margin-top: 6px;
        }
        table.findings-table {
            width: 100%;
            border-collapse: collapse;
            table-layout: auto;
            font-size: 0.87rem;
            color: #1E293B;
        }
        table.findings-table thead th {
            background-color: #F8FAFC;
            color: #334155;
            font-weight: 700;
            text-align: left;
            padding: 10px 12px;
            border-bottom: 2px solid #E2E8F0;
            white-space: normal;
        }
        table.findings-table tbody td {
            padding: 10px 12px;
            border-bottom: 1px solid #F1F5F9;
            vertical-align: top;
            white-space: normal !important;
            word-wrap: break-word;
            overflow-wrap: break-word;
            line-height: 1.45;
        }
        table.findings-table th.col-cat,
        table.findings-table td.col-cat {
            min-width: 138px;
            width: 15%;
            text-align: center !important;
            vertical-align: middle;
            padding-left: 10px;
            padding-right: 10px;
        }
        table.findings-table tbody tr:last-child td {
            border-bottom: none;
        }
        table.findings-table tbody tr:hover {
            background-color: #F8FAFC;
        }
        .tbl-cat-pill {
            display: inline-block;
            padding: 3px 9px;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 700;
            line-height: 1.25;
            white-space: nowrap;
            max-width: 100%;
            box-sizing: border-box;
            text-align: center;
        }
        /* Mini-tabla comparativa ML compacta sin recorte horizontal */
        .mini-ml-wrap {
            width: 100%;
            overflow-x: auto;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            background: #FFFFFF;
        }
        table.mini-ml-table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            font-size: 0.73rem;
            color: #1E293B;
        }
        table.mini-ml-table thead th {
            background-color: #F8FAFC;
            color: #334155;
            font-weight: 700;
            padding: 6px 3px;
            border-bottom: 1.5px solid #E2E8F0;
            text-align: center;
            white-space: nowrap;
        }
        table.mini-ml-table thead th:first-child {
            text-align: left;
            padding-left: 7px;
        }
        table.mini-ml-table tbody td {
            padding: 6px 3px;
            border-bottom: 1px solid #F1F5F9;
            text-align: center;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        table.mini-ml-table tbody td:first-child {
            text-align: left;
            padding-left: 7px;
            font-weight: 600;
        }
        table.mini-ml-table tbody tr:last-child td {
            border-bottom: none;
        }
        .mini-pred-pill {
            display: inline-block;
            padding: 1px 5px;
            border-radius: 999px;
            font-size: 0.68rem;
            font-weight: 700;
            line-height: 1.25;
        }
        /* Banners explicativos del motor híbrido y panel de explicabilidad ML */
        .hybrid-banner-info {
            background-color: #F0F9FF;
            border: 1px solid #BAE6FD;
            border-left: 4px solid #0284C7;
            border-radius: 10px;
            padding: 11px 15px;
            margin: 10px 0 4px 0;
            color: #0C4A6E;
            font-size: 0.88rem;
            line-height: 1.5;
        }
        .hybrid-banner-warn {
            background-color: #FFFBEB;
            border: 1px solid #FDE68A;
            border-left: 4px solid #D97706;
            border-radius: 10px;
            padding: 11px 15px;
            margin: 10px 0 4px 0;
            color: #78350F;
            font-size: 0.88rem;
            line-height: 1.5;
        }
        .ml-exp-header {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
        }
        .ml-rel-pill {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 700;
        }
        .ml-diag-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 8px;
            margin-bottom: 12px;
        }
        .ml-diag-box {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 7px 10px;
        }
        .ml-diag-label {
            font-size: 0.71rem;
            color: #64748B;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }
        .ml-diag-val {
            font-size: 0.96rem;
            font-weight: 700;
            color: #0F172A;
            margin-top: 2px;
        }
        .ml-diag-sub {
            font-size: 0.72rem;
            color: #475569;
        }
        .ml-term-row {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 6px;
            font-size: 0.82rem;
            color: #1E293B;
        }
        .ml-term-name {
            min-width: 125px;
            max-width: 44%;
            font-weight: 600;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .ml-term-bar-bg {
            flex: 1;
            height: 8px;
            background-color: #E2E8F0;
            border-radius: 999px;
            overflow: hidden;
        }
        .ml-term-bar-fill {
            height: 100%;
            border-radius: 999px;
        }
        .ml-term-weight {
            min-width: 72px;
            text-align: right;
            font-size: 0.74rem;
            color: #475569;
            font-variant-numeric: tabular-nums;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- RECURSOS EN CACHÉ ----------
@st.cache_resource(show_spinner=False)
def get_cached_models() -> Dict[str, object]:
    """Carga los 3 pipelines ML entrenados en memoria."""
    loaded = {}
    for key in MODEL_DISPLAY_NAMES:
        model_path = MODELS_DIR / f"{key}.joblib"
        if model_path.exists():
            loaded[key] = load_model(key)
    return loaded


@st.cache_data(show_spinner=False)
def get_cached_metrics() -> dict:
    return load_saved_metrics()


@st.cache_data(show_spinner=False)
def get_cached_dataset() -> pd.DataFrame:
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)
    return pd.DataFrame()


def ensure_artifacts_ready() -> None:
    """Si faltan modelos o métricas, ejecuta generación y entrenamiento automáticamente."""
    missing_models = any(not (MODELS_DIR / f"{k}.joblib").exists() for k in MODEL_DISPLAY_NAMES)
    if missing_models or not METRICS_JSON_PATH.exists() or not DATASET_PATH.exists():
        with st.spinner("Generando dataset sintético balanceado y entrenando los 3 modelos ML..."):
            train_and_evaluate(regenerate_dataset=not DATASET_PATH.exists())
        get_cached_models.clear()
        get_cached_metrics.clear()
        get_cached_dataset.clear()


def render_category_legend() -> None:
    pills = []
    for cat in BIAS_CATEGORIES:
        meta = CATEGORY_INFO[cat]
        pills.append(
            f'<span class="legend-pill" style="background:{meta["bg"]}; color:{meta["text"]}; '
            f'border:1px solid {meta["border"]};">'
            f'<span style="width:9px;height:9px;border-radius:50%;background:{meta["color"]};display:inline-block;"></span>'
            f"{cat}</span>"
        )
    st.markdown(f'<div class="legend-wrap">{"".join(pills)}</div>', unsafe_allow_html=True)


def make_bias_probability_chart(bias_probs: Dict[str, float]) -> go.Figure:
    cats = list(BIAS_CATEGORIES)
    vals = [round(bias_probs.get(c, 0.0) * 100.0, 2) for c in cats]
    colors = [CATEGORY_INFO[c]["color"] for c in cats]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=cats,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside",
            hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Probabilidad por categoría de sesgo (Modelo ML)",
        xaxis=dict(title="Probabilidad (%)", range=[0, max(105, max(vals) + 15)], showgrid=True, gridcolor="#E2E8F0"),
        yaxis=dict(autorange="reversed", title=""),
        height=310,
        margin=dict(l=10, r=40, t=45, b=35),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(size=13, color="#1E293B"),
    )
    return fig


def make_confusion_matrix_figure(cm_matrix: list, labels: list, title: str) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=cm_matrix,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=cm_matrix,
            texttemplate="%{text}",
            textfont={"size": 13},
            hovertemplate="Real: <b>%{y}</b><br>Predicho: <b>%{x}</b><br>Casos: %{z}<extra></extra>",
            showscale=True,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Clase Predicha",
        yaxis_title="Clase Real",
        yaxis=dict(autorange="reversed"),
        height=420,
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


def read_file_bytes(path: Path) -> bytes:
    return path.read_bytes() if path.exists() else b""


def render_findings_html_table(spans) -> str:
    """Renderiza la tabla de frases detectadas en HTML con white-space: normal y columna Categoría centrada."""
    rows_html = []
    for s in spans:
        meta = CATEGORY_INFO[s.category]
        badge = (
            f'<span class="tbl-cat-pill" style="background:{meta["bg"]}; color:{meta["text"]}; '
            f'border:1px solid {meta["border"]};">{html.escape(s.category)}</span>'
        )
        rows_html.append(
            "<tr>"
            f'<td style="font-weight:600; width:21%;">{html.escape(s.text)}</td>'
            f'<td class="col-cat">{badge}</td>'
            f'<td style="width:33%;">{html.escape(s.explanation)}</td>'
            f'<td style="width:31%;">{html.escape(s.suggestion)}</td>'
            "</tr>"
        )
    return (
        '<div class="findings-table-wrap">'
        '<table class="findings-table">'
        "<thead><tr>"
        '<th style="width:21%;">Frase detectada</th>'
        '<th class="col-cat">Categoría</th>'
        '<th style="width:33%;">Motivo del sesgo</th>'
        '<th style="width:31%;">Sugerencia inclusiva</th>'
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table></div>"
    )


def render_mini_ml_comparison_table(comp_results) -> str:
    """Renderiza una mini-tabla HTML compacta con cabeceras abreviadas para evitar recortes."""
    short_headers = {
        "Masculino": "Masc.",
        "Femenino": "Fem.",
        "Excluyente": "Excl.",
        "LGBTQ+": "LGBTQ+",
        "Demográfico": "Demog.",
        "Neutro": "Neutro",
    }
    short_preds = {
        "Masculino": "Masc.",
        "Femenino": "Fem.",
        "Excluyente": "Excl.",
        "LGBTQ+": "LGBTQ+",
        "Demográfico": "Demog.",
        "Neutro": "Neutro",
    }
    rows_html = []
    for m_name, m_res in comp_results:
        pred_cat = m_res.raw_ml_category
        meta = CATEGORY_INFO[pred_cat]
        pred_badge = (
            f'<span class="mini-pred-pill" title="Predicción cruda ML: {html.escape(pred_cat)}" '
            f'style="background:{meta["bg"]}; color:{meta["text"]}; '
            f'border:1px solid {meta["border"]};">{html.escape(short_preds[pred_cat])}</span>'
        )
        prob_cells = []
        for cat in ALL_CATEGORIES:
            pct = m_res.probabilities[cat] * 100.0
            style = "font-weight:700; color:#0F172A;" if cat == pred_cat else "color:#475569;"
            prob_cells.append(f'<td style="{style}">{pct:.1f}%</td>')
        rows_html.append(
            "<tr>"
            f'<td title="{html.escape(m_name)}">{html.escape(m_name)}</td>'
            f"<td>{pred_badge}</td>"
            + "".join(prob_cells)
            + "</tr>"
        )

    header_cells = "".join(
        f'<th title="{html.escape(cat)}">{html.escape(short_headers[cat])}</th>'
        for cat in ALL_CATEGORIES
    )
    return (
        '<div class="mini-ml-wrap">'
        '<table class="mini-ml-table">'
        "<colgroup>"
        '<col style="width:20%;">'
        '<col style="width:14%;">'
        '<col style="width:11%;">'
        '<col style="width:11%;">'
        '<col style="width:11%;">'
        '<col style="width:11%;">'
        '<col style="width:11%;">'
        '<col style="width:11%;">'
        "</colgroup>"
        "<thead><tr>"
        "<th>Modelo</th>"
        '<th title="Predicción cruda del clasificador ML">Pred. ML</th>'
        f"{header_cells}"
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table></div>"
    )


def render_hybrid_status_banner(analysis) -> str:
    """Renderiza un aviso explicativo limpio cuando aplica calibración OOD o alerta de sesgo implícito."""
    explanation = html.escape(analysis.decision_explanation)
    if analysis.decision_mode == "ml_implicit_bias":
        return (
            '<div class="hybrid-banner-warn">'
            "<strong>⚠️ Alerta de posible sesgo implícito (patrón estadístico ML):</strong> "
            f"{explanation}"
            "</div>"
        )
    if analysis.decision_mode == "calibrated_neutral":
        return (
            '<div class="hybrid-banner-info">'
            "<strong>ℹ️ Calibración híbrida aplicada (baja confianza ML / vocabulario fuera de dominio):</strong> "
            f"{explanation}"
            "</div>"
        )
    if analysis.decision_mode == "lexical_explicit" and analysis.predicted_class != analysis.raw_ml_category:
        return (
            '<div class="hybrid-banner-info">'
            "<strong>ℹ️ Priorización de evidencia léxica explícita:</strong> "
            f"{explanation}"
            "</div>"
        )
    return ""


def render_ml_explainability_html(analysis) -> str:
    """Renderiza métricas de incertidumbre, cobertura TF-IDF y términos influyentes del clasificador ML."""
    rel_styles = {
        "alta": ("#DCFCE7", "#166534", "#86EFAC"),
        "media": ("#FEF3C7", "#92400E", "#FCD34D"),
        "baja": ("#FEE2E2", "#991B1B", "#FCA5A5"),
    }
    bg, fg, border = rel_styles.get(analysis.ml_reliability, ("#F1F5F9", "#334155", "#CBD5E1"))
    raw_meta = CATEGORY_INFO.get(analysis.raw_ml_category, CATEGORY_INFO["Neutro"])
    bar_color = raw_meta["color"]

    header_html = (
        '<div class="ml-exp-header">'
        f'<span class="ml-rel-pill" style="background:{bg}; color:{fg}; border:1px solid {border};">'
        f"Confiabilidad ML: {html.escape(analysis.ml_reliability_label)}</span>"
        f'<span style="font-size:0.78rem; color:#475569;">'
        f'Señal cruda: <strong style="color:{raw_meta["text"]};">{html.escape(analysis.raw_ml_category)}</strong> '
        f"({analysis.ml_confidence * 100:.1f}%)</span>"
        "</div>"
    )

    cov_pct = analysis.vocabulary_coverage * 100.0
    diag_grid_html = (
        '<div class="ml-diag-grid">'
        '<div class="ml-diag-box">'
        '<div class="ml-diag-label">Cobertura TF-IDF</div>'
        f'<div class="ml-diag-val">{cov_pct:.1f}%</div>'
        f'<div class="ml-diag-sub">{analysis.recognized_tokens}/{analysis.total_tokens} tokens · '
        f"{analysis.matched_features} n-gramas</div>"
        "</div>"
        '<div class="ml-diag-box">'
        '<div class="ml-diag-label">Confianza máx. ML</div>'
        f'<div class="ml-diag-val">{analysis.ml_confidence * 100:.1f}%</div>'
        f'<div class="ml-diag-sub">Clase cruda: {html.escape(analysis.raw_ml_category)}</div>'
        "</div>"
        '<div class="ml-diag-box">'
        '<div class="ml-diag-label">Margen (Top1 − Top2)</div>'
        f'<div class="ml-diag-val">{analysis.ml_margin * 100:.1f}%</div>'
        '<div class="ml-diag-sub">Separación entre clases</div>'
        "</div>"
        '<div class="ml-diag-box">'
        '<div class="ml-diag-label">Entropía normalizada</div>'
        f'<div class="ml-diag-val">{analysis.ml_entropy:.2f}</div>'
        '<div class="ml-diag-sub">0.00 certeza · 1.00 dispersión</div>'
        "</div>"
        "</div>"
    )

    if analysis.influential_terms:
        term_rows = []
        for item in analysis.influential_terms:
            width_pct = max(8.0, min(100.0, item.influence_score * 100.0))
            node_tag = (
                ' <span style="font-size:0.66rem; background:#EEF2FF; color:#3730A3; '
                'padding:1px 5px; border-radius:4px; font-weight:600;">nodo activo</span>'
                if item.source == "ruta_arbol"
                else ""
            )
            term_rows.append(
                '<div class="ml-term-row">'
                f'<div class="ml-term-name" title="{html.escape(item.term)}">'
                f"{html.escape(item.term)}{node_tag}</div>"
                '<div class="ml-term-bar-bg">'
                f'<div class="ml-term-bar-fill" style="width:{width_pct:.1f}%; background:{bar_color};"></div>'
                "</div>"
                f'<div class="ml-term-weight">TF-IDF {item.tfidf_weight:.2f}</div>'
                "</div>"
            )
        terms_html = (
            '<div style="font-size:0.82rem; font-weight:700; color:#1E293B; margin-bottom:2px;">'
            f"Términos del aviso que más influyeron en {html.escape(analysis.model_name)}"
            "</div>"
            f'<div style="font-size:0.74rem; color:#64748B; margin-bottom:8px;">'
            f"Método: {html.escape(analysis.explainability_method)}</div>"
            + "".join(term_rows)
        )
    else:
        terms_html = (
            '<div style="font-size:0.82rem; color:#64748B; background:#F8FAFC; '
            'border:1px solid #E2E8F0; border-radius:8px; padding:8px 10px;">'
            "Ningún término del texto coincide con el vocabulario entrenado del vectorizador TF-IDF."
            "</div>"
        )

    return header_html + diag_grid_html + terms_html


def render_changes_html_table(changes) -> str:
    """Renderiza la tabla de transformaciones de reescritura en HTML sin recortar texto."""
    rows_html = []
    for ch in changes:
        meta = CATEGORY_INFO.get(ch.category, CATEGORY_INFO["Neutro"])
        badge = (
            f'<span class="tbl-cat-pill" style="background:{meta["bg"]}; color:{meta["text"]}; '
            f'border:1px solid {meta["border"]};">{html.escape(ch.category)}</span>'
        )
        rows_html.append(
            "<tr>"
            f'<td class="col-cat">{badge}</td>'
            f'<td style="width:27%;">{html.escape(ch.original)}</td>'
            f'<td style="width:30%; color:#065F46; font-weight:500;">{html.escape(ch.replacement)}</td>'
            f'<td style="width:28%;">{html.escape(ch.reason)}</td>'
            "</tr>"
        )
    return (
        '<div class="findings-table-wrap">'
        '<table class="findings-table">'
        "<thead><tr>"
        '<th class="col-cat">Categoría</th>'
        '<th style="width:27%;">Expresión original</th>'
        '<th style="width:30%;">Reemplazo inclusivo</th>'
        '<th style="width:28%;">Motivo</th>'
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table></div>"
    )


# ---------- INICIALIZACIÓN ----------
ensure_artifacts_ready()
models_dict = get_cached_models()
metrics_data = get_cached_metrics()
best_model_key = load_best_model_key()

if "job_text" not in st.session_state:
    first_example_name = next(iter(PRELOADED_EXAMPLES.keys()))
    st.session_state.job_text = PRELOADED_EXAMPLES[first_example_name]
    st.session_state.selected_example = first_example_name

if "rewritten_output" not in st.session_state:
    st.session_state.rewritten_output = None

if "last_analyzed_text" not in st.session_state:
    st.session_state.last_analyzed_text = ""


# ---------- BARRA LATERAL ----------
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    model_keys = list(MODEL_DISPLAY_NAMES.keys())
    default_idx = model_keys.index(best_model_key) if best_model_key in model_keys else 0
    selected_model_key = st.selectbox(
        "Clasificador ML activo",
        options=model_keys,
        index=default_idx,
        format_func=lambda k: (
            f"{MODEL_DISPLAY_NAMES[k]} ⭐ (Mejor)" if k == best_model_key else MODEL_DISPLAY_NAMES[k]
        ),
    )

    if metrics_data and selected_model_key in metrics_data.get("models", {}):
        m_info = metrics_data["models"][selected_model_key]["test_metrics"]
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{m_info['accuracy'] * 100:.1f}%")
        c2.metric("F1-Macro", f"{m_info['f1_macro'] * 100:.1f}%")

    append_clause = st.checkbox(
        "Incluir cláusula de igualdad al reescribir",
        value=True,
        help="Añade una declaración de igualdad de oportunidades al final de la oferta mitigada.",
    )

    st.divider()
    st.markdown("### 🏷️ Categorías evaluadas")
    for cat in ALL_CATEGORIES:
        meta = CATEGORY_INFO[cat]
        st.markdown(
            f'<div style="margin-bottom:8px; padding:7px 10px; border-left:4px solid {meta["color"]}; '
            f'background:{meta["bg"]}; border-radius:6px;">'
            f'<strong style="color:{meta["text"]};">{cat}</strong><br>'
            f'<span style="font-size:0.78rem; color:#334155;">{meta["short_desc"]}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    if st.button("🔄 Reentrenar modelos ML", width="stretch"):
        with st.spinner("Reentrenando Random Forest, SVM y Decision Tree..."):
            train_and_evaluate(regenerate_dataset=False)
        get_cached_models.clear()
        get_cached_metrics.clear()
        st.success("Modelos y métricas actualizados correctamente.")
        st.rerun()


# ---------- ENCABEZADO ----------
st.markdown(
    """
    <div class="main-header">
        <h1>⚖️ Plataforma de Detección y Mitigación de Sesgos de Género en Avisos de Empleo</h1>
        <p>
            Análisis NLP en español con TF-IDF + Machine Learning (Random Forest, SVM y Decision Tree),
            resaltado explicable de frases sesgadas y reescritura inclusiva preservando requisitos técnicos.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- PESTAÑAS PRINCIPALES ----------
total_dataset_count = metrics_data.get("dataset", {}).get("total_samples", 1200)
tab_analyzer, tab_metrics, tab_dataset = st.tabs(
    [
        "🔍 Analizador y Mitigador",
        "📊 Comparativa de Modelos ML",
        f"🗂️ Dataset Sintético ({total_dataset_count} avisos)",
    ]
)


# ==================== TAB 1: ANALIZADOR Y MITIGADOR ====================
with tab_analyzer:
    st.markdown("### 1. Ingresa o selecciona una oferta laboral en español")

    col_ex, col_btn_load, col_btn_clear = st.columns([4, 1.3, 1.1])
    with col_ex:
        example_choice = st.selectbox(
            "Ejemplos de prueba precargados",
            options=list(PRELOADED_EXAMPLES.keys()),
            index=list(PRELOADED_EXAMPLES.keys()).index(st.session_state.selected_example)
            if st.session_state.selected_example in PRELOADED_EXAMPLES
            else 0,
            label_visibility="collapsed",
        )
    with col_btn_load:
        if st.button("📋 Cargar ejemplo", width="stretch"):
            st.session_state.selected_example = example_choice
            st.session_state.job_text = PRELOADED_EXAMPLES[example_choice]
            st.session_state.rewritten_output = None
            st.rerun()
    with col_btn_clear:
        if st.button("🧹 Limpiar", width="stretch"):
            st.session_state.job_text = ""
            st.session_state.rewritten_output = None
            st.rerun()

    job_text_input = st.text_area(
        "Pega aquí el texto completo del aviso de empleo:",
        value=st.session_state.job_text,
        height=185,
        placeholder="Ej.: Buscamos un desarrollador senior aguerrido, menor de 30 años, con experiencia en Python, SQL y AWS...",
    )

    if job_text_input != st.session_state.job_text:
        st.session_state.job_text = job_text_input
        st.session_state.rewritten_output = None

    act_col1, act_col2, act_col3 = st.columns([1.4, 1.9, 2.7])
    with act_col1:
        analyze_clicked = st.button("🔎 Analizar oferta", type="primary", width="stretch")
    with act_col2:
        rewrite_clicked = st.button("✨ Generar versión neutra e inclusiva", width="stretch")
    with act_col3:
        st.caption(
            f"Modelo activo: **{MODEL_DISPLAY_NAMES[selected_model_key]}** · "
            "Los requisitos técnicos (lenguajes, herramientas, certificaciones y experiencia) se preservan intactos."
        )

    if not st.session_state.job_text.strip():
        st.info("Pega una oferta laboral o carga uno de los ejemplos precargados para iniciar el análisis.")
    else:
        analysis = analyze_job_posting(
            st.session_state.job_text,
            model_key=selected_model_key,
            pipeline=models_dict.get(selected_model_key),
        )

        # KPIs principales
        st.markdown("### 2. Resultados del diagnóstico de sesgos")
        pred_meta = CATEGORY_INFO[analysis.predicted_class]
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            multi_bias_html = ""
            if len(analysis.detected_categories) >= 2:
                cats_str = ", ".join(analysis.detected_categories)
                multi_bias_html = (
                    '<div class="kpi-card-multi">'
                    f'<strong>Sesgo múltiple detectado:</strong> {html.escape(cats_str)}'
                    "</div>"
                )
            if analysis.decision_mode == "calibrated_neutral":
                if analysis.raw_ml_category != "Neutro":
                    kpi1_sub = (
                        f"Calibrado · ML crudo: {html.escape(analysis.raw_ml_category)} "
                        f"({analysis.ml_confidence * 100:.1f}%)"
                    )
                else:
                    kpi1_sub = (
                        f"Calibrado · Cobertura TF-IDF: {analysis.vocabulary_coverage * 100:.0f}%"
                    )
            elif analysis.decision_mode == "ml_implicit_bias":
                kpi1_sub = (
                    f"Sesgo implícito ML ({analysis.ml_confidence * 100:.1f}%) · Sin frases explícitas"
                )
            elif analysis.decision_mode == "lexical_explicit" and analysis.predicted_class != analysis.raw_ml_category:
                kpi1_sub = (
                    f"Evidencia léxica · ML crudo: {html.escape(analysis.raw_ml_category)} "
                    f"({analysis.ml_confidence * 100:.1f}%)"
                )
            else:
                kpi1_sub = (
                    f"Confianza ML: {analysis.probabilities[analysis.predicted_class] * 100:.1f}% · "
                    f"Cobertura: {analysis.vocabulary_coverage * 100:.0f}%"
                )

            st.markdown(
                f'<div class="kpi-card" style="background:{pred_meta["bg"]}; '
                f'border:1px solid {pred_meta["border"]};">'
                f'<div class="kpi-card-label">DIAGNÓSTICO HÍBRIDO</div>'
                f'<div class="kpi-card-value" style="color:{pred_meta["text"]};">'
                f"{html.escape(analysis.predicted_class)}</div>"
                f'<div class="kpi-card-sub">{kpi1_sub}</div>'
                f"{multi_bias_html}"
                f"</div>",
                unsafe_allow_html=True,
            )
        with kpi2:
            st.metric("Índice de Inclusividad", f"{analysis.inclusivity_score:.1f} / 100")
        with kpi3:
            st.metric("Nivel de Riesgo", analysis.risk_level)
        with kpi4:
            if analysis.spans:
                kpi4_delta = f"{len(analysis.detected_categories)} categoría(s)"
                kpi4_delta_color = "inverse"
            elif analysis.decision_mode == "ml_implicit_bias":
                kpi4_delta = "Sesgo implícito ML"
                kpi4_delta_color = "off"
            else:
                kpi4_delta = "Sin alertas léxicas"
                kpi4_delta_color = "normal"
            st.metric(
                "Frases Sesgadas Detectadas",
                f"{len(analysis.spans)}",
                delta=kpi4_delta,
                delta_color=kpi4_delta_color,
            )

        banner_html = render_hybrid_status_banner(analysis)
        if banner_html:
            st.markdown(banner_html, unsafe_allow_html=True)

        st.markdown("")
        col_left, col_right = st.columns([1.25, 1.0], gap="large")

        with col_left:
            st.markdown("#### 🖍️ Resaltado visual de frases sesgadas")
            render_category_legend()
            st.markdown(analysis.highlighted_html, unsafe_allow_html=True)
            if analysis.spans:
                st.caption("Pasa el cursor sobre cualquier frase resaltada para ver la explicación y sugerencia.")
            elif analysis.decision_mode == "ml_implicit_bias":
                st.warning(
                    "No se detectaron frases sesgadas explícitas en el texto, pero el clasificador ML "
                    "identificó un posible patrón estadístico implícito. Revisa los términos influyentes "
                    "en el panel derecho."
                )
            elif analysis.decision_mode == "calibrated_neutral":
                st.success(
                    "No se detectaron frases sesgadas explícitas en el texto analizado. "
                    "El motor híbrido calibró el diagnóstico final a Neutro ante la baja confianza "
                    "estadística o vocabulario fuera de dominio del modelo ML."
                )
            else:
                st.success("No se detectaron frases sesgadas explícitas en el texto analizado.")

            if analysis.spans:
                st.markdown("#### 📋 Detalle de frases detectadas")
                st.markdown(render_findings_html_table(analysis.spans), unsafe_allow_html=True)

        with col_right:
            st.markdown("#### 📈 Probabilidades por categoría de sesgo")
            fig_probs = make_bias_probability_chart(analysis.bias_probabilities)
            st.plotly_chart(fig_probs, width="stretch")

            neutro_pct = analysis.probabilities["Neutro"] * 100.0
            neutro_meta = CATEGORY_INFO["Neutro"]
            st.markdown(
                f'<div style="padding:10px 14px; border-radius:8px; background:{neutro_meta["bg"]}; '
                f'border:1px solid {neutro_meta["border"]}; color:{neutro_meta["text"]}; margin-bottom:12px;">'
                f'<strong>Probabilidad clase Neutro / Inclusivo (ML crudo):</strong> {neutro_pct:.2f}%'
                f"</div>",
                unsafe_allow_html=True,
            )

            with st.expander(
                f"🔬 Explicabilidad ML y cobertura TF-IDF ({analysis.vocabulary_coverage * 100:.0f}% reconocido)",
                expanded=analysis.decision_mode in ("ml_implicit_bias", "calibrated_neutral"),
            ):
                st.markdown(render_ml_explainability_html(analysis), unsafe_allow_html=True)

            with st.expander("🤖 Comparar predicción entre los 3 modelos ML en esta oferta", expanded=True):
                comp_results = []
                for m_key, m_name in MODEL_DISPLAY_NAMES.items():
                    m_res = analyze_job_posting(
                        st.session_state.job_text,
                        model_key=m_key,
                        pipeline=models_dict.get(m_key),
                    )
                    comp_results.append((m_name, m_res))
                st.markdown(render_mini_ml_comparison_table(comp_results), unsafe_allow_html=True)
                if any(
                    m_res.decision_mode == "calibrated_neutral"
                    or m_res.predicted_class != m_res.raw_ml_category
                    for _, m_res in comp_results
                ):
                    st.caption(
                        "Por transparencia académica, esta tabla conserva la predicción y probabilidades "
                        "crudas de cada clasificador ML; el diagnóstico principal aplica la calibración híbrida."
                    )

            if analysis.technical_terms:
                st.markdown("#### 🛠️ Requisitos técnicos identificados (protegidos)")
                pills_html = "".join(
                    f'<span class="tech-pill">{term}</span>' for term in analysis.technical_terms
                )
                st.markdown(pills_html, unsafe_allow_html=True)

        # ---------- SECCIÓN DE REESCRITURA INCLUSIVA ----------
        if rewrite_clicked:
            st.session_state.rewritten_output = rewrite_inclusive(
                st.session_state.job_text,
                append_equality_clause=append_clause,
            )

        if st.session_state.rewritten_output is not None:
            rw = st.session_state.rewritten_output
            st.divider()
            st.markdown("### 3. Versión reescrita con lenguaje neutro e inclusivo")

            re_analysis = analyze_job_posting(
                rw.rewritten_text,
                model_key=selected_model_key,
                pipeline=models_dict.get(selected_model_key),
            )

            # Impacto antes vs después
            imp1, imp2, imp3, imp4 = st.columns(4)
            delta_incl = round(re_analysis.inclusivity_score - analysis.inclusivity_score, 1)
            delta_neutro = round(
                (re_analysis.probabilities["Neutro"] - analysis.probabilities["Neutro"]) * 100.0,
                1,
            )
            delta_spans = len(re_analysis.spans) - len(analysis.spans)
            total_tech = len(rw.preserved_technical_terms) + len(rw.missing_technical_terms)
            tech_pct = (
                (len(rw.preserved_technical_terms) / total_tech) * 100.0 if total_tech > 0 else 100.0
            )

            imp1.metric(
                "Nuevo Índice de Inclusividad",
                f"{re_analysis.inclusivity_score:.1f} / 100",
                delta=f"{delta_incl:+.1f} pts",
            )
            imp2.metric(
                "Probabilidad Neutro (ML)",
                f"{re_analysis.probabilities['Neutro']*100:.1f}%",
                delta=f"{delta_neutro:+.1f}%",
            )
            imp3.metric(
                "Frases Sesgadas Restantes",
                f"{len(re_analysis.spans)}",
                delta=f"{delta_spans:+d}",
                delta_color="inverse",
            )
            imp4.metric(
                "Requisitos Técnicos Preservados",
                f"{len(rw.preserved_technical_terms)}/{total_tech}" if total_tech else "N/A",
                delta=f"{tech_pct:.0f}% intactos",
            )

            cmp_left, cmp_right = st.columns(2, gap="large")
            with cmp_left:
                st.markdown("#### ❌ Oferta original (sesgos resaltados)")
                st.markdown(analysis.highlighted_html, unsafe_allow_html=True)
            with cmp_right:
                st.markdown("#### ✅ Oferta mitigada (lenguaje neutro e inclusivo)")
                st.markdown(
                    f'<div class="rewritten-box">{rw.rewritten_text}</div>',
                    unsafe_allow_html=True,
                )
                st.download_button(
                    "⬇️ Descargar oferta inclusiva (.txt)",
                    data=rw.rewritten_text.encode("utf-8"),
                    file_name="aviso_empleo_inclusivo.txt",
                    mime="text/plain",
                    width="stretch",
                )

            if rw.preserved_technical_terms:
                st.success(
                    f"Verificación técnica completada: se preservaron intactos "
                    f"{len(rw.preserved_technical_terms)} de {total_tech} requisitos técnicos detectados "
                    f"({', '.join(rw.preserved_technical_terms)})."
                )
            if rw.missing_technical_terms:
                st.warning(
                    f"Atención: verifica los siguientes términos técnicos: {', '.join(rw.missing_technical_terms)}"
                )

            if rw.changes:
                st.markdown("#### 🔁 Transformaciones aplicadas")
                st.markdown(render_changes_html_table(rw.changes), unsafe_allow_html=True)
            else:
                st.info("La oferta ya se encontraba redactada con lenguaje neutro; no se requirieron sustituciones.")


# ==================== TAB 2: COMPARATIVA DE MODELOS ML ====================
with tab_metrics:
    st.markdown("### Evaluación comparativa de clasificadores ML (TF-IDF)")

    if not metrics_data:
        st.warning("No se encontraron métricas guardadas. Ejecuta el entrenamiento desde la barra lateral.")
    else:
        ds_meta = metrics_data["dataset"]
        best_meta = metrics_data["best_model"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Dataset", f"{ds_meta['total_samples']} avisos")
        m2.metric("Train / Test Split", f"{ds_meta['train_samples']} / {ds_meta['test_samples']} (80/20)")
        m3.metric("Clases Evaluadas", f"{len(ds_meta['classes'])} balanceadas")
        m4.metric("Mejor Clasificador", best_meta["display_name"])

        st.markdown("#### 📋 Tabla resumen de métricas (Conjunto de Prueba + CV 5-Fold)")
        summary_df = pd.read_csv(METRICS_SUMMARY_CSV_PATH)
        display_summary = summary_df[
            [
                "modelo",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "cv5_accuracy_mean",
                "cv5_f1_macro_mean",
                "cv5_f1_macro_std",
            ]
        ].rename(
            columns={
                "modelo": "Modelo",
                "accuracy": "Accuracy",
                "precision_macro": "Precision (Macro)",
                "recall_macro": "Recall (Macro)",
                "f1_macro": "F1-Score (Macro)",
                "cv5_accuracy_mean": "CV-5 Accuracy",
                "cv5_f1_macro_mean": "CV-5 F1-Macro",
                "cv5_f1_macro_std": "CV-5 F1 Std",
            }
        )
        st.dataframe(display_summary, width="stretch", hide_index=True)

        # Gráfico comparativo de métricas
        melted = summary_df.melt(
            id_vars=["modelo"],
            value_vars=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
            var_name="Métrica",
            value_name="Valor",
        )
        metric_labels = {
            "accuracy": "Accuracy",
            "precision_macro": "Precision (Macro)",
            "recall_macro": "Recall (Macro)",
            "f1_macro": "F1-Score (Macro)",
        }
        melted["Métrica"] = melted["Métrica"].map(metric_labels)
        fig_comp = px.bar(
            melted,
            x="Métrica",
            y="Valor",
            color="modelo",
            barmode="group",
            text_auto=".3f",
            title="Comparación de desempeño en conjunto de prueba (Test 20%)",
            color_discrete_sequence=["#2563EB", "#7C3AED", "#059669"],
        )
        fig_comp.update_layout(yaxis=dict(range=[0, 1.08]), legend_title="Modelo", height=380)
        st.plotly_chart(fig_comp, width="stretch")

        # Matrices de confusión y métricas por clase
        st.markdown("#### 🧩 Matriz de confusión y reporte por clase")
        cm_model_key = st.selectbox(
            "Selecciona un clasificador para inspeccionar su matriz de confusión:",
            options=list(MODEL_DISPLAY_NAMES.keys()),
            format_func=lambda k: MODEL_DISPLAY_NAMES[k],
            key="cm_selector",
        )
        sel_model_data = metrics_data["models"][cm_model_key]
        cm_col1, cm_col2 = st.columns([1.2, 1.0], gap="large")
        with cm_col1:
            fig_cm = make_confusion_matrix_figure(
                sel_model_data["confusion_matrix"]["matrix"],
                sel_model_data["confusion_matrix"]["labels"],
                f"Matriz de Confusión — {sel_model_data['display_name']}",
            )
            st.plotly_chart(fig_cm, width="stretch")
        with cm_col2:
            st.markdown(f"**Métricas por clase — {sel_model_data['display_name']}**")
            per_cls_df = pd.DataFrame.from_dict(sel_model_data["per_class"], orient="index").reset_index()
            per_cls_df.columns = ["Categoría", "Precision", "Recall", "F1-Score", "Soporte"]
            st.dataframe(per_cls_df, width="stretch", hide_index=True)

        st.divider()
        st.markdown("#### 📥 Descargar reportes exactos para tu informe (JSON / CSV)")
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button(
            "📄 model_metrics.json",
            data=read_file_bytes(METRICS_JSON_PATH),
            file_name="model_metrics.json",
            mime="application/json",
            width="stretch",
        )
        d2.download_button(
            "📊 model_metrics_summary.csv",
            data=read_file_bytes(METRICS_SUMMARY_CSV_PATH),
            file_name="model_metrics_summary.csv",
            mime="text/csv",
            width="stretch",
        )
        d3.download_button(
            "📑 per_class_metrics.csv",
            data=read_file_bytes(PER_CLASS_CSV_PATH),
            file_name="per_class_metrics.csv",
            mime="text/csv",
            width="stretch",
        )
        d4.download_button(
            "🧩 confusion_matrices.csv",
            data=read_file_bytes(CONFUSION_CSV_PATH),
            file_name="confusion_matrices.csv",
            mime="text/csv",
            width="stretch",
        )


# ==================== TAB 3: EXPLORADOR DEL DATASET ====================
with tab_dataset:
    st.markdown("### Explorador del Dataset Sintético Balanceado en Español")
    df_dataset = get_cached_dataset()

    if df_dataset.empty:
        st.warning("Dataset no encontrado.")
    else:
        counts = df_dataset["categoria"].value_counts().reindex(ALL_CATEGORIES).reset_index()
        counts.columns = ["Categoría", "Cantidad"]

        ds_c1, ds_c2 = st.columns([1.1, 1.4], gap="large")
        with ds_c1:
            fig_dist = px.bar(
                counts,
                x="Categoría",
                y="Cantidad",
                color="Categoría",
                text="Cantidad",
                color_discrete_map={cat: CATEGORY_INFO[cat]["color"] for cat in ALL_CATEGORIES},
                title="Distribución balanceada por categoría",
            )
            fig_dist.update_layout(showlegend=False, height=340, yaxis=dict(range=[0, counts["Cantidad"].max() + 25]))
            st.plotly_chart(fig_dist, width="stretch")

        with ds_c2:
            st.markdown("#### Resumen estructural del corpus")
            st.write(
                f"- **Total de avisos sintéticos:** `{len(df_dataset)}` registros únicos.\n"
                f"- **Balance exacto:** `{len(df_dataset) // len(ALL_CATEGORIES)}` ejemplos por cada una de las 6 clases.\n"
                f"- **Cobertura sectorial:** `{df_dataset['sector'].nunique()}` sectores laborales distribuidos equitativamente entre todas las clases para prevenir sesgo temático en TF-IDF.\n"
                f"- **Idioma:** Español (contexto laboral hispanohablante)."
            )
            st.download_button(
                "⬇️ Descargar dataset completo (CSV)",
                data=read_file_bytes(DATASET_PATH),
                file_name="dataset_sesgos_empleo.csv",
                mime="text/csv",
            )

        st.markdown("#### 🔎 Filtrar y explorar ejemplos")
        f1, f2, f3 = st.columns([1.2, 1.2, 2.0])
        with f1:
            cat_filter = st.multiselect("Categoría", options=ALL_CATEGORIES, default=ALL_CATEGORIES)
        with f2:
            sector_options = sorted(df_dataset["sector"].unique().tolist())
            sector_filter = st.multiselect("Sector", options=sector_options, default=[])
        with f3:
            keyword = st.text_input("Buscar palabra clave en el aviso", placeholder="Ej.: Python, edad, señorita, sexo...")

        filtered = df_dataset[df_dataset["categoria"].isin(cat_filter)]
        if sector_filter:
            filtered = filtered[filtered["sector"].isin(sector_filter)]
        if keyword.strip():
            filtered = filtered[filtered["texto"].str.contains(keyword.strip(), case=False, na=False)]

        st.caption(f"Mostrando {len(filtered)} de {len(df_dataset)} avisos.")
        st.dataframe(filtered, width="stretch", hide_index=True, height=400)
