"""Plataforma Web Interactiva para Detección y Mitigación de Sesgos de Género en Avisos de Empleo."""

from __future__ import annotations

import gc
import html
from typing import Dict

import plotly.graph_objects as go
import streamlit as st

from src.bias_analyzer import analyze_job_posting, load_model
from src.config import (
    ALL_CATEGORIES,
    BIAS_CATEGORIES,
    CATEGORY_INFO,
    PRELOADED_EXAMPLES,
    RF_MODEL_PATH,
)
from src.rewriter import rewrite_inclusive

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
        /* Forzar tema claro global en toda la app independientemente del navegador/SO */
        :root, html, body {
            color-scheme: light !important;
            background-color: #FFFFFF !important;
            color: #0F172A !important;
        }
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
        }
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
        }
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div,
        [data-testid="stSidebarContent"] {
            background-color: #F8FAFC !important;
            color: #0F172A !important;
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
            color: #0F172A;
        }
        .stApp [data-testid="stMarkdownContainer"] p,
        .stApp [data-testid="stMarkdownContainer"] li,
        .stApp [data-testid="stWidgetLabel"],
        .stApp [data-testid="stWidgetLabel"] p,
        .stApp label {
            color: #0F172A !important;
        }
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stCaptionContainer"] p {
            color: #475569 !important;
        }
        /* Inputs, textarea, selectbox, expanders, tabs y botones en modo claro */
        .stApp textarea,
        .stApp input,
        .stApp [data-baseweb="textarea"],
        .stApp [data-baseweb="textarea"] > div,
        .stApp [data-baseweb="input"],
        .stApp [data-baseweb="base-input"] {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border-color: #CBD5E1 !important;
            caret-color: #0F172A !important;
        }
        .stApp textarea::placeholder,
        .stApp input::placeholder {
            color: #64748B !important;
        }
        .stApp [data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border-color: #CBD5E1 !important;
        }
        .stApp [data-baseweb="select"] span,
        .stApp [data-baseweb="select"] div,
        .stApp [data-baseweb="select"] svg {
            color: #0F172A !important;
            fill: #0F172A !important;
        }
        [data-baseweb="popover"],
        [data-baseweb="menu"],
        ul[role="listbox"],
        li[role="option"] {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
        }
        li[role="option"]:hover {
            background-color: #F1F5F9 !important;
            color: #0F172A !important;
        }
        .stApp [data-testid="stExpander"],
        .stApp [data-testid="stExpander"] details {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px !important;
            color: #0F172A !important;
        }
        .stApp [data-testid="stExpander"] summary,
        .stApp [data-testid="stExpander"] summary p,
        .stApp [data-testid="stExpander"] summary svg {
            background-color: #F8FAFC !important;
            color: #0F172A !important;
            fill: #0F172A !important;
        }
        .stApp [data-testid="stTabs"] button[role="tab"] {
            color: #475569 !important;
            background-color: transparent !important;
        }
        .stApp [data-testid="stTabs"] button[role="tab"] p {
            color: inherit !important;
        }
        .stApp [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            color: #2563EB !important;
            font-weight: 700 !important;
        }
        .stApp button[kind="secondary"],
        .stApp [data-testid="stDownloadButton"] button {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #CBD5E1 !important;
        }
        .stApp button[kind="secondary"] p,
        .stApp [data-testid="stDownloadButton"] button p {
            color: #0F172A !important;
        }
        .stApp button[kind="secondary"]:hover,
        .stApp [data-testid="stDownloadButton"] button:hover {
            background-color: #F8FAFC !important;
            border-color: #2563EB !important;
            color: #1E3A8A !important;
        }
        .stApp button[kind="primary"] {
            background-color: #2563EB !important;
            color: #FFFFFF !important;
            border: 1px solid #2563EB !important;
        }
        .stApp button[kind="primary"] p {
            color: #FFFFFF !important;
        }
        .stApp [data-testid="stAlert"] p {
            color: #0F172A !important;
        }
        .main-header,
        .stApp [data-testid="stMarkdownContainer"] .main-header {
            background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 60%, #4338CA 100%);
            padding: 1.6rem 2rem;
            border-radius: 14px;
            color: #FFFFFF !important;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.14);
        }
        .main-header h1,
        .stApp [data-testid="stMarkdownContainer"] .main-header h1 {
            margin: 0;
            font-size: 1.85rem;
            font-weight: 700;
            color: #FFFFFF !important;
        }
        .main-header p,
        .stApp [data-testid="stMarkdownContainer"] .main-header p {
            margin: 0.45rem 0 0 0;
            font-size: 0.98rem;
            color: #E2E8F0 !important;
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
        /* Ajuste tipográfico de tarjetas KPI y st.metric con contraste oscuro forzado */
        [data-testid="stMetric"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px;
            padding: 12px 14px;
            min-height: 92px;
            color: #0F172A !important;
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricLabel"] div {
            font-size: 0.78rem !important;
            line-height: 1.25 !important;
            color: #475569 !important;
            white-space: normal !important;
            overflow: visible !important;
        }
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] > div,
        [data-testid="stMetricValue"] * {
            font-size: 1.32rem !important;
            line-height: 1.25 !important;
            font-weight: 700 !important;
            color: #0F172A !important;
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
            background-color: #FFFFFF !important;
            padding: 8px 10px;
            min-height: auto;
        }
        section[data-testid="stSidebar"] [data-testid="stMetricValue"],
        section[data-testid="stSidebar"] [data-testid="stMetricValue"] > div,
        section[data-testid="stSidebar"] [data-testid="stMetricValue"] * {
            font-size: 1.12rem !important;
            color: #0F172A !important;
        }
        .kpi-card {
            padding: 12px 14px;
            border-radius: 10px;
            min-height: 92px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            color: #0F172A;
        }
        .kpi-card-label {
            font-size: 0.74rem;
            color: #475569 !important;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .kpi-card-value {
            font-size: 1.20rem;
            font-weight: 800;
            margin-top: 3px;
            line-height: 1.2;
            color: #0F172A;
            white-space: normal;
            word-break: break-word;
        }
        .kpi-card-sub {
            font-size: 0.76rem;
            color: #334155 !important;
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


# ---------- RECURSOS EN CACHÉ (OPTIMIZADOS PARA BAJO CONSUMO DE RAM) ----------
@st.cache_resource(show_spinner=False)
def get_cached_model():
    """Carga una sola vez en memoria el clasificador preentrenado y comprimido."""
    model = load_model("random_forest")
    gc.collect()
    return model


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


def apply_light_plotly_theme(fig: go.Figure) -> go.Figure:
    """Fuerza fondo blanco y tipografía oscura en cualquier gráfico Plotly."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"),
        title_font=dict(color="#0F172A"),
        legend=dict(
            font=dict(color="#0F172A"),
            title_font=dict(color="#0F172A"),
            bgcolor="#FFFFFF",
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#CBD5E1",
            font=dict(color="#0F172A"),
        ),
    )
    fig.update_xaxes(
        title_font=dict(color="#334155"),
        tickfont=dict(color="#334155"),
        gridcolor="#E2E8F0",
        linecolor="#CBD5E1",
        zerolinecolor="#CBD5E1",
    )
    fig.update_yaxes(
        title_font=dict(color="#334155"),
        tickfont=dict(color="#334155"),
        gridcolor="#E2E8F0",
        linecolor="#CBD5E1",
        zerolinecolor="#CBD5E1",
    )
    return fig


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
            textfont=dict(color="#0F172A", size=12),
            hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Probabilidad por categoría de sesgo", font=dict(color="#0F172A", size=15)),
        xaxis=dict(title="Probabilidad (%)", range=[0, max(105, max(vals) + 15)], showgrid=True),
        yaxis=dict(autorange="reversed", title=""),
        height=310,
        margin=dict(l=10, r=45, t=45, b=35),
        font=dict(size=13, color="#0F172A"),
    )
    return apply_light_plotly_theme(fig)


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


def _sanitize_ui_explanation(text: str) -> str:
    """Elimina menciones técnicas a algoritmos internos en textos mostrados al usuario."""
    cleaned = (
        text.replace("de Random Forest", "del motor de análisis")
        .replace("de Decision Tree", "del motor de análisis")
        .replace("de SVM", "del motor de análisis")
        .replace("Random Forest", "el motor de análisis")
        .replace("Decision Tree", "el motor de análisis")
        .replace("SVM", "el motor de análisis")
        .replace("TF-IDF", "léxica")
        .replace("ML", "estadístico")
    )
    return html.escape(cleaned)


def render_hybrid_status_banner(analysis) -> str:
    """Renderiza un aviso explicativo limpio cuando aplica calibración o alerta de sesgo implícito."""
    explanation = _sanitize_ui_explanation(analysis.decision_explanation)
    if analysis.decision_mode == "ml_implicit_bias":
        return (
            '<div class="hybrid-banner-warn">'
            "<strong>⚠️ Alerta de posible sesgo implícito (patrón estadístico):</strong> "
            f"{explanation}"
            "</div>"
        )
    if analysis.decision_mode == "calibrated_neutral":
        return (
            '<div class="hybrid-banner-info">'
            "<strong>ℹ️ Calibración híbrida aplicada (baja certeza estadística / vocabulario poco frecuente):</strong> "
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
    """Renderiza métricas de incertidumbre, cobertura léxica y términos influyentes del análisis."""
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
        f"Confiabilidad estadística: {html.escape(analysis.ml_reliability_label)}</span>"
        f'<span style="font-size:0.78rem; color:#475569;">'
        f'Señal base: <strong style="color:{raw_meta["text"]};">{html.escape(analysis.raw_ml_category)}</strong> '
        f"({analysis.ml_confidence * 100:.1f}%)</span>"
        "</div>"
    )

    cov_pct = analysis.vocabulary_coverage * 100.0
    diag_grid_html = (
        '<div class="ml-diag-grid">'
        '<div class="ml-diag-box">'
        '<div class="ml-diag-label">Cobertura léxica</div>'
        f'<div class="ml-diag-val">{cov_pct:.1f}%</div>'
        f'<div class="ml-diag-sub">{analysis.recognized_tokens}/{analysis.total_tokens} términos · '
        f"{analysis.matched_features} expresiones</div>"
        "</div>"
        '<div class="ml-diag-box">'
        '<div class="ml-diag-label">Confianza máx.</div>'
        f'<div class="ml-diag-val">{analysis.ml_confidence * 100:.1f}%</div>'
        f'<div class="ml-diag-sub">Señal base: {html.escape(analysis.raw_ml_category)}</div>'
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
            term_rows.append(
                '<div class="ml-term-row">'
                f'<div class="ml-term-name" title="{html.escape(item.term)}">'
                f"{html.escape(item.term)}</div>"
                '<div class="ml-term-bar-bg">'
                f'<div class="ml-term-bar-fill" style="width:{width_pct:.1f}%; background:{bar_color};"></div>'
                "</div>"
                f'<div class="ml-term-weight">Peso {item.tfidf_weight:.2f}</div>'
                "</div>"
            )
        terms_html = (
            '<div style="font-size:0.82rem; font-weight:700; color:#1E293B; margin-bottom:6px;">'
            "Términos del aviso que más influyeron en el análisis"
            "</div>"
            + "".join(term_rows)
        )
    else:
        terms_html = (
            '<div style="font-size:0.82rem; color:#64748B; background:#F8FAFC; '
            'border:1px solid #E2E8F0; border-radius:8px; padding:8px 10px;">'
            "Ningún término del texto coincide con el vocabulario de referencia del analizador."
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


# ---------- INICIALIZACIÓN (SOLO CARGA DE MODELO PREENTRENADO, SIN ENTRENAMIENTO) ----------
if not RF_MODEL_PATH.exists():
    st.error("No se encontró el archivo de modelo preentrenado requerido para el análisis.")
    st.stop()

rf_model = get_cached_model()

if "job_text" not in st.session_state:
    first_example_name = next(iter(PRELOADED_EXAMPLES.keys()))
    st.session_state.job_text = PRELOADED_EXAMPLES[first_example_name]
    st.session_state.selected_example = first_example_name

if "rewritten_output" not in st.session_state:
    st.session_state.rewritten_output = None

if "last_analyzed_text" not in st.session_state:
    st.session_state.last_analyzed_text = ""


# ---------- BARRA LATERAL ----------
selected_model_key = "random_forest"
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
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


# ---------- ENCABEZADO ----------
st.markdown(
    """
    <div class="main-header">
        <h1>⚖️ Plataforma de Detección y Mitigación de Sesgos de Género en Avisos de Empleo</h1>
        <p>
            Análisis inteligente en español para detección explicable de sesgos y reescritura inclusiva preservando requisitos técnicos.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================== ANALIZADOR Y MITIGADOR (VISTA PRINCIPAL ÚNICA) ====================
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
        "Los requisitos técnicos (lenguajes, herramientas, certificaciones y experiencia) se preservan intactos."
    )

if not st.session_state.job_text.strip():
    st.info("Pega una oferta laboral o carga uno de los ejemplos precargados para iniciar el análisis.")
else:
    analysis = analyze_job_posting(
        st.session_state.job_text,
        model_key=selected_model_key,
        pipeline=rf_model,
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
                    f"Calibrado · Señal base: {html.escape(analysis.raw_ml_category)} "
                    f"({analysis.ml_confidence * 100:.1f}%)"
                )
            else:
                kpi1_sub = (
                    f"Calibrado · Cobertura léxica: {analysis.vocabulary_coverage * 100:.0f}%"
                )
        elif analysis.decision_mode == "ml_implicit_bias":
            kpi1_sub = (
                f"Sesgo implícito ({analysis.ml_confidence * 100:.1f}%) · Sin frases explícitas"
            )
        elif analysis.decision_mode == "lexical_explicit" and analysis.predicted_class != analysis.raw_ml_category:
            kpi1_sub = (
                f"Evidencia léxica · Señal base: {html.escape(analysis.raw_ml_category)} "
                f"({analysis.ml_confidence * 100:.1f}%)"
            )
        else:
            kpi1_sub = (
                f"Confianza: {analysis.probabilities[analysis.predicted_class] * 100:.1f}% · "
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
            kpi4_delta = "Sesgo implícito detectado"
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
                "No se detectaron frases sesgadas explícitas en el texto, pero el análisis "
                "identificó un posible patrón estadístico implícito. Revisa los términos influyentes "
                "en el panel derecho."
            )
        elif analysis.decision_mode == "calibrated_neutral":
            st.success(
                "No se detectaron frases sesgadas explícitas en el texto analizado. "
                "El motor híbrido calibró el diagnóstico final a Neutro ante la ausencia de alertas explícitas."
            )
        else:
            st.success("No se detectaron frases sesgadas explícitas en el texto analizado.")

        if analysis.spans:
            st.markdown("#### 📋 Detalle de frases detectadas")
            st.markdown(render_findings_html_table(analysis.spans), unsafe_allow_html=True)

    with col_right:
        st.markdown("#### 📈 Probabilidades por categoría de sesgo")
        fig_probs = make_bias_probability_chart(analysis.bias_probabilities)
        st.plotly_chart(fig_probs, theme=None, width="stretch")

        neutro_pct = analysis.probabilities["Neutro"] * 100.0
        neutro_meta = CATEGORY_INFO["Neutro"]
        st.markdown(
            f'<div style="padding:10px 14px; border-radius:8px; background:{neutro_meta["bg"]}; '
            f'border:1px solid {neutro_meta["border"]}; color:{neutro_meta["text"]}; margin-bottom:12px;">'
            f'<strong>Probabilidad clase Neutro / Inclusivo:</strong> {neutro_pct:.2f}%'
            f"</div>",
            unsafe_allow_html=True,
        )

        with st.expander(
            f"🔬 Explicabilidad y términos influyentes ({analysis.vocabulary_coverage * 100:.0f}% cobertura)",
            expanded=analysis.decision_mode in ("ml_implicit_bias", "calibrated_neutral"),
        ):
            st.markdown(render_ml_explainability_html(analysis), unsafe_allow_html=True)

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
            pipeline=rf_model,
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
            "Probabilidad Neutro",
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
