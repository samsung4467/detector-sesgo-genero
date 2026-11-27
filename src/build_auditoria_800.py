"""Genera auditoria_800.html: auditoría visual interactiva de los 800 avisos reales de Perú (3 portales).

Lee data/validacion_real_resultados.csv (800 filas: 300 Computrabajo, 300 Bumeran, 200 LinkedIn),
construye la matriz comparativa de los 3 portales, filtros rápidos con paginación fluida y
resaltado carácter a carácter de las frases disparadoras, y reemplaza auditoria_200.html por auditoria_800.html.
"""

from __future__ import annotations

import html
import json
from typing import Any, Dict, List

import pandas as pd

from src.bias_analyzer import detect_biased_spans
from src.config import BASE_DIR, DATA_DIR

RESULTADOS_CSV = DATA_DIR / "validacion_real_resultados.csv"
OLD_HTML = BASE_DIR / "auditoria_200.html"
OUTPUT_HTML = BASE_DIR / "auditoria_800.html"

CATEGORY_ORDER = ["Masculino", "Femenino", "Demográfico", "Excluyente", "LGBTQ+", "Neutro"]


def build_highlighted_html(text: str, spans: List[Any]) -> str:
    if not spans:
        return html.escape(text)
    parts: List[str] = []
    cursor = 0
    for sp in sorted(spans, key=lambda s: (s.start, -(s.end - s.start))):
        if sp.start < cursor:
            continue
        parts.append(html.escape(text[cursor : sp.start]))
        cat_slug = (
            sp.category.lower()
            .replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
            .replace("+", "")
        )
        tooltip = f"[{sp.category}] {sp.explanation} | Sugerencia: {sp.suggestion}"
        parts.append(
            f'<mark class="hl hl-{cat_slug}" title="{html.escape(tooltip, quote=True)}">'
            f'{html.escape(text[sp.start : sp.end])}'
            f'<span class="hl-tag">{html.escape(sp.category)}</span>'
            f"</mark>"
        )
        cursor = sp.end
    parts.append(html.escape(text[cursor:]))
    return "".join(parts)


def build_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        texto = str(row["texto"]) if pd.notna(row["texto"]) else ""
        spans = detect_biased_spans(texto)
        just_list: List[Dict[str, str]] = []
        seen_pairs = set()
        for sp in spans:
            key = (sp.category, sp.text.strip().lower())
            if key not in seen_pairs:
                seen_pairs.add(key)
                just_list.append(
                    {
                        "category": sp.category,
                        "phrase": sp.text,
                        "explanation": sp.explanation,
                    }
                )
        records.append(
            {
                "id": str(row["id"]),
                "portal": str(row["portal"]),
                "busqueda": str(row["busqueda"]) if pd.notna(row["busqueda"]) else "",
                "puesto": str(row["puesto"]) if pd.notna(row["puesto"]) else "",
                "categoria_final": str(row["categoria_final"]),
                "categoria_ml_cruda": str(row["categoria_ml_cruda"]),
                "modo_decision": str(row["modo_decision"]),
                "nivel_riesgo": str(row["nivel_riesgo"]),
                "indice_inclusividad": float(row["indice_inclusividad"]),
                "num_frases": int(row["num_frases_sesgadas"]),
                "frases_csv": str(row["frases_sesgadas_detectadas"]) if pd.notna(row["frases_sesgadas_detectadas"]) else "",
                "justificaciones": just_list,
                "texto": texto,
                "texto_len": len(texto),
                "texto_html": build_highlighted_html(texto, spans),
            }
        )
    return records


def build_summary_matrix(df: pd.DataFrame) -> List[Dict[str, Any]]:
    df_c = df[df["portal"] == "Computrabajo Perú"]
    df_b = df[df["portal"] == "Bumeran Perú"]
    df_l = df[df["portal"] == "LinkedIn Perú"]
    n_c, n_b, n_l, n_t = len(df_c), len(df_b), len(df_l), len(df)

    rows = []
    for cat in CATEGORY_ORDER:
        c = int((df_c["categoria_final"] == cat).sum())
        b = int((df_b["categoria_final"] == cat).sum())
        l = int((df_l["categoria_final"] == cat).sum())
        t = c + b + l
        rows.append(
            {
                "categoria": cat,
                "comp_n": c,
                "comp_pct": round(100.0 * c / n_c, 1) if n_c else 0.0,
                "bum_n": b,
                "bum_pct": round(100.0 * b / n_b, 1) if n_b else 0.0,
                "li_n": l,
                "li_pct": round(100.0 * l / n_l, 1) if n_l else 0.0,
                "tot_n": t,
                "tot_pct": round(100.0 * t / n_t, 1) if n_t else 0.0,
            }
        )
    return rows


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Auditoría Visual — 800 Avisos Reales de Perú (3 Portales)</title>
  <style>
    :root { --bg:#F8FAFC; --surface:#FFFFFF; --border:#E2E8F0; --text:#0F172A; --muted:#475569; --navy:#1E3A8A; }
    * { box-sizing: border-box; }
    body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; background:var(--bg); color:var(--text); line-height:1.5; }
    .container { max-width:1520px; margin:0 auto; padding:20px 24px 48px; }
    .header { background:linear-gradient(135deg,#0F172A 0%,#1E3A8A 100%); color:#FFF; border-radius:14px; padding:22px 26px; margin-bottom:18px; }
    .header-top { display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px; }
    .header h1 { margin:0 0 4px; font-size:1.45rem; }
    .header p { margin:0; font-size:0.9rem; color:#CBD5E1; }
    .source-pill { background:rgba(255,255,255,0.14); border:1px solid rgba(255,255,255,0.22); color:#F8FAFC; padding:6px 12px; border-radius:999px; font-size:0.78rem; font-weight:600; }
    .kpi-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(185px,1fr)); gap:12px; margin-top:16px; }
    .kpi-card { background:rgba(255,255,255,0.10); border:1px solid rgba(255,255,255,0.16); border-radius:10px; padding:10px 14px; }
    .kpi-label { font-size:0.74rem; text-transform:uppercase; letter-spacing:0.04em; color:#CBD5E1; }
    .kpi-val { font-size:1.35rem; font-weight:700; color:#FFF; margin-top:2px; }
    .kpi-sub { font-size:0.75rem; color:#E2E8F0; }
    .card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:18px 20px; margin-bottom:18px; box-shadow:0 1px 3px rgba(15,23,42,0.06); }
    .card-title { margin:0 0 12px; font-size:1.02rem; font-weight:700; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; }
    .matrix-wrap, .table-wrap { overflow-x:auto; }
    table.matrix { width:100%; border-collapse:collapse; font-size:0.88rem; }
    table.matrix th { background:#F1F5F9; color:#1E293B; font-weight:700; text-align:center; padding:10px 12px; border-bottom:2px solid #CBD5E1; white-space:nowrap; }
    table.matrix th:first-child, table.matrix td:first-child { text-align:left; }
    table.matrix td { padding:9px 12px; border-bottom:1px solid #E2E8F0; text-align:center; vertical-align:middle; }
    table.matrix tbody tr { cursor:pointer; }
    table.matrix tbody tr:hover { background:#F8FAFC; }
    table.matrix tbody tr.active-row { background:#EFF6FF; }
    table.matrix tfoot td { background:#EFF6FF; font-weight:700; border-top:2px solid #1E3A8A; }
    .bar-track { width:100%; min-width:150px; height:10px; background:#E2E8F0; border-radius:999px; overflow:hidden; display:flex; }
    .bar-comp { background:#0D9488; height:100%; }
    .bar-bum { background:#4F46E5; height:100%; }
    .bar-li { background:#0284C7; height:100%; }
    .bar-legend { display:inline-flex; gap:12px; font-size:0.76rem; color:var(--muted); align-items:center; flex-wrap:wrap; }
    .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:4px; }
    .filter-section { display:flex; flex-direction:column; gap:12px; }
    .filter-row { display:flex; flex-wrap:wrap; align-items:center; gap:8px; }
    .filter-label { font-size:0.78rem; font-weight:700; text-transform:uppercase; color:var(--muted); min-width:78px; }
    .pill-btn { border:1px solid #CBD5E1; background:#FFF; color:#1E293B; padding:6px 12px; border-radius:999px; font-size:0.82rem; font-weight:600; cursor:pointer; }
    .pill-btn:hover { border-color:#2563EB; color:#1D4ED8; }
    .pill-btn.active { background:#1E3A8A; border-color:#1E3A8A; color:#FFF; }
    .pill-btn.cat-masculino.active { background:#1D4ED8; border-color:#1D4ED8; }
    .pill-btn.cat-femenino.active { background:#BE185D; border-color:#BE185D; }
    .pill-btn.cat-demografico.active { background:#B45309; border-color:#B45309; }
    .pill-btn.cat-excluyente.active { background:#B91C1C; border-color:#B91C1C; }
    .pill-btn.cat-neutro.active { background:#15803D; border-color:#15803D; }
    .search-row { display:flex; flex-wrap:wrap; gap:10px; align-items:center; justify-content:space-between; padding-top:8px; border-top:1px solid #F1F5F9; }
    .search-box { flex:1 1 320px; }
    .search-input { width:100%; padding:9px 14px; border:1px solid #CBD5E1; border-radius:8px; font-size:0.88rem; }
    .actions-group { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
    .action-btn { padding:7px 12px; border-radius:8px; border:1px solid #CBD5E1; background:#F8FAFC; color:#1E293B; font-size:0.8rem; font-weight:600; cursor:pointer; }
    .action-btn:hover { background:#EFF6FF; border-color:#93C5FD; }
    .result-counter { font-size:0.84rem; font-weight:700; color:#1E3A8A; background:#EFF6FF; padding:6px 12px; border-radius:8px; }
    .badge { display:inline-block; padding:3px 10px; border-radius:999px; font-size:0.76rem; font-weight:700; white-space:nowrap; border:1px solid transparent; }
    .badge-excluyente { background:#FEE2E2; color:#991B1B; border-color:#FCA5A5; }
    .badge-masculino { background:#DBEAFE; color:#1E40AF; border-color:#93C5FD; }
    .badge-femenino { background:#FCE7F3; color:#9D174D; border-color:#F9A8D4; }
    .badge-demografico { background:#FEF3C7; color:#92400E; border-color:#FCD34D; }
    .badge-lgbtq { background:#EDE9FE; color:#5B21B6; border-color:#C4B5FD; }
    .badge-neutro { background:#DCFCE7; color:#166534; border-color:#86EFAC; }
    .portal-comp { background:#CCFBF1; color:#115E59; border-color:#5EEAD4; }
    .portal-bum { background:#E0E7FF; color:#3730A3; border-color:#A5B4FC; }
    .portal-li { background:#E0F2FE; color:#075985; border-color:#7DD3FC; }
    .risk-alto { color:#B91C1C; font-weight:700; font-size:0.74rem; }
    .risk-moderado { color:#B45309; font-weight:700; font-size:0.74rem; }
    .risk-bajo { color:#15803D; font-weight:700; font-size:0.74rem; }
    table.audit-table { width:100%; border-collapse:collapse; font-size:0.85rem; }
    table.audit-table thead th { position:sticky; top:0; z-index:2; background:#0F172A; color:#FFF; text-align:left; padding:11px 10px; font-size:0.8rem; white-space:nowrap; }
    table.audit-table tbody tr.row-main { border-bottom:1px solid #E2E8F0; vertical-align:top; background:#FFF; }
    table.audit-table tbody tr.row-main:hover { background:#F8FAFC; }
    table.audit-table td { padding:11px 10px; }
    .id-code { font-family:ui-monospace,Consolas,monospace; font-weight:700; font-size:0.82rem; color:#1E3A8A; }
    .busq-tag { display:inline-block; margin-top:4px; font-size:0.71rem; color:#475569; background:#F1F5F9; padding:2px 7px; border-radius:6px; }
    .puesto-title { font-weight:700; color:#0F172A; margin-bottom:4px; }
    .puesto-meta { font-size:0.74rem; color:#64748B; display:flex; gap:8px; flex-wrap:wrap; }
    .just-list { display:flex; flex-direction:column; gap:6px; }
    .just-item { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:6px 8px; font-size:0.78rem; }
    .just-phrase { font-weight:700; margin-right:4px; }
    .just-exp { color:#475569; display:block; margin-top:2px; font-size:0.74rem; }
    .neutral-note { font-size:0.78rem; color:#15803D; background:#F0FDF4; border:1px solid #BBF7D0; border-radius:8px; padding:6px 8px; }
    details.fulltext-box { border:1px solid #CBD5E1; border-radius:8px; background:#FFF; min-width:280px; }
    details.fulltext-box summary { cursor:pointer; padding:7px 10px; font-size:0.79rem; font-weight:700; color:#1E3A8A; background:#F8FAFC; border-radius:8px; list-style:none; display:flex; justify-content:space-between; gap:8px; }
    details.fulltext-box summary::-webkit-details-marker { display:none; }
    details.fulltext-box[open] summary { border-bottom:1px solid #E2E8F0; border-radius:8px 8px 0 0; background:#EFF6FF; }
    .snippet-preview { font-size:0.76rem; color:#475569; padding:6px 10px 8px; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; }
    details.fulltext-box[open] .snippet-preview { display:none; }
    .fulltext-body { padding:12px 14px; font-size:0.83rem; line-height:1.68; color:#1E293B; max-height:360px; overflow-y:auto; white-space:pre-wrap; word-break:break-word; }
    mark.hl { padding:1px 5px; border-radius:4px; font-weight:700; border-bottom:2px solid transparent; cursor:help; }
    mark.hl .hl-tag { font-size:0.64rem; text-transform:uppercase; margin-left:4px; padding:0 4px; border-radius:3px; background:rgba(255,255,255,0.75); }
    mark.hl-excluyente { background:#FEE2E2; color:#991B1B; border-bottom-color:#DC2626; }
    mark.hl-masculino { background:#DBEAFE; color:#1E40AF; border-bottom-color:#2563EB; }
    mark.hl-femenino { background:#FCE7F3; color:#9D174D; border-bottom-color:#DB2777; }
    mark.hl-demografico { background:#FEF3C7; color:#92400E; border-bottom-color:#D97706; }
    mark.hl-lgbtq { background:#EDE9FE; color:#5B21B6; border-bottom-color:#7C3AED; }
    .pager { display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; padding:12px 16px; background:#F8FAFC; border-top:1px solid #E2E8F0; font-size:0.84rem; }
    .empty-state { text-align:center; padding:40px 20px; color:#64748B; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="header-top">
        <div>
          <h1>Auditoría Visual Interactiva — 800 Avisos Reales de Perú (3 Portales)</h1>
          <p>Computrabajo Perú (n=300) · Bumeran Perú (n=300) · LinkedIn Perú (n=200) · Diagnóstico híbrido con reglas léxicas limpias</p>
        </div>
        <div class="source-pill">Fuente: data/validacion_real_resultados.csv · 800/800 textos completos</div>
      </div>
      <div class="kpi-grid">
        <div class="kpi-card"><div class="kpi-label">Muestra Real Total</div><div class="kpi-val">800 avisos</div><div class="kpi-sub">300 Computrabajo · 300 Bumeran · 200 LinkedIn</div></div>
        <div class="kpi-card"><div class="kpi-label">Con Sesgo Detectado</div><div class="kpi-val" id="kpiBiased">—</div><div class="kpi-sub">Cualquier categoría de sesgo</div></div>
        <div class="kpi-card"><div class="kpi-label">Avisos Neutros</div><div class="kpi-val" id="kpiNeutral">—</div><div class="kpi-sub">Sin alertas léxicas explícitas</div></div>
        <div class="kpi-card"><div class="kpi-label">Riesgo Alto</div><div class="kpi-val" id="kpiHighRisk">—</div><div class="kpi-sub">Restricción directa o sesgo múltiple</div></div>
        <div class="kpi-card"><div class="kpi-label">Integridad de Texto</div><div class="kpi-val" id="kpiIntegrity">800 / 800</div><div class="kpi-sub" id="kpiTextStats">—</div></div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">
        <span>Matriz Resumen Comparativa: Computrabajo (n=300) vs. Bumeran (n=300) vs. LinkedIn (n=200)</span>
        <span class="bar-legend">
          <span><span class="dot" style="background:#0D9488;"></span>Computrabajo</span>
          <span><span class="dot" style="background:#4F46E5;"></span>Bumeran</span>
          <span><span class="dot" style="background:#0284C7;"></span>LinkedIn</span>
          <span>· Clic en fila para filtrar</span>
        </span>
      </div>
      <div class="matrix-wrap">
        <table class="matrix">
          <thead>
            <tr>
              <th>Categoría Asignada</th>
              <th>Computrabajo (n=300)</th>
              <th>Bumeran (n=300)</th>
              <th>LinkedIn (n=200)</th>
              <th>Total (n=800)</th>
              <th>Proporción entre Portales</th>
            </tr>
          </thead>
          <tbody id="matrixBody"></tbody>
          <tfoot id="matrixFoot"></tfoot>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="filter-section">
        <div class="filter-row"><span class="filter-label">Categoría:</span><div id="catPills" style="display:flex;flex-wrap:wrap;gap:7px;"></div></div>
        <div class="filter-row">
          <span class="filter-label">Portal:</span><div id="portalPills" style="display:flex;flex-wrap:wrap;gap:7px;"></div>
          <span class="filter-label" style="margin-left:14px;">Riesgo:</span><div id="riskPills" style="display:flex;flex-wrap:wrap;gap:7px;"></div>
        </div>
        <div class="search-row">
          <div class="search-box"><input type="text" id="searchInput" class="search-input" placeholder="Buscar por ID, puesto, término de búsqueda, frase detectada o cualquier palabra del texto..." /></div>
          <div class="actions-group">
            <span class="result-counter" id="resultCounter">Mostrando 800 de 800</span>
            <select id="pageSizeSel" class="action-btn"><option value="50" selected>50 por página</option><option value="100">100 por página</option><option value="9999">Ver todos</option></select>
            <button class="action-btn" id="btnExpandAll" type="button">Expandir página</button>
            <button class="action-btn" id="btnCollapseAll" type="button">Colapsar todos</button>
            <button class="action-btn" id="btnReset" type="button">Limpiar filtros</button>
          </div>
        </div>
      </div>
    </div>

    <div class="card" style="padding:0; overflow:hidden;">
      <div class="table-wrap">
        <table class="audit-table">
          <thead>
            <tr>
              <th style="width:110px;">ID / Búsqueda</th>
              <th style="width:145px;">Portal</th>
              <th style="width:230px;">Puesto</th>
              <th style="width:125px;">Categoría</th>
              <th style="width:300px;">Justificación / Frases Detectadas</th>
              <th style="min-width:360px;">Texto Completo Anonimizado (con resaltado)</th>
            </tr>
          </thead>
          <tbody id="auditBody"></tbody>
        </table>
      </div>
      <div id="emptyState" class="empty-state" style="display:none;">No se encontraron avisos con los filtros aplicados.</div>
      <div class="pager" id="pagerBar">
        <span id="pagerInfo">Página 1</span>
        <div style="display:flex;gap:8px;">
          <button class="action-btn" id="btnPrev" type="button">← Anterior</button>
          <button class="action-btn" id="btnNext" type="button">Siguiente →</button>
        </div>
      </div>
    </div>
  </div>

  <script>
    const SUMMARY_DATA = __SUMMARY_JSON__;
    const RECORDS = __RECORDS_JSON__;
    const state = { category: "Todas", portal: "Todos", risk: "Todos", query: "", page: 1, pageSize: 50, expandAll: false };

    const slugCat = c => (c || "").toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/[^a-z]/g, "");
    const esc = s => String(s || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");

    function initKPIs() {
      const total = RECORDS.length;
      const neutral = RECORDS.filter(r => r.categoria_final === "Neutro").length;
      const biased = total - neutral;
      const highRisk = RECORDS.filter(r => r.nivel_riesgo === "Alto").length;
      const lens = RECORDS.map(r => r.texto_len).sort((a,b) => a - b);
      document.getElementById("kpiBiased").textContent = `${biased} (${(biased*100/total).toFixed(1)}%)`;
      document.getElementById("kpiNeutral").textContent = `${neutral} (${(neutral*100/total).toFixed(1)}%)`;
      document.getElementById("kpiHighRisk").textContent = `${highRisk} (${(highRisk*100/total).toFixed(1)}%)`;
      document.getElementById("kpiIntegrity").textContent = `${RECORDS.filter(r => r.texto.trim().length > 0).length} / ${total}`;
      document.getElementById("kpiTextStats").textContent = `Mín ${lens[0]||0} · Med ${lens[Math.floor(lens.length/2)]||0} · Máx ${lens[lens.length-1]||0} car.`;
    }

    function renderMatrix() {
      const tbody = document.getElementById("matrixBody");
      const tfoot = document.getElementById("matrixFoot");
      tbody.innerHTML = "";
      SUMMARY_DATA.forEach(row => {
        const sl = slugCat(row.categoria);
        const cS = row.tot_n ? (row.comp_n*100/row.tot_n) : 0;
        const bS = row.tot_n ? (row.bum_n*100/row.tot_n) : 0;
        const lS = row.tot_n ? (row.li_n*100/row.tot_n) : 0;
        const tr = document.createElement("tr");
        if (state.category === row.categoria) tr.classList.add("active-row");
        tr.innerHTML = `
          <td><span class="badge badge-${sl}">${esc(row.categoria)}</span></td>
          <td><strong>${row.comp_n}</strong> <span style="color:#64748B;">(${row.comp_pct.toFixed(1)}%)</span></td>
          <td><strong>${row.bum_n}</strong> <span style="color:#64748B;">(${row.bum_pct.toFixed(1)}%)</span></td>
          <td><strong>${row.li_n}</strong> <span style="color:#64748B;">(${row.li_pct.toFixed(1)}%)</span></td>
          <td><strong>${row.tot_n}</strong> <span style="color:#1E3A8A;font-weight:600;">(${row.tot_pct.toFixed(1)}%)</span></td>
          <td><div class="bar-track" title="Computrabajo: ${row.comp_n} | Bumeran: ${row.bum_n} | LinkedIn: ${row.li_n}"><div class="bar-comp" style="width:${cS.toFixed(1)}%"></div><div class="bar-bum" style="width:${bS.toFixed(1)}%"></div><div class="bar-li" style="width:${lS.toFixed(1)}%"></div></div></td>
        `;
        tr.addEventListener("click", () => { state.category = state.category === row.categoria ? "Todas" : row.categoria; state.page = 1; renderAll(); });
        tbody.appendChild(tr);
      });
      const tc = SUMMARY_DATA.reduce((a,r)=>a+r.comp_n,0), tb = SUMMARY_DATA.reduce((a,r)=>a+r.bum_n,0), tl = SUMMARY_DATA.reduce((a,r)=>a+r.li_n,0), tt = tc+tb+tl;
      tfoot.innerHTML = `<tr><td>Total Consolidado</td><td>${tc} (100.0%)</td><td>${tb} (100.0%)</td><td>${tl} (100.0%)</td><td>${tt} (100.0%)</td><td><div class="bar-track"><div class="bar-comp" style="width:${(tc*100/tt).toFixed(1)}%"></div><div class="bar-bum" style="width:${(tb*100/tt).toFixed(1)}%"></div><div class="bar-li" style="width:${(tl*100/tt).toFixed(1)}%"></div></div></td></tr>`;
    }

    function renderFilterPills() {
      const catCounts = {};
      RECORDS.forEach(r => { catCounts[r.categoria_final] = (catCounts[r.categoria_final]||0)+1; });
      const cw = document.getElementById("catPills"); cw.innerHTML = "";
      ["Todas","Masculino","Femenino","Demográfico","Excluyente","Neutro"].forEach(opt => {
        const cnt = opt === "Todas" ? RECORDS.length : (catCounts[opt]||0);
        const btn = document.createElement("button");
        btn.type = "button"; btn.className = `pill-btn cat-${slugCat(opt)} ${state.category===opt?"active":""}`;
        btn.textContent = `${opt} (${cnt})`;
        btn.addEventListener("click", () => { state.category = opt; state.page = 1; renderAll(); });
        cw.appendChild(btn);
      });
      const pw = document.getElementById("portalPills"); pw.innerHTML = "";
      [["Todos", RECORDS.length], ["Computrabajo Perú", RECORDS.filter(r=>r.portal==="Computrabajo Perú").length], ["Bumeran Perú", RECORDS.filter(r=>r.portal==="Bumeran Perú").length], ["LinkedIn Perú", RECORDS.filter(r=>r.portal==="LinkedIn Perú").length]].forEach(([opt,cnt]) => {
        const btn = document.createElement("button");
        btn.type = "button"; btn.className = `pill-btn ${state.portal===opt?"active":""}`;
        btn.textContent = `${opt} (${cnt})`;
        btn.addEventListener("click", () => { state.portal = opt; state.page = 1; renderAll(); });
        pw.appendChild(btn);
      });
      const rw = document.getElementById("riskPills"); rw.innerHTML = "";
      ["Todos","Alto","Moderado","Bajo (Inclusivo)"].forEach(opt => {
        const btn = document.createElement("button");
        btn.type = "button"; btn.className = `pill-btn ${state.risk===opt?"active":""}`;
        btn.textContent = opt;
        btn.addEventListener("click", () => { state.risk = opt; state.page = 1; renderAll(); });
        rw.appendChild(btn);
      });
    }

    function getFiltered() {
      const q = state.query.trim().toLowerCase();
      return RECORDS.filter(r => {
        if (state.category !== "Todas" && r.categoria_final !== state.category) return false;
        if (state.portal !== "Todos" && r.portal !== state.portal) return false;
        if (state.risk !== "Todos" && r.nivel_riesgo !== state.risk) return false;
        if (q) {
          const hay = `${r.id} ${r.portal} ${r.busqueda} ${r.puesto} ${r.categoria_final} ${r.frases_csv} ${r.texto}`.toLowerCase();
          if (!hay.includes(q)) return false;
        }
        return true;
      });
    }

    function renderTable() {
      const allRows = getFiltered();
      const totalPages = Math.max(1, Math.ceil(allRows.length / state.pageSize));
      if (state.page > totalPages) state.page = totalPages;
      const start = (state.page - 1) * state.pageSize;
      const rows = allRows.slice(start, start + state.pageSize);

      document.getElementById("resultCounter").textContent = `Mostrando ${allRows.length} de ${RECORDS.length} avisos`;
      document.getElementById("pagerInfo").textContent = `Página ${state.page} de ${totalPages} (${allRows.length} resultados)`;
      document.getElementById("btnPrev").disabled = state.page <= 1;
      document.getElementById("btnNext").disabled = state.page >= totalPages;

      const tbody = document.getElementById("auditBody");
      const emptyEl = document.getElementById("emptyState");
      tbody.innerHTML = "";
      if (rows.length === 0) { emptyEl.style.display = "block"; return; }
      emptyEl.style.display = "none";

      const frag = document.createDocumentFragment();
      rows.forEach(r => {
        const tr = document.createElement("tr"); tr.className = "row-main";
        const catSlug = slugCat(r.categoria_final);
        const portalCls = r.portal === "Computrabajo Perú" ? "portal-comp" : (r.portal === "Bumeran Perú" ? "portal-bum" : "portal-li");
        const riskCls = r.nivel_riesgo === "Alto" ? "risk-alto" : (r.nivel_riesgo === "Moderado" ? "risk-moderado" : "risk-bajo");
        const justHtml = r.justificaciones.length > 0
          ? `<div class="just-list">` + r.justificaciones.map(j => `<div class="just-item"><span class="badge badge-${slugCat(j.category)}" style="padding:1px 7px;font-size:0.68rem;">${esc(j.category)}</span> <span class="just-phrase">“${esc(j.phrase)}”</span><span class="just-exp">${esc(j.explanation)}</span></div>`).join("") + `</div>`
          : `<div class="neutral-note">Sin frases sesgadas explícitas. Clasificado como <strong>Neutro</strong>.</div>`;
        const previewSnippet = esc(r.texto.slice(0, 210)) + (r.texto.length > 210 ? "…" : "");
        tr.innerHTML = `
          <td><div class="id-code">${esc(r.id)}</div><div class="busq-tag">🔍 ${esc(r.busqueda)}</div></td>
          <td><span class="badge ${portalCls}">${esc(r.portal)}</span></td>
          <td><div class="puesto-title">${esc(r.puesto)}</div><div class="puesto-meta"><span class="${riskCls}">Riesgo: ${esc(r.nivel_riesgo)}</span><span>· Inclusividad: <strong>${r.indice_inclusividad.toFixed(1)}</strong></span></div></td>
          <td><span class="badge badge-${catSlug}">${esc(r.categoria_final)}</span></td>
          <td>${justHtml}</td>
          <td><details class="fulltext-box" ${state.expandAll?"open":""}><summary><span>Ver texto completo (${r.texto_len.toLocaleString("es-PE")} car.)</span><span style="font-size:0.72rem;color:#475569;">${r.num_frases} frase(s) ▾</span></summary><div class="snippet-preview">${previewSnippet}</div><div class="fulltext-body">${r.texto_html}</div></details></td>
        `;
        frag.appendChild(tr);
      });
      tbody.appendChild(frag);
    }

    function renderAll() { renderMatrix(); renderFilterPills(); renderTable(); }
    document.getElementById("searchInput").addEventListener("input", e => { state.query = e.target.value; state.page = 1; renderTable(); });
    document.getElementById("pageSizeSel").addEventListener("change", e => { state.pageSize = parseInt(e.target.value, 10) || 50; state.page = 1; renderTable(); });
    document.getElementById("btnPrev").addEventListener("click", () => { if (state.page > 1) { state.page--; renderTable(); window.scrollTo(0, 420); } });
    document.getElementById("btnNext").addEventListener("click", () => { state.page++; renderTable(); window.scrollTo(0, 420); });
    document.getElementById("btnExpandAll").addEventListener("click", () => { state.expandAll = true; document.querySelectorAll("details.fulltext-box").forEach(d => { d.open = true; }); });
    document.getElementById("btnCollapseAll").addEventListener("click", () => { state.expandAll = false; document.querySelectorAll("details.fulltext-box").forEach(d => { d.open = false; }); });
    document.getElementById("btnReset").addEventListener("click", () => { state.category="Todas"; state.portal="Todos"; state.risk="Todos"; state.query=""; state.page=1; state.expandAll=false; document.getElementById("searchInput").value=""; renderAll(); });

    initKPIs();
    renderAll();
  </script>
</body>
</html>
"""


def main() -> None:
    df = pd.read_csv(RESULTADOS_CSV)
    assert len(df) == 800, f"Se esperaban 800 filas en {RESULTADOS_CSV}, encontradas {len(df)}"
    assert "texto" in df.columns

    summary = build_summary_matrix(df)
    records = build_records(df)

    summary_json = json.dumps(summary, ensure_ascii=False).replace("</", "<\\/")
    records_json = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    final_html = HTML_TEMPLATE.replace("__SUMMARY_JSON__", summary_json).replace("__RECORDS_JSON__", records_json)
    OUTPUT_HTML.write_text(final_html, encoding="utf-8")

    # Renombrar/limpiar auditoria_200.html para dejar auditoria_800.html como versión vigente
    if OLD_HTML.exists():
        OLD_HTML.unlink()

    print(f"=== AUDITORÍA 800 GENERADA: {OUTPUT_HTML} ({OUTPUT_HTML.stat().st_size / 1024:.1f} KB) ===")


if __name__ == "__main__":
    main()
