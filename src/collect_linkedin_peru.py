"""Recolector público (invitado) de ofertas laborales reales en Lima desde LinkedIn Perú (`jobs-guest`).

Usa únicamente los endpoints públicos de búsqueda de invitados de LinkedIn sin autenticación,
con pausas moderadas entre peticiones y detención inmediata ante cualquier authwall/CAPTCHA real.
Guarda los resultados completos en scratch/raw_600/linkedin_lima.json.
"""

from __future__ import annotations

import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Set, Tuple

OUT_PATH = Path(
    r"C:\Users\verde\.gemini\antigravity\brain\8c6c8e80-06b2-44e5-a8d8-62ebb1c9ed98\scratch\raw_600\linkedin_lima.json"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-PE,es;q=0.9,en;q=0.4",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Búsquedas objetivas en Lima, Perú (sin "buena presencia")
QUERY_PLAN: List[Tuple[str, int]] = [
    # Términos dirigidos (sesgos objetivos — toma los disponibles en español en Lima)
    ("varones", 10),
    ("caballeros", 10),
    ("personal masculino", 10),
    ("operario varón", 8),
    ("chofer varón", 8),
    ("señoritas", 10),
    ("anfitriona", 12),
    ("azafata", 10),
    ("entre 20 y 30", 8),
    ("entre 18 y 25", 8),
    ("edad máxima", 8),
    ("abstenerse", 12),
    ("solo peruanos", 8),
    ("sin hijos", 8),
    # Control operativo y corporativo en Lima (alta disponibilidad en LinkedIn Perú)
    ("analista", 38),
    ("asistente administrativo", 36),
    ("vendedor", 32),
    ("operario", 30),
    ("ejecutivo comercial", 36),
    ("ingeniero", 38),
]


def _fetch_html(url: str, max_retries: int = 3) -> str:
    delay = 1.5
    for attempt in range(max_retries):
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=18) as resp:
                final_url = resp.geturl().lower()
                if "/checkpoint/" in final_url or "authwall" in final_url:
                    raise RuntimeError(f"AUTHWALL_OR_CAPTCHA: {final_url}")
                raw = resp.read().decode("utf-8", errors="replace")
                return raw
        except urllib.error.HTTPError as e:
            if e.code == 429:
                if attempt == max_retries - 1:
                    return ""
                time.sleep(delay)
                delay *= 2.0
            elif e.code in (400, 404, 410):
                return ""
            elif e.code == 999:
                raise RuntimeError("LINKEDIN_BLOCK_999")
            else:
                if attempt == max_retries - 1:
                    return ""
                time.sleep(delay)
        except RuntimeError:
            raise
        except Exception:
            if attempt == max_retries - 1:
                return ""
            time.sleep(delay)
    return ""


def _strip_tags(fragment: str) -> str:
    s = re.sub(r"<(?:br|p|li|div|h[1-6])[^>]*>", "\n", fragment, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


def _parse_search_cards(page_html: str) -> List[Dict[str, str]]:
    """Extrae job_id, url, puesto, empresa y ubicación desde seeMoreJobPostings."""
    cards: List[Dict[str, str]] = []
    for block in re.split(r"<li\b", page_html)[1:]:
        m_id = re.search(r"urn:li:jobPosting:(\d+)", block)
        if not m_id:
            m_id = re.search(r"/jobs/view/[^\"']*?(\d{8,})", block)
        if not m_id:
            continue
        job_id = m_id.group(1)

        m_title = re.search(
            r'base-search-card__title[^>]*>(.*?)</h3', block, flags=re.IGNORECASE | re.DOTALL
        )
        m_comp = re.search(
            r'base-search-card__subtitle[^>]*>(.*?)</h4', block, flags=re.IGNORECASE | re.DOTALL
        )
        m_loc = re.search(
            r'job-search-card__location[^>]*>(.*?)</span', block, flags=re.IGNORECASE | re.DOTALL
        )
        title = _strip_tags(m_title.group(1)) if m_title else ""
        comp = _strip_tags(m_comp.group(1)) if m_comp else ""
        loc = _strip_tags(m_loc.group(1)) if m_loc else ""
        cards.append(
            {
                "job_id": job_id,
                "url": f"https://pe.linkedin.com/jobs/view/{job_id}",
                "puesto": title,
                "empresa": comp,
                "ubicacion": loc,
            }
        )
    return cards


def _parse_job_detail(detail_html: str) -> Tuple[str, str, str]:
    """Extrae (puesto, empresa, descripcion_completa) del endpoint jobPosting/{id}."""
    m_title = re.search(
        r'top-card-layout__title[^>]*>(.*?)</h[12]', detail_html, flags=re.IGNORECASE | re.DOTALL
    )
    m_comp = re.search(
        r'topcard__org-name-link[^>]*>(.*?)</a>', detail_html, flags=re.IGNORECASE | re.DOTALL
    )
    if not m_comp:
        m_comp = re.search(
            r'topcard__flavor[^>]*>(.*?)</span>', detail_html, flags=re.IGNORECASE | re.DOTALL
        )
    m_desc = re.search(
        r'show-more-less-html__markup[^>]*>(.*?)</div>', detail_html, flags=re.IGNORECASE | re.DOTALL
    )
    if not m_desc:
        m_desc = re.search(
            r'description__text[^>]*>(.*?)</section>', detail_html, flags=re.IGNORECASE | re.DOTALL
        )

    title = _strip_tags(m_title.group(1)) if m_title else ""
    comp = _strip_tags(m_comp.group(1)) if m_comp else ""
    desc = _strip_tags(m_desc.group(1)) if m_desc else ""
    return title, comp, desc


def _looks_spanish(text: str) -> bool:
    """Filtra avisos mayoritariamente en español."""
    low = " " + re.sub(r"\s+", " ", text.lower()) + " "
    es_hits = sum(
        1
        for w in (
            " de ", " la ", " el ", " en ", " para ", " con ", " experiencia ",
            " requisitos ", " funciones ", " puesto ", " equipo ", " trabajo ",
            " empresa ", " buscamos ", " conocimiento ", " gestión ", " área ",
            " responsabilidades ", " beneficios ", " perfil ",
        )
        if w in low
    )
    return es_hits >= 5


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    target_total = 225

    results: List[Dict[str, str]] = []
    seen_ids: Set[str] = set()

    if OUT_PATH.exists():
        try:
            prev = json.loads(OUT_PATH.read_text(encoding="utf-8"))
            if isinstance(prev, list):
                results = prev
                seen_ids = {str(x.get("job_id", "")) for x in results if x.get("job_id")}
        except Exception:
            pass

    print(f"Iniciando LinkedIn Perú (previos={len(results)}, meta={target_total})...", flush=True)

    for query, quota in QUERY_PLAN:
        if len(results) >= target_total:
            break
        collected_q = 0
        # LinkedIn seeMoreJobPostings devuelve ~10 resultados por página
        max_pages = 8 if quota >= 30 else 3
        for page_i in range(max_pages):
            if collected_q >= quota or len(results) >= target_total:
                break
            start = page_i * 10
            params = urllib.parse.urlencode(
                {
                    "keywords": query,
                    "location": "Lima, Peru",
                    "start": str(start),
                }
            )
            search_url = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?{params}"
            page_html = _fetch_html(search_url)
            time.sleep(0.65)
            if not page_html:
                break
            cards = _parse_search_cards(page_html)
            if not cards:
                break

            for card in cards:
                if collected_q >= quota or len(results) >= target_total:
                    break
                jid = card["job_id"]
                if jid in seen_ids:
                    continue
                seen_ids.add(jid)

                # Preferir ofertas ubicadas en Lima / Perú
                loc_low = (card.get("ubicacion") or "").lower()
                if loc_low and ("lima" not in loc_low and "peru" not in loc_low and "perú" not in loc_low):
                    continue

                detail_url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{jid}"
                detail_html = _fetch_html(detail_url)
                time.sleep(0.55)
                if not detail_html:
                    continue

                d_title, d_comp, d_desc = _parse_job_detail(detail_html)
                puesto = d_title or card["puesto"]
                empresa = d_comp or card["empresa"]
                if not d_desc or len(d_desc) < 180:
                    continue
                if not _looks_spanish(d_desc):
                    continue

                results.append(
                    {
                        "job_id": jid,
                        "portal": "LinkedIn Perú",
                        "busqueda": query,
                        "url": card["url"],
                        "puesto": puesto,
                        "empresa": empresa,
                        "descripcion": d_desc,
                    }
                )
                collected_q += 1

                if len(results) % 10 == 0:
                    OUT_PATH.write_text(
                        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
                    )
                    print(
                        f"  [LinkedIn] acumulado={len(results)}/{target_total} | query='{query}' ({collected_q}/{quota})",
                        flush=True,
                    )

        print(f"[LinkedIn] query='{query}' -> +{collected_q} (total={len(results)})", flush=True)
        OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"=== LINKEDIN COMPLETADO: {len(results)} avisos guardados en {OUT_PATH} ===", flush=True)


if __name__ == "__main__":
    main()
