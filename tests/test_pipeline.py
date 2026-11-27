"""Pruebas automatizadas del pipeline consolidado (1,400 avisos + Random Forest + reglas peruanas)."""

from __future__ import annotations

import json

import pandas as pd

from src.bias_analyzer import analyze_job_posting, detect_biased_spans, extract_technical_terms
from src.config import (
    ALL_CATEGORIES,
    BASE_DIR,
    BIAS_CATEGORIES,
    CONFUSION_CSV_PATH,
    CONSOLIDATED_DATASET_PATH,
    CONSOLIDATED_EXCEL_PATH,
    DATASET_PATH,
    METRICS_JSON_PATH,
    METRICS_SUMMARY_CSV_PATH,
    MODEL_DISPLAY_NAMES,
    PER_CLASS_CSV_PATH,
    PRELOADED_EXAMPLES,
    REAL_PERU_DATASET_PATH,
    RF_MODEL_PATH,
)
from src.rewriter import rewrite_inclusive


def run_all_tests() -> None:
    # 1. Validar modelo fijo Random Forest ligero (< 15 MB) y sin entrenamiento en app.py
    assert list(MODEL_DISPLAY_NAMES.keys()) == ["random_forest"], (
        f"Se esperaba únicamente random_forest en config: {list(MODEL_DISPLAY_NAMES.keys())}"
    )
    assert RF_MODEL_PATH.exists(), f"Falta modelo preentrenado comprimido: {RF_MODEL_PATH}"
    model_size_mb = RF_MODEL_PATH.stat().st_size / (1024 * 1024)
    assert model_size_mb < 15.0, f"El modelo pesa {model_size_mb:.2f} MB (>= 15 MB)"

    import ast
    app_source = (BASE_DIR / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(app_source)
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)
    for forbidden in ("src.train_models", "matplotlib", "seaborn", "openpyxl", "sklearn"):
        assert not any(name == forbidden or name.startswith(forbidden + ".") for name in imported_names), (
            f"app.py no debe importar módulo pesado: {forbidden} ({imported_names})"
        )
    assert "train_and_evaluate" not in app_source, "app.py no debe invocar train_and_evaluate"
    assert "st.tabs" not in app_source, "app.py debe mostrar únicamente el analizador sin pestañas"
    assert "Modelo activo:" not in app_source, "app.py no debe mostrar 'Modelo activo:'"

    # 2. Validar respaldos intactos (1,200 sintéticos y 200 reales) y dataset consolidado (1,400)
    df_synth = pd.read_csv(DATASET_PATH)
    df_real = pd.read_csv(REAL_PERU_DATASET_PATH)
    df_1400 = pd.read_csv(CONSOLIDATED_DATASET_PATH)

    assert len(df_synth) == 1200, f"Respaldo sintético alterado: {len(df_synth)}"
    assert len(df_real) == 200, f"Respaldo real Perú alterado: {len(df_real)}"
    assert len(df_1400) == 1400, f"Dataset consolidado esperado 1400: {len(df_1400)}"
    assert CONSOLIDATED_EXCEL_PATH.exists(), f"Falta respaldo Excel: {CONSOLIDATED_EXCEL_PATH}"

    req_cols = {"id", "origen", "portal", "puesto", "categoria", "nivel_riesgo", "frases_detectadas", "texto"}
    assert req_cols.issubset(set(df_1400.columns)), f"Faltan columnas en 1400: {req_cols - set(df_1400.columns)}"
    assert int(df_1400.isna().sum().sum()) == 0, "Existen valores nulos en dataset_consolidado_1400.csv"
    assert df_1400["texto"].nunique() == 1400, "Existen textos duplicados en dataset_consolidado_1400.csv"
    assert set(df_1400["origen"].unique()) == {"Sintético (1200)", "Real Perú (200)"}

    # 3. Validar métricas Random Forest guardadas sobre 1,400 (split estratificado 80/20)
    for path in [METRICS_JSON_PATH, METRICS_SUMMARY_CSV_PATH, PER_CLASS_CSV_PATH, CONFUSION_CSV_PATH]:
        assert path.exists(), f"Falta archivo de reporte: {path}"

    with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert metrics["dataset"]["total_samples"] == 1400
    assert metrics["dataset"]["train_samples"] == 1120
    assert metrics["dataset"]["test_samples"] == 280
    assert metrics["best_model"]["model_key"] == "random_forest"

    rf_tm = metrics["models"]["random_forest"]["test_metrics"]
    assert 0.85 <= rf_tm["accuracy"] <= 0.97, f"Accuracy inesperada: {rf_tm['accuracy']}"
    assert 0.85 <= rf_tm["f1_macro"] <= 0.97, f"F1-Macro inesperado: {rf_tm['f1_macro']}"

    # 4. Verificar nuevas reglas de limpieza: masculino genérico neutro y restricciones de hombres -> Masculino
    neutral_generic_cases = [
        "Buscamos un analista de sistemas con experiencia en SQL y Python.",
        "Se requiere un ejecutivo comercial para atención de cartera B2B.",
        "Estamos en búsqueda de un vendedor con orientación al cliente.",
        "Necesitamos un arquitecto de software con experiencia en AWS y Docker.",
    ]
    for ad in neutral_generic_cases:
        spans = detect_biased_spans(ad)
        assert len(spans) == 0, f"Masculino genérico NO debe marcarse como sesgo: {[(s.text, s.category) for s in spans]} | {ad}"

    male_explicit_cases = [
        ("Operarios de almacén / Varones - VES", "/ Varones"),
        ("Personal de limpieza - Varón / Surco", "- Varón"),
        ("Reponedor Multifuncional (Varón)", "(Varón)"),
        ("ASISTENTE DE FACTURACIÓN – PERSONAL MASCULINO", "PERSONAL MASCULINO"),
        ("Solo caballeros con secundaria completa", "Solo caballeros"),
        ("Solo hombres para planta industrial", "Solo hombres"),
        ("Buscamos varones para labores de carga", "varones"),
    ]
    for ad, expected_phrase in male_explicit_cases:
        spans = detect_biased_spans(ad)
        assert any(s.category == "Masculino" and expected_phrase.lower() in s.text.lower() for s in spans), (
            f"Esperado Masculino/'{expected_phrase}', obtenido: {[(s.text, s.category) for s in spans]} | {ad}"
        )

    # 5. Validar 6 ejemplos [Real Perú] + 1 único [Caso de prueba] LGBTQ+ en PRELOADED_EXAMPLES
    real_examples = [k for k in PRELOADED_EXAMPLES if k.startswith("[Real Perú]")]
    synth_examples = [k for k in PRELOADED_EXAMPLES if k.startswith("[Caso de prueba]")]
    assert len(real_examples) == 6, f"Se esperaban 6 ejemplos [Real Perú], hay {len(real_examples)}"
    assert len(synth_examples) == 1 and "LGBTQ+" in synth_examples[0], (
        f"Se esperaba exactamente 1 caso sintético LGBTQ+, encontrado: {synth_examples}"
    )
    assert len(PRELOADED_EXAMPLES) == 7, f"Total esperado 7 ejemplos, hay {len(PRELOADED_EXAMPLES)}"

    expected_categories = {
        "[Real Perú] Masculino — Asistente de Facturación (Bumeran)": "Masculino",
        "[Real Perú] Femenino — Anfitriona / Azafata (Bumeran)": "Femenino",
        "[Real Perú] Demográfico — Supervisor/a Mercaderista (Computrabajo)": "Demográfico",
        "[Real Perú] Excluyente — Damas Universitarias / Asesora Comercial (Bumeran)": "Excluyente",
        "[Real Perú] Caso Mixto (Múltiples Sesgos) — Ayudante de Confección Textil (Computrabajo)": "Femenino",
        "[Real Perú] Neutro — Asistente Administrativo SAP (Computrabajo)": "Neutro",
        "[Caso de prueba] Sesgo LGBTQ+ — Profesional en Auditoría Interna": "LGBTQ+",
    }

    for name, text in PRELOADED_EXAMPLES.items():
        res = analyze_job_posting(text, model_key="random_forest")
        assert abs(sum(res.probabilities.values()) - 1.0) < 1e-3, "Probabilidades no suman 1."
        assert set(res.bias_probabilities.keys()) == set(BIAS_CATEGORIES)
        assert res.predicted_class == expected_categories[name], (
            f"Categoría inesperada para '{name}': esperado={expected_categories[name]}, obtenido={res.predicted_class}"
        )
        if "Caso Mixto" in name:
            assert len(res.detected_categories) >= 2, (
                f"Caso Mixto debe detectar >= 2 categorías: {res.detected_categories}"
            )

        rw = rewrite_inclusive(text)
        assert not rw.missing_technical_terms, f"Se perdieron términos técnicos en '{name}': {rw.missing_technical_terms}"
        res_after = analyze_job_posting(rw.rewritten_text, model_key="random_forest")
        assert len(res_after.spans) == 0, (
            f"Quedaron frases sesgadas tras reescribir '{name}': {[s.text for s in res_after.spans]}"
        )
        assert res_after.predicted_class == "Neutro", (
            f"La versión reescrita de '{name}' fue clasificada como {res_after.predicted_class} en lugar de Neutro."
        )

    # 6. Cero falsos positivos en avisos neutros con cláusulas reales de inclusión
    inclusive_neutral_ads = [
        (
            "Buscamos una persona especialista en desarrollo backend con Python, SQL y Docker. "
            "Ofrecemos igualdad de oportunidades sin distinción de género, edad ni orientación."
        ),
        (
            "Convocatoria abierta para profesional en analítica de datos (Power BI y Python). "
            "Llevamos adelante un proceso inclusivo y diverso basado exclusivamente en competencias técnicas."
        ),
        (
            "Nuestro equipo busca profesional en diseño UX/UI con experiencia en Figma. "
            "Garantizamos igualdad de oportunidades y no discriminamos por género, expresión de género, "
            "identidad no binaria, edad, embarazo ni estado de gestación."
        ),
    ]
    for ad in inclusive_neutral_ads:
        spans = detect_biased_spans(ad)
        assert len(spans) == 0, f"Falso positivo de resaltado: {[s.text for s in spans]} | {ad}"
        res = analyze_job_posting(ad, model_key="random_forest")
        assert res.predicted_class == "Neutro", f"Falso positivo ML: {res.predicted_class} | {ad}"

    # 7. Deduplicación de requisitos técnicos
    tech_sample = (
        "Requisitos: dominio de ERP SAP, Microsoft Excel avanzado, CRM Salesforce, "
        "facturación electrónica, gestión documental, control de calidad y gestión de inventarios."
    )
    extracted = extract_technical_terms(tech_sample)
    extracted_lower = [t.lower() for t in extracted]
    for expected in [
        "erp sap",
        "microsoft excel avanzado",
        "crm salesforce",
        "facturación electrónica",
        "gestión documental",
        "control de calidad",
        "gestión de inventarios",
    ]:
        assert expected in extracted_lower, f"Falta término compuesto '{expected}' en {extracted}"
    for redundant in ["erp", "sap", "excel", "crm", "salesforce"]:
        assert redundant not in extracted_lower, f"Sub-término redundante '{redundant}' en {extracted}"

    print(
        f"TODAS LAS PRUEBAS PASARON EXITOSAMENTE | 1400 consolidado (1200+200) | "
        f"RF Acc={rf_tm['accuracy']:.4f}, F1-Macro={rf_tm['f1_macro']:.4f} | "
        f"Ejemplos [Real Perú]={len(real_examples)}"
    )


if __name__ == "__main__":
    run_all_tests()
