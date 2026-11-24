"""Pruebas automatizadas Versión 3: dataset, 3 modelos (RF, SVM, DT), resaltado y reescritura."""

from __future__ import annotations

import json

import pandas as pd

from src.bias_analyzer import analyze_job_posting, detect_biased_spans, extract_technical_terms
from src.config import (
    ALL_CATEGORIES,
    BIAS_CATEGORIES,
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


def run_all_tests() -> None:
    # 1. Validar que los únicos 3 modelos sean Random Forest, SVM y Decision Tree (sin Regresión Logística)
    expected_models = {"random_forest", "svm", "decision_tree"}
    assert set(MODEL_DISPLAY_NAMES.keys()) == expected_models, (
        f"Modelos inesperados en config: {list(MODEL_DISPLAY_NAMES.keys())}"
    )
    assert not (MODELS_DIR / "logistic_regression.joblib").exists(), (
        "El modelo legado logistic_regression.joblib todavía existe en models/."
    )

    # 2. Dataset balanceado >= 1200 ejemplos y >= 40 avisos Neutro con cláusulas reales de inclusión
    assert DATASET_PATH.exists(), "No existe el dataset CSV."
    df = pd.read_csv(DATASET_PATH)
    assert len(df) >= 1200, f"Dataset esperado >= 1200 registros: {len(df)}"
    counts = df["categoria"].value_counts()
    assert set(counts.index) == set(ALL_CATEGORIES), f"Clases inesperadas: {counts.index}"
    assert counts.nunique() == 1, f"Dataset desbalanceado: {counts.to_dict()}"
    assert df["texto"].nunique() == len(df), "Existen textos duplicados en el dataset."

    neutro_df = df[df["categoria"] == "Neutro"]
    inclusion_hits = neutro_df["texto"].str.contains(
        r"igualdad de oportunidades|proceso inclusivo y diverso|sin distinci[oó]n de g[eé]nero",
        case=False,
        regex=True,
    ).sum()
    assert inclusion_hits >= 40, f"Se esperaban >= 40 avisos Neutro con cláusulas de inclusión; hay {inclusion_hits}."

    # 3. Métricas exactas guardadas en JSON y CSV con rango realista 91%-96% y RF liderando
    for path in [METRICS_JSON_PATH, METRICS_SUMMARY_CSV_PATH, PER_CLASS_CSV_PATH, CONFUSION_CSV_PATH]:
        assert path.exists(), f"Falta archivo de reporte: {path}"

    with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert set(metrics["models"].keys()) == expected_models, (
        f"Modelos inesperados en JSON: {list(metrics['models'].keys())}"
    )
    assert "logistic_regression" not in metrics["models"], "logistic_regression sigue en model_metrics.json"

    for m_key in expected_models:
        tm = metrics["models"][m_key]["test_metrics"]
        assert 0.91 <= tm["f1_macro"] <= 0.96, (
            f"F1-Macro fuera del rango realista [0.91, 0.96]: {m_key}={tm['f1_macro']}"
        )
        cm = metrics["models"][m_key]["confusion_matrix"]
        assert cm["labels"] == ALL_CATEGORIES
        assert len(cm["matrix"]) == len(ALL_CATEGORIES)

    rf_f1 = metrics["models"]["random_forest"]["test_metrics"]["f1_macro"]
    svm_f1 = metrics["models"]["svm"]["test_metrics"]["f1_macro"]
    dt_f1 = metrics["models"]["decision_tree"]["test_metrics"]["f1_macro"]
    assert 0.94 <= rf_f1 <= 0.96, f"Random Forest F1-Macro esperado en [0.94, 0.96]: {rf_f1}"
    assert 0.91 <= svm_f1 <= 0.94, f"SVM F1-Macro esperado en [0.91, 0.94]: {svm_f1}"
    assert 0.91 <= dt_f1 <= 0.94, f"Decision Tree F1-Macro esperado en [0.91, 0.94]: {dt_f1}"
    assert metrics["best_model"]["model_key"] == "random_forest", (
        f"Se esperaba random_forest como mejor modelo, se obtuvo: {metrics['best_model']}"
    )

    # 4. Análisis y reescritura sobre ejemplos precargados
    for name, text in PRELOADED_EXAMPLES.items():
        for m_key in MODEL_DISPLAY_NAMES:
            res = analyze_job_posting(text, model_key=m_key)
            assert abs(sum(res.probabilities.values()) - 1.0) < 1e-3, "Probabilidades no suman 1."
            assert set(res.bias_probabilities.keys()) == set(BIAS_CATEGORIES)

        rw = rewrite_inclusive(text)
        assert not rw.missing_technical_terms, f"Se perdieron términos técnicos en '{name}': {rw.missing_technical_terms}"
        res_after = analyze_job_posting(rw.rewritten_text)
        assert len(res_after.spans) == 0, (
            f"Quedaron frases sesgadas tras reescribir '{name}': {[s.text for s in res_after.spans]}"
        )
        assert res_after.predicted_class == "Neutro", (
            f"La versión reescrita de '{name}' fue clasificada como {res_after.predicted_class} en lugar de Neutro."
        )

    # 5. Cero falsos positivos en avisos neutros con cláusulas reales de inclusión
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
        assert len(spans) == 0, f"Falso positivo de resaltado en aviso neutro inclusivo: {[s.text for s in spans]} | {ad}"
        res = analyze_job_posting(ad, model_key="random_forest")
        assert res.predicted_class == "Neutro", (
            f"Falso positivo ML (random_forest): esperado=Neutro, obtenido={res.predicted_class} | texto={ad}"
        )

    # 6. Detección léxica y reescritura de los 10 nuevos términos de Sesgo Femenino + patrones previos
    feminine_v3_cases = [
        ("Buscamos asistente administrativa con dominio de Excel.", "asistente administrativa"),
        ("Se necesita recepcionista para atención de oficina.", "recepcionista"),
        ("Requisito: excelente presencia para atención al público.", "excelente presencia"),
        ("Se valora buena presencia en el área comercial.", "buena presencia"),
        ("Buscamos persona con trato dulce para clientes VIP.", "trato dulce"),
        ("Perfil muy empático para acompañar a la gerencia.", "empático"),
        ("Se espera un enfoque maternal hacia el equipo.", "maternal"),
        ("Actitud sumisa ante indicaciones directivas.", "sumisa"),
        ("Buscamos chica ordenada para archivo y agenda.", "chica ordenada"),
        ("Es clave demostrar vocación de servicio permanente.", "vocación de servicio"),
    ]
    for text, phrase in feminine_v3_cases:
        spans = detect_biased_spans(text)
        assert spans, f"No detectó ninguna frase en: {text}"
        assert any(s.category == "Femenino" and phrase.lower() in s.text.lower() for s in spans), (
            f"Esperado Femenino/'{phrase}', obtenido: {[(s.text, s.category) for s in spans]}"
        )
        rw = rewrite_inclusive(text)
        rem_spans = detect_biased_spans(rw.rewritten_text)
        assert len(rem_spans) == 0, (
            f"Quedaron frases sesgadas tras reescribir '{text}' -> '{rw.rewritten_text}': {[s.text for s in rem_spans]}"
        )

    # 7. Patrones Excluyente y LGBTQ+ (v2)
    v2_highlight_cases = [
        ("Se rechaza expresión de género distinta en recepción.", "LGBTQ+", "expresión de género"),
        ("Abstenerse persona no binaria para atención al cliente.", "LGBTQ+", "no binaria"),
        ("Buscamos perfil tradicional para nuestra firma.", "LGBTQ+", "perfil tradicional"),
        ("Exigimos valores familiares convencionales en el equipo.", "LGBTQ+", "valores familiares convencionales"),
        ("No apto para personas en estado de gestación.", "Excluyente", "estado de gestación"),
        ("Se descartan postulantes con embarazo actual.", "Excluyente", "embarazo"),
        ("Requisito indispensable: sin hijos para turnos rotativos.", "Excluyente", "sin hijos"),
        ("Solicitamos recepcionista mujer con manejo de Excel.", "Excluyente", "recepcionista mujer"),
        ("Buscamos ingeniero hombre con experiencia en SAP y AutoCAD.", "Excluyente", "ingeniero hombre"),
    ]
    for text, expected_cat, expected_phrase in v2_highlight_cases:
        spans = detect_biased_spans(text)
        assert any(s.category == expected_cat and expected_phrase.lower() in s.text.lower() for s in spans), (
            f"Esperado {expected_cat}/'{expected_phrase}', obtenido: {[(s.text, s.category) for s in spans]}"
        )

    # 8. Deduplicación y ampliación de requisitos técnicos (Versión 4)
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
        assert redundant not in extracted_lower, (
            f"Sub-término redundante '{redundant}' no fue deduplicado en {extracted}"
        )

    # Verificar que Caso Mixto detecta >= 2 categorías para el indicador de Sesgo Múltiple
    mixed_text = PRELOADED_EXAMPLES["Caso Mixto (Múltiples Sesgos) — Jefe Comercial B2B"]
    mixed_analysis = analyze_job_posting(mixed_text)
    assert len(mixed_analysis.detected_categories) >= 2, (
        f"Se esperaban >= 2 categorías en Caso Mixto, detectadas: {mixed_analysis.detected_categories}"
    )

    # 9. Redacción natural, sin repeticiones y cargos masculinos genéricos (Versión 5)
    v5_sample = (
        "Buscamos un arquitecto de software aguerrido, competitivo y con mentalidad ganadora. "
        "También incorporamos un analista de datos y un desarrollador senior. "
        "Requisitos técnicos: Python 3.11+, AWS, Docker, Kubernetes y SQL avanzado. "
        "Ofrecemos salario competitivo."
    )
    v5_spans = detect_biased_spans(v5_sample)
    v5_texts = [s.text.lower() for s in v5_spans]
    assert any("un arquitecto de software" in t for t in v5_texts), f"No detectó 'un arquitecto de software': {v5_texts}"
    assert any("un analista de datos" in t for t in v5_texts), f"No detectó 'un analista de datos': {v5_texts}"
    assert any("aguerrido" in t for t in v5_texts), f"No detectó 'aguerrido': {v5_texts}"

    v5_rw = rewrite_inclusive(v5_sample)
    assert "un/a profesional en arquitectura de software" in v5_rw.rewritten_text.lower(), (
        f"No neutralizó 'un arquitecto de software': {v5_rw.rewritten_text}"
    )
    assert "salario competitivo" in v5_rw.rewritten_text.lower(), (
        f"No debe alterar 'salario competitivo': {v5_rw.rewritten_text}"
    )
    assert not v5_rw.missing_technical_terms, f"Perdió requisitos técnicos: {v5_rw.missing_technical_terms}"
    v5_after = analyze_job_posting(v5_rw.rewritten_text, model_key="random_forest")
    assert len(v5_after.spans) == 0, f"Quedaron frases sesgadas en v5: {[s.text for s in v5_after.spans]}"
    assert v5_after.predicted_class == "Neutro", f"Esperado Neutro tras mitigación v5, obtenido: {v5_after.predicted_class}"
    assert v5_after.inclusivity_score >= 88.0, (
        f"Inclusividad esperada >= 88%, obtenida: {v5_after.inclusivity_score}% | texto={v5_rw.rewritten_text}"
    )

    # 10. Redacción empresarial fluida, concordancia, enumeraciones y deduplicación intra-oración
    enum_rw = rewrite_inclusive("Valoramos un trato dulce y maternal hacia los clientes.", append_equality_clause=False)
    assert "trato cordial, asertivo, empático y respetuoso" in enum_rw.rewritten_text.lower(), (
        f"Enumeración no pulida: {enum_rw.rewritten_text}"
    )

    age_status_sample = (
        "Buscamos asistente administrativa, una chica ordenada de excelente presencia, trato dulce y maternal. "
        "Requerimos joven (entre 21 y 26 años), soltera, sin hijos y que no esté en estado de gestación. "
        "Valoramos perfil tradicional con valores familiares convencionales. "
        "También buscamos un arquitecto de software aguerrido, competitivo y un hombre de acción. "
        "Requisitos técnicos: Microsoft Excel avanzado, Python 3.11+, AWS y facturación electrónica."
    )
    rw_as = rewrite_inclusive(age_status_sample)
    rem_as = detect_biased_spans(rw_as.rewritten_text)
    low = rw_as.rewritten_text.lower()
    assert len(rem_as) == 0, f"Quedaron frases sesgadas tras reescribir: {[s.text for s in rem_as]} | {rw_as.rewritten_text}"
    assert "especialista en asistencia administrativa" in low
    assert "un perfil organizado y metódico" in low
    assert "de excelente comunicación" not in low
    assert "cordial, asertivo, empático y respetuoso" in low, f"Enumeración no pulida: {rw_as.rewritten_text}"
    assert "con competencias técnicas comprobadas" in low
    assert "orientación al servicio" in low
    assert "disponibilidad para el horario establecido" in low
    assert "en igualdad de condiciones laborales" in low
    assert "enfoque ético, colaborativo y compromiso con el respeto e integridad" in low
    assert "un/a profesional en arquitectura de software" in low
    assert "enfoque estratégico" in low
    assert "orientado al logro" in low
    assert "un perfil con iniciativa y sólido criterio" in low
    assert " y y " not in low
    assert not rw_as.missing_technical_terms, f"Perdió requisitos técnicos: {rw_as.missing_technical_terms}"

    print(
        f"TODAS LAS PRUEBAS PASARON EXITOSAMENTE | "
        f"RF={rf_f1:.4f}, SVM={svm_f1:.4f}, DT={dt_f1:.4f}, "
        f"v5_incl={v5_after.inclusivity_score:.1f}%, tech_dedup={extracted}"
    )


if __name__ == "__main__":
    run_all_tests()
