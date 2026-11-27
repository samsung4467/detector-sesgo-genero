"""Entrenamiento offline de Random Forest ligero y comprimido para Streamlit Cloud.

Entrena el clasificador sobre data/dataset_consolidado_1400.csv con división estratificada 80/20
(stratify=categoria, random_state=42), optimizado para bajo consumo de RAM:
- TfidfVectorizer: max_features=3000, ngram_range=(1, 2), dtype=float32
- RandomForestClassifier: n_estimators=100, max_depth=25, min_samples_leaf=2
- Exportación comprimida: joblib.dump(modelo, "models/rf_model.joblib", compress=3)

Este script se ejecuta únicamente offline; app.py NUNCA entrena el modelo al iniciar.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    ALL_CATEGORIES,
    BEST_MODEL_META_PATH,
    CONFUSION_CSV_PATH,
    CONSOLIDATED_2000_PATH,
    CONSOLIDATED_DATASET_PATH,
    DATASET_PATH,
    METRICS_JSON_PATH,
    METRICS_SUMMARY_CSV_PATH,
    MODEL_DISPLAY_NAMES,
    MODELS_DIR,
    PER_CLASS_CSV_PATH,
    REPORTS_DIR,
    RF_MODEL_PATH,
)
from src.text_utils import SPANISH_MINIMAL_STOPWORDS, normalize_for_ml

RANDOM_SEED = 42
TEST_SIZE = 0.20


def build_tfidf() -> TfidfVectorizer:
    """Crea el vectorizador TF-IDF ligero (max_features=3000, float32) para español."""
    return TfidfVectorizer(
        preprocessor=normalize_for_ml,
        stop_words=SPANISH_MINIMAL_STOPWORDS,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.90,
        sublinear_tf=True,
        max_features=3000,
        dtype=np.float32,
    )


def build_model_pipelines() -> Dict[str, Pipeline]:
    """Construye el pipeline TF-IDF + Random Forest optimizado para bajo consumo de memoria."""
    return {
        "random_forest": Pipeline(
            [
                ("tfidf", build_tfidf()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=25,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def train_and_evaluate(regenerate_dataset: bool = False) -> Dict[str, Any]:
    """Entrena Random Forest ligero sobre dataset_consolidado_2000.csv (80/20 estratificado)."""
    if CONSOLIDATED_2000_PATH.exists():
        source_path = CONSOLIDATED_2000_PATH
    elif CONSOLIDATED_DATASET_PATH.exists():
        source_path = CONSOLIDATED_DATASET_PATH
    else:
        source_path = DATASET_PATH
    df = pd.read_csv(source_path)

    X = df["texto"].astype(str)
    y = df["categoria"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Limpiar modelos legados/pesados que ya no se usan en producción
    for legacy_name in ("random_forest.joblib", "svm.joblib", "decision_tree.joblib", "logistic_regression.joblib"):
        legacy_path = MODELS_DIR / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    pipelines = build_model_pipelines()

    origin_counts = (
        df["origen"].value_counts().to_dict() if "origen" in df.columns else {"Sintético": int(len(df))}
    )

    metrics_payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(source_path.name),
            "total_samples": int(len(df)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "test_size_ratio": TEST_SIZE,
            "random_seed": RANDOM_SEED,
            "stratified_by": "categoria",
            "classes": ALL_CATEGORIES,
            "class_distribution": {cat: int((y == cat).sum()) for cat in ALL_CATEGORIES},
            "origin_distribution": {str(k): int(v) for k, v in origin_counts.items()},
        },
        "vectorizer": {
            "type": "TfidfVectorizer",
            "ngram_range": [1, 2],
            "min_df": 3,
            "max_df": 0.90,
            "sublinear_tf": True,
            "max_features": 3000,
            "dtype": "float32",
        },
        "models": {},
    }

    summary_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    confusion_rows: List[Dict[str, Any]] = []

    for model_key, pipeline in pipelines.items():
        display_name = MODEL_DISPLAY_NAMES[model_key]

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "f1_macro": "f1_macro",
                "precision_macro": "precision_macro",
                "recall_macro": "recall_macro",
            },
            n_jobs=-1,
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # En inferencia en Streamlit Cloud usar un solo hilo evita picos de RAM por workers joblib
        pipeline.named_steps["clf"].n_jobs = 1

        acc = float(accuracy_score(y_test, y_pred))
        prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        cm = confusion_matrix(y_test, y_pred, labels=ALL_CATEGORIES)
        report_dict = classification_report(
            y_test,
            y_pred,
            labels=ALL_CATEGORIES,
            output_dict=True,
            zero_division=0,
        )

        per_class_clean: Dict[str, Dict[str, float]] = {}
        for cat in ALL_CATEGORIES:
            cat_stats = report_dict[cat]
            per_class_clean[cat] = {
                "precision": round(float(cat_stats["precision"]), 4),
                "recall": round(float(cat_stats["recall"]), 4),
                "f1_score": round(float(cat_stats["f1-score"]), 4),
                "support": int(cat_stats["support"]),
            }
            per_class_rows.append(
                {
                    "model_key": model_key,
                    "modelo": display_name,
                    "categoria": cat,
                    "precision": round(float(cat_stats["precision"]), 4),
                    "recall": round(float(cat_stats["recall"]), 4),
                    "f1_score": round(float(cat_stats["f1-score"]), 4),
                    "support": int(cat_stats["support"]),
                }
            )

        for i, true_label in enumerate(ALL_CATEGORIES):
            row = {
                "model_key": model_key,
                "modelo": display_name,
                "clase_real": true_label,
            }
            for j, pred_label in enumerate(ALL_CATEGORIES):
                row[f"pred_{pred_label}"] = int(cm[i, j])
            confusion_rows.append(row)

        cv_acc_mean = float(np.mean(cv_results["test_accuracy"]))
        cv_acc_std = float(np.std(cv_results["test_accuracy"]))
        cv_f1_mean = float(np.mean(cv_results["test_f1_macro"]))
        cv_f1_std = float(np.std(cv_results["test_f1_macro"]))

        # Guardar modelo comprimido en models/rf_model.joblib
        joblib.dump(pipeline, RF_MODEL_PATH, compress=3)
        size_mb = round(RF_MODEL_PATH.stat().st_size / (1024 * 1024), 3)

        metrics_payload["models"][model_key] = {
            "display_name": display_name,
            "model_file": "models/rf_model.joblib",
            "model_size_mb": size_mb,
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 25,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "class_weight": "balanced",
                "compress": 3,
            },
            "test_metrics": {
                "accuracy": round(acc, 4),
                "precision_macro": round(prec_macro, 4),
                "recall_macro": round(rec_macro, 4),
                "f1_macro": round(f1_macro, 4),
                "precision_weighted": round(prec_weighted, 4),
                "recall_weighted": round(rec_weighted, 4),
                "f1_weighted": round(f1_weighted, 4),
            },
            "cv_5fold_train": {
                "accuracy_mean": round(cv_acc_mean, 4),
                "accuracy_std": round(cv_acc_std, 4),
                "f1_macro_mean": round(cv_f1_mean, 4),
                "f1_macro_std": round(cv_f1_std, 4),
            },
            "per_class": per_class_clean,
            "confusion_matrix": {
                "labels": ALL_CATEGORIES,
                "matrix": cm.tolist(),
            },
        }

        summary_rows.append(
            {
                "model_key": model_key,
                "modelo": display_name,
                "accuracy": round(acc, 4),
                "precision_macro": round(prec_macro, 4),
                "recall_macro": round(rec_macro, 4),
                "f1_macro": round(f1_macro, 4),
                "precision_weighted": round(prec_weighted, 4),
                "recall_weighted": round(rec_weighted, 4),
                "f1_weighted": round(f1_weighted, 4),
                "cv5_accuracy_mean": round(cv_acc_mean, 4),
                "cv5_f1_macro_mean": round(cv_f1_mean, 4),
                "cv5_f1_macro_std": round(cv_f1_std, 4),
            }
        )

    metrics_payload["best_model"] = {
        "model_key": "random_forest",
        "display_name": MODEL_DISPLAY_NAMES["random_forest"],
        "model_file": "models/rf_model.joblib",
        "selection_criterion": "f1_macro (test set 20% estratificado)",
    }

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    with open(BEST_MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload["best_model"], f, ensure_ascii=False, indent=2)

    pd.DataFrame(summary_rows).to_csv(METRICS_SUMMARY_CSV_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(per_class_rows).to_csv(PER_CLASS_CSV_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(confusion_rows).to_csv(CONFUSION_CSV_PATH, index=False, encoding="utf-8-sig")

    return metrics_payload


if __name__ == "__main__":
    results = train_and_evaluate(regenerate_dataset=False)
    info = results["models"]["random_forest"]
    tm = info["test_metrics"]
    cv_m = info["cv_5fold_train"]
    print("=== Random Forest Ligero y Comprimido (Test Set 20%) ===")
    print(
        f"- Archivo: {info['model_file']} ({info['model_size_mb']:.3f} MB) | "
        f"Acc: {tm['accuracy']:.4f} | "
        f"Prec: {tm['precision_macro']:.4f} | "
        f"Rec: {tm['recall_macro']:.4f} | "
        f"F1-Macro: {tm['f1_macro']:.4f} | "
        f"F1-Weighted: {tm['f1_weighted']:.4f} | "
        f"CV5-F1: {cv_m['f1_macro_mean']:.4f} (±{cv_m['f1_macro_std']:.4f})"
    )
