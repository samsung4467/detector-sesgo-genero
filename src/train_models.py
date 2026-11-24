"""Entrenamiento y evaluación comparativa de 3 clasificadores ML con TF-IDF.

Modelos entrenados (Versión 3):
1. Random Forest (RandomForestClassifier)
2. SVM (CalibratedClassifierCV + SVC)
3. Decision Tree (DecisionTreeClassifier)

Guarda:
- Pipelines serializados en models/*.joblib
- Métricas exactas (Accuracy, Precision, Recall, F1-Score, Matriz de Confusión y CV-5)
  en reports/model_metrics.json y archivos CSV en reports/*.csv.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
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
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    ALL_CATEGORIES,
    BEST_MODEL_META_PATH,
    CONFUSION_CSV_PATH,
    DATASET_PATH,
    METRICS_JSON_PATH,
    METRICS_SUMMARY_CSV_PATH,
    MODEL_DISPLAY_NAMES,
    MODELS_DIR,
    PER_CLASS_CSV_PATH,
    REPORTS_DIR,
)
from src.generate_dataset import generate_balanced_dataset
from src.text_utils import SPANISH_MINIMAL_STOPWORDS, normalize_for_ml

RANDOM_SEED = 42
TEST_SIZE = 0.20


def build_tfidf() -> TfidfVectorizer:
    """Crea el vectorizador TF-IDF adaptado para texto en español con señales de sesgo."""
    return TfidfVectorizer(
        preprocessor=normalize_for_ml,
        stop_words=SPANISH_MINIMAL_STOPWORDS,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.90,
        sublinear_tf=True,
        max_features=4500,
    )


def build_model_pipelines() -> Dict[str, Pipeline]:
    """Construye los 3 pipelines TF-IDF + clasificador (Random Forest, SVM y Decision Tree)."""
    return {
        "random_forest": Pipeline(
            [
                ("tfidf", build_tfidf()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=1,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "svm": Pipeline(
            [
                ("tfidf", build_tfidf()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        estimator=SVC(
                            kernel="rbf",
                            C=0.62,
                            gamma="scale",
                            class_weight="balanced",
                            random_state=RANDOM_SEED,
                        ),
                        cv=3,
                        method="sigmoid",
                        ensemble=False,
                    ),
                ),
            ]
        ),
        "decision_tree": Pipeline(
            [
                ("tfidf", build_tfidf()),
                (
                    "clf",
                    DecisionTreeClassifier(
                        criterion="gini",
                        max_depth=20,
                        min_samples_split=4,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    }


def train_and_evaluate(regenerate_dataset: bool = False) -> Dict[str, Any]:
    """Entrena los 3 modelos, evalúa métricas exactas y exporta artefactos JSON/CSV y modelos."""
    if regenerate_dataset or not DATASET_PATH.exists():
        df = generate_balanced_dataset()
    else:
        df = pd.read_csv(DATASET_PATH)

    X = df["texto"].astype(str)
    y = df["categoria"].astype(str)
    stratify_key = (
        df["categoria"].astype(str) + "__" + df["nivel_sesgo"].astype(str)
        if "nivel_sesgo" in df.columns
        else y
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=stratify_key,
        random_state=RANDOM_SEED,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Limpiar posibles modelos legados que ya no forman parte del pipeline
    legacy_lr = MODELS_DIR / "logistic_regression.joblib"
    if legacy_lr.exists():
        legacy_lr.unlink()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    pipelines = build_model_pipelines()

    metrics_payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(DATASET_PATH.name),
            "total_samples": int(len(df)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "test_size_ratio": TEST_SIZE,
            "random_seed": RANDOM_SEED,
            "classes": ALL_CATEGORIES,
            "class_distribution": {
                cat: int((y == cat).sum()) for cat in ALL_CATEGORIES
            },
        },
        "vectorizer": {
            "type": "TfidfVectorizer",
            "ngram_range": [1, 2],
            "min_df": 3,
            "max_df": 0.90,
            "sublinear_tf": True,
            "max_features": 4500,
        },
        "models": {},
    }

    summary_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    confusion_rows: List[Dict[str, Any]] = []

    best_model_key = None
    best_score = -1.0

    for model_key, pipeline in pipelines.items():
        display_name = MODEL_DISPLAY_NAMES[model_key]

        # Validación cruzada 5-fold sobre el conjunto de entrenamiento
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

        # Entrenamiento final en train split y evaluación en test split
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

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

        metrics_payload["models"][model_key] = {
            "display_name": display_name,
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

        model_path = MODELS_DIR / f"{model_key}.joblib"
        joblib.dump(pipeline, model_path)

        composite_score = f1_macro + 0.001 * cv_f1_mean
        if composite_score > best_score:
            best_score = composite_score
            best_model_key = model_key

    metrics_payload["best_model"] = {
        "model_key": best_model_key,
        "display_name": MODEL_DISPLAY_NAMES[best_model_key],
        "selection_criterion": "f1_macro (test set)",
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
    print("=== Resultados de Evaluación (Test Set 20%) ===")
    for key, info in results["models"].items():
        tm = info["test_metrics"]
        cv_m = info["cv_5fold_train"]
        print(
            f"- {info['display_name']:18s} | "
            f"Acc: {tm['accuracy']:.4f} | "
            f"Prec: {tm['precision_macro']:.4f} | "
            f"Rec: {tm['recall_macro']:.4f} | "
            f"F1: {tm['f1_macro']:.4f} | "
            f"CV5-F1: {cv_m['f1_macro_mean']:.4f} (±{cv_m['f1_macro_std']:.4f})"
        )
    print(f"\nMejor modelo: {results['best_model']['display_name']}")
    print(f"Reportes guardados en: {REPORTS_DIR}")
