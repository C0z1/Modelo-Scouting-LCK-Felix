"""Entrenamiento, validación cruzada y búsqueda de hiperparámetros.

Funciones reutilizables que el notebook y el driver orquestan. Todo modelo se
envuelve con el preprocesamiento de la fase final para que la ingeniería de
características y las transformaciones se ajusten SOLO con train dentro de cada
fold (sin fuga).
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE
from .preprocessing import build_preprocessor_base, build_preprocessor_final

SCORING = ["f1", "roc_auc", "precision", "recall", "accuracy"]


# --------------------------------------------------------------------------- #
# Catálogos de modelos                                                         #
# --------------------------------------------------------------------------- #
def modelos_avance() -> dict[str, Any]:
    """Los modelos del avance (para reproducir con el pipeline base)."""
    return {
        "Dummy (baseline)": DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
        "Regresión Logística": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "Árbol de Decisión": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5, random_state=RANDOM_STATE
        ),
    }


def modelos_fase_final() -> dict[str, Any]:
    """Familias de modelos comparadas en la fase final.

    Cubre: trivial (Dummy), lineal (LogReg), ensambles (RF, GB, HistGB),
    márgenes (SVM lineal y RBF) y red neuronal (MLP, opcional).
    """
    return {
        "Dummy (baseline)": DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
        "Regresión Logística": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=None, max_iter=300,
            l2_regularization=1.0, random_state=RANDOM_STATE
        ),
        "SVM lineal": SVC(
            kernel="linear", C=1.0, class_weight="balanced",
            probability=False, random_state=RANDOM_STATE
        ),
        "SVM RBF": SVC(
            kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
            probability=False, random_state=RANDOM_STATE
        ),
        "MLP (red neuronal)": MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3,
            early_stopping=True, n_iter_no_change=10, max_iter=300,
            random_state=RANDOM_STATE
        ),
    }


# --------------------------------------------------------------------------- #
# Construcción de pipelines completos                                          #
# --------------------------------------------------------------------------- #
def make_pipeline_final(clasificador, min_frequency: float | None = None) -> Pipeline:
    """Preprocesamiento fase final + clasificador."""
    pre = build_preprocessor_final(min_frequency=min_frequency)
    return Pipeline(steps=[("pre", pre), ("clf", clasificador)])


def make_pipeline_base(clasificador) -> Pipeline:
    """Preprocesamiento del avance (sin características nuevas) + clasificador."""
    from sklearn.pipeline import Pipeline as SkPipeline
    return SkPipeline(steps=[("preprocessor", build_preprocessor_base()), ("classifier", clasificador)])


# --------------------------------------------------------------------------- #
# Validación cruzada                                                           #
# --------------------------------------------------------------------------- #
def evaluar_cv(
    modelos: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv,
    pipeline_builder=make_pipeline_final,
) -> pd.DataFrame:
    """Evalúa un catálogo de modelos con validación cruzada y devuelve una tabla.

    Mide media y desviación estándar de cada métrica y el tiempo de ajuste/predicción.
    """
    filas = []
    for nombre, modelo in modelos.items():
        pipe = pipeline_builder(modelo)
        t0 = time.perf_counter()
        cvres = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=SCORING,
            n_jobs=-1, return_train_score=False,
        )
        elapsed = time.perf_counter() - t0
        fila = {"Modelo": nombre}
        for m in SCORING:
            fila[f"{m}_mean"] = cvres[f"test_{m}"].mean()
            fila[f"{m}_std"] = cvres[f"test_{m}"].std()
        fila["fit_time_mean"] = cvres["fit_time"].mean()
        fila["score_time_mean"] = cvres["score_time"].mean()
        fila["cv_wall_time_s"] = elapsed
        filas.append(fila)
    return pd.DataFrame(filas).set_index("Modelo")
