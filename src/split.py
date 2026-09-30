"""Partición de datos y estrategia de validación.

Reproduce la partición del avance: 70 / 15 / 15 (train / validación / test),
estratificada por la etiqueta, con semilla 42. La columna `league` (y cualquier
otra con fuga) se elimina ANTES de particionar.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split

from .config import BASE_FEATURES, LEAK_COLS, RANDOM_STATE, TARGET


def separar_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve (X, y) usando solo las columnas base; excluye columnas con fuga."""
    presentes = [c for c in LEAK_COLS if c in df.columns]
    X = df[[c for c in BASE_FEATURES if c in df.columns]].copy()
    y = df[TARGET].copy()
    # Verificación explícita de que no quedó ninguna columna con fuga en X
    fuga_en_X = [c for c in presentes if c in X.columns]
    assert not fuga_en_X, f"Columnas con fuga presentes en X: {fuga_en_X}"
    return X, y


def particion_70_15_15(X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_STATE):
    """70/15/15 estratificado (idéntico al avance)."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def cv_avance(random_state: int = RANDOM_STATE) -> StratifiedKFold:
    """CV del avance: StratifiedKFold(5). Se usa en la reproducción."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)


def cv_final(n_splits: int = 5, n_repeats: int = 3, random_state: int = RANDOM_STATE):
    """CV de la fase final: RepeatedStratifiedKFold.

    Más estable que una sola partición porque promedia el desempeño sobre
    n_splits*n_repeats particiones distintas, reduciendo la varianza de la
    estimación y la dependencia de una división puntual. Nunca toca el conjunto
    de test: toda la selección se hace con train (dentro de estos folds).
    """
    return RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
