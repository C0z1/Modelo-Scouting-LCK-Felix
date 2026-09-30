"""Evaluación: métricas, umbral óptimo, análisis de errores y sesgos."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metricas_completas(y_true, y_pred, y_prob) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }


def tabla_umbral(y_true, y_prob, thresholds=None) -> pd.DataFrame:
    """Precision/Recall/F1 y conteo de FP/FN por umbral."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.20, 0.81, 0.05), 2)
    filas = []
    y_true = np.asarray(y_true)
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        filas.append(
            {
                "Umbral": float(t),
                "Precision": precision_score(y_true, y_pred, zero_division=0),
                "Recall": recall_score(y_true, y_pred, zero_division=0),
                "F1-score": f1_score(y_true, y_pred, zero_division=0),
                "Falsos positivos": int(fp),
                "Falsos negativos": int(fn),
            }
        )
    return pd.DataFrame(filas).set_index("Umbral")


def umbral_por_youden(y_true, y_prob) -> float:
    """Umbral que maximiza el índice de Youden (J = recall + especificidad - 1)."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # Recalcular con barrido fino usando la curva PR no da especificidad; usamos barrido.
    ts = np.linspace(0.05, 0.95, 181)
    y_true = np.asarray(y_true)
    best_t, best_j = 0.5, -1.0
    for t in ts:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        j = tpr + tnr - 1
        if j > best_j:
            best_j, best_t = j, float(round(t, 3))
    return best_t


def umbral_por_f1(y_true, y_prob) -> float:
    ts = np.linspace(0.05, 0.95, 181)
    y_true = np.asarray(y_true)
    best_t, best_f1 = 0.5, -1.0
    for t in ts:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(round(t, 3))
    return best_t


def umbral_min_recall(y_true, y_prob, min_recall: float = 0.90) -> float:
    """Umbral más alto que aún cumple una restricción mínima de recall.

    Prioriza no perder talento (recall alto) sin sacrificar precisión de más.
    """
    ts = np.linspace(0.05, 0.95, 181)
    y_true = np.asarray(y_true)
    candidatos = [
        float(round(t, 3))
        for t in ts
        if recall_score(y_true, (y_prob >= t).astype(int), zero_division=0) >= min_recall
    ]
    return max(candidatos) if candidatos else 0.05


def tabla_errores(X_val: pd.DataFrame, y_true, y_pred, y_prob, cols) -> pd.DataFrame:
    """DataFrame etiquetado con tipo de acierto/error para análisis."""
    d = X_val.copy()
    d["y_true"] = np.asarray(y_true)
    d["y_pred"] = np.asarray(y_pred)
    d["y_prob"] = np.asarray(y_prob)
    tipo = np.select(
        [
            (d.y_true == 1) & (d.y_pred == 1),
            (d.y_true == 0) & (d.y_pred == 0),
            (d.y_true == 0) & (d.y_pred == 1),
            (d.y_true == 1) & (d.y_pred == 0),
        ],
        ["VP", "VN", "FP", "FN"],
        default="?",
    )
    d["tipo"] = tipo
    return d


def analisis_sesgos(df_eval: pd.DataFrame, grupo: str, y_true="y_true", y_pred="y_pred") -> pd.DataFrame:
    """Métricas por subgrupo: tamaño, tasa real, recall, FPR, FNR, precision."""
    filas = []
    for g, sub in df_eval.groupby(grupo):
        yt = sub[y_true].to_numpy()
        yp = sub[y_pred].to_numpy()
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        filas.append(
            {
                "Grupo": g,
                "Tamaño": len(sub),
                "Tasa real (+)": yt.mean(),
                "Tasa pred (+)": yp.mean(),
                "Recall": tp / (tp + fn) if (tp + fn) else np.nan,
                "Precision": tp / (tp + fp) if (tp + fp) else np.nan,
                "FPR": fp / (fp + tn) if (fp + tn) else np.nan,
                "FNR": fn / (fn + tp) if (fn + tp) else np.nan,
            }
        )
    return pd.DataFrame(filas).set_index("Grupo")
