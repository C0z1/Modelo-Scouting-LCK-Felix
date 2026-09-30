"""Generación reproducible del dataset simulado.

IMPORTANTE / TRAZABILIDAD
-------------------------
El dataset del *avance de proyecto* es SIMULADO dentro del propio notebook
(no se descarga Oracle's Elixir real). Para respetar la regla nº1 del CLAUDE.md
("mantener el mismo problema, dataset y variable objetivo") y no romper la
comparabilidad (regla nº3, misma semilla), este módulo reproduce EXACTAMENTE la
misma secuencia de generación de la celda de datos del avance (semilla 42).

Cualquier persona puede reconstruir el dataset ejecutando `generar_dataset()`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_STATE, TARGET


def generar_dataset(n: int = 8000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Reproduce el dataset simulado del avance (misma semilla y secuencia).

    Devuelve el DataFrame *crudo* con la columna `league` (fuente de la etiqueta).
    La eliminación de `league` y la creación de `is_lck` se hacen en `preparar()`.
    """
    np.random.seed(random_state)
    league_dist = np.random.choice(["LCK", "LCK CL"], size=n, p=[0.35, 0.65])
    is_lck = (league_dist == "LCK").astype(int)

    roles = np.random.choice(["top", "jng", "mid", "bot", "sup"], size=n)
    base_gpm = np.where(
        is_lck == 1,
        np.random.normal(400, 40, n),
        np.random.normal(355, 45, n),
    )
    base_csm = np.where(
        is_lck == 1,
        np.random.normal(8.2, 0.9, n),
        np.random.normal(7.4, 1.0, n),
    )
    raw = pd.DataFrame(
        {
            "league": league_dist,
            "position": roles,
            "kills": np.random.poisson(3, n),
            "deaths": np.random.poisson(2.5, n),
            "assists": np.random.poisson(6, n),
            "cspm": np.clip(base_csm + np.random.normal(0, 0.3, n), 2, 12),
            "goldat15": np.random.normal(5800, 600, n).astype(int),
            "xpat15": np.random.normal(6000, 500, n).astype(int),
            "csat15": np.random.normal(100, 15, n).astype(int),
            "golddiffat15": np.where(
                is_lck == 1,
                np.random.normal(180, 400, n),
                np.random.normal(-80, 420, n),
            ),
            "xpdiffat15": np.where(
                is_lck == 1,
                np.random.normal(120, 350, n),
                np.random.normal(-50, 360, n),
            ),
            "earnedgpm": np.clip(base_gpm + np.random.normal(0, 20, n), 200, 600),
            "damageshare": np.clip(np.random.normal(0.22, 0.08, n), 0.02, 0.55),
            "vspm": np.clip(np.random.normal(1.4, 0.4, n), 0.3, 4),
            "killsat15": np.random.poisson(1.5, n),
            "assistsat15": np.random.poisson(2.5, n),
            "deathsat15": np.random.poisson(1.2, n),
            "dpm": np.clip(np.random.normal(530, 120, n), 100, 1000),
            "year": np.random.choice([2022, 2023, 2024], size=n),
        }
    )
    # Faltantes artificiales (~8%) en variables de los 15 minutos (igual al avance)
    for col in ["golddiffat15", "xpdiffat15", "csat15", "goldat15", "xpat15"]:
        idx = np.random.choice(n, size=int(n * 0.08), replace=False)
        raw.loc[idx, col] = np.nan
    return raw


def preparar(raw: pd.DataFrame) -> pd.DataFrame:
    """Crea la etiqueta `is_lck` a partir de `league`. No elimina `league` aún.

    La eliminación de columnas con fuga se realiza al separar X/y (split.py),
    ANTES de particionar, tal como en el avance.
    """
    df = raw[raw["league"].isin(["LCK", "LCK CL"])].copy()
    df[TARGET] = (df["league"] == "LCK").astype(int)
    return df


if __name__ == "__main__":
    d = preparar(generar_dataset())
    print(f"Dataset: {d.shape[0]:,} filas × {d.shape[1]} columnas")
    print(d[TARGET].value_counts(normalize=True).round(3).to_string())
