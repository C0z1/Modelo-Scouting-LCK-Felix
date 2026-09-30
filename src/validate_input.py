"""Validación de datos de entrada para inferencia.

Detecta y reporta problemas SIN detener la ejecución de forma inesperada:
campos faltantes, tipos incorrectos, valores fuera de rango, categorías no
reconocidas, columnas adicionales y estructura incompleta. Devuelve un informe
estructurado; el llamador decide si continúa.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import BASE_FEATURES, BASE_NUM, NUM_RANGES, VALID_POSITIONS


@dataclass
class InformeValidacion:
    ok: bool = True
    errores: list[str] = field(default_factory=list)     # impiden inferir con seguridad
    advertencias: list[str] = field(default_factory=list)  # no bloquean, pero se reportan

    def add_error(self, msg: str) -> None:
        self.errores.append(msg)
        self.ok = False

    def add_warn(self, msg: str) -> None:
        self.advertencias.append(msg)

    def resumen(self) -> str:
        partes = ["VÁLIDO" if self.ok else "CON ERRORES"]
        if self.errores:
            partes.append("Errores:\n  - " + "\n  - ".join(self.errores))
        if self.advertencias:
            partes.append("Advertencias:\n  - " + "\n  - ".join(self.advertencias))
        return "\n".join(partes)


def validar_entrada(df: pd.DataFrame) -> InformeValidacion:
    """Valida un DataFrame de una o varias filas contra el esquema esperado."""
    info = InformeValidacion()

    if df is None or len(df) == 0:
        info.add_error("La entrada está vacía (0 registros).")
        return info

    # 1) Columnas faltantes / adicionales
    faltantes = [c for c in BASE_FEATURES if c not in df.columns]
    for c in faltantes:
        info.add_error(f"Falta la columna requerida: '{c}'.")
    extra = [c for c in df.columns if c not in BASE_FEATURES]
    if extra:
        info.add_warn(f"Columnas adicionales que serán ignoradas: {extra}.")

    # 2) Tipos y rangos numéricos
    for c in BASE_NUM:
        if c not in df.columns:
            continue
        serie = pd.to_numeric(df[c], errors="coerce")
        n_no_num = int(serie.isna().sum() - df[c].isna().sum())
        if n_no_num > 0:
            info.add_error(f"'{c}': {n_no_num} valor(es) no numérico(s).")
        lo, hi = NUM_RANGES.get(c, (-np.inf, np.inf))
        fuera = int(((serie < lo) | (serie > hi)).sum())
        if fuera > 0:
            info.add_warn(f"'{c}': {fuera} valor(es) fuera del rango plausible [{lo}, {hi}].")

    # 3) Categoría de posición
    if "position" in df.columns:
        desconocidas = sorted(set(df["position"].dropna().astype(str)) - set(VALID_POSITIONS))
        if desconocidas:
            info.add_warn(
                f"Posición(es) no reconocida(s): {desconocidas}. "
                f"El OneHotEncoder las tratará como desconocidas (handle_unknown='ignore')."
            )

    return info
