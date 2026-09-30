"""Utilidades de reporte: guardar tablas (CSV + Markdown) y figuras."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import FIG_DIR, TAB_DIR

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"


def guardar_tabla(df: pd.DataFrame, nombre: str, index: bool = True, floatfmt: int = 4) -> Path:
    """Guarda un DataFrame como CSV y como Markdown en reports/tablas/."""
    csv_path = TAB_DIR / f"{nombre}.csv"
    md_path = TAB_DIR / f"{nombre}.md"
    df.to_csv(csv_path, index=index)
    try:
        md = df.round(floatfmt).to_markdown(index=index)
    except Exception:
        md = df.to_markdown(index=index)
    md_path.write_text(md + "\n", encoding="utf-8")
    return csv_path


def guardar_figura(fig, nombre: str) -> Path:
    path = FIG_DIR / f"{nombre}.png"
    fig.savefig(path)
    plt.close(fig)
    return path
