"""Configuración compartida del proyecto (fase final).

Centraliza semilla, rutas y listas de variables para que TODOS los módulos y el
notebook usen exactamente los mismos valores. Esto garantiza reproducibilidad y
comparabilidad con el avance de proyecto.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Semilla global (idéntica al avance de proyecto)                             #
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42

# --------------------------------------------------------------------------- #
# Variable objetivo y fuga de información                                      #
# --------------------------------------------------------------------------- #
TARGET: str = "is_lck"           # 1 = LCK, 0 = LCK Challengers League
LEAK_COLS: list[str] = ["league", "gameid", "playerid", "teamid", "team"]

# --------------------------------------------------------------------------- #
# Variables de ENTRADA al pipeline (stats crudas por partida)                  #
# La ingeniería de características (kda + nuevas) se hace DENTRO del pipeline,  #
# de modo que la inferencia solo requiere estas columnas base.                 #
# --------------------------------------------------------------------------- #
BASE_NUM: list[str] = [
    "kills", "deaths", "assists",
    "cspm", "earnedgpm", "dpm", "damageshare", "vspm",
    "goldat15", "xpat15", "csat15",
    "golddiffat15", "xpdiffat15",
    "killsat15", "assistsat15", "deathsat15",
    "year",
]
BASE_CAT: list[str] = ["position"]
BASE_FEATURES: list[str] = BASE_NUM + BASE_CAT

# Rangos plausibles por variable (para validación de entrada, no para recorte).
NUM_RANGES: dict[str, tuple[float, float]] = {
    "kills": (0, 40), "deaths": (0, 30), "assists": (0, 60),
    "cspm": (0, 14), "earnedgpm": (100, 700), "dpm": (0, 1500),
    "damageshare": (0.0, 0.7), "vspm": (0.0, 6.0),
    "goldat15": (2000, 12000), "xpat15": (2000, 12000), "csat15": (0, 220),
    "golddiffat15": (-6000, 6000), "xpdiffat15": (-6000, 6000),
    "killsat15": (0, 15), "assistsat15": (0, 20), "deathsat15": (0, 12),
    "year": (2015, 2030),
}
VALID_POSITIONS: list[str] = ["top", "jng", "mid", "bot", "sup"]

# --------------------------------------------------------------------------- #
# Características derivadas (creadas por el FeatureEngineer)                    #
# `kda` ya existía en el avance; el resto son NUEVAS de la fase final.         #
# --------------------------------------------------------------------------- #
ENGINEERED_NUM: list[str] = [
    "kda",              # (avance) (kills + assists) / max(deaths, 1)
    "oro_por_cs15",     # NUEVA
    "dominancia15",     # NUEVA
    "participacion15",  # NUEVA
    "impacto_dano",     # NUEVA
]

# Listas que ve el ColumnTransformer DESPUÉS de la ingeniería de características.
NUM_FEATURES: list[str] = BASE_NUM + ENGINEERED_NUM
CAT_FEATURES: list[str] = BASE_CAT

# Orden EXACTO de columnas numéricas del avance (con `kda` en su posición original).
# Se usa solo en la reproducción para garantizar resultados bit a bit idénticos:
# el orden de columnas afecta el submuestreo de variables de los modelos de árbol.
AVANCE_NUM: list[str] = [
    "kills", "deaths", "assists", "cspm", "earnedgpm", "kda", "dpm",
    "damageshare", "vspm", "goldat15", "xpat15", "csat15",
    "golddiffat15", "xpdiffat15", "killsat15", "assistsat15", "deathsat15", "year",
]

# --------------------------------------------------------------------------- #
# Rutas del proyecto                                                           #
# --------------------------------------------------------------------------- #
ROOT: Path = Path(__file__).resolve().parents[1]
DATA_RAW: Path = ROOT / "data" / "raw"
DATA_PROC: Path = ROOT / "data" / "processed"
MODELS_DIR: Path = ROOT / "models"
FIG_DIR: Path = ROOT / "reports" / "figuras"
TAB_DIR: Path = ROOT / "reports" / "tablas"
EXP_DIR: Path = ROOT / "experiments"

for _d in (DATA_RAW, DATA_PROC, MODELS_DIR, FIG_DIR, TAB_DIR, EXP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Métrica principal y umbral                                                   #
# --------------------------------------------------------------------------- #
PRIMARY_METRIC: str = "f1"          # coherente con el avance (F1 + AUC-ROC)
DEFAULT_THRESHOLD: float = 0.50
MODEL_VERSION: str = "1.0.0-fase-final"
