"""Ingeniería de características (fase final).

`FeatureEngineer` es un transformador de scikit-learn SIN estado (row-wise): todas
las variables derivadas se calculan a partir de las stats de la MISMA partida, por
lo que:

* No requiere estadísticas de ajuste  ->  imposible que filtre información de test.
* Se aplica idéntico en train, validación, test e inferencia.
* Es serializable junto con el modelo (se guarda dentro del Pipeline).

Características derivadas
------------------------
| Nueva característica | Variables de origen            | Regla                              | Interpretación                              | Riesgo de fuga |
|----------------------|--------------------------------|------------------------------------|---------------------------------------------|----------------|
| kda (ya en avance)   | kills, assists, deaths         | (kills+assists)/max(deaths,1)      | Eficiencia de combate                       | Nulo           |
| oro_por_cs15  NUEVA  | goldat15, csat15               | goldat15/(csat15+1)                | Oro extraído por unidad de farmeo temprano  | Nulo           |
| dominancia15  NUEVA  | golddiffat15, xpdiffat15       | golddiffat15 + xpdiffat15          | Ventaja total de recursos a los 15 min      | Nulo           |
| participacion15 NUEVA| killsat15, assistsat15         | killsat15 + assistsat15            | Involucramiento en la acción temprana       | Nulo           |
| impacto_dano  NUEVA  | damageshare, dpm               | damageshare * dpm / 100            | Daño absoluto ponderado por cuota de equipo | Nulo           |

Ninguna usa `league` ni `is_lck`, ni información posterior a la partida.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Añade características derivadas a un DataFrame de stats por partida."""

    #: columnas que el transformador espera recibir
    required = [
        "kills", "deaths", "assists",
        "goldat15", "csat15", "golddiffat15", "xpdiffat15",
        "killsat15", "assistsat15", "damageshare", "dpm",
    ]

    def fit(self, X, y=None):  # noqa: D401 - sin estado
        return self

    def transform(self, X):
        X = X.copy()
        deaths = X["deaths"].replace(0, 1)
        X["kda"] = (X["kills"] + X["assists"]) / deaths
        X["oro_por_cs15"] = X["goldat15"] / (X["csat15"].abs() + 1)
        X["dominancia15"] = X["golddiffat15"] + X["xpdiffat15"]
        X["participacion15"] = X["killsat15"] + X["assistsat15"]
        X["impacto_dano"] = X["damageshare"] * X["dpm"] / 100.0
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        nuevas = ["kda", "oro_por_cs15", "dominancia15", "participacion15", "impacto_dano"]
        return np.array(base + [c for c in nuevas if c not in base])
