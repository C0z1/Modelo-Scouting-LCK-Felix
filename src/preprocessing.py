"""Preprocesamiento consolidado (Pipeline + ColumnTransformer).

Provee dos constructores:

* `build_preprocessor_base()`  -> reproduce EXACTAMENTE el preprocesamiento del
  avance (sin características nuevas), para las secciones de reproducción.
* `build_preprocessor_final()` -> preprocesamiento de la fase final: incluye la
  ingeniería de características DENTRO del pipeline.

Garantías (CLAUDE.md §3):
  (a) el ajuste (imputación, escalamiento, OHE) se hace solo con train,
  (b) el test permanece separado,
  (c) las mismas reglas se aplican en inferencia,
  (d) es serializable junto con el modelo.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import AVANCE_NUM, BASE_CAT, BASE_NUM, CAT_FEATURES, NUM_FEATURES
from .features import FeatureEngineer


def _num_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _cat_pipeline(min_frequency: float | None = None) -> Pipeline:
    ohe_kwargs = dict(handle_unknown="ignore", sparse_output=False)
    if min_frequency is not None:
        # Agrupa categorías poco frecuentes (robustez ante categorías raras).
        ohe_kwargs["min_frequency"] = min_frequency
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(**ohe_kwargs)),
        ]
    )


def build_preprocessor_base() -> ColumnTransformer:
    """Preprocesamiento del avance: variables base (con `kda` precalculada fuera).

    Usa el orden EXACTO de columnas del avance (AVANCE_NUM) para reproducir los
    resultados bit a bit.
    """
    return ColumnTransformer(
        transformers=[
            ("num", _num_pipeline(), AVANCE_NUM),
            ("cat", _cat_pipeline(), BASE_CAT),
        ]
    )


def build_column_transformer_final(min_frequency: float | None = None) -> ColumnTransformer:
    """Solo el ColumnTransformer de la fase final (sin el FeatureEngineer).

    Útil para pipelines de imbalanced-learn, que no admiten un Pipeline anidado
    como paso intermedio: allí se usan FeatureEngineer y este CT como pasos planos.
    """
    return ColumnTransformer(
        transformers=[
            ("num", _num_pipeline(), NUM_FEATURES),
            ("cat", _cat_pipeline(min_frequency=min_frequency), CAT_FEATURES),
        ]
    )


def build_preprocessor_final(min_frequency: float | None = None) -> Pipeline:
    """Preprocesamiento de la fase final: FeatureEngineer + ColumnTransformer.

    El resultado es un `Pipeline` (no un `ColumnTransformer`) porque la ingeniería
    de características es el primer paso. Se puede anteponer a cualquier clasificador.
    """
    ct = build_column_transformer_final(min_frequency=min_frequency)
    return Pipeline(steps=[("fe", FeatureEngineer()), ("ct", ct)])


def get_output_feature_names(fitted_preprocessor) -> list[str]:
    """Nombres de columnas de salida tras el preprocesamiento (para importancias/SHAP)."""
    # fitted_preprocessor es el Pipeline [fe, ct] ya ajustado
    ct = fitted_preprocessor.named_steps["ct"] if hasattr(fitted_preprocessor, "named_steps") else fitted_preprocessor
    names: list[str] = []
    for name, trans, cols in ct.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if name == "num":
            names.extend(cols)
        elif name == "cat":
            ohe = trans.named_steps["ohe"]
            names.extend(ohe.get_feature_names_out(cols).tolist())
    return names
