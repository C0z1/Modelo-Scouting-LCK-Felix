"""Script INDEPENDIENTE de inferencia y verificación de carga.

Uso como script:
    python -m src.predict --input registros.csv
    python -m src.predict --input registros.json --umbral 0.45

Uso como módulo:
    from src.predict import cargar_modelo, predecir
    art = cargar_modelo()
    salida = predecir(df, art)

⚠️ ADVERTENCIA DE SEGURIDAD
Cargar archivos serializados (joblib/pickle) de origen desconocido puede ejecutar
código arbitrario. Cargue únicamente artefactos de ORIGEN CONFIABLE y verifique su
integridad con el hash SHA-256 registrado en models/metadata.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

# Permite ejecutar el archivo tanto como módulo (-m src.predict) como directo.
try:
    from .config import BASE_FEATURES, MODELS_DIR, MODEL_VERSION
    from .validate_input import validar_entrada
except ImportError:  # ejecución directa
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import BASE_FEATURES, MODELS_DIR, MODEL_VERSION
    from src.validate_input import validar_entrada

MENSAJE_REVISION = "Resultado estimado; requiere revisión humana."


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def cargar_modelo(models_dir: Path = MODELS_DIR, verificar_hash: bool = True) -> dict:
    """Carga el pipeline serializado y sus metadatos.

    Devuelve un dict con: pipeline, umbral, version, fecha, metadata.
    """
    pipe_path = models_dir / "pipeline_final.joblib"
    meta_path = models_dir / "metadata.json"
    if not pipe_path.exists():
        raise FileNotFoundError(f"No se encontró el artefacto: {pipe_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    if verificar_hash and meta.get("sha256_pipeline"):
        real = _sha256(pipe_path)
        if real != meta["sha256_pipeline"]:
            raise ValueError(
                "El hash del artefacto NO coincide con el registrado. "
                "No se carga por seguridad e integridad."
            )

    pipeline = joblib.load(pipe_path)
    return {
        "pipeline": pipeline,
        "umbral": float(meta.get("umbral", 0.5)),
        "version": meta.get("version", MODEL_VERSION),
        "fecha": meta.get("fecha_entrenamiento", "desconocida"),
        "metadata": meta,
    }


def predecir(df: pd.DataFrame, artefacto: dict, umbral: float | None = None) -> pd.DataFrame:
    """Aplica el pipeline y el umbral. Devuelve una tabla lista para el usuario final."""
    umbral = float(artefacto["umbral"] if umbral is None else umbral)

    info = validar_entrada(df)
    # Nunca detiene de forma inesperada: reporta y sólo bloquea si faltan columnas base.
    faltan_base = [c for c in BASE_FEATURES if c not in df.columns]
    if faltan_base:
        raise ValueError(
            "No se puede inferir: faltan columnas requeridas -> "
            f"{faltan_base}. Informe de validación:\n{info.resumen()}"
        )

    X = df[BASE_FEATURES].copy()
    proba = artefacto["pipeline"].predict_proba(X)[:, 1]
    clas = (proba >= umbral).astype(int)

    salida = pd.DataFrame(
        {
            "probabilidad_LCK": proba.round(4),
            "clasificacion": ["LCK" if c == 1 else "LCK CL" for c in clas],
            "umbral_utilizado": umbral,
            "version_modelo": artefacto["version"],
            "fecha_modelo": artefacto["fecha"],
            "mensaje": MENSAJE_REVISION,
        }
    )
    salida.attrs["validacion"] = info
    return salida


def _leer_entrada(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = [data]
        return pd.DataFrame(data)
    return pd.read_csv(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Inferencia de scouting LCK (requiere revisión humana).")
    ap.add_argument("--input", required=True, help="Archivo CSV o JSON con registros.")
    ap.add_argument("--umbral", type=float, default=None, help="Umbral opcional (0-1).")
    ap.add_argument("--models-dir", default=str(MODELS_DIR))
    args = ap.parse_args(argv)

    art = cargar_modelo(Path(args.models_dir))
    df = _leer_entrada(Path(args.input))
    info = validar_entrada(df)
    if info.advertencias or info.errores:
        print("=== Validación de entrada ===")
        print(info.resumen())
        print()

    salida = predecir(df, art, umbral=args.umbral)
    print("=== Resultado (requiere revisión humana) ===")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(salida.to_string(index=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
