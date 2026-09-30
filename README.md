# Modelo de Scouting LCK — Fase Final

Clasificación binaria para predecir si un jugador tiene perfil de **LCK** (liga principal) o de **LCK
Challengers League** (academia) a partir de estadísticas de rendimiento **individual por partida**, como
**apoyo** —nunca sustituto— a las decisiones de scouting.

> Esta es la **fase final** del proyecto: continúa el *avance* (notebook original `LCK_Scouting_Model.ipynb`)
> y lo lleva a una solución **trazable, reproducible, interpretable y defendible para una prueba controlada**,
> siguiendo el flujo de `CLAUDE.md`.

## ⚠️ Uso responsable
Ninguna decisión sobre un jugador debe automatizarse solo con la predicción. **Todo resultado requiere
revisión humana.** La influencia de una variable **no** implica causalidad.

## Nota sobre los datos
El dataset del avance es **simulado dentro del propio notebook** (semilla 42). Esta fase lo **reproduce
de forma modular y verificable** (`src/data_simulada.py`) para mantener el mismo problema, dataset y
variable objetivo, y no romper la comparabilidad. Con datos reales de Oracle's Elixir los resultados
pueden variar.

## Estructura

```
.
├── CLAUDE.md                     # rúbrica de la fase final
├── README.md
├── requirements.txt              # versiones exactas
├── data/
│   ├── raw/dataset_simulado.csv  # dataset reproducible
│   └── processed/                # splits train/val/test
├── notebooks/
│   └── 01_fase_final.ipynb       # notebook principal (orquesta y narra las 31 secciones)
├── src/
│   ├── config.py                 # semilla, rutas, listas de variables
│   ├── data_simulada.py          # generación reproducible del dataset
│   ├── features.py               # FeatureEngineer (4 características nuevas)
│   ├── preprocessing.py          # Pipeline + ColumnTransformer
│   ├── split.py                  # partición 70/15/15 y estrategias de CV
│   ├── train.py                  # catálogos de modelos, CV, pipelines
│   ├── evaluate.py               # métricas, umbral, errores, sesgos
│   ├── predict.py                # inferencia independiente + verificación de carga
│   ├── validate_input.py         # validación de datos de entrada
│   ├── reporting.py              # guardado de tablas/figuras
│   └── run_pipeline.py           # driver: ejecuta TODO y genera artefactos
├── models/                       # pipeline_final.joblib + metadata.json (hash)
├── reports/
│   ├── tablas/                   # todas las tablas (CSV + Markdown)
│   ├── figuras/                  # ROC, PR, matrices, importancias, SHAP
│   ├── ficha_modelo.md
│   └── reporte_ejecutivo.md
├── experiments/registro_experimentos.csv
└── examples/                     # entradas de ejemplo (CSV/JSON)
```

## Cómo ejecutar

```bash
pip install -r requirements.txt

# 1) Regenerar todo (tablas, figuras, modelo serializado). ~5 min.
python -m src.run_pipeline

# 2) Notebook principal (orquesta y narra)
jupyter notebook notebooks/01_fase_final.ipynb

# 3) Inferencia sobre nuevos registros
python -m src.predict --input examples/ejemplo_entrada.csv
python -m src.predict --input examples/ejemplo_entrada.json --umbral 0.45
```

## Resultados (resumen)

| | Modelo final | Validación (CV) | Prueba |
|---|---|---|---|
| **Regresión Logística** | `class_weight=balanced` | F1 ≈ 0.692 · AUC ≈ 0.844 | F1 0.680 · AUC 0.832 · Recall 0.738 |

- **Umbral operativo:** 0.40 (máx. F1 con recall ≥ 0.80, elegido en validación).
- **Factores dominantes:** `earnedgpm`, `cspm`, `golddiffat15`, `xpdiffat15`.
- **Familias comparadas:** Dummy, Regresión Logística, Random Forest, Gradient Boosting,
  HistGradientBoosting, SVM (lineal/RBF) y MLP. Los modelos lineales encabezan en estos datos.

## Reproducibilidad y trazabilidad
- `random_state=42` en splits, CV, modelos y búsquedas.
- Reproducción del avance verificada **bit a bit** (`reports/tablas/06_reproduccion_baselines`).
- Artefacto serializado con **hash SHA-256** y verificación de carga automática.
- Versiones exactas de bibliotecas en `requirements.txt`.

## Documentación
- **Técnica/operativa:** `reports/ficha_modelo.md`
- **No técnica:** `reports/reporte_ejecutivo.md`
- **Rúbrica y reglas:** `CLAUDE.md`
