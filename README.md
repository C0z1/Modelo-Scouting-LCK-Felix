# Modelo de Scouting LCK

Clasificación binaria para predecir si un jugador pertenece a la LCK o a la LCK Challengers League, usando estadísticas de rendimiento individual por partida.

## Contexto

En Corea del Sur, los equipos de League of Legends tienen que decidir cada temporada qué jugadores de academia están listos para competir en la liga principal (LCK). Este proyecto busca apoyar esa decisión con un modelo de machine learning entrenado con datos históricos de Oracle's Elixir (2022–2024).

## Dataset

- **Fuente:** [Oracle's Elixir](https://oracleselixir.com)
- **Cobertura:** Temporadas 2022, 2023 y 2024
- **Registros:** ~8,000 actuaciones individuales (LCK + LCK CL)
- **Variables clave:** `cspm`, `earnedgpm`, `golddiffat15`, `xpdiffat15`, `dpm`, `damageshare`, `position`

## Estructura del notebook

| Sección | Contenido |
|---------|-----------|
| 1–3 | Definición del problema, pregunta de análisis y descripción del dataset |
| 4 | Diccionario de variables |
| 5–7 | EDA, calidad de datos y desbalance de clases |
| 8–10 | Fuga de información, partición y pipeline de preprocesamiento |
| 11–13 | Baseline, comparación de modelos y métricas |
| 14–15 | Análisis de FP/FN y umbrales |
| 16–19 | Modelo final, plan de mejora y conclusiones |

## Modelos evaluados

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

La selección final se basa en AUC-ROC y F1-Score en el conjunto de validación, priorizando Recall de la clase LCK para minimizar falsos negativos.

## Requisitos

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
scipy
```

## Uso

Abre el notebook en Google Colab o Jupyter y sigue las instrucciones de cada sección. Los datos simulados ya están incluidos para desarrollo; para producción, descarga el CSV real desde Oracle's Elixir y ajusta la celda de carga.

---

Proyecto desarrollado para el curso de Aprendizaje Automático.
