# Ficha del Modelo — Modelo de Scouting LCK (Fase Final)

> Resumen técnico y operativo. Complementa el notebook `notebooks/01_fase_final.ipynb`
> y el reporte ejecutivo `reports/reporte_ejecutivo.md`.

## Identificación
- **Nombre:** Modelo de Scouting LCK
- **Versión:** 1.0.0-fase-final
- **Tipo:** Clasificación binaria supervisada
- **Modelo final:** Regresión Logística (`class_weight="balanced"`)
- **Artefacto:** `models/pipeline_final.joblib` (+ `models/metadata.json` con hash SHA-256)
- **Fecha de entrenamiento:** 2026-09-29
- **Semilla:** 42 · **scikit-learn:** 1.8.0 · **Python:** 3.11

## Propósito
Estimar, a partir de estadísticas de rendimiento **individual por partida**, la probabilidad de que un
jugador tenga un perfil de **LCK** (liga principal) frente a **LCK Challengers League** (academia), para
**apoyar** al cuerpo de scouting en la priorización de jugadores a revisar de cara a una promoción.

## Usuarios previstos
Analistas y coaches de scouting de organizaciones de LoL. **No** es una herramienta para directivos que
busquen automatizar decisiones de contratación.

## Usos permitidos
- Priorizar qué jugadores de academia revisar con más detalle.
- Generar una señal cuantitativa adicional al juicio experto.

## Usos NO recomendados
- Decidir ascensos, despidos o salarios de forma automática.
- Sustituir la evaluación humana o el ojo del coach.
- Comparar jugadores entre regiones/ligas no incluidas en el entrenamiento.

## Datos
- **Fuente:** dataset **simulado reproducible** (`src/data_simulada.py`, semilla 42), inspirado en la
  estructura de Oracle's Elixir para LCK/LCK CL 2022–2024. *(El avance de proyecto ya usaba datos
  simulados; se conservan para mantener la comparabilidad.)*
- **Tamaño:** 8000 registros (jugador × partida). **Balance:** 35.6 % LCK / 64.4 % LCK CL (1.81:1).
- **Variable objetivo:** `is_lck` (1 = LCK, 0 = LCK CL).
- **Variables:** 17 numéricas base + `position` + 4 derivadas (`oro_por_cs15`, `dominancia15`,
  `participacion15`, `impacto_dano`) + `kda`.
- **Excluidas por fuga:** `league` (origen de la etiqueta), `gameid`, `playerid`, `teamid/team`.

## Métricas
- **Validación (CV RepeatedStratifiedKFold 5×3):** F1 ≈ 0.692, AUC-ROC ≈ 0.844.
- **Prueba (una sola vez):** Accuracy 0.753, Precision 0.631, **Recall 0.738**, F1 0.680, **AUC-ROC 0.832**.
- Supera con holgura al baseline (`DummyClassifier`, AUC 0.5).

## Umbral
- **Operativo: 0.40** (criterio: máximo F1 sujeto a recall ≥ 0.80, elegido en **validación**).
- Prioriza el **recall** (no perder talento) sobre la precisión. **No debe adoptarse sin revisión
  organizacional:** el punto óptimo depende del costo real de cada error y de la capacidad de scouting.

## Interpretabilidad
- Coeficientes (features estandarizadas) + permutation importance + SHAP (`LinearExplainer`).
- **Factores dominantes:** `earnedgpm`, `cspm`, `golddiffat15`, `xpdiffat15`.
- **Limitación:** asociación, **no causalidad**. Un coeficiente alto no implica relación causal.

## Limitaciones
- **Datos simulados:** con datos reales los resultados pueden variar.
- **Una partida por jugador:** no considera sinergia de equipo ni consistencia entre partidas.
- **Meta cambiante:** LoL varía por parche; un modelo de 2022 puede no servir en 2024.
- **Sesgo de selección:** los jugadores de LCK CL que ascienden ya son los mejores de su liga.

## Análisis de sesgos
- **Por posición:** recall entre 0.64 (mid) y 0.82 (jng/sup).
- **Por nivel de gasto (`earnedgpm`):** recall de 0.33 (bajo) a 0.91 (alto).
- Las diferencias reflejan que `earnedgpm` es el predictor dominante; **no** se concluye discriminación.
  Mitigación: métricas por subgrupo, posible umbral por rol y más contexto antes de cualquier uso real.

## Riesgos principales
Falsos negativos (perder talento), falsos positivos (evaluar de más), drift por cambios de meta, datos
incompletos, uso automático indebido y sesgo por grupo. Ver la tabla completa en el notebook (§21).

## Mecanismo de inferencia
- `src/predict.py` (CSV/JSON) → probabilidad, clasificación, umbral, versión/fecha y el mensaje
  *"Resultado estimado; requiere revisión humana."*
- `src/validate_input.py` valida la entrada (faltantes, tipos, rangos, categorías, columnas extra) sin
  detener la ejecución de forma inesperada.

## Plan de monitoreo (resumen)
Faltantes (> 15 %), recall (caída > 5 pts), data drift (PSI > 0.2), tiempo de respuesta (> 500 ms) y
categorías nuevas. Frecuencia mensual/por lote. Ver §27 del notebook.

## Criterios de actualización (resumen)
Recalibrar/umbral/reentrenar/suspender ante: caída sostenida de recall, aumento de FN, drift, categorías
nuevas, cambio en la definición de "listo" o brechas excesivas entre grupos. Ver §28 del notebook.

## Responsable de revisión
Felix Yael — con supervisión del responsable de datos y el coach de academia.

## Advertencia de seguridad
Cargar archivos `joblib`/`pickle` de fuentes desconocidas puede ejecutar código arbitrario. Cargar solo
artefactos de origen confiable y verificar el hash SHA-256 (`models/metadata.json`).
