# CLAUDE.md — Proyecto de predicción de abandono de clientes (Fase final)

## Contexto

Este proyecto es la **fase final** de un modelo de clasificación para predecir el abandono de clientes (churn). Continúa el trabajo del *avance de proyecto*. El objetivo es pasar de una línea base inicial a una solución más completa, **trazable, reproducible, interpretable y defendible para una prueba controlada**.

Idioma de trabajo: **español** (documentación, tablas, comentarios, reportes y mensajes al usuario).

## Reglas globales (no negociables)

1. **Mantener el mismo problema, dataset, variable objetivo y criterio de éxito** definidos en el avance de proyecto.
2. **No sustituir el avance anterior sin explicar qué cambió y por qué.** Todo cambio se registra en la bitácora de cambios.
3. **No romper la comparabilidad** con el avance: misma semilla, misma estrategia de validación y mismas métricas para los modelos de referencia.
4. **Todo ajuste (imputación, escalamiento, codificación, selección de variables, hiperparámetros) se hace solo con datos de entrenamiento**, dentro de un `Pipeline` de scikit-learn y/o dentro de la validación cruzada.
5. **El conjunto de prueba se usa una sola vez**, únicamente después de terminar la selección y el ajuste de modelos.
6. **Sin fuga de información:** no usar información posterior al abandono ni atributos que revelen directa o indirectamente la variable objetivo.
7. **Toda modificación se justifica con evidencia técnica** (métricas, gráficas, pruebas), no con intuición.
8. **No elegir el modelo final solo por la métrica más alta:** considerar estabilidad, interpretabilidad, costo, operación y riesgo.
9. **Ninguna decisión sobre un cliente debe automatizarse solo con la predicción:** siempre hay revisión humana.
10. **La influencia de una variable no demuestra causalidad**, y una diferencia estadística entre grupos no demuestra discriminación por sí sola.
11. Todo artefacto debe poder ser reconstruido por otra persona (qué se probó, con qué configuración y con qué resultado).

## Estructura sugerida del repositorio

```
.
├── CLAUDE.md
├── data/
│   ├── raw/                  # dataset original (no modificar)
│   └── processed/            # splits train / val / test
├── notebooks/
│   └── 01_fase_final.ipynb   # notebook principal
├── src/
│   ├── preprocessing.py      # pipeline + ingeniería de características
│   ├── train.py              # entrenamiento, CV, búsqueda de hiperparámetros
│   ├── evaluate.py           # métricas, umbral, errores, sesgos
│   ├── predict.py            # script independiente de inferencia
│   └── validate_input.py     # validación de datos de entrada
├── models/                   # artefactos serializados (joblib) + metadatos
├── reports/
│   ├── tablas/               # todas las tablas en CSV/Markdown
│   ├── figuras/              # ROC, PR, matrices de confusión, importancias
│   ├── ficha_modelo.md
│   └── reporte_ejecutivo.md
├── experiments/
│   └── registro_experimentos.csv
└── requirements.txt          # versiones exactas de bibliotecas
```

## Flujo de trabajo (en orden)

### 0. Recuperar y verificar la fase anterior

Reunir y confirmar que existen: dataset original, diccionario de variables, análisis exploratorio, diagnóstico de calidad, variable objetivo, separación predictores/objetivo, conjuntos train/validación/prueba, semilla, pipeline de preprocesamiento, modelo baseline, modelos iniciales, métricas, matrices de confusión, curvas ROC, modelo preliminar seleccionado, plan de mejora y tabla de trazabilidad.

**Tabla de verificación inicial** (una fila por elemento: Dataset, Pipeline, Baseline, Modelo preliminar, Métricas, Informe del avance):

| Elemento | Archivo o ubicación | Versión | Estado | Acción requerida |
|---|---|---|---|---|

Antes de continuar, **corregir cualquier problema que impida reproducir los resultados** de la fase anterior.

### 1. Bitácora de cambios

Registrar cada modificación desde el avance de proyecto (filas: Datos, Preprocesamiento, Modelos, Métricas, Umbral, Interpretabilidad):

| Elemento modificado | Situación en avance | Cambio realizado | Justificación | Impacto esperado |
|---|---|---|---|---|

### 2. Revisión de calidad de datos

Verificar que el dataset conserve las condiciones del avance: valores faltantes, duplicados, tipos de datos, categorías nuevas o inconsistentes, valores fuera de rango, variables excluidas, balance de clases, posible fuga de información y correspondencia entre columnas y pipeline.

Si hay cambios adicionales en la preparación de datos, documentarlos:

| Variable afectada | Tratamiento en avance | Tratamiento nuevo | Razón del cambio | Efecto en registros o variables |
|---|---|---|---|---|

Ningún cambio debe impedir comparar con el avance.

### 3. Pipeline de preprocesamiento consolidado

Usar `Pipeline` + `ColumnTransformer`. Integrar según corresponda:

- **Numéricas:** imputación, escalamiento, transformación de distribuciones, tratamiento justificado de atípicos, nuevas variables.
- **Categóricas:** imputación, codificación, manejo de categorías desconocidas (`handle_unknown`), agrupación de categorías poco frecuentes.
- **Binarias/ordinales:** tratamiento coherente con su significado analítico.

Verificar: (a) ajuste solo con train, (b) test separado, (c) mismas reglas en inferencia, (d) serializable junto con el modelo.

### 4. Ingeniería de características (mínimo 2 nuevas)

Ejemplos: número total de servicios contratados, proporción cargos acumulados / antigüedad, indicador de contrato de corto plazo, combinación método de pago × facturación electrónica, indicador de cliente nuevo, dependencia de servicios adicionales.

| Nueva característica | Variables de origen | Regla de construcción | Interpretación | Riesgo de fuga |
|---|---|---|---|---|

### 5. Estrategia de validación cruzada

Usar `StratifiedKFold` o `RepeatedStratifiedKFold` (u otra justificada).

| Parámetro | Valor |
|---|---|
| Método de validación | |
| Número de particiones | |
| Número de repeticiones | |
| Semilla | |
| Métrica principal | |
| Métricas secundarias | |

Explicar por qué es más estable que una sola partición y cómo evita usar el conjunto de prueba durante la selección.

### 6. Modelos de referencia (reproducir)

Re-ejecutar `DummyClassifier`, regresión logística y el segundo modelo del avance, **con el mismo pipeline, semilla, validación y métricas**. Registrar si los resultados coinciden con el avance; si no, explicar la causa probable.

### 7. Modelo de ensamble (mínimo 1)

`RandomForestClassifier`, `GradientBoostingClassifier` o `HistGradientBoostingClassifier` (XGBoost/LightGBM opcionales si no requieren configuración compleja). Explicar: principio de funcionamiento, ventajas frente a modelos iniciales, riesgo de sobreajuste, hiperparámetros principales, costo computacional e interpretabilidad. Evaluar con la misma validación.

### 8. Máquina de soporte vectorial (SVM)

Explorar al menos kernel **lineal** y **RBF o polinomial**; ajustar `C` y, si aplica, `gamma`. Integrar en el pipeline con escalamiento. Explicar cómo los hiperparámetros afectan margen, complejidad de la frontera, sobreajuste y generalización.

### 9. Comparación entre familias de modelos

Mínimo: Dummy (trivial), regresión logística (lineal), árbol/ensamble, SVM (márgenes). Opcional: `MLPClassifier` (si se usa, explicar arquitectura, activación, número de capas, criterio de parada y limitaciones).

| Modelo | Métrica media | Desviación estándar | Tiempo de entrenamiento | Tiempo de inferencia |
|---|---|---|---|---|

### 10. Selección o reducción de características (mínimo 1 técnica)

`SelectKBest`, RFE, `SelectFromModel`, L1, importancia, eliminación por correlación o PCA — **dentro del pipeline o ajustada solo con train**. Comparar contra el modelo completo. Si se usa PCA, aclarar que los componentes pierden la interpretación directa.

| Configuración | Número de características | Métrica | Tiempo | Interpretabilidad |
|---|---|---|---|---|
| Modelo completo | | | | |
| Modelo reducido | | | | |

### 11. Ajuste de hiperparámetros

`GridSearchCV`, `RandomizedSearchCV` u otro procedimiento sistemático, con espacio **acotado y viable**. Ajustar al menos un modelo avanzado (idealmente ensamble y SVM). **Sin usar el conjunto de prueba.** Explicar espacio de búsqueda, número de combinaciones, criterio de selección, costo y resultados.

| Modelo | Hiperparámetros explorados | Mejor configuración | Métrica de validación |
|---|---|---|---|

### 12. Registro de experimentos

Consolidar **todos** los experimentos (MLflow, CSV o tabla estructurada):

| ID | Fecha | Dataset | Pipeline | Modelo | Hiperparámetros | Validación | Métrica principal | Resultados |
|---|---|---|---|---|---|---|---|---|

### 13. Selección de finalistas (mínimo 2)

Considerar promedio de CV, variabilidad, recall, precision, F1, ROC-AUC, complejidad, tiempos, interpretabilidad y facilidad de despliegue. **Los pesos suman 100 %.**

| Criterio | Peso | Modelo A | Modelo B | Justificación |
|---|---|---|---|---|
| Recall | | | | |
| Precision | | | | |
| F1-score | | | | |
| ROC-AUC | | | | |
| Estabilidad | | | | |
| Interpretabilidad | | | | |
| Costo operativo | | | | |

### 14. Evaluación en prueba (solo finalistas, una vez)

Accuracy, precision, recall, F1, ROC-AUC, matriz de confusión, curva ROC y curva precision-recall.

| Modelo finalista | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|

Explicar si prueba confirma validación, muestra pérdida razonable, revela sobreajuste o cambia la selección.

### 15. Análisis de errores

Examinar falsos positivos y falsos negativos con ejemplos representativos:

| Caso | Clase real | Predicción | Probabilidad | Tipo de error | Características relevantes |
|---|---|---|---|---|---|

Explicar patrones comunes, variables poco representadas, perfiles donde falla el modelo y acciones para reducir errores.

### 16. Optimización del umbral

Usar curva PR, índice de Youden, costo esperado, restricción mínima de recall u otro criterio justificado.

| Umbral | Precision | Recall | F1-score | Falsos positivos | Falsos negativos |
|---|---|---|---|---|---|
| 0.50 | | | | | |

Elegir un umbral operativo y explicar qué prioriza, qué costo reduce, qué consecuencias puede tener y **por qué no debe adoptarse sin revisión organizacional**. El umbral se elige con datos de validación, no de prueba.

### 17. Modelo final

| Criterio | Evidencia | Conclusión |
|---|---|---|
| Desempeño | | |
| Generalización | | |
| Errores | | |
| Interpretabilidad | | |
| Complejidad | | |
| Operación | | |
| Riesgo | | |

Justificar por qué es defendible para una **prueba controlada**, no solo por su métrica.

### 18. Interpretabilidad global

Coeficientes, importancia, permutation importance, SHAP u otro método justificado. Entregar: ranking de variables, visualización, interpretación de los principales factores y limitaciones del método. Aclarar que **no implica causalidad**.

### 19. Explicaciones individuales

Al menos un VP, un VN, un FP y un FN, con SHAP, LIME o equivalente.

| Caso | Resultado real | Predicción | Factores principales | Interpretación |
|---|---|---|---|---|
| Verdadero positivo | | | | |
| Verdadero negativo | | | | |
| Falso positivo | | | | |
| Falso negativo | | | | |

La explicación sirve para entender al modelo, **no para justificar automáticamente una decisión sobre el cliente**.

### 20. Evaluación de sesgos

Comparar al menos dos grupos (antigüedad, tipo de contrato, nivel de gasto, método de pago, servicios contratados, etc.) con tasa de predicción positiva, recall, precision, FPR y FNR.

| Grupo | Tamaño | Tasa de abandono | Recall | FPR | FNR |
|---|---|---|---|---|---|

Explicar diferencias, posibles causas, limitaciones y medidas de revisión/mitigación. **No concluir discriminación solo por una diferencia estadística**; relacionar con contexto, variables disponibles y calidad de datos.

### 21. Riesgos y condiciones de uso

| Riesgo | Causa | Consecuencia | Probabilidad | Impacto | Mitigación |
|---|---|---|---|---|---|
| Falsos negativos | | | | | |
| Falsos positivos | | | | | |
| Drift | | | | | |
| Datos incompletos | | | | | |
| Uso automático | | | | | |
| Sesgo | | | | | |

Definir además: usos permitidos, usos no recomendados, necesidad de revisión humana, responsables de supervisión y condiciones para suspender el modelo.

### 22. Serialización

Guardar el pipeline **completo** (preprocesamiento + modelo) con Joblib/Pickle.

| Artefacto | Archivo | Versión | Fecha | Biblioteca |
|---|---|---|---|---|
| Pipeline final | | | | |
| Modelo | | | | |
| Codificación de objetivo | | | | |
| Umbral | | | | |

> ⚠️ **Advertencia de seguridad:** cargar archivos serializados (`pickle`/`joblib`) de fuentes desconocidas puede ejecutar código arbitrario. Solo cargar artefactos de origen confiable y verificar su integridad (p. ej., hash).

### 23. Verificación de carga

Script/notebook **independiente** (`src/predict.py`) que: cargue el artefacto, reciba uno o varios registros nuevos, valide la estructura, aplique el preprocesamiento, calcule la probabilidad, aplique el umbral y devuelva la clasificación. **Verificar que las predicciones coincidan con las obtenidas antes de serializar.**

### 24. Mecanismo mínimo de inferencia

Mínimo aceptado: script o notebook funcional con entrada estructurada (CSV/JSON). Alternativas: Streamlit, Flask o FastAPI. Debe mostrar:

- Probabilidad estimada de abandono.
- Clasificación resultante.
- Umbral utilizado.
- Fecha o versión del modelo.
- Mensaje: *"Resultado estimado; requiere revisión humana."*

No mostrar al usuario final detalles innecesarios del pipeline.

### 25. Validación de datos de entrada

Detectar: campos faltantes, tipos incorrectos, valores fuera de rango, categorías no reconocidas, columnas adicionales y estructura incompleta. **Informar el problema sin detener la ejecución de forma inesperada.**

| Caso | Entrada | Resultado esperado | Resultado obtenido |
|---|---|---|---|
| Registro válido | | | |
| Campo faltante | | | |
| Categoría desconocida | | | |
| Tipo incorrecto | | | |
| Valor extremo | | | |

### 26. Simulación de inferencias (≥ 5 clientes)

Perfiles: bajo riesgo, riesgo medio, alto riesgo, categoría poco frecuente, caso cercano al umbral.

| Cliente | Probabilidad | Clasificación | Factores relevantes | Observación |
|---|---|---|---|---|

Explicar cómo se usaría cada resultado en un proceso de revisión humana y qué decisiones **no** deben automatizarse solo con la predicción.

### 27. Plan de monitoreo

Cubrir calidad de datos (faltantes, categorías desconocidas, cambios de rango, volumen, errores de entrada), distribución (data drift, tasa de clientes en riesgo), desempeño (precision, recall, F1, ROC-AUC, FPR, FNR) y operación (tiempo de respuesta, errores, disponibilidad, versión activa).

| Indicador | Frecuencia | Umbral de alerta | Acción |
|---|---|---|---|
| Valores faltantes | | | |
| Recall | | | |
| Data drift | | | |
| Tiempo de respuesta | | | |
| Categorías nuevas | | | |

Puede ser propuesta documentada o simulación básica; no se requiere plataforma productiva.

### 28. Criterios de actualización

Definir cuándo **revisar, recalibrar, cambiar umbral, reentrenar o suspender**, con criterios concretos: caída sostenida de recall, aumento de falsos negativos, cambio significativo de distribución, nuevas categorías, pérdida de disponibilidad, diferencias excesivas entre grupos, cambio en el proceso comercial o en la definición de abandono. Indicar **quién revisa** y **qué acción** corresponde en cada caso.

### 29. Trazabilidad final

| Elemento | Detalle |
|---|---|
| Versión del dataset | |
| Fuente | |
| Fecha de consulta | |
| Variables utilizadas | |
| Variables excluidas | |
| Pipeline | |
| Modelos evaluados | |
| Hiperparámetros | |
| Métricas de validación | |
| Métricas de prueba | |
| Umbral | |
| Modelo final | |
| Archivo serializado | |
| Bibliotecas | |
| Fecha de entrenamiento | |
| Archivo principal | |
| Responsable | |

### 30. Ficha del modelo (`reports/ficha_modelo.md`)

Incluir: nombre y versión, propósito, usuarios previstos, usos permitidos, usos no recomendados, dataset, variable objetivo, métricas, umbral, limitaciones, riesgos, análisis de sesgos, mecanismo de inferencia, plan de monitoreo, criterios de actualización y responsable de revisión. Debe funcionar como resumen técnico y operativo.

### 31. Reporte ejecutivo (`reports/reporte_ejecutivo.md`)

Audiencia **no técnica**. Incluir: problema atendido, utilidad, principales resultados, significado de los errores, clientes que requieren revisión, limitaciones, riesgos, recomendación de uso y condiciones de monitoreo. **Sin código, funciones ni hiperparámetros innecesarios.** Traducir la evidencia en una recomendación clara y prudente.

## Convenciones técnicas

- Python + scikit-learn; fijar `random_state` en todo (splits, CV, modelos, búsquedas) y registrar la semilla.
- Fijar y registrar versiones exactas de bibliotecas (`requirements.txt`).
- Guardar cada tabla en `reports/tablas/` (CSV) además de mostrarla en el notebook.
- Guardar cada figura en `reports/figuras/` con nombre descriptivo.
- Medir tiempos de entrenamiento e inferencia de forma consistente (misma máquina, mismos datos).
- Usar `n_jobs` con cuidado y presupuestar el costo de las búsquedas de hiperparámetros.
- Código con funciones reutilizables en `src/`; el notebook orquesta y narra, no duplica lógica.
- Antes de cualquier evaluación en test, confirmar explícitamente que la selección y el ajuste ya terminaron.

## Checklist de entrega

- [ ] Reproducción de la fase anterior verificada y documentada
- [ ] Bitácora de cambios completa
- [ ] Pipeline consolidado y sin fuga de información
- [ ] ≥ 2 características nuevas documentadas
- [ ] Estrategia de CV definida y justificada
- [ ] Baselines re-ejecutados y comparados con el avance
- [ ] Ensamble, SVM y (opcional) MLP evaluados con CV
- [ ] Selección/reducción de características comparada
- [ ] Ajuste de hiperparámetros sin usar test
- [ ] Registro de experimentos consolidado
- [ ] ≥ 2 finalistas con matriz de decisión (pesos = 100 %)
- [ ] Evaluación en test (una sola vez) + análisis de errores
- [ ] Umbral operativo justificado
- [ ] Modelo final con decisión documentada
- [ ] Interpretabilidad global y local (VP, VN, FP, FN)
- [ ] Análisis de sesgos entre ≥ 2 grupos
- [ ] Registro de riesgos y condiciones de uso
- [ ] Pipeline serializado + advertencia de seguridad
- [ ] Script de carga independiente con predicciones verificadas
- [ ] Mecanismo de inferencia con validación de entrada y casos de prueba
- [ ] Simulación con ≥ 5 clientes
- [ ] Plan de monitoreo y criterios de actualización
- [ ] Tabla de trazabilidad final
- [ ] Ficha del modelo
- [ ] Reporte ejecutivo
