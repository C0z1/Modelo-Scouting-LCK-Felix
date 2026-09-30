# Reporte Ejecutivo — Modelo de Scouting LCK

*Documento para audiencia no técnica. Traduce la evidencia en una recomendación clara y prudente.*

## El problema

Cada temporada, los equipos de LoL en Corea deben decidir qué jugadores de su academia (LCK Challengers
League) están listos para subir a la liga principal (LCK). Hoy esa decisión es sobre todo subjetiva. La
pregunta de este proyecto es si los **datos de rendimiento por partida** pueden **apoyar** esa decisión.

## Qué hace el modelo

A partir de estadísticas de una partida (farmeo, oro, daño, ventajas a los 15 minutos, posición), el
modelo estima **qué tan parecido es el rendimiento de un jugador al de la LCK**. Devuelve una
probabilidad y una clasificación orientativa. **Es una ayuda para priorizar a quién revisar, no un
veredicto.**

## Para qué sirve

- Señalar, entre muchos jugadores de academia, **cuáles conviene revisar con más atención**.
- Aportar una segunda opinión cuantitativa al juicio del cuerpo de scouting.

## Principales resultados

- El modelo distingue razonablemente bien ambos niveles: **acierta ~83 de cada 100** comparaciones
  (AUC-ROC 0.83) y **detecta al 74 %** de los jugadores con perfil de LCK.
- Los factores que más pesan son coherentes con el sentido común del juego: **oro por minuto**,
  **eficiencia de farmeo (CS/min)** y las **ventajas de oro y experiencia a los 15 minutos**.
- Un modelo simple e **interpretable** (regresión logística) igualó a alternativas más complejas, por lo
  que se eligió por ser más transparente y fácil de operar.

## Qué significan los errores

- **No detectar a un jugador listo (falso negativo):** es el error más costoso — se puede perder un
  talento. El modelo se ajustó para **minimizar este error** (por eso prioriza el "recall").
- **Señalar a un jugador que no está listo (falso positivo):** menos grave — solo implica dedicarle una
  revisión que confirmará el juicio humano.

## Jugadores que requieren revisión

El modelo entrega una lista priorizada. Los casos **cercanos al umbral** (ni claramente LCK ni claramente
academia) son precisamente los que **más necesitan ojo humano**. Ningún jugador debe ser ascendido o
descartado solo por el número del modelo.

## Limitaciones (importante)

- Los datos usados son **simulados**; con datos reales los resultados pueden cambiar.
- El modelo mira **una partida a la vez** y a cada jugador **de forma individual**: no ve la química de
  equipo ni la consistencia a lo largo de la temporada.
- El juego **cambia con cada parche**, así que el modelo debe revisarse con el tiempo.
- Funciona mejor con jugadores de perfil ofensivo (más oro/farmeo) y peor con roles de bajo oro (p. ej.
  soportes), lo que debe tenerse en cuenta al interpretarlo.

## Riesgos y uso responsable

- **Nunca** debe automatizarse una decisión sobre un jugador solo con la predicción.
- **Siempre** hay revisión humana, a cargo del cuerpo de scouting.
- Debe **suspenderse** si su desempeño cae de forma sostenida o si cambia la forma de jugar de la liga.

## Recomendación

Usar el modelo como una **herramienta de apoyo en una prueba controlada**: que acompañe (no reemplace) al
scouting, con seguimiento de su desempeño y revisión periódica. Es defendible no por tener la mejor
métrica, sino por su **equilibrio entre desempeño, transparencia, bajo costo y prudencia en el uso**.

## Condiciones de monitoreo

Revisar mensualmente la calidad de los datos, la estabilidad de los aciertos y si la liga ha cambiado.
Ante cualquier señal de deterioro, recalibrar o reentrenar antes de seguir usándolo.
