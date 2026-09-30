| Caso               | Real   | Predicción   |   Prob. | Factores principales (SHAP)                                            |
|:-------------------|:-------|:-------------|--------:|:-----------------------------------------------------------------------|
| Verdadero positivo | LCK    | LCK          |   0.88  | cspm(+1.16); earnedgpm(+0.82); impacto_dano(-0.29); damageshare(+0.26) |
| Verdadero negativo | LCK CL | LCK CL       |   0.149 | cspm(-1.34); xpdiffat15(+0.64); golddiffat15(-0.56); earnedgpm(-0.19)  |
| Falso positivo     | LCK CL | LCK          |   0.694 | earnedgpm(+1.59); xpdiffat15(+0.50); oro_por_cs15(-0.45); cspm(-0.37)  |
| Falso negativo     | LCK    | LCK CL       |   0.212 | earnedgpm(-1.29); xpdiffat15(+1.19); cspm(-0.85); golddiffat15(-0.15)  |
