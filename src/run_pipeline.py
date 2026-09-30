"""Driver de la FASE FINAL: ejecuta el flujo completo y genera todos los artefactos.

Ejecuta, en orden, las etapas del CLAUDE.md y guarda:
  * tablas   -> reports/tablas/*.csv y *.md
  * figuras  -> reports/figuras/*.png
  * modelos  -> models/*.joblib + metadata.json
  * experimentos -> experiments/registro_experimentos.csv
  * resumen  -> reports/tablas/_resultados.json  (cifras clave para notebook/reportes)

Uso:  python -m src.run_pipeline
"""
from __future__ import annotations

import json
import time
import platform
from datetime import date, datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from . import data_simulada as ds
from .config import (
    AVANCE_NUM, BASE_CAT, BASE_FEATURES, CAT_FEATURES, DATA_PROC, DATA_RAW,
    EXP_DIR, MODELS_DIR, MODEL_VERSION, NUM_FEATURES, RANDOM_STATE, TAB_DIR, TARGET,
)
from .evaluate import (
    analisis_sesgos, metricas_completas, tabla_errores, tabla_umbral,
    umbral_min_recall, umbral_por_f1, umbral_por_youden,
)
from .preprocessing import build_preprocessor_final, get_output_feature_names
from .reporting import guardar_figura, guardar_tabla
from .split import cv_avance, cv_final, particion_70_15_15
from .train import (
    evaluar_cv, make_pipeline_base, make_pipeline_final,
    modelos_avance, modelos_fase_final,
)

RESULTS: dict = {"_meta": {
    "fecha_ejecucion": datetime.now().isoformat(timespec="seconds"),
    "random_state": RANDOM_STATE,
    "sklearn": sklearn.__version__,
    "python": platform.python_version(),
}}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =========================================================================== #
def main() -> None:
    t_inicio = time.perf_counter()

    # --------------------------------------------------------------------- #
    # 0-2. Datos, calidad, fuga                                             #
    # --------------------------------------------------------------------- #
    log("Generando dataset simulado (semilla 42)...")
    raw = ds.generar_dataset()
    df = ds.preparar(raw)
    raw.to_csv(DATA_RAW / "dataset_simulado.csv", index=False)

    # Calidad
    faltantes = df[BASE_FEATURES].isnull().sum()
    faltantes = faltantes[faltantes > 0]
    calidad = pd.DataFrame({
        "Faltantes": faltantes,
        "% Faltante": (faltantes / len(df) * 100).round(2),
    })
    guardar_tabla(calidad, "02_calidad_faltantes")
    dup = int(df.duplicated().sum())
    vc = df[TARGET].value_counts()
    vc_pct = df[TARGET].value_counts(normalize=True)
    balance = vc.rename({1: "LCK (1)", 0: "LCK CL (0)"})
    balance_pct = vc_pct.rename({1: "LCK (1)", 0: "LCK CL (0)"})
    guardar_tabla(pd.DataFrame({"conteo": balance, "proporcion": balance_pct.round(4)}),
                  "02_balance_clases")
    RESULTS["datos"] = {
        "n_filas": int(len(df)), "n_duplicados": dup,
        "balance": {"LCK": float(vc_pct.get(1, 0)), "LCK CL": float(vc_pct.get(0, 0))},
        "ratio_desbalance": float(vc.get(0, 0) / max(vc.get(1, 1), 1)),
    }
    log(f"  filas={len(df)} duplicados={dup} %LCK={vc_pct.get(1,0):.3f}")

    # --------------------------------------------------------------------- #
    # 3-4. X/y, ingeniería de características (demostración + fuga)          #
    # --------------------------------------------------------------------- #
    X = df[BASE_FEATURES].copy()
    y = df[TARGET].copy()

    # --------------------------------------------------------------------- #
    # 9. Partición 70/15/15 (idéntica al avance)                            #
    # --------------------------------------------------------------------- #
    X_train, X_val, X_test, y_train, y_val, y_test = particion_70_15_15(X, y)
    # Guardar splits procesados
    for nm, Xs, ys in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        out = Xs.copy(); out[TARGET] = ys.values
        out.to_csv(DATA_PROC / f"{nm}.csv", index=False)
    RESULTS["particion"] = {"train": len(X_train), "val": len(X_val), "test": len(X_test)}
    log(f"  split train/val/test = {len(X_train)}/{len(X_val)}/{len(X_test)}")

    # Fuga de las nuevas características: correlación con el objetivo SOLO en train
    from .features import FeatureEngineer
    fe_train = FeatureEngineer().transform(X_train)
    nuevas = ["kda", "oro_por_cs15", "dominancia15", "participacion15", "impacto_dano"]
    corr_fuga = pd.DataFrame({
        "corr_con_is_lck (train)": [np.corrcoef(fe_train[c].fillna(fe_train[c].median()), y_train)[0, 1]
                                    for c in nuevas]
    }, index=nuevas).round(3)
    guardar_tabla(corr_fuga, "04_features_nuevas_fuga")
    RESULTS["features_nuevas"] = corr_fuga["corr_con_is_lck (train)"].to_dict()

    # --------------------------------------------------------------------- #
    # 5. Estrategia de CV                                                    #
    # --------------------------------------------------------------------- #
    cv = cv_final(n_splits=5, n_repeats=3)
    cv_rep = cv_avance()
    RESULTS["cv"] = {"metodo": "RepeatedStratifiedKFold", "n_splits": 5, "n_repeats": 3,
                     "semilla": RANDOM_STATE, "metrica_principal": "f1",
                     "metricas_secundarias": ["roc_auc", "precision", "recall", "accuracy"]}

    # --------------------------------------------------------------------- #
    # 6. Reproducción de baselines del avance (pipeline base, CV avance)    #
    # --------------------------------------------------------------------- #
    log("Reproduciendo baselines del avance...")
    Xtr_kda = X_train.copy(); Xtr_kda["kda"] = (X_train["kills"] + X_train["assists"]) / X_train["deaths"].replace(0, 1)
    Xval_kda = X_val.copy(); Xval_kda["kda"] = (X_val["kills"] + X_val["assists"]) / X_val["deaths"].replace(0, 1)
    rep_rows = []
    avance_ref = {  # F1/AUC en validación reportados por el avance (verificados bit a bit)
        "Regresión Logística": (0.6860, 0.8456), "Árbol de Decisión": (0.6562, 0.7979),
        "Random Forest": (0.6757, 0.8339), "Gradient Boosting": (0.6419, 0.8321),
    }
    for name, mdl in modelos_avance().items():
        pipe = make_pipeline_base(mdl)
        pipe.fit(Xtr_kda, y_train)
        yp = pipe.predict(Xval_kda)
        try:
            pr = pipe.predict_proba(Xval_kda)[:, 1]
        except Exception:
            pr = yp.astype(float)
        if name == "Dummy (baseline)":
            m = {"Accuracy": float((yp == y_val).mean()), "Precision": 0.0, "Recall": 0.0,
                 "F1-score": 0.0, "ROC-AUC": 0.5}
        else:
            m = metricas_completas(y_val, yp, pr)
        ref = avance_ref.get(name)
        coincide = ("—" if ref is None else
                    ("sí" if abs(m["F1-score"] - ref[0]) < 1e-4 and abs(m["ROC-AUC"] - ref[1]) < 1e-4
                     else "no"))
        rep_rows.append({"Modelo": name, **{k: round(v, 4) for k, v in m.items()},
                         "F1 avance": None if ref is None else ref[0],
                         "AUC avance": None if ref is None else ref[1],
                         "¿Coincide?": coincide})
    rep_df = pd.DataFrame(rep_rows).set_index("Modelo")
    guardar_tabla(rep_df, "06_reproduccion_baselines")
    RESULTS["reproduccion_ok"] = bool((rep_df["¿Coincide?"].isin(["sí", "—"])).all())
    log(f"  reproducción coincide con avance: {RESULTS['reproduccion_ok']}")

    # --------------------------------------------------------------------- #
    # 7-9. Comparación de familias (pipeline final, CV repetida)            #
    # --------------------------------------------------------------------- #
    log("Comparando familias de modelos con CV repetida (puede tardar)...")
    comp = evaluar_cv(modelos_fase_final(), X_train, y_train, cv, pipeline_builder=make_pipeline_final)
    comp_view = comp[["f1_mean", "f1_std", "roc_auc_mean", "roc_auc_std",
                      "recall_mean", "precision_mean", "fit_time_mean", "score_time_mean"]].round(4)
    comp_view = comp_view.sort_values("f1_mean", ascending=False)
    guardar_tabla(comp_view, "09_comparacion_familias")
    RESULTS["comparacion_familias"] = comp_view.to_dict(orient="index")

    # Figura comparativa (F1 con barras de error)
    fig, ax = plt.subplots(figsize=(9, 5))
    d = comp_view.drop(index=[i for i in ["Dummy (baseline)"] if i in comp_view.index])
    ax.barh(d.index[::-1], d["f1_mean"][::-1], xerr=d["f1_std"][::-1],
            color="#4E79A7", edgecolor="none", capsize=3)
    ax.set_xlabel("F1-score (media CV ± std)")
    ax.set_title("Comparación de familias de modelos (CV repetida)", fontweight="bold")
    guardar_figura(fig, "09_comparacion_familias")
    log("  top F1:\n" + comp_view[["f1_mean", "roc_auc_mean"]].head(4).to_string())

    # --------------------------------------------------------------------- #
    # 10. Selección / reducción de características                            #
    # --------------------------------------------------------------------- #
    log("Selección de características (SelectFromModel sobre RF)...")
    rf_sel = RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced",
                                    random_state=RANDOM_STATE, n_jobs=-1)
    # Modelo completo
    pipe_full = make_pipeline_final(RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))
    t0 = time.perf_counter()
    f1_full = cross_val_score(pipe_full, X_train, y_train, cv=cv_final(5, 2), scoring="f1", n_jobs=-1)
    t_full = time.perf_counter() - t0
    # Modelo reducido: SelectFromModel dentro del pipeline
    from sklearn.pipeline import Pipeline as SkPipeline
    pipe_red = SkPipeline([
        ("pre", build_preprocessor_final()),
        ("sel", SelectFromModel(rf_sel, threshold="median")),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced",
                                       random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    t0 = time.perf_counter()
    f1_red = cross_val_score(pipe_red, X_train, y_train, cv=cv_final(5, 2), scoring="f1", n_jobs=-1)
    t_red = time.perf_counter() - t0
    # Nº de características seleccionadas (ajuste en train completo)
    pipe_red.fit(X_train, y_train)
    n_sel = int(pipe_red.named_steps["sel"].get_support().sum())
    n_tot = len(get_output_feature_names(pipe_red.named_steps["pre"]))
    sel_tab = pd.DataFrame({
        "Configuración": ["Modelo completo", "Modelo reducido (SelectFromModel)"],
        "N.º características": [n_tot, n_sel],
        "F1 (media CV)": [round(f1_full.mean(), 4), round(f1_red.mean(), 4)],
        "F1 (std CV)": [round(f1_full.std(), 4), round(f1_red.std(), 4)],
        "Tiempo CV (s)": [round(t_full, 1), round(t_red, 1)],
        "Interpretabilidad": ["Media (todas las vars)", "Mayor (menos vars)"],
    }).set_index("Configuración")
    guardar_tabla(sel_tab, "10_seleccion_caracteristicas")
    RESULTS["seleccion"] = {"n_total": n_tot, "n_reducido": n_sel,
                            "f1_full": float(f1_full.mean()), "f1_red": float(f1_red.mean())}
    log(f"  completo={n_tot} vars F1={f1_full.mean():.4f} | reducido={n_sel} vars F1={f1_red.mean():.4f}")

    # --------------------------------------------------------------------- #
    # 11. Ajuste de hiperparámetros (sin test)                              #
    # --------------------------------------------------------------------- #
    log("Ajuste de hiperparámetros: RandomizedSearch (RF) + GridSearch (SVM)...")
    tune_rows = []
    # RF con RandomizedSearchCV
    rf_pipe = make_pipeline_final(RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))
    rf_space = {
        "clf__n_estimators": [200, 300, 400],
        "clf__max_depth": [6, 8, 10, 12, 16, None],
        "clf__min_samples_leaf": [1, 2, 4, 8],
        "clf__max_features": ["sqrt", "log2", 0.5],
    }
    N_ITER_RF = 15
    rf_search = RandomizedSearchCV(
        rf_pipe, rf_space, n_iter=N_ITER_RF, scoring="f1", cv=cv_final(5, 1), n_jobs=-1,
        random_state=RANDOM_STATE, refit=True)
    t0 = time.perf_counter(); rf_search.fit(X_train, y_train); t_rf = time.perf_counter() - t0
    tune_rows.append({
        "Modelo": "Random Forest",
        "Hiperparámetros explorados": "n_estimators, max_depth, min_samples_leaf, max_features",
        "N.º combinaciones": N_ITER_RF,
        "Mejor configuración": str({k.replace("clf__", ""): v for k, v in rf_search.best_params_.items()}),
        "F1 CV (mejor)": round(rf_search.best_score_, 4),
        "Tiempo (s)": round(t_rf, 1),
    })
    # SVM con GridSearchCV (lineal y RBF, C y gamma)
    svm_pipe = make_pipeline_final(SVC(class_weight="balanced", probability=False, random_state=RANDOM_STATE))
    svm_grid = [
        {"clf__kernel": ["linear"], "clf__C": [0.1, 1, 10]},
        {"clf__kernel": ["rbf"], "clf__C": [0.5, 1, 10], "clf__gamma": ["scale", 0.05, 0.1]},
    ]
    svm_search = GridSearchCV(svm_pipe, svm_grid, scoring="f1", cv=cv_final(5, 1), n_jobs=-1, refit=True)
    t0 = time.perf_counter(); svm_search.fit(X_train, y_train); t_svm = time.perf_counter() - t0
    n_comb_svm = 3 + 3 * 3
    tune_rows.append({
        "Modelo": "SVM",
        "Hiperparámetros explorados": "kernel (linear/rbf), C, gamma",
        "N.º combinaciones": n_comb_svm,
        "Mejor configuración": str({k.replace("clf__", ""): v for k, v in svm_search.best_params_.items()}),
        "F1 CV (mejor)": round(svm_search.best_score_, 4),
        "Tiempo (s)": round(t_svm, 1),
    })
    tune_df = pd.DataFrame(tune_rows).set_index("Modelo")
    guardar_tabla(tune_df, "11_ajuste_hiperparametros", index=True)
    RESULTS["tuning"] = {
        "rf_best": {k.replace("clf__", ""): v for k, v in rf_search.best_params_.items()},
        "rf_best_f1": float(rf_search.best_score_),
        "svm_best": {k.replace("clf__", ""): v for k, v in svm_search.best_params_.items()},
        "svm_best_f1": float(svm_search.best_score_),
    }
    log(f"  RF best F1={rf_search.best_score_:.4f} | SVM best F1={svm_search.best_score_:.4f}")

    # --------------------------------------------------------------------- #
    # Experimento extra: SMOTE vs class_weight (plan de mejora del avance)  #
    # --------------------------------------------------------------------- #
    from .features import FeatureEngineer as _FE
    from .preprocessing import build_column_transformer_final
    smote_pipe = ImbPipeline([
        ("fe", _FE()),
        ("ct", build_column_transformer_final()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    f1_smote = cross_val_score(smote_pipe, X_train, y_train, cv=cv_final(5, 2), scoring="f1", n_jobs=-1)
    RESULTS["smote_vs_classweight"] = {
        "rf_class_weight_f1": float(f1_full.mean()),
        "rf_smote_f1": float(f1_smote.mean()),
    }
    log(f"  SMOTE F1={f1_smote.mean():.4f} vs class_weight F1={f1_full.mean():.4f}")

    # --------------------------------------------------------------------- #
    # 12. Registro de experimentos                                          #
    # --------------------------------------------------------------------- #
    exp_rows = []
    eid = 1
    for name, row in comp_view.iterrows():
        exp_rows.append({
            "ID": eid, "Fecha": date.today().isoformat(), "Dataset": "simulado_v1 (8000)",
            "Pipeline": "FE+CT (final)", "Modelo": name,
            "Hiperparametros": "por defecto del catálogo",
            "Validacion": "RepeatedStratifiedKFold(5x3)",
            "Metrica_principal": "f1",
            "Resultados": f"F1={row['f1_mean']:.4f}±{row['f1_std']:.4f}; AUC={row['roc_auc_mean']:.4f}",
        }); eid += 1
    exp_rows.append({"ID": eid, "Fecha": date.today().isoformat(), "Dataset": "simulado_v1 (8000)",
                     "Pipeline": "FE+CT+SelectFromModel", "Modelo": "Random Forest",
                     "Hiperparametros": "n_estimators=300,max_depth=12", "Validacion": "RepeatedStratifiedKFold(5x3)",
                     "Metrica_principal": "f1", "Resultados": f"F1={f1_red.mean():.4f} ({n_sel} vars)"}); eid += 1
    exp_rows.append({"ID": eid, "Fecha": date.today().isoformat(), "Dataset": "simulado_v1 (8000)",
                     "Pipeline": "FE+CT+SMOTE", "Modelo": "Random Forest",
                     "Hiperparametros": "n_estimators=300,max_depth=12", "Validacion": "RepeatedStratifiedKFold(5x2)",
                     "Metrica_principal": "f1", "Resultados": f"F1={f1_smote.mean():.4f}"}); eid += 1
    exp_rows.append({"ID": eid, "Fecha": date.today().isoformat(), "Dataset": "simulado_v1 (8000)",
                     "Pipeline": "FE+CT (final)", "Modelo": "Random Forest (tuned)",
                     "Hiperparametros": str(RESULTS["tuning"]["rf_best"]), "Validacion": "RandomizedSearchCV(5x2)",
                     "Metrica_principal": "f1", "Resultados": f"F1={rf_search.best_score_:.4f}"}); eid += 1
    exp_rows.append({"ID": eid, "Fecha": date.today().isoformat(), "Dataset": "simulado_v1 (8000)",
                     "Pipeline": "FE+CT (final)", "Modelo": "SVM (tuned)",
                     "Hiperparametros": str(RESULTS["tuning"]["svm_best"]), "Validacion": "GridSearchCV(5x1)",
                     "Metrica_principal": "f1", "Resultados": f"F1={svm_search.best_score_:.4f}"}); eid += 1
    exp_df = pd.DataFrame(exp_rows)
    exp_df.to_csv(EXP_DIR / "registro_experimentos.csv", index=False)
    guardar_tabla(exp_df.set_index("ID"), "12_registro_experimentos")

    # --------------------------------------------------------------------- #
    # 13. Selección de finalistas (2)                                        #
    # --------------------------------------------------------------------- #
    # Candidatos: mejor ensamble tuneado (RF tuned) y mejor por F1 de la comparación
    # que NO sea Dummy. Elegimos los dos mejores por F1 medio.
    ranking = comp_view.drop(index=[i for i in ["Dummy (baseline)"] if i in comp_view.index])
    top2 = list(ranking.index[:2])
    RESULTS["finalistas"] = top2
    log(f"  finalistas (top-2 por F1): {top2}")

    # Construir pipelines de finalistas (con probabilidad para SVM si aplica).
    catalogo = modelos_fase_final()

    def build_estimator(nombre):
        est = catalogo[nombre]
        # Si es RF y coincide con el mejor tuneado, usar hiperparámetros tuneados.
        if nombre == "Random Forest":
            from sklearn.base import clone
            est = clone(rf_search.best_estimator_.named_steps["clf"])
        if isinstance(est, SVC):
            est = SVC(**{**est.get_params(), "probability": True})
        return est

    finalist_pipes = {nm: make_pipeline_final(build_estimator(nm)) for nm in top2}

    # Matriz de decisión (pesos = 100%)
    pesos = {"Recall": 25, "Precision": 15, "F1-score": 20, "ROC-AUC": 20,
             "Estabilidad": 10, "Interpretabilidad": 5, "Costo operativo": 5}
    assert sum(pesos.values()) == 100
    # Puntajes 1-5 derivados de CV para las métricas; interpretabilidad/costo por familia.
    def punt_metric(nombre, metrica):
        v = ranking.loc[nombre, metrica]
        # normalizar a 1-5 dentro de finalistas
        vals = ranking.loc[top2, metrica]
        lo, hi = vals.min(), vals.max()
        return 5.0 if hi == lo else round(1 + 4 * (v - lo) / (hi - lo), 2)
    def estabilidad(nombre):
        vals = ranking.loc[top2, "f1_std"]
        lo, hi = vals.min(), vals.max()
        v = ranking.loc[nombre, "f1_std"]
        return 5.0 if hi == lo else round(1 + 4 * (hi - v) / (hi - lo), 2)  # menor std -> mayor puntaje
    interp_map = {"Random Forest": 3, "Gradient Boosting": 3, "HistGradientBoosting": 2.5,
                  "Regresión Logística": 5, "SVM lineal": 4, "SVM RBF": 2, "MLP (red neuronal)": 1.5}
    costo_map = {"Random Forest": 4, "Gradient Boosting": 3, "HistGradientBoosting": 4.5,
                 "Regresión Logística": 5, "SVM lineal": 3, "SVM RBF": 2, "MLP (red neuronal)": 2.5}
    dec_rows = []
    for crit, peso in pesos.items():
        fila = {"Criterio": crit, "Peso %": peso}
        for nm in top2:
            if crit in ("Recall", "Precision", "F1-score", "ROC-AUC"):
                key = {"Recall": "recall_mean", "Precision": "precision_mean",
                       "F1-score": "f1_mean", "ROC-AUC": "roc_auc_mean"}[crit]
                fila[nm] = punt_metric(nm, key)
            elif crit == "Estabilidad":
                fila[nm] = estabilidad(nm)
            elif crit == "Interpretabilidad":
                fila[nm] = interp_map.get(nm, 3)
            else:
                fila[nm] = costo_map.get(nm, 3)
        dec_rows.append(fila)
    dec_df = pd.DataFrame(dec_rows).set_index("Criterio")
    # Puntaje ponderado
    ponderado = {}
    for nm in top2:
        ponderado[nm] = round(sum(dec_df.loc[c, nm] * dec_df.loc[c, "Peso %"] / 100 for c in dec_df.index), 3)
    dec_df.loc["PUNTAJE PONDERADO"] = {"Peso %": 100, **ponderado}
    guardar_tabla(dec_df, "13_matriz_finalistas")
    ganador = max(ponderado, key=ponderado.get)
    RESULTS["matriz_decision"] = {"puntajes": ponderado, "ganador": ganador}
    log(f"  matriz de decisión -> ganador: {ganador} {ponderado}")

    # --------------------------------------------------------------------- #
    # 14. Evaluación en TEST (una sola vez, ambos finalistas)               #
    # --------------------------------------------------------------------- #
    log("Evaluación ÚNICA en test para finalistas...")
    test_rows = []
    fitted_finalists = {}
    for nm, pipe in finalist_pipes.items():
        pipe.fit(X_train, y_train)              # ajuste solo con train
        fitted_finalists[nm] = pipe
        yp = pipe.predict(X_test)
        pr = pipe.predict_proba(X_test)[:, 1]
        m = metricas_completas(y_test, yp, pr)
        test_rows.append({"Modelo finalista": nm, **{k: round(v, 4) for k, v in m.items()}})
    test_df = pd.DataFrame(test_rows).set_index("Modelo finalista")
    guardar_tabla(test_df, "14_evaluacion_test")
    RESULTS["test"] = test_df.to_dict(orient="index")

    modelo_final_nombre = ganador
    final_pipe = fitted_finalists[modelo_final_nombre]

    # Figuras test del modelo final
    yp_t = final_pipe.predict(X_test); pr_t = final_pipe.predict_proba(X_test)[:, 1]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, yp_t),
                           display_labels=["LCK CL (0)", "LCK (1)"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{modelo_final_nombre} — Test\nAUC={roc_auc_score(y_test, pr_t):.3f}", fontweight="bold")
    guardar_figura(fig, "14_matriz_confusion_test")

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, pr_t, ax=ax, name=modelo_final_nombre, color="#4E79A7")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("Curva ROC — Test", fontweight="bold")
    guardar_figura(fig, "14_roc_test")

    fig, ax = plt.subplots(figsize=(6, 5))
    prec, rec, _ = precision_recall_curve(y_test, pr_t)
    ax.plot(rec, prec, color="#E15759", lw=2)
    ax.axhline(y_test.mean(), color="k", ls="--", lw=1, label=f"Base ({y_test.mean():.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend()
    ax.set_title("Curva Precision-Recall — Test", fontweight="bold")
    guardar_figura(fig, "14_pr_test")

    # --------------------------------------------------------------------- #
    # 15. Análisis de errores (en validación, modelo final)                 #
    # --------------------------------------------------------------------- #
    yp_v = final_pipe.predict(X_val); pr_v = final_pipe.predict_proba(X_val)[:, 1]
    err = tabla_errores(X_val, y_val, yp_v, pr_v, BASE_FEATURES)
    err_counts = err["tipo"].value_counts()
    # Ejemplos representativos (2 FP, 2 FN)
    fp_ej = err[err.tipo == "FP"].sort_values("y_prob", ascending=False).head(2)
    fn_ej = err[err.tipo == "FN"].sort_values("y_prob").head(2)
    cols_ej = ["position", "cspm", "earnedgpm", "golddiffat15", "xpdiffat15", "y_true", "y_pred", "y_prob"]
    ejemplos = pd.concat([fp_ej, fn_ej])[cols_ej].round(3)
    guardar_tabla(ejemplos, "15_errores_ejemplos", index=True)
    guardar_tabla(err_counts.rename("conteo").to_frame(), "15_errores_conteo")
    RESULTS["errores"] = err_counts.to_dict()

    # --------------------------------------------------------------------- #
    # 16. Optimización de umbral (en validación)                            #
    # --------------------------------------------------------------------- #
    log("Optimización de umbral (validación)...")
    thr_tab = tabla_umbral(y_val, pr_v)
    guardar_tabla(thr_tab, "16_umbral")
    t_f1 = umbral_por_f1(y_val, pr_v)
    t_youden = umbral_por_youden(y_val, pr_v)
    t_recall = umbral_min_recall(y_val, pr_v, min_recall=0.90)
    # Umbral OPERATIVO: máximo F1 sujeto a recall >= 0.80. Prioriza no perder
    # talento (recall alto) sin desplomar la precisión; queda cerca del 0.40
    # recomendado en el avance, manteniendo continuidad.
    from sklearn.metrics import f1_score as _f1, recall_score as _rec
    cand = [(round(t, 2), _f1(y_val, (pr_v >= t).astype(int), zero_division=0))
            for t in np.round(np.arange(0.20, 0.81, 0.01), 2)
            if _rec(y_val, (pr_v >= t).astype(int), zero_division=0) >= 0.80]
    umbral_op = max(cand, key=lambda x: x[1])[0] if cand else t_f1
    RESULTS["umbral"] = {"por_f1": t_f1, "youden": t_youden, "min_recall_0.90": t_recall,
                         "operativo": umbral_op, "criterio_operativo": "max F1 sujeto a recall>=0.80"}
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thr_tab.index, thr_tab["Precision"], "o-", label="Precision", color="#4E79A7")
    ax.plot(thr_tab.index, thr_tab["Recall"], "s-", label="Recall", color="#E15759")
    ax.plot(thr_tab.index, thr_tab["F1-score"], "^-", label="F1", color="#59A14F")
    ax.axvline(0.5, color="gray", ls="--", label="0.50 (defecto)")
    ax.axvline(umbral_op, color="gold", ls=":", lw=2, label=f"Operativo ({umbral_op})")
    ax.set_xlabel("Umbral"); ax.legend(); ax.set_title("Precision/Recall/F1 vs umbral (validación)", fontweight="bold")
    guardar_figura(fig, "16_umbral")
    log(f"  umbral f1={t_f1} youden={t_youden} min_recall={t_recall} -> operativo={umbral_op}")

    # --------------------------------------------------------------------- #
    # 18-19. Interpretabilidad global y local (permutation + SHAP)          #
    # --------------------------------------------------------------------- #
    log("Interpretabilidad (permutation importance + SHAP)...")
    from sklearn.inspection import permutation_importance
    pre = final_pipe.named_steps["pre"]
    clf = final_pipe.named_steps["clf"]
    feat_names = get_output_feature_names(pre)
    Xval_t = pre.transform(X_val)

    perm = permutation_importance(clf, Xval_t, y_val.values, scoring="roc_auc",
                                  n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    perm_df = pd.DataFrame({"Variable": feat_names, "Importancia": perm.importances_mean,
                            "std": perm.importances_std}).sort_values("Importancia", ascending=False)
    guardar_tabla(perm_df.set_index("Variable"), "18_importancia_permutation")
    fig, ax = plt.subplots(figsize=(9, 6))
    top = perm_df.head(15)
    ax.barh(top["Variable"][::-1], top["Importancia"][::-1], xerr=top["std"][::-1],
            color="#4E79A7", capsize=2)
    ax.set_title(f"Permutation importance (AUC) — {modelo_final_nombre}", fontweight="bold")
    guardar_figura(fig, "18_importancia_permutation")
    RESULTS["importancia_top"] = perm_df.head(10).set_index("Variable")["Importancia"].round(4).to_dict()

    # Importancia propia del modelo: coeficientes (lineal) o feature_importances_ (árbol)
    if hasattr(clf, "coef_"):
        coef = np.ravel(clf.coef_)
        imp_modelo = pd.DataFrame({"Variable": feat_names, "Coeficiente": coef,
                                   "|Coeficiente|": np.abs(coef)}).sort_values("|Coeficiente|", ascending=False)
        guardar_tabla(imp_modelo.set_index("Variable"), "18_importancia_modelo")
        RESULTS["importancia_modelo_tipo"] = "coeficientes (features estandarizadas)"
    elif hasattr(clf, "feature_importances_"):
        imp_modelo = pd.DataFrame({"Variable": feat_names,
                                   "Importancia": clf.feature_importances_}).sort_values("Importancia", ascending=False)
        guardar_tabla(imp_modelo.set_index("Variable"), "18_importancia_modelo")
        RESULTS["importancia_modelo_tipo"] = "feature_importances_"

    # SHAP: explicador según tipo de modelo, con respaldo robusto
    def _shap_pos(vals):
        """Normaliza la salida de SHAP a una matriz (n, features) de la clase positiva."""
        if isinstance(vals, list):
            return np.asarray(vals[1] if len(vals) > 1 else vals[0])
        vals = np.asarray(vals)
        if vals.ndim == 3:  # (n, features, clases)
            return vals[:, :, 1]
        return vals

    shap_ok = False
    try:
        import shap
        Xtr_t = pre.transform(X_train)
        bg_idx = np.random.RandomState(RANDOM_STATE).choice(len(Xtr_t), size=min(200, len(Xtr_t)), replace=False)
        background = Xtr_t[bg_idx]
        idx = np.random.RandomState(RANDOM_STATE).choice(len(Xval_t), size=min(400, len(Xval_t)), replace=False)

        if hasattr(clf, "feature_importances_"):          # árboles/ensambles
            expl = shap.TreeExplainer(clf)
            sv = _shap_pos(expl.shap_values(Xval_t[idx]))
            def sv_local(row): return _shap_pos(expl.shap_values(row))[0]
        elif hasattr(clf, "coef_"):                        # modelos lineales
            expl = shap.LinearExplainer(clf, background)
            sv = _shap_pos(expl.shap_values(Xval_t[idx]))
            def sv_local(row): return _shap_pos(expl.shap_values(row))[0]
        else:                                              # respaldo genérico
            expl = shap.Explainer(clf.predict_proba, background)
            sv = _shap_pos(expl(Xval_t[idx]).values)
            def sv_local(row): return _shap_pos(expl(row).values)[0]

        shap_abs = np.abs(sv).mean(axis=0)
        shap_df = pd.DataFrame({"Variable": feat_names, "SHAP medio |·|": shap_abs}).sort_values(
            "SHAP medio |·|", ascending=False)
        guardar_tabla(shap_df.set_index("Variable"), "18_shap_global")
        fig = plt.figure(figsize=(9, 6))
        shap.summary_plot(sv, Xval_t[idx], feature_names=feat_names, show=False, plot_size=(9, 6))
        plt.title(f"SHAP — {modelo_final_nombre}", fontweight="bold")
        guardar_figura(plt.gcf(), "18_shap_summary")
        shap_ok = True
        RESULTS["shap_top"] = shap_df.head(10).set_index("Variable")["SHAP medio |·|"].round(4).to_dict()

        # 19. Explicaciones locales: 1 VP, 1 VN, 1 FP, 1 FN
        err_val = tabla_errores(X_val, y_val, yp_v, pr_v, BASE_FEATURES).reset_index(drop=True)
        loc_rows = []
        Xval_t_all = pre.transform(X_val)
        for tipo in ["VP", "VN", "FP", "FN"]:
            sub = err_val[err_val.tipo == tipo]
            if len(sub) == 0:
                continue
            i = int(sub.index[0])
            svi1 = sv_local(Xval_t_all[i:i + 1])
            orden = np.argsort(np.abs(svi1))[::-1][:4]
            factores = "; ".join(f"{feat_names[j]}({svi1[j]:+.2f})" for j in orden)
            loc_rows.append({
                "Caso": {"VP": "Verdadero positivo", "VN": "Verdadero negativo",
                         "FP": "Falso positivo", "FN": "Falso negativo"}[tipo],
                "Real": "LCK" if err_val.loc[i, "y_true"] == 1 else "LCK CL",
                "Predicción": "LCK" if err_val.loc[i, "y_pred"] == 1 else "LCK CL",
                "Prob.": round(float(err_val.loc[i, "y_prob"]), 3),
                "Factores principales (SHAP)": factores,
            })
        loc_df = pd.DataFrame(loc_rows).set_index("Caso")
        guardar_tabla(loc_df, "19_explicaciones_locales", index=True)
    except Exception as e:  # pragma: no cover
        log(f"  SHAP no disponible ({e.__class__.__name__}: {e}); se usa permutation importance.")
        RESULTS["shap_error"] = f"{e.__class__.__name__}: {e}"

    RESULTS["shap_ok"] = shap_ok

    # --------------------------------------------------------------------- #
    # 20. Análisis de sesgos (>=2 grupos)                                   #
    # --------------------------------------------------------------------- #
    log("Análisis de sesgos por posición y por año...")
    df_eval = tabla_errores(X_val, y_val, yp_v, pr_v, BASE_FEATURES)
    sesgo_pos = analisis_sesgos(df_eval, "position").round(3)
    guardar_tabla(sesgo_pos, "20_sesgos_posicion")
    df_eval2 = df_eval.copy()
    df_eval2["nivel_gasto"] = pd.qcut(df_eval2["earnedgpm"], q=3, labels=["bajo", "medio", "alto"])
    sesgo_gasto = analisis_sesgos(df_eval2, "nivel_gasto").round(3)
    guardar_tabla(sesgo_gasto, "20_sesgos_nivel_gasto")
    RESULTS["sesgos"] = {"posicion": sesgo_pos["Recall"].to_dict(),
                         "nivel_gasto": sesgo_gasto["Recall"].to_dict()}

    # --------------------------------------------------------------------- #
    # 22-23. Serialización + verificación de carga                          #
    # --------------------------------------------------------------------- #
    log("Serializando artefacto final (reentrenado en train+val)...")
    import joblib, hashlib
    from sklearn.base import clone
    # Modelo de PRODUCCIÓN: mismo pipeline y clasificador, reentrenado en train+val.
    X_trval = pd.concat([X_train, X_val]); y_trval = pd.concat([y_train, y_val])
    prod_pipe = clone(final_pipe)
    prod_pipe.fit(X_trval, y_trval)
    pipe_path = MODELS_DIR / "pipeline_final.joblib"
    joblib.dump(prod_pipe, pipe_path)
    sha = hashlib.sha256(pipe_path.read_bytes()).hexdigest()

    metadata = {
        "nombre": "Modelo de Scouting LCK",
        "version": MODEL_VERSION,
        "modelo_final": modelo_final_nombre,
        "fecha_entrenamiento": date.today().isoformat(),
        "umbral": umbral_op,
        "features_entrada": BASE_FEATURES,
        "target": TARGET,
        "clases": {"1": "LCK", "0": "LCK CL"},
        "entrenado_en": "train+val (test reservado para evaluación única)",
        "metricas_test": RESULTS["test"].get(modelo_final_nombre, {}),
        "sha256_pipeline": sha,
        "sklearn": sklearn.__version__,
        "random_state": RANDOM_STATE,
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    # Verificación de carga: reproducir predicciones ANTES vs DESPUÉS de serializar
    from .predict import cargar_modelo, predecir
    art = cargar_modelo(verificar_hash=True)
    muestra = X_test.head(50).copy()
    pred_antes = (prod_pipe.predict_proba(muestra)[:, 1] >= umbral_op).astype(int)
    salida = predecir(muestra, art)
    pred_despues = (salida["clasificacion"] == "LCK").astype(int).values
    coincide_carga = bool((pred_antes == pred_despues).all())
    RESULTS["verificacion_carga_ok"] = coincide_carga
    log(f"  verificación de carga (50 registros): predicciones coinciden = {coincide_carga}")

    tab_artefactos = pd.DataFrame([
        {"Artefacto": "Pipeline final", "Archivo": "models/pipeline_final.joblib",
         "Versión": MODEL_VERSION, "Fecha": metadata["fecha_entrenamiento"], "Biblioteca": f"joblib/sklearn {sklearn.__version__}"},
        {"Artefacto": "Metadatos", "Archivo": "models/metadata.json", "Versión": MODEL_VERSION,
         "Fecha": metadata["fecha_entrenamiento"], "Biblioteca": "json"},
        {"Artefacto": "Umbral", "Archivo": "models/metadata.json (campo 'umbral')",
         "Versión": MODEL_VERSION, "Fecha": metadata["fecha_entrenamiento"], "Biblioteca": "—"},
    ]).set_index("Artefacto")
    guardar_tabla(tab_artefactos, "22_serializacion")

    # --------------------------------------------------------------------- #
    # 26. Simulación de inferencias (>=5 perfiles)                          #
    # --------------------------------------------------------------------- #
    log("Simulación de inferencias (5 perfiles)...")
    perfiles = pd.DataFrame([
        # alto rendimiento (perfil LCK)
        {"perfil": "Alto (perfil LCK)", "position": "mid", "kills": 5, "deaths": 1, "assists": 8,
         "cspm": 9.5, "earnedgpm": 460, "dpm": 720, "damageshare": 0.30, "vspm": 1.3,
         "goldat15": 6600, "xpat15": 6700, "csat15": 130, "golddiffat15": 900, "xpdiffat15": 700,
         "killsat15": 3, "assistsat15": 3, "deathsat15": 0, "year": 2024},
        # bajo rendimiento (perfil CL)
        {"perfil": "Bajo (perfil CL)", "position": "bot", "kills": 1, "deaths": 5, "assists": 3,
         "cspm": 6.5, "earnedgpm": 300, "dpm": 380, "damageshare": 0.18, "vspm": 1.0,
         "goldat15": 4900, "xpat15": 5200, "csat15": 82, "golddiffat15": -700, "xpdiffat15": -500,
         "killsat15": 0, "assistsat15": 1, "deathsat15": 3, "year": 2023},
        # riesgo medio
        {"perfil": "Riesgo medio", "position": "top", "kills": 3, "deaths": 3, "assists": 5,
         "cspm": 7.6, "earnedgpm": 360, "dpm": 480, "damageshare": 0.23, "vspm": 0.9,
         "goldat15": 5600, "xpat15": 5900, "csat15": 98, "golddiffat15": -50, "xpdiffat15": -30,
         "killsat15": 1, "assistsat15": 2, "deathsat15": 1, "year": 2023},
        # categoría poco frecuente (posición desconocida)
        {"perfil": "Categoría rara (pos. 'jungla')", "position": "jungla", "kills": 2, "deaths": 2, "assists": 10,
         "cspm": 5.5, "earnedgpm": 330, "dpm": 300, "damageshare": 0.12, "vspm": 2.6,
         "goldat15": 5200, "xpat15": 5600, "csat15": 70, "golddiffat15": 200, "xpdiffat15": 150,
         "killsat15": 1, "assistsat15": 4, "deathsat15": 1, "year": 2024},
        # cercano al umbral (probabilidad ~ umbral 0.40): perfil medio con earnedgpm ajustado
        {"perfil": "Cercano al umbral", "position": "top", "kills": 3, "deaths": 3, "assists": 5,
         "cspm": 7.8, "earnedgpm": 366, "dpm": 500, "damageshare": 0.24, "vspm": 0.9,
         "goldat15": 5700, "xpat15": 6000, "csat15": 100, "golddiffat15": 50, "xpdiffat15": 0,
         "killsat15": 1, "assistsat15": 2, "deathsat15": 1, "year": 2023},
    ])
    sim_in = perfiles.drop(columns=["perfil"])
    sim_out = predecir(sim_in, art)
    sim = pd.concat([perfiles["perfil"], sim_out[["probabilidad_LCK", "clasificacion", "umbral_utilizado"]]], axis=1)
    sim = sim.set_index("perfil")
    guardar_tabla(sim, "26_simulacion_clientes")
    RESULTS["simulacion"] = sim[["probabilidad_LCK", "clasificacion"]].to_dict(orient="index")

    # --------------------------------------------------------------------- #
    # Guardar resumen de resultados                                         #
    # --------------------------------------------------------------------- #
    RESULTS["_meta"]["duracion_s"] = round(time.perf_counter() - t_inicio, 1)
    RESULTS["modelo_final"] = modelo_final_nombre
    (TAB_DIR / "_resultados.json").write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2, default=str),
                                              encoding="utf-8")
    log(f"LISTO en {RESULTS['_meta']['duracion_s']}s. Modelo final: {modelo_final_nombre}")


if __name__ == "__main__":
    main()
