import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error


def metricas_por_departamento(df_distritos_info, y_train, y_pred_total):
    """
    Calcula RMSE y MAE por departamento directamente sobre los valores crudos,
    con la misma rigurosidad que la métrica global.
    """
    departamentos = df_distritos_info["departamento"].values
    registros = []

    for dep in np.unique(departamentos):
        mask = departamentos == dep
        rmse = np.sqrt(mean_squared_error(y_train[mask], y_pred_total[mask]))
        mae  = mean_absolute_error(y_train[mask], y_pred_total[mask])
        registros.append({"departamento": dep, "rmse": round(rmse, 6), "mae": round(mae, 6)})

    return (
        pd.DataFrame(registros)
        .sort_values(["mae", "rmse"], ascending=False)
        .reset_index(drop=True)
    )


def evaluar_arima(X_train, y_train, df_distritos_info, window_size, order, exportar=False, ruta=None):
    p, d, q = order
    n_distritos = X_train.shape[0]
    horizonte   = y_train.shape[1]

    y_pred_total = []
    registros    = []
    fallbacks    = 0

    for i in range(n_distritos):
        history = X_train[i, :, 0].tolist()
        preds   = []

        for t in range(horizonte):
            ventana = history[-window_size:]
            try:
                yhat = float(ARIMA(ventana, order=order).fit().forecast()[0])
            except Exception:
                yhat = ventana[-1]
                fallbacks += 1

            preds.append(yhat)
            history.append(y_train[i, t])

        y_pred_total.append(preds)
        rmse_i = np.sqrt(mean_squared_error(y_train[i], preds))
        mae_i  = mean_absolute_error(y_train[i], preds)

        info = df_distritos_info.iloc[i]
        registros.append({
            "geocode":      info["geocode"],
            "departamento": info["departamento"],
            "distrito":     info["distrito"],
            "rmse":         round(rmse_i, 6),
            "mae":          round(mae_i,  6),
        })

    y_pred_total = np.array(y_pred_total)
    rmse_global  = np.sqrt(mean_squared_error(y_train, y_pred_total))
    mae_global   = mean_absolute_error(y_train, y_pred_total)

    if fallbacks:
        print(f"  [WARN] {fallbacks} predicciones cayeron a persistencia.")

    df_metricas = pd.DataFrame(registros).sort_values(["mae", "rmse"], ascending=False)
    df_dep      = metricas_por_departamento(df_distritos_info, y_train, y_pred_total)

    if exportar and ruta:
        modelo_tag = f"ARIMA_WF_w{window_size}_p{p}_d{d}_q{q}"

        df_metricas.to_csv(ruta, index=False)
        print(f"[OK] Distritos: {ruta}")

        ruta_dep = ruta.replace(".csv", "_departamento.csv")
        df_dep.to_csv(ruta_dep, index=False)
        print(f"[OK] Departamentos: {ruta_dep}")

        ruta_global = ruta.replace(".csv", "_global.csv")
        pd.DataFrame([{
            "modelo": modelo_tag,
            "rmse":   round(rmse_global, 6),
            "mae":    round(mae_global,  6),
        }]).to_csv(ruta_global, index=False)
        print(f"[OK] Global: {ruta_global}")

    return {
        "modelo":         f"ARIMA_WF_w{window_size}",
        "rmse":           rmse_global,
        "mae":            mae_global,
        "y_pred":         y_pred_total,
        "df_metricas":    df_metricas,
        "df_dep":         df_dep,
        "rmse_distritos": df_metricas["rmse"].values,
    }


def grid_search(X_train, y_train, df_distritos_info, p_values, d_values, q_values, window_values, ruta_base):
    resultados = []
    best_score = float("inf")
    best_cfg   = None

    total  = len(window_values) * len(p_values) * len(d_values) * len(q_values)
    actual = 0

    for w in window_values:
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    actual += 1
                    order = (p, d, q)
                    print(f"[{actual}/{total}] ARIMA{order} | window={w}")

                    try:
                        res  = evaluar_arima(X_train, y_train, df_distritos_info, w, order)
                        rmse = res["rmse"]
                        mae  = res["mae"]

                        resultados.append({
                            "window": w, "p": p, "d": d, "q": q,
                            "rmse": round(rmse, 6), "mae": round(mae, 6),
                        })

                        if rmse < best_score:
                            best_score = rmse
                            best_cfg   = (w, p, d, q)

                        print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}")

                    except Exception as e:
                        print(f"  [WARN] ARIMA{order} w={w} falló: {e}")

    if not resultados:
        raise RuntimeError("Grid search ARIMA: ninguna configuración pudo ajustarse.")

    df_resumen = pd.DataFrame(resultados).sort_values(["window", "rmse"]).reset_index(drop=True)

    for w in df_resumen["window"].unique():
        df_w   = df_resumen[df_resumen["window"] == w].copy()
        ruta_w = ruta_base.replace(".csv", f"_w{w}_grid.csv")
        df_w.to_csv(ruta_w, index=False)
        print(f"[OK] Grid ventana {w}: {ruta_w}")

    df_mejores = (
        df_resumen
        .groupby("window", sort=True)
        .first()
        .reset_index()
    )
    ruta_mejores = ruta_base.replace(".csv", "_mejores_por_ventana.csv")
    df_mejores.to_csv(ruta_mejores, index=False)

    print(f"[OK] Mejor config global: ARIMA{best_cfg[1:]} window={best_cfg[0]}  RMSE={best_score:.4f}")
    print(f"\n{df_mejores[['window','p','d','q','rmse','mae']].to_string(index=False)}")

    return best_cfg, df_mejores


def boxplot_ventanas(X_train, y_train, df_distritos_info, df_mejores, ruta_base):
    """Boxplot de distribución de RMSE por distrito — mejor config de cada ventana."""
    df_ord = df_mejores.sort_values("window").reset_index(drop=True)

    datos  = []
    labels = []

    for _, row in df_ord.iterrows():
        w, p, d, q = int(row["window"]), int(row["p"]), int(row["d"]), int(row["q"])
        res = evaluar_arima(X_train, y_train, df_distritos_info, w, (p, d, q))
        datos.append(res["rmse_distritos"])
        labels.append(f"w={w}\nARIMA({p},{d},{q})")

    n = len(datos)
    _, ax = plt.subplots(figsize=(max(6, n * 2.5), 5))
    ax.boxplot(datos, labels=labels, patch_artist=True)
    ax.set_xlabel("Ventana")
    ax.set_ylabel("RMSE por distrito")
    ax.set_title("Distribución de errores — mejor config por ventana")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    ruta_boxplot = ruta_base.replace(".csv", "boxplot_ventanas.png")
    plt.savefig(ruta_boxplot, dpi=150)
    plt.close()
    print(f"[OK] Boxplot ventanas: {ruta_boxplot}")


def analisis_departamentos(X_train, y_train, df_distritos_info, df_mejores, ruta_base):
    """Exporta CSV por departamento para la mejor config de cada ventana."""
    for _, row in df_mejores.sort_values("window").iterrows():
        w, p, d, q = int(row["window"]), int(row["p"]), int(row["d"]), int(row["q"])
        res = evaluar_arima(X_train, y_train, df_distritos_info, w, (p, d, q))

        ruta_dep = ruta_base.replace(".csv", f"_w{w}_departamento.csv")
        res["df_dep"].to_csv(ruta_dep, index=False)
        print(f"[OK] Departamentos w={w}: {ruta_dep}")


def pipeline_arima(X_train, y_train, df_distritos_info, ruta_base, p_values, d_values, q_values, window_values):
    """
    Pipeline completo ARIMA. Ejecuta en orden:
      1. Grid search: todas las combinaciones (p,d,q) × ventanas
      2. Boxplot: distribución de RMSE por distrito para la mejor config de cada ventana
      3. Análisis por departamento: CSV por departamento para cada ventana
      4. Exporta resultados de la mejor configuración global

    Returns:
        dict con modelo, rmse, mae, y_pred, df_metricas, df_dep, rmse_distritos
    """

    print("\n" + "=" * 60)
    print(" PIPELINE ARIMA ")
    print("=" * 60)

    best_cfg, df_mejores = grid_search(
        X_train, y_train, df_distritos_info,
        p_values, d_values, q_values, window_values, ruta_base,
    )

    print("\n[INFO] Generando boxplot por ventana...")
    boxplot_ventanas(X_train, y_train, df_distritos_info, df_mejores, ruta_base)

    print("\n[INFO] Análisis por departamento (todas las ventanas)...")
    analisis_departamentos(X_train, y_train, df_distritos_info, df_mejores, ruta_base)

    best_w, best_p, best_d, best_q = best_cfg
    ruta_best = ruta_base.replace(".csv", f"_best_w{best_w}_p{best_p}_d{best_d}_q{best_q}.csv")

    print(f"\n[INFO] Exportando mejor config: w={best_w} ARIMA({best_p},{best_d},{best_q})")
    resultado = evaluar_arima(
        X_train, y_train, df_distritos_info,
        best_w, (best_p, best_d, best_q),
        exportar=True, ruta=ruta_best,
    )

    ruta_ypred = ruta_base.replace(".csv", "_best_ypred.npy")
    np.save(ruta_ypred, resultado["y_pred"])
    print(f"[OK] y_pred ARIMA: {ruta_ypred}")

    print(f"[RESULT] RMSE={resultado['rmse']:.4f}  MAE={resultado['mae']:.4f}")
    return resultado
