import json
import warnings
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

def obtener_ventana(history, window_size):
    if window_size is None:
        return history
    return history[-window_size:]


def tag_ventana(window_size):
    return "full" if window_size is None else str(window_size)


def parse_ventana(valor):
    if str(valor) == "full":
        return None
    return int(valor)


def calcular_metricas(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return rmse, mae


def metricas_por_departamento(df_distritos_info, y_true_total, y_pred_total):
    departamentos = df_distritos_info["departamento"].values
    registros = []

    for dep in np.unique(departamentos):
        mask = departamentos == dep

        rmse, mae = calcular_metricas(
            y_true_total[mask],
            y_pred_total[mask],
        )

        registros.append({
            "departamento": dep,
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
        })

    return (
        pd.DataFrame(registros)
        .sort_values(["mae", "rmse"], ascending=False)
        .reset_index(drop=True)
    )


def evaluar_arima(X_train, y_train, df_distritos_info, window_size, order):
    p, d, q = order

    n_distritos = X_train.shape[0]
    horizonte = y_train.shape[1]

    y_pred_total = []
    registros = []
    fallbacks = 0

    for i in range(n_distritos):
        history = X_train[i, :, 0].tolist()
        preds = []

        for t in range(horizonte):
            ventana = obtener_ventana(history, window_size)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    modelo = ARIMA(ventana, order=order).fit()
                    yhat = float(modelo.forecast()[0])
            except Exception:
                yhat = float(ventana[-1])
                fallbacks += 1

            preds.append(yhat)
            history.append(float(y_train[i, t]))

        y_pred_total.append(preds)

        rmse_i, mae_i = calcular_metricas(y_train[i], preds)

        info = df_distritos_info.iloc[i]
        registros.append({
            "geocode": info["geocode"],
            "departamento": info["departamento"],
            "distrito": info["distrito"],
            "rmse": round(rmse_i, 6),
            "mae": round(mae_i, 6),
        })

    y_pred_total = np.array(y_pred_total)

    rmse_global, mae_global = calcular_metricas(y_train, y_pred_total)

    if fallbacks:
        print(f"  [WARN] {fallbacks} predicciones cayeron a persistencia.")

    df_distrito = (
        pd.DataFrame(registros)
        .sort_values(["mae", "rmse"], ascending=False)
        .reset_index(drop=True)
    )

    df_departamento = metricas_por_departamento(
        df_distritos_info,
        y_train,
        y_pred_total,
    )

    w_tag = tag_ventana(window_size)
    modelo = f"ARIMA_WF_w{w_tag}_p{p}_d{d}_q{q}"

    return {
        "modelo": modelo,
        "window": w_tag,
        "p": p,
        "d": d,
        "q": q,
        "rmse": rmse_global,
        "mae": mae_global,
        "y_pred": y_pred_total,
        "df_distrito": df_distrito,
        "df_departamento": df_departamento,
        "rmse_distritos": df_distrito["rmse"].values,
    }


def grid_search_arima(
    X_train,
    y_train,
    df_distritos_info,
    p_values,
    d_values,
    q_values,
    window_values,
):
    resultados = []

    combinaciones = list(product(window_values, p_values, d_values, q_values))
    total = len(combinaciones)

    for idx, (w, p, d, q) in enumerate(combinaciones, start=1):
        order = (p, d, q)

        print(f"[{idx}/{total}] ARIMA{order} | window={w}")

        try:
            res = evaluar_arima(X_train, y_train, df_distritos_info, w, order)
            resultados.append({
                "modelo": res["modelo"],
                "window": tag_ventana(w),
                "p": p,
                "d": d,
                "q": q,
                "rmse": round(res["rmse"], 6),
                "mae": round(res["mae"], 6),
            })

            print(f"  RMSE={res['rmse']:.4f}  MAE={res['mae']:.4f}")

        except Exception as e:
            print(f"  [WARN] ARIMA{order} w={w} falló: {e}")

    if not resultados:
        raise RuntimeError("Grid search ARIMA: ninguna configuración pudo ajustarse.")

    return (
        pd.DataFrame(resultados)
        .sort_values("rmse")
        .reset_index(drop=True)
    )


def guardar_resultados_grid(df_resultados, ruta_base):
    ruta_resultados = ruta_base.replace(".csv", "_resultados.csv")
    df_resultados.to_csv(ruta_resultados, index=False)
    print(f"[OK] Resultados ARIMA guardados: {ruta_resultados}")

    df_mejores_por_ventana = (
        df_resultados
        .sort_values("rmse")
        .groupby("window", sort=True)
        .first()
        .reset_index()
    )

    ruta_mejores = ruta_base.replace(".csv", "_mejores_por_ventana.csv")
    df_mejores_por_ventana.to_csv(ruta_mejores, index=False)
    print(f"[OK] Mejores por ventana:     {ruta_mejores}")

    return df_mejores_por_ventana


def graficar_boxplot_ventanas(
    X_train,
    y_train,
    df_distritos_info,
    df_mejores_por_ventana,
    ruta_base,
):
    datos = []
    labels = []

    for _, row in df_mejores_por_ventana.sort_values("window").iterrows():
        w = parse_ventana(row["window"])
        w_tag = tag_ventana(w)
        p = int(row["p"])
        d = int(row["d"])
        q = int(row["q"])

        res = evaluar_arima(X_train, y_train, df_distritos_info, w, (p, d, q))

        datos.append(res["rmse_distritos"])
        labels.append(f"w={w_tag}\nARIMA({p},{d},{q})")

    n = len(datos)

    fig, ax = plt.subplots(figsize=(max(6, n * 2.5), 5))
    ax.boxplot(datos, labels=labels, patch_artist=True)
    ax.set_xlabel("Ventana")
    ax.set_ylabel("RMSE por distrito")
    ax.set_title("Distribución de errores — mejor config por ventana")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()

    ruta_boxplot = ruta_base.replace(".csv", "_boxplot_ventanas.png")
    fig.savefig(ruta_boxplot, dpi=150)
    plt.close(fig)

    print(f"[OK] Boxplot ventanas:        {ruta_boxplot}")


# ============================================================
# FASE 1: Grid search exploratorio ARIMA
# ============================================================

def pipeline_arima(
    X_train,
    y_train,
    df_distritos_info,
    ruta_base,
    p_values,
    d_values,
    q_values,
    window_values,
):
    """
    Fase 1: búsqueda exploratoria ARIMA. Genera:
    - _resultados.csv
    - _top5_configuraciones.csv
    - _mejores_por_ventana.csv
    - _boxplot_ventanas.png

    No evalúa la configuración final. No guarda modelo.
    """
    print("\n[INFO] Pipeline ARIMA — Fase 1: búsqueda exploratoria")
    print("=" * 60)

    df_resultados = grid_search_arima(
        X_train, y_train, df_distritos_info,
        p_values, d_values, q_values, window_values,
    )

    df_mejores_por_ventana = guardar_resultados_grid(df_resultados, ruta_base)

    df_top5 = df_resultados.head(5).copy()
    ruta_top5 = ruta_base.replace(".csv", "_top5_configuraciones.csv")
    df_top5.to_csv(ruta_top5, index=False)
    print(f"[OK] Top 5 configuraciones guardadas: {ruta_top5}")

    print("\n[INFO] Generando boxplot por ventana...")
    graficar_boxplot_ventanas(
        X_train, y_train, df_distritos_info, df_mejores_por_ventana, ruta_base,
    )

    top1 = df_resultados.iloc[0]
    print("\n[OK] Top 1 exploratorio. Revisar CSVs antes de elegir config final:")
    print(f"     Modelo: ARIMA({int(top1['p'])},{int(top1['d'])},{int(top1['q'])}) | window={top1['window']}")
    print(f"     RMSE:   {top1['rmse']}")
    print(f"     MAE:    {top1['mae']}")

    return {
        "grid_resultados":     df_resultados,
        "top5":                df_top5,
        "mejores_por_ventana": df_mejores_por_ventana,
    }


# ============================================================
# FASE 2: Evaluación final con configuración elegida
# ============================================================

def construir_df_predicciones_arima(resultado, y_train, df_distritos_info, anios=None):
    n_distritos, horizonte = y_train.shape
    y_pred = resultado["y_pred"]
    modelo = resultado["modelo"]

    registros = []
    for i in range(n_distritos):
        info = df_distritos_info.iloc[i]
        for t in range(horizonte):
            y_true_val = float(y_train[i, t])
            y_pred_val = float(y_pred[i, t])
            error      = y_pred_val - y_true_val
            registros.append({
                "modelo":       modelo,
                "geocode":      info["geocode"],
                "departamento": info["departamento"],
                "distrito":     info["distrito"],
                "horizonte":    t + 1,
                "anio":         anios[t] if anios is not None else None,
                "y_true":       y_true_val,
                "y_pred":       y_pred_val,
                "error":        error,
                "abs_error":    abs(error),
                "squared_error": error ** 2,
            })

    df = pd.DataFrame(registros)
    if anios is None:
        df = df.drop(columns=["anio"])
    return df


def evaluar_config_final_arima(
    X_train,
    y_train,
    df_distritos_info,
    final_config,
    ruta_base,
    anios=None,
):
    """
    Fase 2: evaluación final ARIMA con la configuración elegida. Genera:
    - _final_config.json
    - _final_global.csv
    - _final_distrito.csv
    - _final_departamento.csv
    - _final_predicciones.csv
    - _final_ypred.npy
    """
    print("\n[INFO] ARIMA — Fase 2: evaluación final")
    print("=" * 60)

    window = parse_ventana(str(final_config["window"]))
    p      = int(final_config["p"])
    d      = int(final_config["d"])
    q      = int(final_config["q"])

    print(f"[INFO] Config: ARIMA({p},{d},{q}) | window={tag_ventana(window)}")

    resultado = evaluar_arima(X_train, y_train, df_distritos_info, window, (p, d, q))

    ruta_config = ruta_base.replace(".csv", "_final_config.json")
    config_dict = {
        "modelo": resultado["modelo"],
        "window": tag_ventana(window),
        "p":      p,
        "d":      d,
        "q":      q,
        "rmse":   round(resultado["rmse"], 6),
        "mae":    round(resultado["mae"],  6),
    }
    with open(ruta_config, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)
    print(f"[OK] Config final:         {ruta_config}")

    ruta_global = ruta_base.replace(".csv", "_final_global.csv")
    pd.DataFrame([{
        "modelo": resultado["modelo"],
        "rmse":   round(resultado["rmse"], 6),
        "mae":    round(resultado["mae"],  6),
    }]).to_csv(ruta_global, index=False)
    print(f"[OK] Métricas globales:    {ruta_global}")

    ruta_dist = ruta_base.replace(".csv", "_final_distrito.csv")
    resultado["df_distrito"].to_csv(ruta_dist, index=False)
    print(f"[OK] Por distrito:         {ruta_dist}")

    ruta_dep = ruta_base.replace(".csv", "_final_departamento.csv")
    resultado["df_departamento"].to_csv(ruta_dep, index=False)
    print(f"[OK] Por departamento:     {ruta_dep}")

    df_predicciones = construir_df_predicciones_arima(resultado, y_train, df_distritos_info, anios)

    ruta_pred = ruta_base.replace(".csv", "_final_predicciones.csv")
    df_predicciones.to_csv(ruta_pred, index=False)
    print(f"[OK] Predicciones largas:  {ruta_pred}")

    ruta_ypred = ruta_base.replace(".csv", "_final_ypred.npy")
    np.save(ruta_ypred, resultado["y_pred"])
    print(f"[OK] y_pred walk-forward:  {ruta_ypred}")

    print(f"\n[OK] Resultado final ARIMA:")
    print(f"     Modelo: {resultado['modelo']}")
    print(f"     RMSE:   {resultado['rmse']:.6f}")
    print(f"     MAE:    {resultado['mae']:.6f}")

    return {
        "modelo":           resultado["modelo"],
        "rmse":             resultado["rmse"],
        "mae":              resultado["mae"],
        "y_pred":           resultado["y_pred"],
        "config":           config_dict,
        "df_predicciones":  df_predicciones,
        "df_departamento":  resultado["df_departamento"],
        "df_distrito":      resultado["df_distrito"],
    }
