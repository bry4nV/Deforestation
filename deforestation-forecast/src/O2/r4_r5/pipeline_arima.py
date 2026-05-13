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

    for w in sorted(df_resultados["window"].unique()):
        df_w = (
            df_resultados[df_resultados["window"] == w]
            .sort_values("rmse")
            .reset_index(drop=True)
        )
        ruta_w = ruta_base.replace(".csv", f"_w{w}_grid.csv")
        df_w.to_csv(ruta_w, index=False)
        print(f"[OK] Grid ventana {w}:        {ruta_w}")

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


def guardar_departamentos_por_ventana(
    X_train,
    y_train,
    df_distritos_info,
    df_mejores_por_ventana,
    ruta_base,
):
    for _, row in df_mejores_por_ventana.sort_values("window").iterrows():
        w = parse_ventana(row["window"])
        w_tag = tag_ventana(w)
        p = int(row["p"])
        d = int(row["d"])
        q = int(row["q"])

        res = evaluar_arima(X_train, y_train, df_distritos_info, w, (p, d, q))

        ruta_dep = ruta_base.replace(".csv", f"_w{w_tag}_departamento.csv")
        res["df_departamento"].to_csv(ruta_dep, index=False)

        print(f"[OK] Departamentos w={w_tag}:     {ruta_dep}")


def guardar_mejor_resultado(resultado, ruta_base):
    ruta_distrito = ruta_base.replace(".csv", "_mejor_distrito.csv")
    resultado["df_distrito"].to_csv(ruta_distrito, index=False)
    print(f"[OK] Por distrito:            {ruta_distrito}")

    ruta_departamento = ruta_base.replace(".csv", "_mejor_departamento.csv")
    resultado["df_departamento"].to_csv(ruta_departamento, index=False)
    print(f"[OK] Por departamento:        {ruta_departamento}")

    ruta_global = ruta_base.replace(".csv", "_global.csv")
    pd.DataFrame([{
        "modelo": resultado["modelo"],
        "rmse": round(resultado["rmse"], 6),
        "mae": round(resultado["mae"], 6),
    }]).to_csv(ruta_global, index=False)
    print(f"[OK] Métricas globales:       {ruta_global}")

    ruta_ypred = ruta_base.replace(".csv", "_mejor_ypred.npy")
    np.save(ruta_ypred, resultado["y_pred"])
    print(f"[OK] y_pred walk-forward:     {ruta_ypred}")


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
    print("\n[INFO] Pipeline ARIMA — búsqueda de hiperparámetros")
    print("=" * 60)

    df_resultados = grid_search_arima(
        X_train,
        y_train,
        df_distritos_info,
        p_values,
        d_values,
        q_values,
        window_values,
    )

    df_mejores_por_ventana = guardar_resultados_grid(df_resultados, ruta_base)

    print("\n[INFO] Generando boxplot por ventana...")
    graficar_boxplot_ventanas(
        X_train,
        y_train,
        df_distritos_info,
        df_mejores_por_ventana,
        ruta_base,
    )

    print("\n[INFO] Generando métricas por departamento por ventana...")
    guardar_departamentos_por_ventana(
        X_train,
        y_train,
        df_distritos_info,
        df_mejores_por_ventana,
        ruta_base,
    )

    mejor_fila = df_resultados.iloc[0]
    best_w = parse_ventana(mejor_fila["window"])
    best_w_tag = tag_ventana(best_w)
    best_p = int(mejor_fila["p"])
    best_d = int(mejor_fila["d"])
    best_q = int(mejor_fila["q"])

    print(
        "\n[INFO] Evaluando mejor configuración global: "
        f"w={best_w_tag} ARIMA({best_p},{best_d},{best_q})"
    )

    mejor_resultado = evaluar_arima(
        X_train,
        y_train,
        df_distritos_info,
        best_w,
        (best_p, best_d, best_q),
    )

    guardar_mejor_resultado(mejor_resultado, ruta_base)

    print(
        f"[OK] Mejor config: {mejor_resultado['modelo']}  "
        f"RMSE={mejor_resultado['rmse']:.4f}  "
        f"MAE={mejor_resultado['mae']:.4f}"
    )

    return {
        "modelo": mejor_resultado["modelo"],
        "rmse": mejor_resultado["rmse"],
        "mae": mejor_resultado["mae"],
        "y_pred": mejor_resultado["y_pred"],
    }
