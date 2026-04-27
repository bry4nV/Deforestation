import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error


def pipeline_arima(
    X_train,
    y_train,
    df_distritos_info,
    ruta_modelo_arima,
    window_size=5,
    order=(1,1,1)
):
    """
    Modelo ARIMA con walk-forward y ventana deslizante.
    """

    print(f"[INFO] Ejecutando ARIMA (walk-forward, ventana={window_size})...")

    n_distritos = X_train.shape[0]
    horizonte = y_train.shape[1]

    y_pred_total = []
    registros = []

    # ------------------------------------------------------------------
    # 🔹 Loop por distrito
    # ------------------------------------------------------------------
    for i in range(n_distritos):

        history = X_train[i, :, 0].tolist()
        preds = []

        # ------------------------------------------------------------------
        # 🔹 Walk-forward
        # ------------------------------------------------------------------
        for t in range(horizonte):

            ventana = history[-window_size:]

            try:
                model = ARIMA(ventana, order=order)
                model_fit = model.fit()
                yhat = model_fit.forecast()[0]
            except:
                # fallback (persistencia)
                yhat = ventana[-1]

            preds.append(yhat)

            # actualizar con valor real
            history.append(y_train[i, t])

        y_pred_total.append(preds)

        # ------------------------------------------------------------------
        # 🔹 Métricas por distrito
        # ------------------------------------------------------------------
        rmse = np.sqrt(mean_squared_error(y_train[i], preds))
        mae = mean_absolute_error(y_train[i], preds)

        info = df_distritos_info.iloc[i]

        registros.append({
            "geocode": info["geocode"],
            "departamento": info["departamento"],
            "distrito": info["distrito"],
            "rmse": rmse,
            "mae": mae
        })

    y_pred_total = np.array(y_pred_total)

    # ------------------------------------------------------------------
    # 🔹 Métricas globales
    # ------------------------------------------------------------------
    rmse_global = np.sqrt(mean_squared_error(y_train, y_pred_total))
    mae_global = mean_absolute_error(y_train, y_pred_total)

    print(f"[RESULT] RMSE global: {rmse_global:.4f}")
    print(f"[RESULT] MAE global: {mae_global:.4f}")

    # ------------------------------------------------------------------
    # 🔹 DataFrames
    # ------------------------------------------------------------------
    df_metricas = pd.DataFrame(registros)

    df_dep = (
        df_metricas
        .groupby("departamento")[["rmse", "mae"]]
        .mean()
        .reset_index()
    )

    df_metricas = df_metricas.sort_values(by=["mae", "rmse"], ascending=False)
    df_dep = df_dep.sort_values(by=["mae", "rmse"], ascending=False)

    # ------------------------------------------------------------------
    # 🔹 Exportar
    # ------------------------------------------------------------------
    df_metricas.to_csv(ruta_modelo_arima, index=False)
    print(f"[OK] Métricas por distrito: {ruta_modelo_arima}")

    ruta_dep = ruta_modelo_arima.replace(".csv", "_departamento.csv")
    df_dep.to_csv(ruta_dep, index=False)
    print(f"[OK] Métricas por departamento: {ruta_dep}")

    ruta_global = ruta_modelo_arima.replace(".csv", "_global.csv")

    df_global = pd.DataFrame([{
        "modelo": f"ARIMA_WF_w{window_size}",
        "rmse": rmse_global,
        "mae": mae_global
    }])

    df_global.to_csv(ruta_global, index=False)
    print(f"[OK] Métricas globales: {ruta_global}")

    return {
        "modelo": f"ARIMA_WF_w{window_size}",
        "rmse": rmse_global,
        "mae": mae_global,
        "y_pred": y_pred_total
    }