"""
Verificación empírica de que el error de pronóstico sobre el nivel de
pct_bosque (reportado en la Tabla 25) es idéntico al error sobre la
deforestación derivada (variación interanual), bajo el protocolo
walk-forward de R6.

Esta identidad se cumple porque cada predicción usa como ancla el valor
REAL del año anterior, nunca una predicción encadenada del propio modelo:

    Δ_pred(t) = pct_real(t-1) - pct_pred(t)
    Δ_real(t) = pct_real(t-1) - pct_real(t)
    Δ_pred(t) - Δ_real(t) = pct_real(t) - pct_pred(t) = -error_nivel(t)

Esa cancelación del ancla es lo que hace que RMSE/MAE coincidan entre nivel
y deforestación. Esto depende de que el ancla sea siempre real -- ver
evaluar_arima() en pipeline_arima.py, evaluar_geografico() en
pipeline_mlp.py / pipeline_lstm.py / pipeline_cnn.py, y pipeline_persistencia()
en pipeline_persistencia.py: los cinco usan history.append(valor_real),
nunca history.append(predicción).

No reentrena ni recalcula nada -- solo lee las predicciones ya guardadas de
Fase 2 (las mismas que generaron la Tabla 25) y confirma la identidad fila
por fila.
"""

import os

import numpy as np
import pandas as pd

from O1.config import SERIES_ENTRENAMIENTO_DIR
from O2.config import ARIMA_DIR, CNN_DIR, COMPARACION_DIR, LSTM_DIR, MLP_DIR, PERSISTENCIA_DIR

ARCHIVOS_PREDICCIONES = {
    "Persistencia": os.path.join(PERSISTENCIA_DIR, "persistencia_resultados_predicciones.csv"),
    "ARIMA":        os.path.join(ARIMA_DIR, "arima_final_predicciones.csv"),
    "MLP":          os.path.join(MLP_DIR, "mlp_final_predicciones.csv"),
    "LSTM":         os.path.join(LSTM_DIR, "lstm_final_predicciones.csv"),
    "CNN1D":        os.path.join(CNN_DIR, "cnn_final_predicciones.csv"),
}


def verificar_identidad_deforestacion(ruta_series_real: str, tolerancia: float = 1e-9) -> pd.DataFrame:
    anchor_2019 = pd.read_csv(ruta_series_real, dtype={"geocode": str})
    anchor_2019 = (
        anchor_2019[anchor_2019["anio"] == 2019][["geocode", "pct_bosque"]]
        .rename(columns={"pct_bosque": "anchor"})
        .set_index("geocode")["anchor"]
    )

    resumen = []
    for nombre, ruta in ARCHIVOS_PREDICCIONES.items():
        if not os.path.exists(ruta):
            print(f"[SKIP] {nombre}: no existe {ruta}")
            continue

        df = (
            pd.read_csv(ruta, dtype={"geocode": str})
            .sort_values(["geocode", "horizonte"])
            .reset_index(drop=True)
        )

        df["anchor"] = df.groupby("geocode")["y_true"].shift(1)
        falta = df["horizonte"] == 1
        df.loc[falta, "anchor"] = df.loc[falta, "geocode"].map(anchor_2019)

        delta_pred = df["anchor"] - df["y_pred"]
        delta_real = df["anchor"] - df["y_true"]
        delta_error = delta_pred - delta_real

        identico = bool(np.allclose(delta_error, -df["error"], atol=tolerancia))

        rmse_nivel = float(np.sqrt(np.mean(df["error"] ** 2)))
        mae_nivel = float(np.mean(df["error"].abs()))
        rmse_deforestacion = float(np.sqrt(np.mean(delta_error ** 2)))
        mae_deforestacion = float(np.mean(delta_error.abs()))

        resumen.append({
            "modelo": nombre,
            "identidad_confirmada": identico,
            "rmse_nivel": round(rmse_nivel, 6),
            "rmse_deforestacion": round(rmse_deforestacion, 6),
            "mae_nivel": round(mae_nivel, 6),
            "mae_deforestacion": round(mae_deforestacion, 6),
            "filas_verificadas": len(df),
        })

    df_resumen = pd.DataFrame(resumen)
    print(df_resumen.to_string(index=False))

    if not df_resumen["identidad_confirmada"].all():
        raise AssertionError(
            "La identidad nivel = deforestación no se cumple para todos los modelos. "
            "Revisar si el walk-forward encadena predicciones propias en algún punto."
        )

    return df_resumen


def main():
    ruta_series = os.path.join(SERIES_ENTRENAMIENTO_DIR, "distritos_entrenamiento.csv")
    df_resumen = verificar_identidad_deforestacion(ruta_series)

    ruta_salida = os.path.join(COMPARACION_DIR, "verificacion_deforestacion.csv")
    df_resumen.to_csv(ruta_salida, index=False)
    print(f"\n[OK] {ruta_salida}")


if __name__ == "__main__":
    main()
