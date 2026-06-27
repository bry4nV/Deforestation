"""
R13 — Pronósticos de pérdida de bosque en las 20 zonas de generalización
espacial, para ARIMA (O2) y CNN (O3/R11) — los dos modelos ya finalizados.

Mismo protocolo walk-forward oracle/teacher-forced que ya usa todo el
proyecto (evaluar_arima en O2, _evaluar_geografico en R11): en cada uno de
los 5 años de prueba (2020-2024) se pronostica un paso adelante usando el
valor real del año anterior, nunca la propia predicción retroalimentada.
Así el RMSE/MAE de generalización es directamente comparable contra el
RMSE/MAE "en muestra" ya reportado por O2 y R11.
"""

import logging
import os

import numpy as np
import pandas as pd

from O2.r4_r5.construir_dataset import cargar_series, construir_dataset_estadistico
from O2.r4_r5.pipeline_arima import construir_df_predicciones_arima, evaluar_arima
from O3.r11.cargar_panel import cargar_panel
from O3.r11.escalador import cargar_escalador, escalar_panel
from O3.r11.pipeline_cnn import _evaluar_geografico as evaluar_cnn_geografico
from O3.utils import guardar_csv
from O3.r11.transformaciones import aplicar_transformaciones

from O4.config import (
    ANIO_INICIO,
    ANIO_FIN,
    PANEL_GENERALIZACION_CSV,
    R13_DIR,
    TAMANIO_ENTRENAMIENTO,
)
from O4.r12_r13_r14.modelos import cargar_cnn_entrenado, obtener_orden_arima

logger = logging.getLogger(__name__)

HORIZONTE = (ANIO_FIN - ANIO_INICIO + 1) - TAMANIO_ENTRENAMIENTO  # 5 años: 2020-2024
ANIOS_TEST = list(range(ANIO_INICIO + TAMANIO_ENTRENAMIENTO, ANIO_FIN + 1))


def _rutas(nombre_modelo: str) -> dict:
    base = os.path.join(R13_DIR, f"{nombre_modelo}_generalizacion")
    return {
        "global": f"{base}_global.csv",
        "distrito": f"{base}_distrito.csv",
        "departamento": f"{base}_departamento.csv",
        "predicciones": f"{base}_predicciones.csv",
        "ypred": f"{base}_ypred.npy",
    }


def pronosticar_arima() -> dict:
    rutas = _rutas("arima")
    if os.path.exists(rutas["global"]):
        logger.info("[SKIP] Pronósticos ARIMA de generalización ya existen.")
        df_gbl = pd.read_csv(rutas["global"])
        row = df_gbl.iloc[0]
        return {"modelo": row["modelo"], "rmse": float(row["rmse"]), "mae": float(row["mae"]),
                "y_pred": np.load(rutas["ypred"]), "etiqueta": "ARIMA"}

    logger.info("=" * 70)
    logger.info("ARIMA — Pronósticos de generalización espacial (20 zonas nuevas)")

    series, df_distritos_info = cargar_series(PANEL_GENERALIZACION_CSV)
    X_train, y_train = construir_dataset_estadistico(series, train_size=TAMANIO_ENTRENAMIENTO, horizon=HORIZONTE)

    order, window = obtener_orden_arima()
    resultado = evaluar_arima(X_train, y_train, df_distritos_info, window, order)

    df_predicciones = construir_df_predicciones_arima(resultado, y_train, df_distritos_info, anios=ANIOS_TEST)

    guardar_csv(resultado["df_distrito"], rutas["distrito"])
    guardar_csv(resultado["df_departamento"], rutas["departamento"])
    guardar_csv(df_predicciones, rutas["predicciones"])
    np.save(rutas["ypred"], resultado["y_pred"])
    guardar_csv(pd.DataFrame([{
        "modelo": resultado["modelo"], "rmse": round(resultado["rmse"], 6), "mae": round(resultado["mae"], 6),
    }]), rutas["global"])

    logger.info(f"[OK] ARIMA generalización — RMSE={resultado['rmse']:.6f}  MAE={resultado['mae']:.6f}")
    return {"modelo": resultado["modelo"], "rmse": resultado["rmse"], "mae": resultado["mae"],
            "y_pred": resultado["y_pred"], "etiqueta": "ARIMA", "df_distrito": resultado["df_distrito"]}


def pronosticar_cnn() -> dict:
    rutas = _rutas("cnn")
    if os.path.exists(rutas["global"]):
        logger.info("[SKIP] Pronósticos CNN de generalización ya existen.")
        df_gbl = pd.read_csv(rutas["global"])
        row = df_gbl.iloc[0]
        return {"modelo": row["modelo"], "rmse": float(row["rmse"]), "mae": float(row["mae"]),
                "y_pred": np.load(rutas["ypred"]), "etiqueta": "CNN"}

    logger.info("=" * 70)
    logger.info("CNN (R11) — Pronósticos de generalización espacial (20 zonas nuevas)")

    panel, df_distritos_info = cargar_panel(ruta_csv=PANEL_GENERALIZACION_CSV)
    panel_transformado = aplicar_transformaciones(panel)

    escalador = cargar_escalador()
    panel_escalado = escalar_panel(panel_transformado, escalador)

    modelo, config = cargar_cnn_entrenado()
    window_size = int(config["window_size"])

    df_distrito, df_departamento, rmse_global, mae_global, y_pred_total, df_predicciones = evaluar_cnn_geografico(
        modelo, panel_escalado, df_distritos_info, window_size, TAMANIO_ENTRENAMIENTO, escalador,
        anios_test=ANIOS_TEST, modelo_nombre=f"{config['modelo']}_GENERALIZACION",
    )

    guardar_csv(df_distrito, rutas["distrito"])
    guardar_csv(df_departamento, rutas["departamento"])
    guardar_csv(df_predicciones, rutas["predicciones"])
    np.save(rutas["ypred"], y_pred_total)
    guardar_csv(pd.DataFrame([{
        "modelo": config["modelo"], "rmse": round(rmse_global, 6), "mae": round(mae_global, 6),
    }]), rutas["global"])

    logger.info(f"[OK] CNN generalización — RMSE={rmse_global:.6f}  MAE={mae_global:.6f}")
    return {"modelo": config["modelo"], "rmse": rmse_global, "mae": mae_global,
            "y_pred": y_pred_total, "etiqueta": "CNN", "df_distrito": df_distrito}


def pipeline_pronosticos() -> list:
    return [pronosticar_arima(), pronosticar_cnn()]
