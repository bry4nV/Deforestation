"""
R13 — Pronósticos de pérdida de bosque en las 20 zonas de generalización
espacial, con el CNN1D extendido (O3/R11) — el modelo final del proyecto.

Mismo protocolo walk-forward oracle/teacher-forced que ya usa todo el
proyecto (_evaluar_geografico en R11): en cada uno de los 5 años de prueba
(2020-2024) se pronostica un paso adelante usando el valor real del año
anterior, nunca la propia predicción retroalimentada. Así el RMSE/MAE de
generalización es directamente comparable contra el RMSE/MAE "en muestra"
ya reportado por R11.
"""

import logging
import os

import numpy as np
import pandas as pd

from O1.config import PIXEL_AREA_KM2
from O3.r11.cargar_panel import cargar_panel
from O3.r11.escalador import cargar_escalador, escalar_panel
from O3.r11.pipeline_cnn import _evaluar_geografico as evaluar_cnn_geografico
from O3.r11.pronostico_2025 import generar_predicciones
from O3.utils import guardar_csv
from O3.r11.transformaciones import aplicar_transformaciones

from O4.config import (
    ANIO_FIN,
    ANIO_INICIO,
    PANEL_GENERALIZACION_CSV,
    R11_CNN_MODEL_PTH,
    R13_DIR,
    TAMANIO_ENTRENAMIENTO,
)
from O4.r12_r13_r14.modelos import cargar_cnn_entrenado

logger = logging.getLogger(__name__)

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


def pronosticar_2025_generalizacion() -> pd.DataFrame:
    """
    Pronóstico 2025 (sin reentrenamiento, ancla = pct_bosque_real_2024) para
    las 20 zonas de generalización espacial. Produce deforestacion_2025.csv
    con predicciones, fracción de cobertura perdida y área absoluta en km².
    """
    ruta = os.path.join(R13_DIR, "cnn_generalizacion_deforestacion_2025.csv")
    if os.path.exists(ruta):
        logger.info("[SKIP] Pronóstico 2025 de generalización ya existe.")
        return pd.read_csv(ruta, dtype={"geocode": str})

    logger.info("=" * 70)
    logger.info("CNN (R11) — Pronóstico 2025 en las 20 zonas de generalización")

    panel, df_distritos_info = cargar_panel(ruta_csv=PANEL_GENERALIZACION_CSV)
    panel_transformado = aplicar_transformaciones(panel)
    escalador = cargar_escalador()
    panel_escalado = escalar_panel(panel_transformado, escalador)

    df = generar_predicciones(
        panel_escalado, panel, df_distritos_info,
        rutas_modelo={"cnn": R11_CNN_MODEL_PTH}, escalador=escalador,
        anio_anchor=ANIO_FIN,
    )

    df["deforestacion_2025_cnn"] = df["pct_bosque_real_2024"] - df[f"cnn_pred_{ANIO_FIN + 1}"]

    panel_csv = pd.read_csv(PANEL_GENERALIZACION_CSV, dtype={"geocode": str})
    pix_2024 = (
        panel_csv[panel_csv["anio"] == ANIO_FIN][["geocode", "pix_total"]]
        .rename(columns={"pix_total": "pix_total_2024"})
    )
    df = df.merge(pix_2024, on="geocode", how="left")
    df["area_km2_2024"] = df["pix_total_2024"] * PIXEL_AREA_KM2
    df["deforestacion_2025_cnn_km2"] = df["area_km2_2024"] * df["deforestacion_2025_cnn"]
    df = df.drop(columns=["pix_total_2024"])

    guardar_csv(df, ruta)
    logger.info(f"[OK] {ruta} — {len(df)} zonas de generalización.")
    return df


def pipeline_pronosticos() -> dict:
    resultado = pronosticar_cnn()
    pronosticar_2025_generalizacion()
    return resultado
