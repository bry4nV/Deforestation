"""
R12 — Conjunto de datos de las nuevas zonas para evaluar generalización
espacial.

No construye nada desde cero: O1 ya separó 20 distritos como
"generalización espacial" (nunca usados en el entrenamiento de O2 ni de
O3/R11) y O3 ya les integró las 6 variables locales + pct_bosque. Este
módulo solo verifica y documenta, con números concretos, los 2 IOV de R12:

  1. Las nuevas zonas no se usaron en el entrenamiento del modelo.
  2. Datos de entrada completos para el 100% de las nuevas zonas.
"""

import logging
import os

import geopandas as gpd
import pandas as pd

from O3.r11.cargar_panel import cargar_panel
from O3.utils import guardar_csv

from O4.config import (
    DISTRITOS_GENERALIZACION_GPKG,
    PANEL_ENTRENAMIENTO_CSV,
    PANEL_GENERALIZACION_CSV,
    REPORTE_R12_CSV,
)

logger = logging.getLogger(__name__)


def construir_dataset_generalizacion() -> pd.DataFrame:
    if os.path.exists(REPORTE_R12_CSV):
        logger.info("[SKIP] R12 ya verificado — cargando reporte existente")
        return pd.read_csv(REPORTE_R12_CSV)

    if not os.path.exists(DISTRITOS_GENERALIZACION_GPKG):
        raise RuntimeError(
            f"No se encontró el gpkg de generalización: {DISTRITOS_GENERALIZACION_GPKG}\n"
            "Ejecuta primero O1 (pipeline_seleccion_distritos / series_temporales)."
        )

    gdf_generalizacion = gpd.read_file(DISTRITOS_GENERALIZACION_GPKG)
    logger.info(f"Geometrías cargadas: {len(gdf_generalizacion)} distritos")

    # cargar_panel() ya valida internamente que no haya NaN en las columnas
    # predictoras (lanza RuntimeError si las hay) — si esto carga sin error,
    # el IOV de completitud ya queda demostrado, no solo asumido.
    panel_train, info_train = cargar_panel(ruta_csv=PANEL_ENTRENAMIENTO_CSV)
    panel_gen, info_gen = cargar_panel(ruta_csv=PANEL_GENERALIZACION_CSV)

    geocodes_train = set(info_train["geocode"])
    geocodes_gen = set(info_gen["geocode"])
    solapados = sorted(geocodes_train & geocodes_gen)

    n_zonas = len(geocodes_gen)
    n_anios = panel_gen.shape[1]
    n_filas = n_zonas * n_anios

    if solapados:
        raise RuntimeError(
            f"IOV de R12 incumplido: {len(solapados)} zonas de generalización "
            f"ya estaban en entrenamiento: {solapados}"
        )

    if len(gdf_generalizacion) != n_zonas:
        logger.warning(
            f"[WARN] El gpkg tiene {len(gdf_generalizacion)} geometrías pero el "
            f"panel tiene {n_zonas} distritos — revisar consistencia."
        )

    reporte = pd.DataFrame([{
        "n_zonas_generalizacion": n_zonas,
        "n_anios_por_zona": n_anios,
        "n_filas_panel": n_filas,
        "n_geocodes_solapados_con_entrenamiento": len(solapados),
        "pct_completitud_variables_entrada": 100.0,
        "n_geometrias_gpkg": len(gdf_generalizacion),
        "fuente_geometrias": DISTRITOS_GENERALIZACION_GPKG,
        "fuente_panel": PANEL_GENERALIZACION_CSV,
    }])

    guardar_csv(reporte, REPORTE_R12_CSV)
    logger.info(
        f"[OK] R12 verificado: {n_zonas} zonas, 0 solapadas con entrenamiento, "
        f"100% completas ({n_filas} filas sin NaN)."
    )
    return reporte
