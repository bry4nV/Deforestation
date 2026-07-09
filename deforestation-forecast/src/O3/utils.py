import logging
import os
from datetime import datetime

import pandas as pd

from O3.config import (
    ANIOS,
    ANP_SHP,
    BASE_DIR,
    CARRETERAS_DEPARTAMENTAL_SHP,
    CARRETERAS_NACIONAL_SHP,
    CARRETERAS_VECINAL_SHP,
    CLASES_AGROPECUARIA,
    CLASES_RIOS_LAGOS,
    CLASES_URBANO,
    CRS_GEOG,
    CRS_PROYECTADO,
    DISTRITOS_ALTO_CAMBIO_GPKG,
    DISTRITOS_ENTRENAMIENTO_CSV,
    DISTRITOS_GENERALIZACION_CSV,
    LOGS_DIR,
    MAPBIOMAS_AMAZONIA_PATRON,
    O3_INTERIM_DIR,
    RIOS_SHP,
    SRTM_TILES,
    VARIABLES_LOCALES_RAW_DIR,
)

logger = logging.getLogger(__name__)


def guardar_csv(df, ruta):
    """Guarda DataFrame o GeoDataFrame como CSV eliminando la columna geometry si existe."""
    if hasattr(df, "drop") and "geometry" in df.columns:
        df = df.drop(columns=["geometry"])
    df.to_csv(ruta, index=False, encoding="utf-8")
    logger.info(f"[OK] CSV guardado: {ruta}")


def guardar_metadatos(meta_dict, ruta):
    """Guarda un diccionario de metadatos como CSV de una fila."""
    pd.DataFrame([meta_dict]).to_csv(ruta, index=False, encoding="utf-8")
    logger.info(f"[OK] Metadatos guardados: {ruta}")


def log_config():
    """Registra todos los parametros activos de config.py al inicio de la ejecucion."""
    logger.info("=" * 60)
    logger.info("Configuracion activa O3:")
    logger.info(f"  BASE_DIR:              {BASE_DIR}")
    logger.info(f"  O3_INTERIM_DIR:        {O3_INTERIM_DIR}")
    logger.info(f"  ANIOS:                 {ANIOS[0]}–{ANIOS[-1]} ({len(ANIOS)} anos)")
    logger.info(f"  CRS_PROYECTADO:        {CRS_PROYECTADO}")
    logger.info(f"  CRS_GEOG:              {CRS_GEOG}")
    logger.info(f"  CLASES_AGROPECUARIA:   {sorted(CLASES_AGROPECUARIA)}")
    logger.info(f"  CLASES_RIOS_LAGOS:     {sorted(CLASES_RIOS_LAGOS)}")
    logger.info(f"  CLASES_URBANO:         {sorted(CLASES_URBANO)}")
    logger.info(f"  Tiles SRTM:            {len(SRTM_TILES)} archivos")
    logger.info("=" * 60)


def validar_fuentes():
    """
    Verifica fuentes requeridas y reporta todas las rutas faltantes juntas.
    """
    fuentes = {
        "ANP shapefile":                      ANP_SHP,
        "Carreteras nacional shapefile":      CARRETERAS_NACIONAL_SHP,
        "Carreteras departamental shapefile": CARRETERAS_DEPARTAMENTAL_SHP,
        "Carreteras vecinal shapefile":       CARRETERAS_VECINAL_SHP,
        "Rios shapefile":                     RIOS_SHP,
        "Distritos entrenamiento CSV":        DISTRITOS_ENTRENAMIENTO_CSV,
        "Distritos generalizacion CSV":       DISTRITOS_GENERALIZACION_CSV,
        "Distritos alto cambio GPKG":         DISTRITOS_ALTO_CAMBIO_GPKG,
        "MapBiomas 1985 (primer ano)":        MAPBIOMAS_AMAZONIA_PATRON.format(anio=1985),
        "MapBiomas 2024 (ultimo ano)":        MAPBIOMAS_AMAZONIA_PATRON.format(anio=2024),
    }

    faltantes = []
    for nombre, ruta in fuentes.items():
        if not os.path.exists(ruta):
            faltantes.append(f"  - {nombre}: {ruta}")

    if not SRTM_TILES:
        faltantes.append(
            f"  - Tiles SRTM: ninguno encontrado en "
            f"{os.path.join(VARIABLES_LOCALES_RAW_DIR, 'elevacion')}"
        )

    if faltantes:
        raise RuntimeError(
            "Fuentes faltantes — verifica las rutas en config.py:\n"
            + "\n".join(faltantes)
        )

    logger.info(f"[OK] Todas las fuentes verificadas ({len(fuentes)} archivos + {len(SRTM_TILES)} tiles SRTM)")


def iniciar_log_archivo(nombre_pipeline: str) -> str:
    """Crea un log con timestamp en outputs/logs y lo conecta al root logger."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ruta_log = os.path.join(LOGS_DIR, f"{nombre_pipeline}_run_debug_{timestamp}.log")

    handler = logging.FileHandler(ruta_log, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(handler)

    logger.info(f"[OK] Log de esta corrida: {ruta_log}")
    return ruta_log
