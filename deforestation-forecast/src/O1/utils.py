import logging

from O1.config import (
    ANIOS,
    UMBRAL_AMAZONIA,
    N_DISTRITOS_ALTO_CAMBIO,
    TAMANIO_ENTRENAMIENTO,
    SEMILLA_SPLIT,
    PIXEL_AREA_KM2,
    TAMANIO_TILE,
)


def guardar_csv(gdf, ruta_csv):
    df = gdf.copy()
    if "geometry" in df.columns:
        df = df.drop(columns="geometry")
    df.to_csv(ruta_csv, index=False)


def log_config():
    """Registra los parámetros activos del pipeline para trazabilidad de ejecución."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("CONFIGURACIÓN ACTIVA DEL PIPELINE O1")
    logger.info("=" * 60)
    logger.info(f"  Años:                {min(ANIOS)} – {max(ANIOS)}  ({len(ANIOS)} años)")
    logger.info(f"  Umbral Amazonía:     {UMBRAL_AMAZONIA * 100:.0f}% del área distrital")
    logger.info(f"  Distritos top-N:     {N_DISTRITOS_ALTO_CAMBIO}")
    logger.info(f"  Split entrenamiento: {TAMANIO_ENTRENAMIENTO * 100:.0f}% / {(1 - TAMANIO_ENTRENAMIENTO) * 100:.0f}%")
    logger.info(f"  Semilla split:       {SEMILLA_SPLIT}")
    logger.info(f"  Área píxel:          {PIXEL_AREA_KM2} km²  (resolución 30 m)")
    logger.info(f"  Tamaño tile:         {TAMANIO_TILE} px")
    logger.info("=" * 60)
