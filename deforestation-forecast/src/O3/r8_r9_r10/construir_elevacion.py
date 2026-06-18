import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterstats import zonal_stats

from O3.config import (
    CRS_GEOG,
    ELEVACION_CSV,
    ELEVACION_META,
    ELEVACION_MOSAIC,
    GPKG_COL_DEPARTAMENTO,
    GPKG_COL_DISTRITO,
    GPKG_COL_GEOCODE,
    SRTM_NODATA,
    SRTM_TILES,
)
from O3.utils import guardar_csv, guardar_metadatos

logger = logging.getLogger(__name__)


def crear_mosaico_dem(distritos_gdf):
    """Fusiona los tiles SRTM recortados a la extensión de los distritos."""
    logger.info(f"  Creando mosaico DEM de {len(SRTM_TILES)} tiles SRTM...")
    logger.info("  (puede tomar varios minutos — solo se ejecuta una vez)")

    bounds = tuple(distritos_gdf.total_bounds)  # xmin, ymin, xmax, ymax

    datasets = [rasterio.open(p) for p in SRTM_TILES]
    try:
        mosaic, out_transform = merge(datasets, bounds=bounds, nodata=SRTM_NODATA)
        out_meta = datasets[0].meta.copy()
    finally:
        for ds in datasets:
            ds.close()
    out_meta.update(
        {
            "driver":    "GTiff",
            "height":    mosaic.shape[1],
            "width":     mosaic.shape[2],
            "transform": out_transform,
            "crs":       CRS_GEOG,
            "nodata":    SRTM_NODATA,
            "compress":  "lzw",
        }
    )
    with rasterio.open(ELEVACION_MOSAIC, "w", **out_meta) as dst:
        dst.write(mosaic)

    logger.info(f"  [OK] Mosaico DEM guardado: {ELEVACION_MOSAIC}")
    logger.info(f"       Dimensiones: {mosaic.shape[2]} × {mosaic.shape[1]} px")


def construir_elevacion(distritos_gdf):
    """Calcula elevación media (m) por distrito usando el DEM SRTM 30m.

    Crea un mosaico de los tiles SRTM recortado a la extensión de los
    distritos (una sola vez) y luego aplica zonal_stats. Output: una fila
    por distrito, sin columna anio.
    """
    if os.path.exists(ELEVACION_CSV) and os.path.exists(ELEVACION_META):
        logger.info("[SKIP] Elevación ya calculada — cargando CSV existente")
        return pd.read_csv(ELEVACION_CSV)

    if not SRTM_TILES:
        raise RuntimeError("No se encontraron tiles SRTM — revisa config.py")

    if not os.path.exists(ELEVACION_MOSAIC):
        crear_mosaico_dem(distritos_gdf)
    else:
        logger.info(f"  Mosaico DEM ya existe: {ELEVACION_MOSAIC}")

    logger.info("Calculando elevación media por distrito...")
    stats = zonal_stats(
        distritos_gdf,
        ELEVACION_MOSAIC,
        stats=["mean", "median", "std", "min", "max", "count"],
        nodata=SRTM_NODATA,
    )

    registros = []
    for stat, (_, row) in zip(stats, distritos_gdf.iterrows()):
        mean_val = stat.get("mean")
        median_val = stat.get("median")
        std_val = stat.get("std")

        registros.append({
            "geocode":        row[GPKG_COL_GEOCODE],
            "departamento":   row[GPKG_COL_DEPARTAMENTO],
            "distrito":       row[GPKG_COL_DISTRITO],
            "elev_media_m":   mean_val if mean_val is not None else np.nan,
            "elev_mediana_m": median_val if median_val is not None else np.nan,
            "elev_std_m":     std_val if std_val is not None else np.nan,
            "elev_min_m":     stat.get("min"),
            "elev_max_m":     stat.get("max"),
            "elev_count_px":  stat.get("count"),
        })

    df = pd.DataFrame(registros)

    n_sin_datos = int(df["elev_media_m"].isna().sum())
    if n_sin_datos:
        logger.warning(f"  {n_sin_datos} distritos sin cobertura SRTM válida (elev_media_m = NaN)")

    guardar_csv(df, ELEVACION_CSV)
    guardar_metadatos(
        {
            "variable":              "elevacion",
            "fuente":                "SRTM SRTMGL1 30m (NASA/USGS)",
            "n_tiles_srtm":          len(SRTM_TILES),
            "mosaico":               ELEVACION_MOSAIC,
            "nodata_srtm":           SRTM_NODATA,
            "n_distritos":           len(df),
            "n_distritos_sin_datos": n_sin_datos,
            "elev_media_global_m":   df["elev_media_m"].mean(skipna=True),
            "elev_mediana_global_m": df["elev_mediana_m"].median(skipna=True),
            "elev_std_promedio_m":   df["elev_std_m"].mean(skipna=True),
            "elev_min_global_m":     df["elev_min_m"].min(skipna=True),
            "elev_max_global_m":     df["elev_max_m"].max(skipna=True),
            "fecha_procesamiento":   datetime.now().isoformat(),
        },
        ELEVACION_META,
    )
    logger.info(
        f"[OK] Elevación calculada: {len(df)} distritos, "
        f"media global {df['elev_media_m'].mean():.0f} m"
        + (f", {n_sin_datos} sin cobertura SRTM" if n_sin_datos else "")
    )
    return df
