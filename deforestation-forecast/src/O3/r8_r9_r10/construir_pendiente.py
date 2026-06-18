import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform
from rasterio.warp import reproject as warp_reproject
from rasterstats import zonal_stats
from scipy.ndimage import convolve as _convolve

from O3.config import (
    CRS_PROYECTADO,
    ELEVACION_MOSAIC,
    GPKG_COL_DEPARTAMENTO,
    GPKG_COL_DISTRITO,
    GPKG_COL_GEOCODE,
    PENDIENTE_CSV,
    PENDIENTE_META,
    PENDIENTE_RASTER,
    SLOPE_NODATA,
    SRTM_NODATA,
)
from O3.utils import guardar_csv, guardar_metadatos

logger = logging.getLogger(__name__)


def calcular_slope_raster():
    """Calcula pendiente en grados con el algoritmo de Horn (1981).

    Mismo método que gdaldem slope, QGIS y ArcGIS: ventana 3×3 ponderada
    aplicada sobre el DEM reproyectado a UTM (píxeles cuadrados en metros).
    El mosaico puede superar 4 GB como float32 → se escribe como BigGeoTIFF.
    """
    logger.info(f"  Calculando raster de pendiente desde {ELEVACION_MOSAIC}...")
    logger.info(f"  Reproyectando DEM a {CRS_PROYECTADO} (píxeles en metros)...")

    with rasterio.open(ELEVACION_MOSAIC) as src:
        src_meta = src.meta.copy()

        transform_utm, width_utm, height_utm = calculate_default_transform(
            src.crs,
            CRS_PROYECTADO,
            src.width,
            src.height,
            *src.bounds,
        )

        dem_utm = np.full((height_utm, width_utm), np.nan, dtype=np.float64)

        warp_reproject(
            source=rasterio.band(src, 1),
            destination=dem_utm,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform_utm,
            dst_crs=CRS_PROYECTADO,
            src_nodata=SRTM_NODATA,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    px_m = abs(transform_utm.a)
    py_m = abs(transform_utm.e)

    logger.info(
        f"  DEM proyectado: {width_utm}×{height_utm} px — "
        f"píxel {px_m:.2f}×{py_m:.2f} m"
    )

    # Horn (1981): kernels ponderados 3×3 idénticos a gdaldem slope.
    # _convolve invierte el kernel, pero al elevar al cuadrado el signo no afecta
    # la magnitud de la pendiente.
    kern_dx = np.array([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]], dtype=np.float64) / (8.0 * px_m)
    kern_dy = np.array([[ 1., 2., 1.],
                        [ 0., 0., 0.],
                        [-1.,-2.,-1.]], dtype=np.float64) / (8.0 * py_m)

    dz_dx = _convolve(dem_utm, kern_dx, mode="nearest")
    dz_dy = _convolve(dem_utm, kern_dy, mode="nearest")

    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    slope = np.clip(slope, 0.0, 90.0)

    slope = np.where(
        np.isnan(dem_utm) | np.isnan(slope),
        SLOPE_NODATA,
        slope,
    )

    meta_utm = src_meta.copy()
    meta_utm.update({
        "driver":    "GTiff",
        "crs":       CRS_PROYECTADO,
        "transform": transform_utm,
        "width":     width_utm,
        "height":    height_utm,
        "dtype":     rasterio.float32,
        "nodata":    SLOPE_NODATA,
        "count":     1,
        "compress":  "lzw",
        "bigtiff":   "YES",
    })

    with rasterio.open(PENDIENTE_RASTER, "w", **meta_utm) as dst:
        dst.write(slope.astype(rasterio.float32), 1)

    logger.info(f"  [OK] Raster de pendiente guardado (BigGeoTIFF): {PENDIENTE_RASTER}")


def construir_pendiente(distritos_gdf):
    """Calcula estadísticas distritales de pendiente en grados.

    Usa el raster de pendiente derivado del DEM SRTM y reproyecta los distritos
    al CRS del raster antes de aplicar zonal_stats.
    """
    if os.path.exists(PENDIENTE_CSV) and os.path.exists(PENDIENTE_META):
        logger.info("[SKIP] Pendiente ya calculada — cargando CSV existente")
        return pd.read_csv(PENDIENTE_CSV)

    if not os.path.exists(ELEVACION_MOSAIC):
        raise RuntimeError(
            f"Mosaico DEM no encontrado: {ELEVACION_MOSAIC}\n"
            "Ejecuta primero construir_elevacion()."
        )

    if not os.path.exists(PENDIENTE_RASTER):
        calcular_slope_raster()
    else:
        try:
            with rasterio.open(PENDIENTE_RASTER) as _chk:
                _valido = _chk.count > 0 and _chk.width > 0
        except Exception:
            _valido = False
        if _valido:
            logger.info(f"  Raster de pendiente ya existe: {PENDIENTE_RASTER}")
        else:
            logger.warning(
                "  Raster de pendiente corrupto o incompleto — eliminando y recalculando"
            )
            os.remove(PENDIENTE_RASTER)
            calcular_slope_raster()

    logger.info("Calculando estadísticas de pendiente por distrito...")

    # El raster de pendiente está en CRS_PROYECTADO.
    # Por tanto, las geometrías deben estar en el mismo CRS antes de zonal_stats.
    logger.info(f"  CRS original distritos: {distritos_gdf.crs}")
    logger.info(f"  CRS raster pendiente / cálculo: {CRS_PROYECTADO}")

    distritos_zonal = distritos_gdf.to_crs(CRS_PROYECTADO)

    stats = zonal_stats(
        distritos_zonal,
        PENDIENTE_RASTER,
        stats=["mean", "median", "std", "min", "max", "count"],
        nodata=SLOPE_NODATA,
    )

    registros = []
    for stat, (_, row) in zip(stats, distritos_gdf.iterrows()):
        mean_val = stat.get("mean")
        median_val = stat.get("median")
        std_val = stat.get("std")

        registros.append({
            "geocode":                row[GPKG_COL_GEOCODE],
            "departamento":           row[GPKG_COL_DEPARTAMENTO],
            "distrito":               row[GPKG_COL_DISTRITO],
            "pendiente_media_deg":    mean_val if mean_val is not None else np.nan,
            "pendiente_mediana_deg":  median_val if median_val is not None else np.nan,
            "pendiente_std_deg":      std_val if std_val is not None else np.nan,
            "pendiente_min_deg":      stat.get("min"),
            "pendiente_max_deg":      stat.get("max"),
            "pendiente_count_px":     stat.get("count"),
        })

    df = pd.DataFrame(registros)

    n_sin_datos = int(df["pendiente_media_deg"].isna().sum())
    if n_sin_datos:
        logger.warning(
            f"  {n_sin_datos} distritos sin cobertura válida "
            f"(pendiente_media_deg = NaN)"
        )

    guardar_csv(df, PENDIENTE_CSV)

    guardar_metadatos(
        {
            "variable":                    "pendiente",
            "fuente":                      "SRTM SRTMGL1 30m (derivado de elevacion)",
            "metodo":                      "Horn (1981): DEM reproyectado a UTM + ventana 3×3 ponderada (gdaldem slope equivalente)",
            "unidad":                      "grados",
            "crs_original_distritos":       str(distritos_gdf.crs),
            "crs_calculo":                 str(CRS_PROYECTADO),
            "mosaico_dem":                 ELEVACION_MOSAIC,
            "raster_pendiente":            PENDIENTE_RASTER,
            "nodata_pendiente":            SLOPE_NODATA,
            "n_distritos":                 len(df),
            "n_distritos_sin_datos":       n_sin_datos,
            "pendiente_media_global_deg":   df["pendiente_media_deg"].mean(skipna=True),
            "pendiente_mediana_global_deg": df["pendiente_mediana_deg"].median(skipna=True),
            "pendiente_std_promedio_deg":   df["pendiente_std_deg"].mean(skipna=True),
            "pendiente_min_global_deg":     df["pendiente_min_deg"].min(skipna=True),
            "pendiente_max_global_deg":     df["pendiente_max_deg"].max(skipna=True),
            "fecha_procesamiento":          datetime.now().isoformat(),
        },
        PENDIENTE_META,
    )

    logger.info(
        f"[OK] Pendiente calculada: {len(df)} distritos, "
        f"media global {df['pendiente_media_deg'].mean(skipna=True):.2f}°"
        + (f", {n_sin_datos} sin cobertura válida" if n_sin_datos else "")
    )

    return df