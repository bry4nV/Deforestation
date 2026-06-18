import logging
import os
from datetime import datetime

import pandas as pd
from rasterstats import zonal_stats

from O3.config import (
    ANIOS,
    MAPBIOMAS_AMAZONIA_PATRON,
    CLASES_URBANO,
    CLASE_NO_OBSERVADO,
    URBANO_CSV,
    URBANO_META,
    GPKG_COL_GEOCODE,
    GPKG_COL_DEPARTAMENTO,
    GPKG_COL_DISTRITO,
)
from O3.utils import guardar_csv, guardar_metadatos

logger = logging.getLogger(__name__)


def construir_urbano(distritos_gdf):
    """Calcula la fraccion de infraestructura urbana (clase 24) por distrito y anio (1985-2024).

    Usa los rasters MapBiomas C3 de O1 (peru_amazonia_YYYY.tif).
    Denominador: pixeles validos = todos excepto clase 0 (fondo, excluido via nodata)
    y clase 27 (no observado).
    """
    if os.path.exists(URBANO_CSV) and os.path.exists(URBANO_META):
        logger.info("[SKIP] Urbano ya calculado — cargando CSV existente")
        return pd.read_csv(URBANO_CSV)

    n_distritos = len(distritos_gdf)
    logger.info(f"Calculando urbano 1985-2024 para {n_distritos} distritos...")

    registros = []
    for anio in ANIOS:
        ruta = MAPBIOMAS_AMAZONIA_PATRON.format(anio=anio)
        stats = zonal_stats(distritos_gdf, ruta, categorical=True, nodata=0)
        for stat, (_, row) in zip(stats, distritos_gdf.iterrows()):
            conteos = stat or {}
            pixeles_validos = sum(v for k, v in conteos.items() if k != CLASE_NO_OBSERVADO)
            pixeles_clase   = sum(conteos.get(c, 0) for c in CLASES_URBANO)
            pct = round(pixeles_clase / pixeles_validos, 6) if pixeles_validos > 0 else 0.0
            registros.append({
                "geocode":      row[GPKG_COL_GEOCODE],
                "departamento": row[GPKG_COL_DEPARTAMENTO],
                "distrito":     row[GPKG_COL_DISTRITO],
                "anio":         anio,
                "pct_urbano":   pct,
            })
        logger.info(f"  {anio}: {n_distritos} distritos")

    df = pd.DataFrame(registros)
    guardar_csv(df, URBANO_CSV)
    guardar_metadatos(
        {
            "variable":            "urbano",
            "fuente":              MAPBIOMAS_AMAZONIA_PATRON,
            "clases_mapbiomas":    str(sorted(CLASES_URBANO)),
            "n_distritos":         n_distritos,
            "n_anios":             len(ANIOS),
            "anio_inicio":         ANIOS[0],
            "anio_fin":            ANIOS[-1],
            "n_registros":         len(df),
            "fecha_procesamiento": datetime.now().isoformat(),
        },
        URBANO_META,
    )
    logger.info(f"[OK] Urbano calculado: {len(df)} registros")
    return df
