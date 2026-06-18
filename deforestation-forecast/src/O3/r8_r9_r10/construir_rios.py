import logging
import os
from datetime import datetime

import geopandas as gpd
import pandas as pd

from O3.config import (
    CRS_PROYECTADO,
    GPKG_COL_DEPARTAMENTO,
    GPKG_COL_DISTRITO,
    GPKG_COL_GEOCODE,
    RIOS_CSV,
    RIOS_META,
    RIOS_SHP,
)
from O3.utils import guardar_csv, guardar_metadatos

logger = logging.getLogger(__name__)


def construir_rios(distritos_gdf):
    """Calcula km e densidad de red hidrográfica por distrito (estático — ANA).

    Intersecta los ejes de ríos (Rios.shp) con cada distrito en UTM y calcula:
      km_rios              — longitud total de ríos dentro del distrito
      area_utm_km2         — área del distrito en UTM (insumo para densidad)
      densidad_rios_km_km2 — drainage density = km_rios / area_utm_km2

    Output: una fila por distrito, sin columna anio.
    """
    if os.path.exists(RIOS_CSV) and os.path.exists(RIOS_META):
        logger.info("[SKIP] Ríos ya calculados — cargando CSV existente")
        return pd.read_csv(RIOS_CSV)

    logger.info("Calculando km de ríos por distrito...")

    rios_gdf = gpd.read_file(RIOS_SHP)
    logger.info(f"  Segmentos de ríos cargados: {len(rios_gdf)}")

    distritos_utm = distritos_gdf.to_crs(CRS_PROYECTADO)
    rios_utm      = rios_gdf.to_crs(CRS_PROYECTADO)

    # Sanear geometrías inválidas antes del overlay
    rios_utm["geometry"]      = rios_utm.geometry.make_valid()
    distritos_utm["geometry"] = distritos_utm.geometry.make_valid()

    logger.info("  Intersectando ríos con distritos...")
    interseccion_raw = gpd.overlay(
        rios_utm[["geometry"]],
        distritos_utm[[GPKG_COL_GEOCODE, "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    es_lineal     = interseccion_raw.geometry.geom_type.isin(["LineString", "MultiLineString"])
    n_descartadas = int((~es_lineal).sum())
    if n_descartadas:
        logger.info(
            f"  Descartadas {n_descartadas} geometrías degeneradas "
            f"(puntos en límites de distrito — esperado)"
        )
    interseccion = interseccion_raw[es_lineal].copy()
    km_por_distrito = (
        interseccion.groupby(GPKG_COL_GEOCODE)["geometry"]
        .apply(lambda g: g.length.sum() / 1000)
        .reset_index()
        .rename(columns={"geometry": "km_rios", GPKG_COL_GEOCODE: "geocode"})
    )

    # Área de cada distrito en km² (en UTM, mismo CRS que la intersección)
    area_por_distrito = (
        distritos_utm[[GPKG_COL_GEOCODE, "geometry"]]
        .assign(area_utm_km2=lambda g: g.geometry.area / 1e6)
        [[GPKG_COL_GEOCODE, "area_utm_km2"]]
        .rename(columns={GPKG_COL_GEOCODE: "geocode"})
    )

    # Left-merge para que los distritos sin ríos queden con 0.0 explícito
    base = (
        distritos_gdf[[GPKG_COL_GEOCODE, GPKG_COL_DEPARTAMENTO, GPKG_COL_DISTRITO]]
        .rename(columns={
            GPKG_COL_GEOCODE:      "geocode",
            GPKG_COL_DEPARTAMENTO: "departamento",
            GPKG_COL_DISTRITO:     "distrito",
        })
    )
    df = base.merge(km_por_distrito, on="geocode", how="left")
    df = df.merge(area_por_distrito,  on="geocode", how="left")
    df["km_rios"]              = df["km_rios"].fillna(0.0)
    df["densidad_rios_km_km2"] = df["km_rios"] / df["area_utm_km2"]

    guardar_csv(df, RIOS_CSV)
    guardar_metadatos(
        {
            "variable":              "rios",
            "fuente":                RIOS_SHP,
            "crs_calculo":           CRS_PROYECTADO,
            "n_segmentos":           len(rios_gdf),
            "n_distritos":           len(df),
            "km_total":              round(df["km_rios"].sum(), 2),
            "densidad_media":        round(df["densidad_rios_km_km2"].mean(), 6),
            "n_distritos_sin_rios":  int((df["km_rios"] == 0.0).sum()),
            "fecha_procesamiento":   datetime.now().isoformat(),
        },
        RIOS_META,
    )
    logger.info(
        f"[OK] Ríos calculados: {len(df)} distritos, "
        f"{df['km_rios'].sum():.1f} km total, "
        f"densidad media {df['densidad_rios_km_km2'].mean():.4f} km/km²"
    )
    return df
