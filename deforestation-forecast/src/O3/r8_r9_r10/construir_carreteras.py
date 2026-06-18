import logging
import os
from datetime import datetime

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union as _unary_union

from O3.config import (
    CARRETERAS_CSV,
    CARRETERAS_DEPARTAMENTAL_SHP,
    CARRETERAS_META,
    CARRETERAS_NACIONAL_SHP,
    CARRETERAS_VECINAL_SHP,
    CRS_PROYECTADO,
    GPKG_COL_DEPARTAMENTO,
    GPKG_COL_DISTRITO,
    GPKG_COL_GEOCODE,
)
from O3.utils import guardar_csv, guardar_metadatos

logger = logging.getLogger(__name__)


def construir_carreteras(distritos_gdf):
    """Calcula km y densidad de red vial por distrito (estático — MTC diciembre 2018).

    Concatena tres capas (nacional, departamental, vecinal), las intersecta con
    cada distrito en UTM y calcula:
      km_carreteras              — longitud total de carreteras dentro del distrito
      area_utm_km2               — área del distrito en UTM (insumo para densidad)
      densidad_carreteras_km_km2 — road density = km_carreteras / area_utm_km2

    Output: una fila por distrito, sin columna anio.
    """
    if os.path.exists(CARRETERAS_CSV) and os.path.exists(CARRETERAS_META):
        logger.info("[SKIP] Carreteras ya calculadas — cargando CSV existente")
        return pd.read_csv(CARRETERAS_CSV)

    logger.info("Calculando km de carreteras por distrito...")

    logger.info("  Cargando redes viales (nacional, departamental, vecinal)...")
    capas = [
        gpd.read_file(CARRETERAS_NACIONAL_SHP),
        gpd.read_file(CARRETERAS_DEPARTAMENTAL_SHP),
        gpd.read_file(CARRETERAS_VECINAL_SHP),
    ]
    carreteras_gdf = gpd.GeoDataFrame(
        pd.concat(capas, ignore_index=True),
        geometry="geometry",
        crs=capas[0].crs,
    )
    logger.info(f"  Total segmentos de carretera: {len(carreteras_gdf)}")

    distritos_utm  = distritos_gdf.to_crs(CRS_PROYECTADO)
    carreteras_utm = carreteras_gdf.to_crs(CRS_PROYECTADO)

    # Sanear geometrías inválidas antes del overlay
    carreteras_utm["geometry"] = carreteras_utm.geometry.make_valid()
    distritos_utm["geometry"]  = distritos_utm.geometry.make_valid()

    # Eliminar duplicados WKB exactos (segmentos idénticos entre capas o dentro de una capa)
    n_antes = len(carreteras_utm)
    wkb_series = carreteras_utm.geometry.apply(lambda g: g.wkb)
    carreteras_utm = carreteras_utm[~wkb_series.duplicated()].copy()
    n_dedup = n_antes - len(carreteras_utm)
    if n_dedup:
        logger.info(f"  Eliminados {n_dedup} segmentos duplicados (WKB exacto)")

    # Unificar la red topológicamente: fusiona tramos concurrentes para que la
    # longitud por distrito represente red única, no suma de capas superpuestas
    logger.info("  Unificando red vial (unary_union) — puede tomar varios minutos...")
    red_unificada = _unary_union(carreteras_utm.geometry.tolist())
    carreteras_utm = (
        gpd.GeoDataFrame(geometry=[red_unificada], crs=CRS_PROYECTADO)
        .explode(index_parts=False)
        .reset_index(drop=True)
    )
    logger.info(f"  Red unificada: {len(carreteras_utm)} segmentos")

    logger.info("  Intersectando red vial con distritos (puede tomar varios minutos)...")
    interseccion_raw = gpd.overlay(
        carreteras_utm[["geometry"]],
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
        .rename(columns={"geometry": "km_carreteras", GPKG_COL_GEOCODE: "geocode"})
    )

    # Área de cada distrito en km² (en UTM, mismo CRS que la intersección)
    area_por_distrito = (
        distritos_utm[[GPKG_COL_GEOCODE, "geometry"]]
        .assign(area_utm_km2=lambda g: g.geometry.area / 1e6)
        [[GPKG_COL_GEOCODE, "area_utm_km2"]]
        .rename(columns={GPKG_COL_GEOCODE: "geocode"})
    )

    # Left-merge para que los distritos sin carreteras queden con 0.0 explícito
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
    df["km_carreteras"]              = df["km_carreteras"].fillna(0.0)
    df["densidad_carreteras_km_km2"] = df["km_carreteras"] / df["area_utm_km2"]

    guardar_csv(df, CARRETERAS_CSV)
    guardar_metadatos(
        {
            "variable":                    "carreteras",
            "fuente":                      "MTC Red Vial Nacional/Departamental/Vecinal dic-2018",
            "crs_calculo":                 CRS_PROYECTADO,
            "n_segmentos_raw":             len(carreteras_gdf),
            "n_segmentos_dedup":           len(carreteras_gdf) - n_dedup,
            "n_duplicados_eliminados":     n_dedup,
            "metodo_unificacion":          "unary_union topológico tras drop_duplicates WKB",
            "n_distritos":                 len(df),
            "km_total":                    round(df["km_carreteras"].sum(), 2),
            "densidad_media":              round(df["densidad_carreteras_km_km2"].mean(), 6),
            "n_distritos_sin_carreteras":  int((df["km_carreteras"] == 0.0).sum()),
            "fecha_procesamiento":         datetime.now().isoformat(),
        },
        CARRETERAS_META,
    )
    logger.info(
        f"[OK] Carreteras calculadas: {len(df)} distritos, "
        f"{df['km_carreteras'].sum():.1f} km total, "
        f"densidad media {df['densidad_carreteras_km_km2'].mean():.4f} km/km²"
    )
    return df
