import logging
import os
from datetime import datetime

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

from O3.config import (
    ANIOS,
    ANP_ACP_COL_FECHA,
    ANP_ACP_SHP,
    ANP_ACR_COL_FECHA,
    ANP_ACR_SHP,
    ANP_COL_FECHA,
    ANP_COL_FECHA_STD,
    ANP_CSV,
    ANP_META,
    ANP_RESUMEN_ANUAL_CSV,
    ANP_SHP,
    ANP_ZR_COL_FECHA,
    ANP_ZR_SHP,
    CRS_PROYECTADO,
    GPKG_COL_DEPARTAMENTO,
    GPKG_COL_DISTRITO,
    GPKG_COL_GEOCODE,
)
from O3.utils import guardar_csv, guardar_metadatos

logger = logging.getLogger(__name__)

# (ruta, campo_fecha_original, tipo) para las cuatro capas ANP
CAPAS_ANP = [
    (ANP_SHP,     ANP_COL_FECHA,     "Nacional"),
    (ANP_ZR_SHP,  ANP_ZR_COL_FECHA,  "ZonaReservada"),
    (ANP_ACR_SHP, ANP_ACR_COL_FECHA, "ACR"),
    (ANP_ACP_SHP, ANP_ACP_COL_FECHA, "ACP"),
]


def cargar_y_estandarizar_anp():
    """Carga las cuatro capas ANP y devuelve (GeoDataFrame unificado, conteos_por_capa).

    Estandariza el campo de fecha a ANP_COL_FECHA_STD ('felec'), agrega una
    columna 'tipo' para trazabilidad, y filtra filas con fecha nula.
    Retorna también un dict {tipo: n_poligonos} con los conteos reales de cada capa.
    """
    partes = []
    conteos = {}
    for shp, col_fecha, tipo in CAPAS_ANP:
        gdf = gpd.read_file(shp)
        gdf[ANP_COL_FECHA_STD] = pd.to_datetime(gdf[col_fecha], errors="coerce")
        gdf["tipo"] = tipo
        gdf = gdf[[ANP_COL_FECHA_STD, "tipo", "geometry"]].dropna(subset=[ANP_COL_FECHA_STD])
        conteos[tipo] = len(gdf)
        partes.append(gdf)
        logger.info(f"  {tipo}: {len(gdf)} polígonos")

    combinado = gpd.GeoDataFrame(
        pd.concat(partes, ignore_index=True),
        geometry="geometry",
        crs=partes[0].crs,
    )
    return combinado, conteos


def construir_anp(distritos_gdf):
    """Calcula fracción de área de cada distrito cubierta por ANP por año (1985-2024).

    Unifica las cuatro capas SERNANP (Nacional, Zonas Reservadas, ACR, ACP) y
    acumula año a año las ANP establecidas hasta cada anio (felec.year <= anio).

    Solapamiento entre capas (p.ej. ACP dentro de ANP Nacional) se resuelve con
    unary_union por distrito, que produce la unión real sin doble conteo.

    Estrategia incremental:
        - Pre-calcula la intersección (todos los ANPs) × (todos los distritos)
          una sola vez → fragmentos con (felec, tipo, geocode, geometry).
        - Mantiene un dict {geocode → union_geometry} con el estado acumulado.
        - Cada año solo actualiza los distritos que tienen ANPs nuevas (felec.year == anio);
          los demás reutilizan la unión del año anterior sin recomputar.

    Output: 200 distritos × 40 años = 8 000 filas.
    """
    if os.path.exists(ANP_CSV) and os.path.exists(ANP_META):
        logger.info("[SKIP] ANP ya calculada — cargando CSV existente")
        return pd.read_csv(ANP_CSV)

    logger.info("Calculando fracción de ANP por distrito y año...")
    logger.info("  Cargando y estandarizando las cuatro capas ANP...")
    anp_gdf, conteos_por_capa = cargar_y_estandarizar_anp()
    logger.info(f"  Total polígonos ANP unificados: {len(anp_gdf)}")

    distritos_utm = distritos_gdf.to_crs(CRS_PROYECTADO)
    anp_utm       = anp_gdf.to_crs(CRS_PROYECTADO)

    anp_utm["geometry"]       = anp_utm.geometry.make_valid()
    distritos_utm["geometry"] = distritos_utm.geometry.make_valid()

    # Área de cada distrito en m²
    area_distritos = (
        distritos_utm.set_index(GPKG_COL_GEOCODE)["geometry"]
        .area.to_dict()
    )

    # Intersección total (todos los ANPs) × (todos los distritos) — una sola vez.
    logger.info("  Pre-calculando intersección ANP × distritos (una sola vez)...")
    anp_interseccion = gpd.overlay(
        distritos_utm[[GPKG_COL_GEOCODE, "geometry"]],
        anp_utm[[ANP_COL_FECHA_STD, "tipo", "geometry"]],
        how="intersection",
    )
    anp_interseccion["anio_est"] = anp_interseccion[ANP_COL_FECHA_STD].dt.year.clip(lower=ANIOS[0])
    logger.info(f"  Fragmentos de intersección: {len(anp_interseccion)}")

    # Lookups de metadatos por geocode
    geocodes       = list(distritos_gdf[GPKG_COL_GEOCODE])
    geocode_a_dep  = dict(zip(distritos_gdf[GPKG_COL_GEOCODE], distritos_gdf[GPKG_COL_DEPARTAMENTO]))
    geocode_a_dist = dict(zip(distritos_gdf[GPKG_COL_GEOCODE], distritos_gdf[GPKG_COL_DISTRITO]))

    # Pre-agrupar fragmentos por año de establecimiento para acceso O(1) en el loop
    fragmentos_por_anio = {
        anio_est: grupo
        for anio_est, grupo in anp_interseccion.groupby("anio_est")
    }

    # Estado acumulado: geocode → unión geométrica de todos los fragmentos activos
    # Se actualiza incrementalmente: solo cuando hay ANPs nuevas en ese distrito ese año.
    union_por_distrito: dict = {}

    registros = []
    for anio in ANIOS:
        # Fragmentos que se activan exactamente este año
        nuevos = fragmentos_por_anio.get(anio, None)

        if nuevos is not None and len(nuevos) > 0:
            for geocode, grupo in nuevos.groupby(GPKG_COL_GEOCODE):
                nuevas_geoms = list(grupo["geometry"])
                anterior = union_por_distrito.get(geocode)
                if anterior is not None:
                    union_por_distrito[geocode] = unary_union([anterior] + nuevas_geoms)
                else:
                    union_por_distrito[geocode] = unary_union(nuevas_geoms)

        for geocode in geocodes:
            area_total = area_distritos.get(geocode, 0)
            union_geom = union_por_distrito.get(geocode)
            area_anp   = union_geom.area if union_geom is not None else 0.0
            pct = round(area_anp / area_total, 6) if area_total > 0 else 0.0
            registros.append({
                "geocode":      geocode,
                "departamento": geocode_a_dep[geocode],
                "distrito":     geocode_a_dist[geocode],
                "anio":         anio,
                "pct_anp":      pct,
            })

        if anio % 10 == 0 or anio == ANIOS[-1]:
            logger.info(f"  Procesado hasta {anio}...")

    df = pd.DataFrame(registros)
    df["tiene_anp"] = (df["pct_anp"] > 0).astype(int)
    guardar_csv(df, ANP_CSV)

    # Resumen anual: expansión temporal de la cobertura ANP
    resumen_anual = (
        df.groupby("anio")
        .agg(
            n_distritos_con_anp   =("tiene_anp", "sum"),
            pct_distritos_con_anp =("tiene_anp", "mean"),
            pct_anp_promedio      =("pct_anp",   "mean"),
            pct_anp_max           =("pct_anp",   "max"),
        )
        .reset_index()
    )
    resumen_anual["n_distritos_con_anp"]   = resumen_anual["n_distritos_con_anp"].astype(int)
    resumen_anual["pct_distritos_con_anp"] = resumen_anual["pct_distritos_con_anp"].round(6)
    resumen_anual["pct_anp_promedio"]      = resumen_anual["pct_anp_promedio"].round(6)
    resumen_anual["pct_anp_max"]           = resumen_anual["pct_anp_max"].round(6)
    guardar_csv(resumen_anual, ANP_RESUMEN_ANUAL_CSV)

    # Estadísticos estructurales para los metadatos
    resumen_distrital      = df.groupby("geocode")["pct_anp"].max()
    n_distritos_alguna_vez = int(resumen_distrital.gt(0).sum())
    n_distritos_nunca      = int(resumen_distrital.eq(0).sum())
    n_registros_con_anp    = int(df["pct_anp"].gt(0).sum())
    n_registros_sin_anp    = int(df["pct_anp"].eq(0).sum())
    con_anp_mask           = df["pct_anp"] > 0
    anio_min_con_anp       = int(df.loc[con_anp_mask, "anio"].min()) if con_anp_mask.any() else None
    anio_max_con_anp       = int(df.loc[con_anp_mask, "anio"].max()) if con_anp_mask.any() else None

    guardar_metadatos(
        {
            "variable":                 "anp",
            "capas":                    str([t for _, _, t in CAPAS_ANP]),
            "n_poligonos_por_capa":     str(conteos_por_capa),
            "n_poligonos_total":        len(anp_gdf),
            "crs_calculo":              CRS_PROYECTADO,
            "metodo_solapamiento":      "unary_union incremental por distrito",
            "acp_fecad":                "ignorada — se asume renovacion continua",
            "n_distritos":              len(geocodes),
            "n_distritos_con_anp":      len(union_por_distrito),
            "n_distritos_alguna_vez_anp": n_distritos_alguna_vez,
            "n_distritos_nunca_anp":    n_distritos_nunca,
            "n_registros_con_anp":      n_registros_con_anp,
            "n_registros_sin_anp":      n_registros_sin_anp,
            "pct_anp_promedio_serie":   round(float(df["pct_anp"].mean()), 6),
            "pct_anp_max_serie":        round(float(df["pct_anp"].max()), 6),
            "anio_min_con_anp":         anio_min_con_anp,
            "anio_max_con_anp":         anio_max_con_anp,
            "resumen_anual_generado":   True,
            "n_anios":                  len(ANIOS),
            "anio_inicio":              ANIOS[0],
            "anio_fin":                 ANIOS[-1],
            "n_registros":              len(df),
            "fecha_procesamiento":      datetime.now().isoformat(),
        },
        ANP_META,
    )
    logger.info(f"[OK] ANP calculada: {len(df)} registros, {len(union_por_distrito)} distritos con ANP")
    return df
