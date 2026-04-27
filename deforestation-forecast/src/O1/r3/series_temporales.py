import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats
from O1.config import NODATA, ANIOS, SEMILLA_SPLIT, TAMANIO_ENTRENAMIENTO
import numpy as np


def extraer_series(gdf, rutas_mapas_reclasificados):
    registros = []

    for ruta, anio in zip(rutas_mapas_reclasificados, ANIOS):

        print(f"[INFO] Año {anio}")

        stats = zonal_stats(
            gdf.geometry,
            ruta,
            categorical=True,
            nodata=NODATA
        )

        for idx, s in enumerate(stats):

            s = s or {}
            fila = gdf.iloc[idx]

            pix_bosque = int(s.get(1, 0))
            pix_no_bosque = int(s.get(0, 0))
            pix_total = pix_bosque + pix_no_bosque

            registros.append({
                "geocode": fila["GEOCODE"],
                "departamento": fila["LEVEL_2"],
                "distrito": fila["LEVEL_4"],
                "anio": anio,
                "pix_total": pix_total,
                "pix_bosque": pix_bosque,
                "pix_no_bosque": pix_no_bosque,
                "pct_bosque": pix_bosque / pix_total if pix_total > 0 else 0,
                "pct_no_bosque": pix_no_bosque / pix_total if pix_total > 0 else 0
            })

    df = pd.DataFrame(registros)
    df = df.sort_values(["geocode", "departamento", "distrito", "anio"]).reset_index(drop=True)

    return df


def guardar_series(df, gdf, ruta_csv, ruta_stats):
    
    df.to_csv(ruta_csv, index=False)
    print(f"[OK] CSV guardado: {ruta_csv}")

    ruta_gpkg = ruta_csv.replace(".csv", ".gpkg")
    gdf.to_file(ruta_gpkg, driver="GPKG")
    print(f"[OK] GPKG guardado: {ruta_gpkg}")

    if ruta_stats:
        stats = (
            df.groupby("geocode")[["pct_bosque", "pct_no_bosque"]]
            .agg(["mean", "min", "max"])
            .reset_index()
        )

        stats.columns = [
            "geocode",
            "pct_bosque_mean",
            "pct_bosque_min",
            "pct_bosque_max",
            "pct_no_bosque_mean",
            "pct_no_bosque_min",
            "pct_no_bosque_max"
        ]

        stats.to_csv(ruta_stats, index=False)
        print(f"[OK] Stats guardado: {ruta_stats}")


def split_aleatorio(gdf):
    
    np.random.seed(SEMILLA_SPLIT)

    indices = np.random.permutation(len(gdf))
    cantidad_train = int(len(gdf) * TAMANIO_ENTRENAMIENTO)

    idx_train = indices[:cantidad_train]
    idx_test = indices[cantidad_train:]

    gdf_train = gdf.iloc[idx_train].copy()
    gdf_test = gdf.iloc[idx_test].copy()

    return gdf_train, gdf_test


def pipeline_extraer_series_temporales(
    rutas_mapas_reclasificados,
    ruta_distritos_alto_cambio,
    ruta_series_entrenamiento,
    ruta_estadisticas_series_entrenamiento,
    ruta_series_generalizacion,
    ruta_estadisticas_series_generalizacion
):

    print("\n" + "="*70)
    print(" SERIES TEMPORALES (BOSQUE / NO BOSQUE)")
    print("="*70 + "\n")

    rutas_mapas_reclasificados = sorted(rutas_mapas_reclasificados)

    if len(rutas_mapas_reclasificados) != len(ANIOS):
        raise ValueError("Número de rutas y años no coincide")

    gdf = gpd.read_file(ruta_distritos_alto_cambio)

    gdf_train, gdf_generalizacion = split_aleatorio(gdf)

    print(f"[INFO] Entrenamiento: {len(gdf_train)} distritos")
    print(f"[INFO] Generalización: {len(gdf_generalizacion)} distritos")

    df_train = extraer_series(gdf_train, rutas_mapas_reclasificados)

    guardar_series(
        df_train,
        gdf_train,
        ruta_series_entrenamiento,
        ruta_estadisticas_series_entrenamiento
    )

    df_gen = extraer_series(gdf_generalizacion, rutas_mapas_reclasificados)

    guardar_series(
        df_gen,
        gdf_generalizacion,
        ruta_series_generalizacion,
        ruta_estadisticas_series_generalizacion
    )