import os, rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.mask import mask
from O1.config import ANIOS, CRS_PROYECTADO

def guardar_csv(gdf, ruta_csv):
    df = gdf.copy()
    if "geometry" in df.columns:
        df = df.drop(columns="geometry")
    
    df.to_csv(ruta_csv, index=False)

def identificar_distritos_amazonia_interseccion(ruta_biomas_peru, ruta_distritos_peru, ruta_distritos_amazonia_delimitados):
    
    biomas = gpd.read_file(ruta_biomas_peru)
    distritos = gpd.read_file(ruta_distritos_peru)

    if biomas.crs != distritos.crs:
        raise ValueError("CRS diferentes entre biomas y distritos")
    
    biomas["geometry"] = biomas.geometry.buffer(0)
    distritos["geometry"] = distritos.geometry.buffer(0)

    # Filtrar Amazonía
    amazonia = biomas.loc[biomas["NAME"] == "[Amazonía]"].copy()

    amazonia = amazonia.to_crs(CRS_PROYECTADO)
    distritos = distritos.to_crs(CRS_PROYECTADO)

    amazonia_union = amazonia.geometry.union_all()

    # Quedarse con los que intersectan
    distritos_intersectados = distritos[distritos.intersects(amazonia_union)].copy()

    distritos_intersectados["area_total"] = distritos_intersectados.geometry.area
    distritos_intersectados["area_intersectada"] = distritos_intersectados.geometry.intersection(amazonia_union).area
    distritos_intersectados["porcentaje_amazonia"] = distritos_intersectados["area_intersectada"] / distritos_intersectados["area_total"]

    distritos_umbral = distritos_intersectados[distritos_intersectados["porcentaje_amazonia"] > 0.50].copy()
    distritos_umbral = distritos_umbral.to_crs("EPSG:4326")

    distritos_umbral.to_file(ruta_distritos_amazonia_delimitados, driver="GPKG", encoding="utf-8")
    ruta_estadisticas_csv = ruta_distritos_amazonia_delimitados.replace(".gpkg", ".csv")
    guardar_csv(distritos_umbral, ruta_estadisticas_csv)

    print(f"Total distritos amazónicos: {len(distritos_umbral)}")
    print(f"Shapefile guardado en: {ruta_distritos_amazonia_delimitados}")

    return distritos_umbral

def recortar_mapas_amazonia(gdf_amazonia, carpeta_mapas_raw, carpeta_mapas_amazonia):
    
    geometria = [gdf_amazonia.geometry.union_all()]

    for anio in ANIOS:

        mapa_raw = f"peru_collection3_integration_v1-classification_{anio}.tif"
        ruta_raster_raw = os.path.join(carpeta_mapas_raw, mapa_raw)

        if not os.path.exists(ruta_raster_raw):
            print(f"[WARN] No existe raster para el año {anio}")
            continue

        nombre_salida = f"peru_amazonia_{anio}.tif"
        ruta_salida = os.path.join(carpeta_mapas_amazonia, nombre_salida)

        if os.path.exists(ruta_salida):
            print(f"[SKIP] Ya existe: {nombre_salida}")
            continue

        print(f"[PROCESANDO] {mapa_raw} → {nombre_salida}")

        with rasterio.open(ruta_raster_raw) as src:

            out_image, out_transform = mask(src, geometria, crop=True)

            out_meta = src.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            with rasterio.open(ruta_salida, "w", **out_meta) as dest:
                dest.write(out_image)

    print("[OK] Recorte completado")

def pipeline_delimitacion_amazonia(
    ruta_biomas_peru,
    ruta_distritos_peru,
    ruta_distritos_amazonia,
    carpeta_mapas_raw,
    carpeta_mapas_amazonia
):
    print("\n" + "="*70)
    print(" PIPELINE AMAZONÍA")
    print("="*70)

    gdf_amazonia = identificar_distritos_amazonia_interseccion(ruta_biomas_peru, ruta_distritos_peru, ruta_distritos_amazonia)
    recortar_mapas_amazonia(gdf_amazonia, carpeta_mapas_raw, carpeta_mapas_amazonia)

    print("\n[OK] Pipeline Amazonía completado")