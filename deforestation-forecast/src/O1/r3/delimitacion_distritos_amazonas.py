import os
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from O1.config import (
    CRS_PROYECTADO, NODATA
)

def guardar_csv(gdf, ruta_csv):
    df = gdf.copy()
    if "geometry" in df.columns:
        df = df.drop(columns="geometry")
    
    df.to_csv(ruta_csv, index=False)

def identificar_distritos_cobertura_min95(ruta_biomas_peru, ruta_distritos_peru, ruta_distritos_amazonia_delimitados):
    
    biomas = gpd.read_file(ruta_biomas_peru)
    distritos = gpd.read_file(ruta_distritos_peru)

    if biomas.crs != distritos.crs:
        raise ValueError("CRS diferentes entre biomas y distritos")
    
    biomas["geometry"] = biomas.buffer(0)
    distritos["geometry"] = distritos.buffer(0)

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

    # Filtrar
    distritos_umbral = distritos_intersectados[distritos_intersectados["porcentaje_amazonia"] >= 0.95].copy()
    distritos_umbral = distritos_umbral.to_crs("EPSG:4326")
    distritos_umbral.to_file(ruta_distritos_amazonia_delimitados, driver="GPKG", encoding="utf-8")
    ruta_estadisticas_csv = ruta_distritos_amazonia_delimitados.replace(".gpkg", ".csv")
    guardar_csv(distritos_umbral, ruta_estadisticas_csv)

    print(f"Total distritos amazónicos: {len(distritos_umbral)}")
    print(f"Shapefile guardado en: {ruta_distritos_amazonia_delimitados}")

def pipeline_delimitacion_distritos_amazonia(ruta_biomas_peru, ruta_distritos_peru, ruta_distritos_amazonia_delimitados):
    print("\n" + "="*70)
    print(" INICIANDO DELIMITACIÓN DE DISTRITOS AMAZÓNICOS")
    print("="*70)
    
    identificar_distritos_cobertura_min95(ruta_biomas_peru, ruta_distritos_peru, ruta_distritos_amazonia_delimitados)

    print("\n" + "="*70)
    print(" DELIMITACIÓN DE DISTRITOS AMAZÓNICOS SELECCIONADOS COMPLETADA")
    print("="*70)