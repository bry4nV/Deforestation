import os
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from O1.config import (
    COBERTURA_MIN95_DIR, AREA_MINIMA_DIR,  BOSQUE_MINIMO_DIR,
    CRS_PROYECTADO, NODATA
)

def guardar_csv(gdf, ruta_csv):
    df = gdf.copy()
    if "geometry" in df.columns:
        df = df.drop(columns="geometry")
    
    df.to_csv(ruta_csv, index=False)

def identificar_distritos_cobertura_min95(ruta_biomas_peru, ruta_distritos_peru, ruta_distritos_cobertura_min95):
    
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
    distritos_umbral.to_file(ruta_distritos_cobertura_min95, driver="GPKG", encoding="utf-8")
    ruta_estadisticas_csv = ruta_distritos_cobertura_min95.replace(".gpkg", ".csv")
    guardar_csv(distritos_umbral, ruta_estadisticas_csv)

    print(f"Total distritos amazónicos: {len(distritos_umbral)}")
    print(f"Shapefile guardado en: {ruta_distritos_cobertura_min95}")

    return distritos_umbral

def identificar_distritos_area_minima(ruta_mapa_inicial, distritos_cobertura_min, ruta_distritos_area_minima):

    distritos = distritos_cobertura_min.copy()

    estadisticas = zonal_stats(
        distritos.geometry,
        ruta_mapa_inicial,
        stats=["count"],
        nodata=NODATA,
        all_touched=True
    )

    # Contar píxeles válidos (no nodata) por distrito
    distritos["pixeles"] = [s["count"] if s else 0 for s in estadisticas]

    # Umbral dinámico: percentil 10
    umbral = distritos["pixeles"].quantile(0.10)

    distritos_umbral = distritos[distritos["pixeles"] >= umbral].copy()

    distritos_umbral.to_file(ruta_distritos_area_minima, driver="GPKG", encoding="utf-8")
    ruta_estadisticas_csv = ruta_distritos_area_minima.replace(".gpkg", ".csv")
    guardar_csv(distritos_umbral, ruta_estadisticas_csv)

    print(f"Umbral (P10): {umbral}")
    print(f"Total distritos tras filtro de área mínima: {len(distritos_umbral)}")
    print(f"Shapefile guardado en: {ruta_distritos_area_minima}")

    return distritos_umbral


def identificar_distritos_bosque_minimo(ruta_mapa_inicial, distritos_area_minima, ruta_distritos_bosque_minimo):
    
    distritos = distritos_area_minima.copy()

    estadisticas = zonal_stats(
        distritos.geometry,
        ruta_mapa_inicial,
        categorical=True,
        nodata=NODATA,
        all_touched=True
    )

    # Extraer solo píxeles de clase 1 (bosque)
    distritos["pixeles_bosque_inicial"] = [
        (s or {}).get(1, 0) for s in estadisticas
    ]

    distritos["porcentaje_bosque_inicial"] = distritos.apply(
        lambda row: row["pixeles_bosque_inicial"] / row["pixeles"] if row["pixeles"] > 0 else 0,
        axis=1
    )

    # Umbral dinámico: percentil 10 de bosque
    umbral = distritos["porcentaje_bosque_inicial"].quantile(0.10)

    distritos_umbral = distritos[distritos["porcentaje_bosque_inicial"] >= umbral].copy()

    distritos_umbral.to_file(ruta_distritos_bosque_minimo, driver="GPKG", encoding="utf-8")
    ruta_estadisticas_csv = ruta_distritos_bosque_minimo.replace(".gpkg", ".csv")
    guardar_csv(distritos_umbral, ruta_estadisticas_csv)

    print(f"Umbral bosque (P10): {umbral}")
    print(f"Total distritos tras filtro de bosque mínimo: {len(distritos_umbral)}")
    print(f"Shapefile guardado en: {ruta_distritos_bosque_minimo}")


def pipeline_delimitacion_distritos_amazonia(ruta_biomas_peru, ruta_distritos_peru, ruta_mapa_inicial, ruta_distritos_amazonia_delimitados):
    print("\n" + "="*70)
    print(" INICIANDO DELIMITACIÓN DE DISTRITOS AMAZÓNICOS")
    print("="*70)

    ruta_distritos_cobertura_min95 = os.path.join(COBERTURA_MIN95_DIR, "distritos_cobertura_min95.gpkg")
    ruta_distritos_area_minima = os.path.join(AREA_MINIMA_DIR, "distritos_area_minima.gpkg")

    distritos_cobertura_min = identificar_distritos_cobertura_min95(ruta_biomas_peru, ruta_distritos_peru, ruta_distritos_cobertura_min95)
    distritos_area_minima = identificar_distritos_area_minima(ruta_mapa_inicial, distritos_cobertura_min, ruta_distritos_area_minima)
    identificar_distritos_bosque_minimo(ruta_mapa_inicial, distritos_area_minima, ruta_distritos_amazonia_delimitados)

    print("\n" + "="*70)
    print(" DELIMITACIÓN DE DISTRITOS AMAZÓNICOS SELECCIONADOS COMPLETADA")
    print("="*70)