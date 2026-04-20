"""
Zonificación de cambios de cobertura por distrito amazónico.

Este módulo realiza:
1. Intersección del raster de cambios con cada distrito
2. Cálculo de píxeles cambiados vs totales
3. Generación de Excel con estadísticas
4. Generación de shapefile con densidad de cambios para gvSIG
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from O1.config import NODATA

def leer_distritos(ruta_distritos):
    gdf = gpd.read_file(ruta_distritos)
    return gdf

def calcular_metricas_por_distrito(ruta_mapa_cambios, gdf_distritos):
    with rasterio.open(ruta_mapa_cambios) as src:
        if gdf_distritos.crs != src.crs:
            raise ValueError(f"CRS no coincide. Raster: {src.crs}, Distritos: {gdf_distritos.crs}")

    stats = zonal_stats(
        gdf_distritos.geometry,
        ruta_mapa_cambios,
        categorical=True,
        nodata=NODATA,
        all_touched=True
    )

    resultados = []
    for i, s in enumerate(stats):
        s = s or {}

        pixeles_no_cambiados = int(s.get(0, 0))
        pixeles_cambiados = int(s.get(1, 0))
        pixeles_validos = pixeles_no_cambiados + pixeles_cambiados
        porcentaje_cambio = (pixeles_cambiados / pixeles_validos * 100) if pixeles_validos > 0 else None

        resultados.append({
            "GEOCODE": gdf_distritos.iloc[i]["GEOCODE"],
            "pixeles_validos": pixeles_validos,
            "pixeles_cambiados": pixeles_cambiados,
            "pixeles_no_cambiados": pixeles_no_cambiados,
            "porcentaje_cambio": porcentaje_cambio
        })

    return pd.DataFrame(resultados)


def guardar_mapa_cambios_distrito(gdf_distritos, df_metricas, ruta_salida):
    gdf_final = gdf_distritos.merge(df_metricas, on="GEOCODE", how="left")
    gdf_final.to_file(ruta_salida, driver="GPKG", encoding="utf-8")
    print(f"[OK] GPKG guardado: {ruta_salida}")

    return gdf_final

def exportar_csv_distritos(gdf, ruta_csv_base):
    df = gdf.drop(columns="geometry").copy()

    df = df.sort_values("porcentaje_cambio", ascending=False).reset_index(drop=True)

    df["pixeles_cambiados"] = df["pixeles_cambiados"].astype(int)
    df["pixeles_validos"] = df["pixeles_validos"].astype(int)
    df["pixeles_no_cambiados"] = df["pixeles_no_cambiados"].astype(int)
    df["porcentaje_cambio"] = df["porcentaje_cambio"].round(2)

    df = df.rename(columns={
        "LEVEL_2": "Departamento",
        "LEVEL_4": "Distrito",
        "pixeles_cambiados": "Píxeles Cambio",
        "pixeles_no_cambiados": "Píxeles No Cambio",
        "pixeles_validos": "Píxeles Totales",
        "porcentaje_cambio": "% Cambio"
    })

    ruta_distritos = ruta_csv_base.replace(".csv", "_distritos.csv")
    df.to_csv(ruta_distritos, index=False, encoding="utf-8-sig")

    print(f"[OK] CSV distritos: {ruta_distritos}")

def exportar_csv_resumen(gdf, ruta_csv_base):
    total_valido = gdf["pixeles_validos"].sum()
    total_cambio = gdf["pixeles_cambiados"].sum()
    pct_global = (total_cambio / total_valido * 100) if total_valido > 0 else 0

    resumen = pd.DataFrame({
        "Métrica": [
            "Total distritos",
            "Cambio máximo (%)",
            "Cambio mínimo (%)",
            "Cambio promedio (%)",
            "Total píxeles válidos",
            "Total píxeles cambio",
            "Porcentaje global (%)"
        ],
        "Valor": [
            len(gdf),
            f"{gdf['porcentaje_cambio'].max():.2f}" if gdf["porcentaje_cambio"].notna().any() else None,
            f"{gdf['porcentaje_cambio'].min():.2f}" if gdf["porcentaje_cambio"].notna().any() else None,
            f"{gdf['porcentaje_cambio'].mean():.2f}" if gdf["porcentaje_cambio"].notna().any() else None,
            int(total_valido),
            int(total_cambio),
            f"{pct_global:.2f}"
        ]
    })

    ruta_resumen = ruta_csv_base.replace(".csv", "_resumen.csv")
    resumen.to_csv(ruta_resumen, index=False, encoding="utf-8-sig")

    print(f"[OK] CSV resumen: {ruta_resumen}")


def pipeline_zonificacion_distrito(
    ruta_mapa_cambios,
    ruta_distritos_amazonia,
    ruta_mapa_cambios_distrito,
    ruta_estadisticas_cambios_distrito
):
    """
    Pipeline completo de zonificación por distrito.
    
    Args:
        ruta_mapa_cambios (str): Ruta del mapa de cambios
        ruta_distritos_amazonia (str): Ruta al shapefile de distritos
        ruta_mapa_cambios_distrito (str): Ruta donde guardar resultados
        ruta_estadisticas_cambios_distrito (str): Ruta donde guardar estadísticas
    """
    print("\n" + "="*70)
    print(" ZONIFICACIÓN DE CAMBIOS POR DISTRITO")
    print("="*70 + "\n")
    
    gdf_distritos = leer_distritos(ruta_distritos_amazonia)
    df_metricas = calcular_metricas_por_distrito(ruta_mapa_cambios, gdf_distritos)

    gdf_distritos_cambios = guardar_mapa_cambios_distrito(gdf_distritos, df_metricas, ruta_mapa_cambios_distrito)
    exportar_csv_distritos(gdf_distritos_cambios, ruta_estadisticas_cambios_distrito)
    exportar_csv_resumen(gdf_distritos_cambios, ruta_estadisticas_cambios_distrito)

    print("\n" + "="*70)
    print(" ZONIFICACIÓN POR DISTRITO COMPLETADA")
    print("="*70)