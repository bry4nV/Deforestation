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
import warnings
from rasterstats import zonal_stats
from O1.config import NODATA

warnings.filterwarnings('ignore')

def calcular_estadisticas_por_distrito(ruta_mapa_cambios, gdf_distritos):
    
    # Asegurar índice limpio
    gdf = gdf_distritos.reset_index(drop=True).copy()
    gdf["id"] = gdf.index
    nombres = gdf["LEVEL_4"].values

    # Validar CRS (más directo)
    with rasterio.open(ruta_mapa_cambios) as src:
        if gdf.crs != src.crs:
            raise ValueError(f"CRS no coincide. Raster: {src.crs}, Distritos: {gdf.crs}")


    # Zonal stats categórico
    estadisticas = zonal_stats(
        gdf.geometry,
        ruta_mapa_cambios,
        categorical=True,
        nodata=NODATA,
        all_touched=True
    )

    resultados = []
    for i, s in enumerate(estadisticas):
        s = s or {}

        c0 = int(s.get(0, 0))
        c1 = int(s.get(1, 0))
        total = c0 + c1

        pct = (c1 / total * 100) if total > 0 else None

        resultados.append({
            "id": i,
            "distrito": nombres[i],  # 👈 AQUÍ
            "pixeles_validos": total,
            "pixeles_cambiados": c1,
            "pixeles_no_cambiados": c0,
            "porcentaje_cambio": pct
        })

    df_estadisticas = pd.DataFrame(resultados)

    print(f"[OK] {len(df_estadisticas)} distritos procesados")

    return df_estadisticas


def leer_shapefile_distritos(ruta_shapefile):

    print(f"[INFO] Leyendo shapefile de distritos: {ruta_shapefile}")
    gdf = gpd.read_file(ruta_shapefile)
    
    if 'LEVEL_4' not in gdf.columns:
        print("[WARN] Columna 'LEVEL_4' no encontrada. Columnas disponibles:")
        print(gdf.columns.tolist())
    
    return gdf


def clasificar_intensidad_cambio(df_estadisticas):
    """
    Clasifica la intensidad de cambio en categorías usando cuantiles (quintiles).
    """

    df_estadisticas["categoria_cambio"] = pd.qcut(
        df_estadisticas["porcentaje_cambio"],
        q=5,
        labels=["Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto"],
        duplicates="drop"  # evita errores si hay valores repetidos
    )

    return df_estadisticas


def guardar_mapa_cambios_distrito(gdf_distritos, df_estadisticas, ruta_salida):

    gdf = gdf_distritos.reset_index(drop=True).copy()
    gdf["id"] = gdf.index

    gdf_final = gdf.merge(df_estadisticas, on="id", how="left")

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    gdf_final.to_file(ruta_salida, encoding="utf-8")

    print(f"[OK] Shapefile guardado: {ruta_salida}")

def exportar_estadisticas_cambios_distrito(df_estadisticas, ruta_estadisticas_cambios_distrito):
    """
    Exporta estadísticas por distrito y resumen global en CSV.
    """
    print("[INFO] Generando CSV con estadísticas...")
    
    # Ordenar por porcentaje de cambio descendente
    df_estadisticas_ordenadas = df_estadisticas.sort_values("porcentaje_cambio", ascending=False).reset_index(drop=True)
    
    # Formatear columnas
    df_fmt = df_estadisticas_ordenadas.copy()
    df_fmt["pixeles_cambiados"] = df_fmt["pixeles_cambiados"].astype(int)
    df_fmt["pixeles_validos"] = df_fmt["pixeles_validos"].astype(int)
    df_fmt["porcentaje_cambio"] = df_fmt["porcentaje_cambio"].round(2)

    df_fmt = df_fmt.rename(columns={
        "distrito": "Distrito",
        "pixeles_cambiados": "Píxeles Cambio",
        "pixeles_validos": "Píxeles Totales",
        "porcentaje_cambio": "% Cambio",
        "categoria_cambio": "Categoría"
    })

    total_valido = df_estadisticas_ordenadas["pixeles_validos"].sum()
    total_cambio = df_estadisticas_ordenadas["pixeles_cambiados"].sum()

    pct_global = (total_cambio / total_valido * 100) if total_valido > 0 else 0

    resumen_estadisticas = pd.DataFrame({
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
            len(df_estadisticas_ordenadas),
            f"{df_estadisticas_ordenadas['porcentaje_cambio'].max():.2f}",
            f"{df_estadisticas_ordenadas['porcentaje_cambio'].min():.2f}",
            f"{df_estadisticas_ordenadas['porcentaje_cambio'].mean():.2f}",
            int(total_valido),
            int(total_cambio),
            f"{pct_global:.2f}"
        ]
    })

    # Exportar
    ruta_distritos = ruta_estadisticas_cambios_distrito.replace(".csv", "_distritos.csv")
    ruta_resumen = ruta_estadisticas_cambios_distrito.replace(".csv", "_resumen.csv")

    df_fmt.to_csv(ruta_distritos, index=False, encoding="utf-8-sig")
    resumen_estadisticas.to_csv(ruta_resumen, index=False, encoding="utf-8-sig")

    print(f"[OK] CSV distritos: {ruta_distritos}")
    print(f"[OK] CSV resumen: {ruta_resumen}")


def pipeline_zonificacion_distrito(ruta_mapa_cambios, ruta_distritos_amazonia, ruta_mapa_cambios_distrito, ruta_estadisticas_cambios_distrito):
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
    
    gdf_distritos = leer_shapefile_distritos(ruta_distritos_amazonia)

    df_estadisticas = calcular_estadisticas_por_distrito(ruta_mapa_cambios, gdf_distritos)    
    df_estadisticas = clasificar_intensidad_cambio(df_estadisticas)
    
    guardar_mapa_cambios_distrito(gdf_distritos, df_estadisticas, ruta_mapa_cambios_distrito)
    
    # Generar CSV
    exportar_estadisticas_cambios_distrito(df_estadisticas, ruta_estadisticas_cambios_distrito)
    
    print("\n" + "="*70)
    print(" ZONIFICACIÓN POR DISTRITO COMPLETADA")
    print("="*70)
    print(f"\n[OK] Archivos generados:")
    print(f"  1. Shapefile densidad: {ruta_mapa_cambios_distrito}")
    print(f"  2. CSV estadísticas: {ruta_estadisticas_cambios_distrito}\n")