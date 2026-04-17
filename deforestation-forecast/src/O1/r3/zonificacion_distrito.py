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

warnings.filterwarnings('ignore')

def calcular_estadisticas_por_distrito(
    ruta_raster,
    gdf_distritos,
    nodata=255,
    all_touched=True
):
    
    print("\n" + "="*70)
    print(" CALCULANDO ESTADÍSTICAS POR DISTRITO (OPTIMIZADO)")
    print("="*70 + "\n")

    # Asegurar índice limpio
    gdf = gdf_distritos.reset_index(drop=True).copy()
    gdf["id"] = gdf.index

    with rasterio.open(ruta_raster) as src:
        raster_crs = src.crs
        valores = np.unique(src.read(1))

    print("[INFO] Valores únicos en raster:", valores)
    
    # Validar valores del raster
    valores_validos = {0, 1, 255}
    valores_inesperados = set(valores) - valores_validos
    if valores_inesperados:
        print(f"[WARN] Raster contiene valores inesperados: {valores_inesperados}")
    else:
        print("[OK] Valores esperados en raster: 0 (sin cambio), 1 (cambio), 255 (nodata)")

    if gdf.crs != raster_crs:
        print("[WARN] Reproyectando distritos al CRS del raster...")
        gdf = gdf.to_crs(raster_crs)

    # Zonal stats categórico
    stats = zonal_stats(
        vectors=gdf.geometry,
        raster=ruta_raster,
        categorical=True,
        nodata=nodata,
        all_touched=all_touched
    )

    resultados = []

    for i, s in enumerate(stats):
        if s is None:
            s = {}

        c0 = int(s.get(0, 0))
        c1 = int(s.get(1, 0))

        total = c0 + c1

        pct = (c1 / total * 100) if total > 0 else None

        resultados.append({
            "id": i,
            "pix_valid": total,
            "pix_change": c1,
            "pix_nochg": c0,
            "pct_change": pct
        })

    df_stats = pd.DataFrame(resultados)
    df_stats["pct_change"] = df_stats["pct_change"].fillna(0)

    print(f"[OK] {len(df_stats)} distritos procesados")

    return df_stats


def leer_shapefile_distritos(ruta_shapefile):
    """
    Lee el shapefile de distritos amazónicos.
    
    Args:
        ruta_shapefile (str): Ruta al shapefile de distritos
    
    Returns:
        GeoDataFrame: Distritos amazónicos
    """
    print(f"[INFO] Leyendo shapefile de distritos: {ruta_shapefile}")
    gdf = gpd.read_file(ruta_shapefile)
    print(f"[OK] {len(gdf)} distritos cargados. CRS: {gdf.crs}")
    
    # Validar columnas disponibles
    print(f"[INFO] Columnas disponibles: {gdf.columns.tolist()}")
    if 'LEVEL_4' not in gdf.columns:
        print("[WARN] Columna 'LEVEL_4' no encontrada. Columnas disponibles:")
        print(gdf.columns.tolist())
    
    return gdf


def asignar_categorias_cambio(df_stats):
    """
    Asigna categoría de cambio (Bajo, Medio, Alto) basada en quintiles.
    
    Args:
        df_stats (pd.DataFrame): Estadísticas con columna 'pct_cambio'
    
    Returns:
        pd.DataFrame: Con columna 'categoria_cambio' añadida
    """
    print("[INFO] Asignando categorías de cambio basadas en cuantiles...")
    
    
    try:
        df_stats["categoria_cambio"] = pd.qcut(
            df_stats["pct_change"],
            q=5,
            labels=["Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto"]
        )
    except:
        df_stats["categoria_cambio"] = "Sin clasificación"
    
    
    return df_stats


def generar_shapefile_densidad(gdf_distritos, df_stats, ruta_salida):

    gdf = gdf_distritos.reset_index(drop=True).copy()
    gdf["id"] = gdf.index

    gdf_final = gdf.merge(df_stats, on="id", how="left")
    gdf_final = gdf_final.rename(columns={
        "pix_change": "pix_chg",
        "pix_valid": "pix_val",
        "pct_change": "pct_chg"
    })
    
    # Agregar nombre del distrito si existe columna LEVEL_4
    if 'LEVEL_4' in gdf_final.columns:
        gdf_final = gdf_final.rename(columns={'LEVEL_4': 'distrito'})

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    gdf_final.to_file(ruta_salida, encoding="utf-8")

    print(f"[OK] Shapefile guardado: {ruta_salida}")

    return gdf_final

def generar_excel_estadisticas(df_stats, gdf_distritos, ruta_salida):
    """
    Genera Excel con estadísticas por distrito.
    
    Args:
        df_stats (pd.DataFrame): Estadísticas calculadas
        gdf_distritos (GeoDataFrame): GeoDataFrame con info de distritos
        ruta_salida (str): Ruta donde guardar el Excel
    """
    print("[INFO] Generando Excel con estadísticas...")
    
    # Ordenar por porcentaje de cambio descendente
    df_export = df_stats.sort_values('pct_change', ascending=False).reset_index(drop=True)
    
    # Agregar nombre del distrito si existe
    if 'LEVEL_4' in gdf_distritos.columns:
        gdf_temp = gdf_distritos.reset_index(drop=True)[['LEVEL_4']].copy()
        gdf_temp['id'] = gdf_temp.index
        df_export = df_export.merge(gdf_temp, on='id', how='left')
        df_export = df_export.rename(columns={'LEVEL_4': 'distrito'})
    
    # Formatear columnas
    df_export_fmt = df_export.copy()
    df_export_fmt['pix_change'] = df_export_fmt['pix_change'].astype(int)
    df_export_fmt['pix_valid'] = df_export_fmt['pix_valid'].astype(int)
    df_export_fmt['pct_change'] = df_export_fmt['pct_change'].round(2)
    
    # Renombrar columnas para legibilidad
    rename_cols = {
        'pix_change': 'Píxeles Cambio',
        'pix_valid': 'Píxeles Totales',
        'pct_change': '% Cambio',
        'categoria_cambio': 'Categoría'
    }
    if 'distrito' in df_export_fmt.columns:
        rename_cols['distrito'] = 'Distrito'
    
    df_export_fmt = df_export_fmt.rename(columns=rename_cols)

    total_valid = df_export['pix_valid'].sum()

    pct_global = (
        df_export['pix_change'].sum() / total_valid * 100
        if total_valid > 0 else 0
    )
    
    # Guardar Excel
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    
    # Crear DataFrame de resumen
    estadistica_resumen = pd.DataFrame({
        'Métrica': [
            'Total de distritos',
            'Cambio máximo (%)',
            'Cambio mínimo (%)',
            'Cambio promedio (%)',
            'Total píxeles válidos',
            'Total píxeles cambio',
            'Porcentaje global cambio (%)'
        ],
        'Valor': [
            len(df_export),
            f"{df_export['pct_change'].max():.2f}",
            f"{df_export['pct_change'].min():.2f}",
            f"{df_export['pct_change'].mean():.2f}",
            int(df_export['pix_valid'].sum()),
            int(df_export['pix_change'].sum()),
            f"{pct_global:.2f}"
        ]
    })

    # Rutas CSV
    ruta_csv_distritos = ruta_salida.replace(".xlsx", "_distritos.csv")
    ruta_csv_resumen = ruta_salida.replace(".xlsx", "_resumen.csv")

    # Exportar
    df_export_fmt.to_csv(ruta_csv_distritos, index=False, encoding="utf-8-sig")
    estadistica_resumen.to_csv(ruta_csv_resumen, index=False, encoding="utf-8-sig")

    print(f"[OK] CSV distritos: {ruta_csv_distritos}")
    print(f"[OK] CSV resumen: {ruta_csv_resumen}")
    
    # Imprimir resumen
    print("\n" + "="*70)
    print(" RESUMEN DE CAMBIOS POR DISTRITO")
    print("="*70)
    print(df_export_fmt.head(10).to_string(index=False))
    print("...")
    print(f"\nTotal: {len(df_export)} distritos")
    print(f"Cambio promedio: {df_export['pct_change'].mean():.2f}%")


def pipeline_zonificacion_distrito(
    ruta_raster_cambios,
    ruta_shapefile_distritos,
    directorio_salida
):
    """
    Pipeline completo de zonificación por distrito.
    
    Args:
        ruta_raster_cambios (str): Ruta al raster de cambios
        ruta_shapefile_distritos (str): Ruta al shapefile de distritos
        directorio_salida (str): Directorio donde guardar resultados
    """
    print("\n" + "="*70)
    print(" ZONIFICACIÓN DE CAMBIOS POR DISTRITO")
    print("="*70 + "\n")
    
    # Crear directorio de salida
    os.makedirs(directorio_salida, exist_ok=True)
    
    # Leer datos
    gdf_distritos = leer_shapefile_distritos(ruta_shapefile_distritos)
    
    # Calcular estadísticas
    df_stats = calcular_estadisticas_por_distrito(
        ruta_raster_cambios,
        gdf_distritos
    )    
    # Asignar categorías
    df_stats = asignar_categorias_cambio(df_stats)
    
    # Generar shapefile con densidad
    ruta_shp_densidad = os.path.join(directorio_salida, "distritos_densidad_cambios.shp")
    generar_shapefile_densidad(gdf_distritos, df_stats, ruta_shp_densidad)
    
    # Generar Excel
    ruta_excel = os.path.join(directorio_salida, "distritos_estadisticas_cambios.xlsx")
    generar_excel_estadisticas(df_stats, gdf_distritos, ruta_excel)
    
    print("\n" + "="*70)
    print(" ZONIFICACIÓN POR DISTRITO COMPLETADA")
    print("="*70)
    print(f"\n[OK] Archivos generados:")
    print(f"  1. Shapefile densidad: {ruta_shp_densidad}")
    print(f"  2. Excel estadísticas: {ruta_excel}\n")
    
    return df_stats
