"""
Zonificación de parches de deforestación mediante componentes conectados.

Este módulo identifica zonas de deforestación agrupando píxeles contiguos
que hayan experimentado cambios (bosque <=> no bosque), usando el algoritmo de
componentes conectados con conectividad 8.
"""
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import label
from skimage.measure import regionprops
import os
from O1.config import (NODATA)

def identificar_zonas_por_conectividad(ruta_mapa_cambios, pixeles_min):
    """
    Identifica zonas de deforestación mediante componentes conectados.
    
    Parameters:
    -----------
    ruta_mapa_cambios : str
        Ruta al mapa de cambios (1=cambió, 0=sin cambio)
    pixeles_min : float
        Número mínimo de píxeles para considerar una zona válida
    
    Returns:
    --------
    zonas_labels : np.ndarray
        Raster con IDs de zonas (0=sin zona, 1,2,3,...=zonas)
    zonas_info : list
        Lista de diccionarios con información de cada zona
    meta : dict
        Metadatos del raster (transform, crs, etc.)
    """
    
    print("\n" + "="*70)
    print("ZONIFICACIÓN POR COMPONENTES CONECTADOS")
    print("="*70 + "\n")
    
    # ========================================
    # PASO 1: Cargar mapa de cambios
    # ========================================
    
    with rasterio.open(ruta_mapa_cambios) as src:
        mapa_cambios = src.read(1)
        transform = src.transform
        meta = src.meta.copy()
    
    print(f"[INFO] Mapa cargado: {mapa_cambios.shape[1]} x {mapa_cambios.shape[0]} píxeles")
    
    # Máscara de píxeles con cambio
    mascara_valida = (mapa_cambios != NODATA)
    mascara_cambio = (mapa_cambios == 1) & mascara_valida
    n_pixels_cambio = np.sum(mascara_cambio)
    
    print(f"[INFO] Píxeles con cambio: {n_pixels_cambio:,}")
    
    if n_pixels_cambio == 0:
        zonas_vacias = np.zeros_like(mapa_cambios, dtype=np.int32)
        zonas_vacias[mapa_cambios == NODATA] = NODATA
        return zonas_vacias, [], meta
    
    # ========================================
    # PASO 2: Calcular área de píxel
    # ========================================
    
    # Para EPSG:4326, aproximación simple
    # Resolución en grados
    res_deg_x = abs(transform.a)
    res_deg_y = abs(transform.e)

    # Latitud representativa (centro del raster)
    h, w = mapa_cambios.shape
    _, lat = transform * (w // 2, h // 2)

    # Conversión grados → metros
    m_per_deg_lat = 110574.0
    m_per_deg_lon = 111320.0 * np.cos(np.deg2rad(lat))

    # Resolución en metros
    res_m_x = res_deg_x * m_per_deg_lon
    res_m_y = res_deg_y * m_per_deg_lat
    
    # Área
    area_pixel_m2 = res_m_x * res_m_y
    area_pixel_km2 = area_pixel_m2 / 1e6

    print(f"\n[INFO] Resolución espacial:")
    print(f"  Grados: {res_deg_x:.6f}° x {res_deg_y:.6f}°")
    print(f"  Metros: {res_m_x:.2f} m x {res_m_y:.2f} m")
    print(f"  Área/píxel: {area_pixel_m2:.0f} m² ({area_pixel_km2:.6f} km²)")
    
    # ========================================
    # PASO 3: Aplicar componentes conectados
    # ========================================
    
    print(f"\n[INFO] Aplicando componentes conectados (conectividad 8)...")
    
    # Estructura de conectividad 8 (incluye diagonales)
    estructura_8 = np.ones((3, 3), dtype=np.int8)

    estructura_4 = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=np.int8)
    
    # Identificar componentes
    zonas_cambio, n_zonas_cambio = label(mascara_cambio, structure=estructura_8)
    
    print(f"[OK] Componentes identificados: {n_zonas_cambio:,}")
    
    # ========================================
    # PASO 4: Filtrar por cantidad de píxeles
    # ========================================
    
    print(f"\n[INFO] Filtrando zonas por cantidad de píxeles...")
    print(f"  Píxeles mínimos: {pixeles_min}")
    
    # Calcular propiedades de cada parche
    propiedades_zonas = regionprops(zonas_cambio)
    
    # Filtrar y renumerar
    zonas_cambio_filtradas = np.zeros_like(zonas_cambio, dtype=np.int32)
    zonas_cambio_filtradas_info = []
    zona_id = 1
    
    n_descartados_pequeños = 0
    
    for region in propiedades_zonas:

        n_pixeles_region = region.area
        area_km2 = region.area * area_pixel_km2
        
        # Verificar si supera la cantidad de píxeles mínimos
        if n_pixeles_region < pixeles_min:
            n_descartados_pequeños += 1
            continue
        
        print(f"Evaluated zone {region.label}: {region.area} pixels")

        # Asignar zona
        mascara_zona = (zonas_cambio == region.label)
        zonas_cambio_filtradas[mascara_zona] = zona_id
        
        # Calcular centroide en coordenadas geográficas
        centroid_y, centroid_x = region.centroid
        geo_x, geo_y = transform * (centroid_x, centroid_y)
        
        # Calcular bounding box
        minr, minc, maxr, maxc = region.bbox
        
        zonas_cambio_filtradas_info.append({
            'zona_id': zona_id,
            'area_km2': area_km2,
            'n_pixels': region.area,
            'centroid_lat': geo_y,
            'centroid_lon': geo_x,
            'bbox_minr': minr,
            'bbox_minc': minc,
            'bbox_maxr': maxr,
            'bbox_maxc': maxc,
        })
        
        zona_id += 1

        if zona_id == 255:
            zona_id += 1
    
    n_zonas_validas = len(zonas_cambio_filtradas_info)
    
    print(f"\n[OK] Filtrado completado:")
    print(f"  Zonas válidas:        {n_zonas_validas:,}")
    print(f"  Descartadas (< {pixeles_min} km²): {n_descartados_pequeños:,}")
    
    # ========================================
    # PASO 5: Estadísticas de zonas
    # ========================================
    
    if n_zonas_validas > 0:
        areas = [z['area_km2'] for z in zonas_cambio_filtradas_info]
        
        print(f"\n" + "="*70)
        print("ESTADÍSTICAS DE ZONAS")
        print("="*70)
        print(f"  Total zonas:       {n_zonas_validas:,}")
        n_pixels = [z['n_pixels'] for z in zonas_cambio_filtradas_info]
        print(f"Mediana: {np.median(areas):.2f} km² ({np.median(n_pixels):.0f} px) | "
              f"Mín: {np.min(areas):.2f} km² ({np.min(n_pixels):.0f} px) | "
              f"Máx: {np.max(areas):.2f} km² ({np.max(n_pixels):.0f} px)")
        
        print("="*70 + "\n")
    else:
        print("\n[WARN] No se identificaron zonas válidas.")

    # ========================================
    # PASO 6: Propagar NODATA (CRÍTICO)
    # ========================================

    mascara_nodata = (mapa_cambios == NODATA)
    zonas_cambio_filtradas[mascara_nodata] = NODATA
    zonas_cambio_filtradas = zonas_cambio_filtradas.astype(np.int32)

    return zonas_cambio_filtradas, zonas_cambio_filtradas_info, meta


def exportar_zonas_cambios(zonas_cambio, meta_original, ruta_salida):
    """
    Guarda el mapa de zonas como raster GeoTIFF.
    """

    meta = meta_original.copy()
    meta.update({
        'dtype': 'int32',
        'nodata': NODATA,
        'count': 1
    })
    
    with rasterio.open(ruta_salida, 'w', **meta) as dst:
        dst.write(zonas_cambio.astype('int32'), 1)
    
    print(f"[OK] Zonas guardadas: {ruta_salida}")


def exportar_estadisticas_zonas(zonas_info, ruta_salida):
    """
    Exporta estadísticas de zonas a CSV.
    """
    
    if not zonas_info:
        print("[WARN] No hay zonas para exportar.")
        return
    
    df = pd.DataFrame(zonas_info)
    
    # Ordenar por área (descendente)
    df = df.sort_values('n_pixels', ascending=False)
    
    # Guardar
    df.to_csv(ruta_salida, index=False, float_format='%.6f')
    
    print(f"[OK] Estadísticas exportadas: {ruta_salida}")
    print(f"     {len(df)} zonas × {len(df.columns)} columnas")


def visualizar_distribucion_areas(zonas_info, ruta_salida=None):
    """
    Genera histograma de distribución de áreas de zonas.
    """
    import matplotlib.pyplot as plt
    
    if not zonas_info:
        print("[WARN] No hay zonas para visualizar.")
        return
    
    areas = [z['area_km2'] for z in zonas_info]
    
    plt.figure(figsize=(10, 6))
    plt.hist(areas, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Área (km²)')
    plt.ylabel('Frecuencia')
    plt.title(f'Distribución de Áreas de Zonas (n={len(areas)})')
    plt.grid(True, alpha=0.3)
    
    # Líneas de referencia
    plt.axvline(np.median(areas), color='r', linestyle='--', label=f'Mediana: {np.median(areas):.2f} km²')
    plt.axvline(np.mean(areas), color='g', linestyle='--', label=f'Media: {np.mean(areas):.2f} km²')
    plt.legend()
    
    if ruta_salida:
        plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
        print(f"[OK] Gráfico guardado: {ruta_salida}")
    else:
        plt.show()
    
    plt.close()


def pipeline_zonificacion(ruta_mapa_cambios, ruta_salida, pixeles_min=50):
    """
    Pipeline completo de zonificación.
    
    Parameters:
    -----------
    ruta_mapa_cambios : str
        Ruta al mapa de cambios
    ruta_salida : str
        Directorio de salida
    pixeles_min : float
        Número mínimo de píxeles por zona
    
    Returns:
    --------
    dict : Resumen de resultados
    """
    
    print("\n" + "="*70)
    print(" PIPELINE DE ZONIFICACIÓN")
    print("="*70)
    print(f"  Input:      {os.path.basename(ruta_mapa_cambios)}")
    print(f"  Output dir: {ruta_salida}")
    print(f"  Píxeles min:   {pixeles_min}")
    print("="*70 + "\n")
    
    # Paso 1: Identificar zonas
    zonas_cambio, zonas_info, meta = identificar_zonas_por_conectividad(
        ruta_mapa_cambios,
        pixeles_min=pixeles_min
    )
    
    # Paso 2: Guardar mapa de zonas
    ruta_salida_zonas = os.path.join(ruta_salida, "zonas_cambios_conectividad_8.tif")
    exportar_zonas_cambios(zonas_cambio, meta, ruta_salida_zonas)
    
    # Paso 3: Exportar estadísticas
    ruta_estadisticas_zonas_cambio = os.path.join(ruta_salida, "estadisticas_zonas_cambios_conectividad_8.csv")
    exportar_estadisticas_zonas(zonas_info, ruta_estadisticas_zonas_cambio)
    
    # Paso 4: Generar histograma
    ruta_histograma = os.path.join(ruta_salida, "distribucion_areas_zonas_cambios_conectividad_8.png")
    visualizar_distribucion_areas(zonas_info, ruta_histograma)
    
    # Resumen
    area_total = sum(z['area_km2'] for z in zonas_info) if zonas_info else 0
    
    print("\n" + "="*70)
    print(" ZONIFICACIÓN COMPLETADA ")
    print("="*70)
    print(f"\n  Zonas identificadas:  {len(zonas_info):,}")
    print(f"  Área total:           {area_total:.2f} km²")
    print(f"\n  Outputs:")
    print(f"    - Mapa de zonas:    {ruta_salida_zonas}")
    print(f"    - Estadísticas CSV: {ruta_estadisticas_zonas_cambio}")
    print(f"    - Histograma:       {ruta_histograma}")
    print("\n" + "="*70 + "\n")