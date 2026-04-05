"""
Zonificación de parches de deforestación mediante componentes conectados.

Este módulo identifica zonas de deforestación agrupando píxeles contiguos
que hayan experimentado cambios (bosque <=> no bosque), usando el algoritmo de
componentes conectados con conectividad 8.
"""
import numpy as np
import rasterio
from scipy.ndimage import label
from skimage.measure import regionprops
import os


def identificar_zonas_por_conectividad(ruta_mapa_cambios, pixeles_min=20):
    """
    Identifica zonas de deforestación mediante componentes conectados.
    
    Parameters:
    -----------
    ruta_mapa_cambios : str
        Ruta al mapa de cambios (1=cambió, 0=sin cambio)
    pixeles_min : float
        Número mínimo de píxeles para considerar una zona válida (default: 20)
    
    Returns:
    --------
    zonas_labels : np.ndarray
        Raster con IDs de zonas (0=sin zona, 1,2,3,...=zonas)
    zonas_info : list
        Lista de diccionarios con información de cada zona
    transform : affine.Affine
        Transformación geoespacial
    crs : CRS
        Sistema de coordenadas
    """
    
    print("\n" + "="*70)
    print("ZONIFICACIÓN POR COMPONENTES CONECTADOS")
    print("="*70 + "\n")
    
    # ========================================
    # PASO 1: Cargar mapa de cambios
    # ========================================
    
    with rasterio.open(ruta_mapa_cambios) as src:
        cambios = src.read(1)
        transform = src.transform
        crs = src.crs
        meta = src.meta
    
    print(f"[INFO] Mapa cargado: {cambios.shape[1]} x {cambios.shape[0]} píxeles")
    
    # Máscara de píxeles con cambio
    mascara_cambio = (cambios == 1)
    n_pixels_cambio = np.sum(mascara_cambio)
    
    print(f"[INFO] Píxeles con cambio: {n_pixels_cambio:,}")
    
    if n_pixels_cambio == 0:
        print("\n[WARN] No hay píxeles con cambio. No se pueden identificar zonas.")
        return np.zeros_like(cambios, dtype=np.int32), [], transform, crs
    
    # ========================================
    # PASO 2: Calcular área de píxel
    # ========================================
    
    # Para EPSG:4326, aproximación simple
    res_deg_x = abs(transform.a)
    res_deg_y = abs(transform.e)

    print(f"\n[INFO] Resolución espacial:")
    print(f"  Grados:    {res_deg_x:.6f}° x {res_deg_y:.6f}°")

    res_deg = abs(transform[0])
    res_km = res_deg * 111  # 1 grado ≈ 111 km
    area_pixel_km2 = res_km ** 2
    
    print(f"\n[INFO] Resolución espacial:")
    print(f"  Grados:    {res_deg:.6f}°")
    print(f"  Metros:    {res_km * 1000:.1f} m")
    print(f"  Área/píxel: {area_pixel_km2 * 1e6:.0f} m² ({area_pixel_km2:.6f} km²)")
    
    # ========================================
    # PASO 3: Aplicar componentes conectados
    # ========================================
    
    print(f"\n[INFO] Aplicando componentes conectados (conectividad 8)...")
    
    # Estructura de conectividad 8 (incluye diagonales)
    structure_8 = np.ones((3, 3), dtype=np.int8)
    
    # Identificar componentes
    labels_raw, n_parches_raw = label(mascara_cambio, structure=structure_8)
    
    print(f"[OK] Componentes identificados: {n_parches_raw:,}")
    
    # ========================================
    # PASO 4: Filtrar por área
    # ========================================
    
    print(f"\n[INFO] Filtrando parches por área...")
    print(f"  Píxeles mínimos: {pixeles_min}")
    
    # Calcular propiedades de cada parche
    props = regionprops(labels_raw)
    
    # Filtrar y renumerar
    zonas_labels = np.zeros_like(labels_raw, dtype=np.int32)
    zonas_info = []
    zona_id = 1
    
    n_descartados_pequeños = 0
    n_descartados_grandes = 0
    
    for region in props:
        area_km2 = region.area * area_pixel_km2
        
        # Verificar criterios
        if area_km2 < pixeles_min:
            n_descartados_pequeños += 1
            continue  # Descartar (demasiado pequeño)
        
        # Asignar zona
        mask_zona = (labels_raw == region.label)
        zonas_labels[mask_zona] = zona_id
        
        # Calcular centroide en coordenadas geográficas
        centroid_y, centroid_x = region.centroid
        geo_x = transform[2] + centroid_x * transform[0]
        geo_y = transform[5] + centroid_y * transform[4]
        
        # Calcular bounding box
        minr, minc, maxr, maxc = region.bbox
        
        zonas_info.append({
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
    
    n_zonas_validas = len(zonas_info)
    
    print(f"\n[OK] Filtrado completado:")
    print(f"  Zonas válidas:        {n_zonas_validas:,}")
    print(f"  Descartadas (< {pixeles_min} km²): {n_descartados_pequeños:,}")
    
    # ========================================
    # PASO 5: Estadísticas de zonas
    # ========================================
    
    if n_zonas_validas > 0:
        areas = [z['area_km2'] for z in zonas_info]
        
        print(f"\n" + "="*70)
        print("ESTADÍSTICAS DE ZONAS")
        print("="*70)
        print(f"  Total zonas:       {n_zonas_validas:,}")
        print(f"\n  Área (km²):")
        print(f"    Media:           {np.mean(areas):.2f}")
        print(f"    Mediana:         {np.median(areas):.2f}")
        print(f"    Mínima:          {np.min(areas):.2f}")
        print(f"    Máxima:          {np.max(areas):.2f}")
        print(f"    Desv. estándar:  {np.std(areas):.2f}")
        
        print(f"\n  Número de píxeles:")
        n_pixels = [z['n_pixels'] for z in zonas_info]
        print(f"    Media:           {np.mean(n_pixels):.0f}")
        print(f"    Mediana:         {np.median(n_pixels):.0f}")
        
        # Top 10 zonas más grandes
        print(f"\n  Top 10 zonas más grandes:")
        sorted_zonas = sorted(zonas_info, key=lambda x: x['area_km2'], reverse=True)
        for i, zona in enumerate(sorted_zonas[:10], 1):
            flag = " [!MUY GRANDE]" if zona['es_muy_grande'] else ""
            print(f"    {i}. Zona {zona['zona_id']}: {zona['area_km2']:.2f} km² "
                  f"({zona['n_pixels']:,} px){flag}")
        
        print("="*70 + "\n")
    else:
        print("\n[WARN] No se identificaron zonas válidas.")
    
    return zonas_labels, zonas_info, transform, crs


def guardar_zonas(zonas_labels, transform, crs, output_path):
    """
    Guarda el mapa de zonas como raster GeoTIFF.
    """
    meta = {
        'driver': 'GTiff',
        'dtype': 'int32',
        'nodata': 0,
        'width': zonas_labels.shape[1],
        'height': zonas_labels.shape[0],
        'count': 1,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(zonas_labels.astype('int32'), 1)
    
    print(f"[OK] Zonas guardadas: {output_path}")


def exportar_estadisticas_zonas(zonas_info, output_path):
    """
    Exporta estadísticas de zonas a CSV.
    """
    import pandas as pd
    
    if not zonas_info:
        print("[WARN] No hay zonas para exportar.")
        return
    
    df = pd.DataFrame(zonas_info)
    
    # Ordenar por área (descendente)
    df = df.sort_values('area_km2', ascending=False)
    
    # Guardar
    df.to_csv(output_path, index=False, float_format='%.6f')
    
    print(f"[OK] Estadísticas exportadas: {output_path}")
    print(f"     {len(df)} zonas × {len(df.columns)} columnas")


def visualizar_distribucion_areas(zonas_info, output_path=None):
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
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Gráfico guardado: {output_path}")
    else:
        plt.show()
    
    plt.close()


def pipeline_zonificacion(ruta_mapa_cambios, output_dir, pixeles_min=20):
    """
    Pipeline completo de zonificación.
    
    Parameters:
    -----------
    ruta_mapa_cambios : str
        Ruta al mapa de cambios
    output_dir : str
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
    print(f"  Output dir: {output_dir}")
    print(f"  Píxeles min:   {pixeles_min}")
    print("="*70 + "\n")
    
    # Paso 1: Identificar zonas
    zonas_labels, zonas_info, transform, crs = identificar_zonas_por_conectividad(
        ruta_mapa_cambios,
        pixeles_min=pixeles_min
    )
    
    # Paso 2: Guardar mapa de zonas
    output_zonas = os.path.join(output_dir, "zonas_conectividad8.tif")
    guardar_zonas(zonas_labels, transform, crs, output_zonas)
    
    # Paso 3: Exportar estadísticas
    output_stats = os.path.join(output_dir, "estadisticas_zonas.csv")
    exportar_estadisticas_zonas(zonas_info, output_stats)
    
    # Paso 4: Generar histograma
    output_hist = os.path.join(output_dir, "distribucion_areas_zonas.png")
    visualizar_distribucion_areas(zonas_info, output_hist)
    
    # Resumen
    resumen = {
        'n_zonas': len(zonas_info),
        'area_total_km2': sum([z['area_km2'] for z in zonas_info]) if zonas_info else 0,
        'area_media_km2': np.mean([z['area_km2'] for z in zonas_info]) if zonas_info else 0,
        'output_raster': output_zonas,
        'output_csv': output_stats,
        'output_grafico': output_hist
    }
    
    print("\n" + "="*70)
    print(" ZONIFICACIÓN COMPLETADA ✓")
    print("="*70)
    print(f"\n  Zonas identificadas:  {resumen['n_zonas']:,}")
    print(f"  Área total:           {resumen['area_total_km2']:.2f} km²")
    print(f"  Área media por zona:  {resumen['area_media_km2']:.2f} km²")
    print(f"\n  Outputs:")
    print(f"    - Mapa de zonas:    {output_zonas}")
    print(f"    - Estadísticas CSV: {output_stats}")
    print(f"    - Histograma:       {output_hist}")
    print("\n" + "="*70 + "\n")
    
    return resumen