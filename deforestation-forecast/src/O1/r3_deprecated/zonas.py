import numpy as np
import rasterio
from skimage.measure import regionprops


def renumerar_zonas_secuencial(zonas_labels):
    """
    Renumera zonas de forma secuencial (1, 2, 3, ...) eliminando gaps.
    
    Returns:
    --------
    np.ndarray : Zonas renumeradas
    dict : Mapeo {id_antiguo: id_nuevo}
    """
    zonas_ids = np.unique(zonas_labels)
    zonas_ids = zonas_ids[zonas_ids > 0]
    
    zonas_renumeradas = np.zeros_like(zonas_labels)
    mapeo = {}
    
    for nuevo_id, viejo_id in enumerate(zonas_ids, start=1):
        zonas_renumeradas[zonas_labels == viejo_id] = nuevo_id
        mapeo[int(viejo_id)] = nuevo_id
    
    return zonas_renumeradas, mapeo


def verificar_contigüidad(zonas_labels):
    """
    Verifica que cada zona sea una componente conexa única.
    Si una zona tiene múltiples componentes desconectadas, las separa.
    
    Returns:
    --------
    np.ndarray : Zonas con contigüidad corregida
    """
    from scipy.ndimage import label as scipy_label
    
    print("\n[INFO] Verificando contigüidad de zonas...")
    
    zonas_ids = np.unique(zonas_labels)
    zonas_ids = zonas_ids[zonas_ids > 0]
    
    zonas_corregidas = np.zeros_like(zonas_labels)
    nuevo_id_global = 1
    zonas_divididas = 0
    
    for zona_id in zonas_ids:
        mask_zona = (zonas_labels == zona_id)
        
        # Etiquetar componentes dentro de esta zona
        componentes, n_comp = scipy_label(mask_zona, structure=np.ones((3, 3)))
        
        if n_comp == 1:
            # Zona contigua, OK
            zonas_corregidas[mask_zona] = nuevo_id_global
            nuevo_id_global += 1
        else:
            # Zona fragmentada, dividir en múltiples zonas
            print(f"  Zona {zona_id} fragmentada en {n_comp} componentes → dividiendo")
            for comp_id in range(1, n_comp + 1):
                mask_comp = (componentes == comp_id)
                zonas_corregidas[mask_comp] = nuevo_id_global
                nuevo_id_global += 1
            zonas_divididas += 1
    
    if zonas_divididas > 0:
        print(f"[OK] {zonas_divididas} zonas divididas por falta de contigüidad")
    else:
        print("  Todas las zonas son contiguas")
    
    return zonas_corregidas


def calcular_estadisticas_finales(zonas_labels, perdida_stack, bosque_stack, transform):
    """
    Calcula estadísticas completas de zonas finales.
    
    Returns:
    --------
    dict : Estadísticas por zona
    """
    zonas_ids = np.unique(zonas_labels)
    zonas_ids = zonas_ids[zonas_ids > 0]
    
    area_pixel_km2 = (abs(transform[0]) ** 2) / 1e6
    
    stats = {}
    
    for zid in zonas_ids:
        mask_zona = (zonas_labels == zid)
        n_pixels = np.sum(mask_zona)
        area_km2 = n_pixels * area_pixel_km2
        
        # Bosque remanente
        bosque_actual = (bosque_stack[-1] == 1) & mask_zona
        area_bosque_km2 = np.sum(bosque_actual) * area_pixel_km2
        pct_bosque = (area_bosque_km2 / area_km2 * 100) if area_km2 > 0 else 0
        
        # Serie temporal de pérdida
        perdida_zona = perdida_stack[:, mask_zona]
        perdida_por_anio = np.sum(perdida_zona == 1, axis=1)
        
        # Centroide
        props = regionprops((zonas_labels == zid).astype(int))
        if props:
            centroid_px = props[0].centroid
            # Convertir a coordenadas geográficas
            centroid_x = transform[2] + centroid_px[1] * transform[0]
            centroid_y = transform[5] + centroid_px[0] * transform[4]
        else:
            centroid_x, centroid_y = np.nan, np.nan
        
        stats[int(zid)] = {
            'zona_id': int(zid),
            'area_km2': float(area_km2),
            'n_pixels': int(n_pixels),
            'bosque_remanente_km2': float(area_bosque_km2),
            'bosque_remanente_pct': float(pct_bosque),
            'perdida_total_px': int(np.sum(perdida_por_anio)),
            'anios_activos': int(np.sum(perdida_por_anio > 0)),
            'perdida_media_anual_px': float(np.mean(perdida_por_anio)),
            'perdida_max_anual_px': int(np.max(perdida_por_anio)),
            'centroid_x': float(centroid_x),
            'centroid_y': float(centroid_y)
        }
    
    return stats


def seleccionar_zonas_finales(zonas_regularizadas, perdida_stack, bosque_stack, transform, params):
    """
    Pipeline de selección de zonas finales.
    
    Returns:
    --------
    np.ndarray : Zonas finales
    dict : Estadísticas de zonas finales
    """
    print("\n" + "="*60)
    print("FASE 7: SELECCIÓN DE ZONAS FINALES")
    print("="*60)
    
    # Verificar contigüidad
    zonas_contiguas = verificar_contigüidad(zonas_regularizadas)
    
    # Renumerar secuencialmente
    zonas_finales, mapeo = renumerar_zonas_secuencial(zonas_contiguas)
    
    n_zonas_finales = len(np.unique(zonas_finales)) - 1  # excluir 0
    print(f"\n[OK] Zonas finales: {n_zonas_finales}")
    
    # Calcular estadísticas
    stats = calcular_estadisticas_finales(zonas_finales, perdida_stack, bosque_stack, transform)
    
    # Resumen estadístico
    areas = [s['area_km2'] for s in stats.values()]
    bosque_pcts = [s['bosque_remanente_pct'] for s in stats.values()]
    
    print(f"\n  Estadísticas de área:")
    print(f"    Media:   {np.mean(areas):.1f} km²")
    print(f"    Mediana: {np.median(areas):.1f} km²")
    print(f"    Min:     {np.min(areas):.1f} km²")
    print(f"    Max:     {np.max(areas):.1f} km²")
    
    print(f"\n  Bosque remanente:")
    print(f"    Media:   {np.mean(bosque_pcts):.1f}%")
    print(f"    Mediana: {np.median(bosque_pcts):.1f}%")
    print(f"    Min:     {np.min(bosque_pcts):.1f}%")
    print(f"    Max:     {np.max(bosque_pcts):.1f}%")
    
    print("="*60 + "\n")
    
    return zonas_finales, stats


def guardar_zonas_finales(zonas, transform, crs, output_path):
    """
    Guarda las zonas finales como raster GeoTIFF.
    """
    meta = {
        'driver': 'GTiff',
        'dtype': 'int32',
        'nodata': 0,
        'width': zonas.shape[1],
        'height': zonas.shape[0],
        'count': 1,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(zonas.astype('int32'), 1)
    
    print(f"[OK] Zonas finales guardadas: {output_path}")


def exportar_estadisticas_csv(stats, output_path):
    """
    Exporta estadísticas de zonas a CSV.
    """
    import pandas as pd
    
    df = pd.DataFrame.from_dict(stats, orient='index')
    df = df.sort_values('zona_id')
    
    df.to_csv(output_path, index=False, float_format='%.4f')
    
    print(f"[OK] Estadísticas exportadas: {output_path}")
