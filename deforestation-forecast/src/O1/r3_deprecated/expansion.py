import numpy as np
from scipy.ndimage import distance_transform_edt, label


def calcular_bosque_remanente_actual(bosque_stack):
    """
    Calcula bosque remanente en el último año disponible.
    
    Parameters:
    -----------
    bosque_stack : np.ndarray
        Stack temporal de bosque (tiempo, y, x)
    
    Returns:
    --------
    np.ndarray : Máscara de bosque remanente (True = bosque actual)
    """
    # Último año
    bosque_actual = bosque_stack[-1] == 1
    
    return bosque_actual


def expandir_nucleos_sobre_bosque(nucleos_labels, bosque_remanente, transform, params):
    """
    Expande cada núcleo sobre bosque remanente contiguo.
    
    Parameters:
    -----------
    nucleos_labels : np.ndarray
        Labels de núcleos (0=fondo, 1,2,3...=núcleos)
    bosque_remanente : np.ndarray
        Máscara de bosque actual (True=bosque)
    transform : affine.Affine
        Transformación geoespacial
    params : dict
        Parámetros: distancia_max_expansion_km, area_objetivo_zona_km2
    
    Returns:
    --------
    np.ndarray : Labels de zonas expandidas
    """
    print("\n" + "="*60)
    print("FASE 5: EXPANSIÓN A ZONAS CANDIDATAS")
    print("="*60)
    
    # Calcular resolución espacial
    pixel_size_m = abs(transform[0])  # asume cuadrados
    area_pixel_km2 = (pixel_size_m ** 2) / 1e6
    
    distancia_max_px = int(params['distancia_max_expansion_km'] * 1000 / pixel_size_m)
    area_objetivo_px = int(params['area_objetivo_zona_km2'] / area_pixel_km2)
    
    print(f"[INFO] Distancia máxima expansión: {params['distancia_max_expansion_km']} km = {distancia_max_px} px")
    print(f"[INFO] Área objetivo zona: {params['area_objetivo_zona_km2']} km² = {area_objetivo_px:,} px")
    
    # Identificar núcleos únicos
    nucleos_ids = np.unique(nucleos_labels)
    nucleos_ids = nucleos_ids[nucleos_ids > 0]  # excluir fondo
    
    if len(nucleos_ids) == 0:
        print("[WARN] No hay núcleos para expandir")
        return nucleos_labels.copy()
    
    # Inicializar zonas expandidas con núcleos originales
    zonas = nucleos_labels.copy()
    
    # Para cada núcleo
    for nucleo_id in nucleos_ids:
        # Máscara del núcleo actual
        mask_nucleo = (nucleos_labels == nucleo_id)
        area_nucleo_px = np.sum(mask_nucleo)
        
        # Si el núcleo ya es muy grande, no expandir
        if area_nucleo_px >= area_objetivo_px * 2:
            print(f"[INFO] Núcleo {nucleo_id}: ya es grande ({area_nucleo_px:,} px), no se expande")
            continue
        
        # Calcular distancia desde el núcleo
        distancia = distance_transform_edt(~mask_nucleo)
        
        # Identificar píxeles candidatos para expansión:
        # - Deben ser bosque remanente
        # - Deben estar dentro de distancia máxima
        # - No deben pertenecer a otro núcleo/zona
        candidatos = (
            bosque_remanente & 
            (distancia <= distancia_max_px) & 
            (distancia > 0) &
            (zonas == 0)
        )
        
        if not np.any(candidatos):
            print(f"[INFO] Núcleo {nucleo_id}: sin bosque candidato en vecindad")
            continue
        
        # Ordenar candidatos por distancia (más cercanos primero)
        indices_candidatos = np.argwhere(candidatos)
        distancias_candidatos = distancia[candidatos]
        orden = np.argsort(distancias_candidatos)
        indices_ordenados = indices_candidatos[orden]
        
        # Agregar píxeles hasta alcanzar área objetivo
        area_actual = area_nucleo_px
        agregados = 0
        
        for idx in indices_ordenados:
            if area_actual >= area_objetivo_px:
                break
            
            y, x = idx
            zonas[y, x] = nucleo_id
            area_actual += 1
            agregados += 1
        
        area_final_km2 = area_actual * area_pixel_km2
        print(f"[OK] Núcleo {nucleo_id}: {area_nucleo_px:,} → {area_actual:,} px ({area_final_km2:.2f} km²) [+{agregados:,}]")
    
    print("="*60 + "\n")
    
    return zonas


def calcular_bloque_forestal(bosque_remanente):
    """
    Identifica bloques forestales contiguos.
    Útil para restringir expansión dentro del mismo bloque.
    
    Returns:
    --------
    np.ndarray : Labels de bloques forestales
    """
    # Etiquetar componentes conexas de bosque
    bloques, n_bloques = label(bosque_remanente, structure=np.ones((3, 3)))
    
    print(f"[INFO] Bloques forestales identificados: {n_bloques}")
    
    return bloques


def estadisticas_expansion(nucleos_labels, zonas_labels, transform):
    """
    Calcula estadísticas antes/después de expansión.
    """
    area_pixel_km2 = (abs(transform[0]) ** 2) / 1e6
    
    print("\n" + "="*60)
    print("ESTADÍSTICAS DE EXPANSIÓN")
    print("="*60)
    
    nucleos_ids = np.unique(nucleos_labels)
    nucleos_ids = nucleos_ids[nucleos_ids > 0]
    
    for nid in nucleos_ids[:10]:  # mostrar primeros 10
        area_nucleo = np.sum(nucleos_labels == nid) * area_pixel_km2
        area_zona = np.sum(zonas_labels == nid) * area_pixel_km2
        incremento = area_zona - area_nucleo
        pct_incremento = (incremento / area_nucleo * 100) if area_nucleo > 0 else 0
        
        print(f"  Zona {nid}: {area_nucleo:.1f} → {area_zona:.1f} km² (+{incremento:.1f} km², +{pct_incremento:.1f}%)")
    
    print("="*60 + "\n")


def guardar_zonas(zonas, transform, crs, output_path):
    """
    Guarda las zonas candidatas como raster GeoTIFF.
    """
    import rasterio
    
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
    
    print(f"[OK] Zonas candidatas guardadas: {output_path}")
