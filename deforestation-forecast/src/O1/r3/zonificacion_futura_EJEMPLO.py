"""
PRÓXIMO PASO: Zonificación basada en mapa de cambios

Este archivo muestra cómo usar el mapa de cambios para identificar zonas.
NO ejecutar aún - es un esqueleto/plantilla para desarrollo futuro.
"""
import numpy as np
import rasterio
from scipy.ndimage import label
from skimage.measure import regionprops


def zonas_desde_cambios_EJEMPLO(mapa_cambios_path, area_min_km2=50):
    """
    EJEMPLO/PLANTILLA: Identifica zonas a partir del mapa de cambios.
    
    Estrategia:
    1. Extraer píxeles con cambio=1
    2. Identificar componentes conexas (grupos contiguos)
    3. Filtrar por área mínima
    4. Etiquetar cada grupo como zona
    
    Parameters:
    -----------
    mapa_cambios_path : str
        Ruta al mapa de cambios generado
    area_min_km2 : float
        Área mínima para considerar una zona válida
    
    Returns:
    --------
    zonas_labels : np.ndarray
        Raster con IDs de zonas (0=sin zona, 1,2,3...=zonas)
    """
    
    print("\n[EJEMPLO] Identificación de zonas desde mapa de cambios")
    
    # Leer mapa de cambios
    with rasterio.open(mapa_cambios_path) as src:
        cambios = src.read(1)
        transform = src.transform
        crs = src.crs
        meta = src.meta
    
    # Calcular área de píxel (aproximada en EPSG:4326)
    # Para EPSG:4326: 1 grado ≈ 111 km
    # Resolución típica: 0.00027 grados ≈ 30m
    res_deg = abs(transform[0])
    res_km = res_deg * 111  # Aproximación
    area_pixel_km2 = res_km ** 2
    
    print(f"  Resolución: {res_deg:.6f}° ≈ {res_km*1000:.1f}m")
    print(f"  Área píxel: {area_pixel_km2:.6f} km²")
    
    # Paso 1: Extraer solo píxeles con cambio
    mask_cambio = (cambios == 1)
    
    print(f"  Píxeles con cambio: {np.sum(mask_cambio):,}")
    
    # Paso 2: Identificar componentes conexas
    # Conectividad 8 (incluye diagonales)
    labels, n_componentes = label(mask_cambio, structure=np.ones((3, 3)))
    
    print(f"  Componentes conexas encontradas: {n_componentes}")
    
    # Paso 3: Calcular propiedades de cada componente
    props = regionprops(labels)
    
    # Paso 4: Filtrar por área mínima
    area_min_px = area_min_km2 / area_pixel_km2
    
    print(f"  Área mínima: {area_min_km2} km² = {area_min_px:.0f} píxeles")
    
    zonas_labels = np.zeros_like(labels)
    zonas_validas = []
    zona_id = 1
    
    for region in props:
        area_km2 = region.area * area_pixel_km2
        
        if region.area >= area_min_px:
            # Zona válida
            zonas_labels[labels == region.label] = zona_id
            
            zonas_validas.append({
                'zona_id': zona_id,
                'area_km2': area_km2,
                'n_pixels': region.area,
                'centroid': region.centroid
            })
            
            zona_id += 1
    
    print(f"\n  Zonas válidas (>= {area_min_km2} km²): {len(zonas_validas)}")
    
    if len(zonas_validas) > 0:
        print("\n  Top 10 zonas por área:")
        sorted_zonas = sorted(zonas_validas, key=lambda x: x['area_km2'], reverse=True)
        for i, zona in enumerate(sorted_zonas[:10], 1):
            print(f"    {i}. Zona {zona['zona_id']}: {zona['area_km2']:.2f} km² ({zona['n_pixels']:,} px)")
    
    return zonas_labels, zonas_validas, meta


def exportar_zonas_EJEMPLO(zonas_labels, meta, output_path):
    """
    EJEMPLO: Exporta zonas a raster.
    """
    meta_zonas = meta.copy()
    meta_zonas.update({
        'dtype': 'int32',
        'nodata': 0
    })
    
    with rasterio.open(output_path, 'w', **meta_zonas) as dst:
        dst.write(zonas_labels.astype('int32'), 1)
    
    print(f"\n[OK] Zonas guardadas: {output_path}")


# ============================================================================
# EJEMPLO DE USO (NO EJECUTAR AÚN)
# ============================================================================

def ejemplo_uso():
    """
    Muestra cómo usar las funciones anteriores.
    """
    
    # Rutas de ejemplo
    mapa_cambios = "data/interim/O1/mapas-cambios/mapa_cambios_1985_2024.tif"
    output_zonas = "data/interim/O1/zonas/zonas_desde_cambios.tif"
    
    # Identificar zonas
    zonas_labels, zonas_info, meta = zonas_desde_cambios_EJEMPLO(
        mapa_cambios, 
        area_min_km2=50
    )
    
    # Guardar
    exportar_zonas_EJEMPLO(zonas_labels, meta, output_zonas)
    
    # Análisis adicional
    print("\n[INFO] Próximos pasos:")
    print("  1. Visualizar zonas en QGIS")
    print("  2. Calcular serie temporal de pérdida por zona")
    print("  3. Construir panel zona-año")
    print("  4. Agregar covariables espaciales")


# ============================================================================
# ALTERNATIVA: GRID ADAPTATIVO
# ============================================================================

def grid_adaptativo_EJEMPLO(mapa_cambios_path, grid_size_km=10):
    """
    ALTERNATIVA: Crea grilla regular que se adapta a densidad de cambios.
    
    Estrategia:
    1. Dividir área en celdas regulares (ej. 10×10 km)
    2. Para cada celda, calcular densidad de cambios
    3. Mantener solo celdas con suficiente señal
    4. Cada celda = una zona
    
    Ventajas:
    - Zonas de tamaño comparable
    - Más fácil de interpretar
    - No depende de conectividad
    
    Desventajas:
    - Menos fiel a dinámica real
    - Puede cortar frentes de cambio
    """
    
    print("\n[EJEMPLO] Grid adaptativo basado en densidad de cambios")
    print(f"  Tamaño de grilla: {grid_size_km} km")
    
    # TODO: Implementar lógica
    # 1. Calcular número de celdas según extensión
    # 2. Para cada celda, contar píxeles con cambio
    # 3. Filtrar celdas con densidad mínima
    # 4. Etiquetar celdas válidas como zonas
    
    print("  [TODO] Implementación pendiente")


# ============================================================================
# ALTERNATIVA: CLUSTERING ESPACIAL
# ============================================================================

def clustering_espacial_EJEMPLO(mapa_cambios_path, eps_km=5, min_samples=100):
    """
    ALTERNATIVA: Usa DBSCAN para clustering espacial de píxeles con cambio.
    
    Estrategia:
    1. Extraer coordenadas de píxeles con cambio=1
    2. Aplicar DBSCAN sobre coordenadas
    3. Cada cluster = una zona
    
    Requiere: scikit-learn
    
    Parámetros DBSCAN:
    - eps: distancia máxima entre píxeles del mismo cluster (km)
    - min_samples: número mínimo de píxeles para formar cluster
    
    Ventajas:
    - Detecta formas arbitrarias
    - No requiere especificar número de clusters
    - Maneja ruido/outliers
    
    Desventajas:
    - Más lento
    - Parámetros sensibles
    - Requiere convertir coordenadas a sistema métrico
    """
    
    print("\n[EJEMPLO] Clustering DBSCAN sobre píxeles con cambio")
    print(f"  eps={eps_km} km, min_samples={min_samples}")
    
    # TODO: Implementar con sklearn.cluster.DBSCAN
    # 1. Extraer coordenadas (x,y) de píxeles con cambio=1
    # 2. Convertir a km (o reproyectar)
    # 3. Aplicar DBSCAN
    # 4. Asignar labels de cluster al raster
    
    print("  [TODO] Implementación pendiente")


if __name__ == "__main__":
    print("="*70)
    print("PLANTILLA PARA ZONIFICACIÓN DESDE MAPA DE CAMBIOS")
    print("="*70)
    print("\n[INFO] Este archivo es solo un ejemplo/plantilla.")
    print("[INFO] Muestra 3 estrategias posibles:")
    print("  1. Componentes conexas (simple, fiel a dinámica)")
    print("  2. Grid adaptativo (regular, fácil de interpretar)")
    print("  3. Clustering DBSCAN (flexible, detecta formas)")
    print("\n[INFO] Para implementar, descomentar y adaptar según necesidades.")
    print("="*70 + "\n")
