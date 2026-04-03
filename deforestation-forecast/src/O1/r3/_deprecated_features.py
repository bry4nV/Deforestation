import numpy as np
import rasterio
from scipy.ndimage import uniform_filter, binary_dilation
from rasterio.windows import Window


def calcular_features_por_tiles(raster_paths_perdida, raster_paths_bosque, radio_densidad=2, tile_size=5000):
    """
    Calcula features procesando el raster por tiles espaciales para evitar problemas de memoria.
    
    Parameters:
    -----------
    raster_paths_perdida : list
        Lista de rutas a rasters de pérdida
    raster_paths_bosque : list
        Lista de rutas a rasters de bosque
    radio_densidad : int
        Radio para densidad local
    tile_size : int
        Tamaño del tile en píxeles (default: 5000x5000)
    
    Returns:
    --------
    dict : Features calculados
    """
    print(f"\n[INFO] Calculando features por tiles (tile_size={tile_size}x{tile_size})...")
    
    with rasterio.open(raster_paths_bosque[0]) as src:
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs
        profile = src.profile
    
    n_years = len(raster_paths_bosque)
    print(f"[INFO] Dimensiones: {width} x {height} píxeles, {n_years} años")
    print(f"[INFO] Memoria requerida (completo): {(n_years * height * width / 1e9):.2f} GB")
    
    # Calcular número de tiles
    n_tiles_x = (width + tile_size - 1) // tile_size
    n_tiles_y = (height + tile_size - 1) // tile_size
    total_tiles = n_tiles_x * n_tiles_y
    
    print(f"[INFO] Procesando en {n_tiles_x} x {n_tiles_y} = {total_tiles} tiles")
    
    # Inicializar arrays para features (se llenan tile por tile)
    features_global = {
        'freq': np.zeros((height, width), dtype=np.float32),
        'recencia': np.zeros((height, width), dtype=np.float32),
        'persist_bosque': np.zeros((height, width), dtype=np.float32),
        'exposicion': np.zeros((height, width), dtype=np.float32),
        'densidad_local': np.zeros((height, width), dtype=np.float32)
    }
    
    # Procesar tile por tile
    tile_idx = 0
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            tile_idx += 1
            
            # Definir ventana del tile con overlap para densidad local
            overlap = radio_densidad * 2
            
            col_off = tx * tile_size
            row_off = ty * tile_size
            
            # Tile con overlap
            col_off_overlap = max(0, col_off - overlap)
            row_off_overlap = max(0, row_off - overlap)
            
            width_tile = min(tile_size + 2*overlap, width - col_off_overlap)
            height_tile = min(tile_size + 2*overlap, height - row_off_overlap)
            
            window = Window(col_off_overlap, row_off_overlap, width_tile, height_tile)
            
            # Tile sin overlap (para escribir resultados)
            col_off_inner = col_off
            row_off_inner = row_off
            width_inner = min(tile_size, width - col_off)
            height_inner = min(tile_size, height - row_off)
            
            # Índices dentro del tile con overlap
            inner_col_start = col_off - col_off_overlap
            inner_row_start = row_off - row_off_overlap
            inner_col_end = inner_col_start + width_inner
            inner_row_end = inner_row_start + height_inner
            
            print(f"  Tile {tile_idx}/{total_tiles}: "
                  f"row={row_off_inner}, col={col_off_inner}, "
                  f"size={width_inner}x{height_inner}")
            
            # Cargar datos del tile para todos los años
            perdida_tile = np.zeros((n_years - 1, height_tile, width_tile), dtype=np.uint8)
            bosque_tile = np.zeros((n_years, height_tile, width_tile), dtype=np.uint8)
            
            for t, path_bosque in enumerate(raster_paths_bosque):
                with rasterio.open(path_bosque) as src:
                    bosque_tile[t] = src.read(1, window=window)
            
            for t, path_perdida in enumerate(raster_paths_perdida):
                with rasterio.open(path_perdida) as src:
                    perdida_tile[t] = src.read(1, window=window)
            
            # Calcular features para este tile
            features_tile = calcular_features_tile(perdida_tile, bosque_tile, radio_densidad)
            
            # Extraer región interna (sin overlap) y escribir en features globales
            for fname, fdata in features_tile.items():
                features_global[fname][row_off_inner:row_off_inner+height_inner, 
                                       col_off_inner:col_off_inner+width_inner] = \
                    fdata[inner_row_start:inner_row_end, inner_col_start:inner_col_end]
            
            # Liberar memoria
            del perdida_tile, bosque_tile, features_tile
    
    print(f"[OK] Features calculados exitosamente")
    
    return features_global, transform, crs


def calcular_features_tile(perdida_stack, bosque_stack, radio_densidad=2):
    """
    Calcula features para un tile espacial.
    """
    n_years = bosque_stack.shape[0]
    n_years_perdida = perdida_stack.shape[0]
    
    # Máscara válida
    valid_mask = (perdida_stack[0] != 255) & (bosque_stack[0] != 255)
    
    # Feature 1: Frecuencia de pérdida
    freq = np.sum(perdida_stack == 1, axis=0).astype(np.float32)
    actividad = (freq >= 1).astype(np.float32)
    
    # Feature 2: Recencia
    recencia = np.zeros(perdida_stack.shape[1:], dtype=np.float32)
    for t in range(n_years_perdida):
        mask = (perdida_stack[t] == 1) & valid_mask
        recencia[mask] = t
    recencia = recencia / max(1, n_years_perdida - 1)
    
    # Feature 3: Persistencia bosque
    persist_bosque = np.sum(bosque_stack == 1, axis=0).astype(np.float32) / n_years
    
    # Feature 4: Exposición al borde
    exposicion = np.zeros(bosque_stack.shape[1:], dtype=np.float32)
    for t in range(n_years):
        nobosque = (bosque_stack[t] == 0) & valid_mask
        borde_dilated = binary_dilation(nobosque, iterations=1)
        en_borde = borde_dilated & (bosque_stack[t] == 1) & valid_mask
        exposicion += en_borde.astype(np.float32)
    exposicion = exposicion / n_years
    
    # Feature 5: Densidad local
    window_size = 2 * radio_densidad + 1
    densidad_local = uniform_filter(actividad, size=window_size, mode='constant', cval=0.0)
    
    # Aplicar máscara
    for arr in [freq, recencia, persist_bosque, exposicion, densidad_local]:
        arr[~valid_mask] = np.nan
    
    return {
        'freq': freq,
        'recencia': recencia,
        'persist_bosque': persist_bosque,
        'exposicion': exposicion,
        'densidad_local': densidad_local
    }

def normalizar_feature(feature, percentil_min=5, percentil_max=95):
    """
    Normaliza un feature al rango [0, 1] usando percentiles robustos.
    """
    valid_vals = feature[~np.isnan(feature)]
    if len(valid_vals) == 0:
        return feature
    
    vmin = np.percentile(valid_vals, percentil_min)
    vmax = np.percentile(valid_vals, percentil_max)
    
    feature_norm = (feature - vmin) / (vmax - vmin + 1e-10)
    feature_norm = np.clip(feature_norm, 0, 1)
    
    return feature_norm


def guardar_features(features_dict, transform, crs, output_dir):
    """
    Guarda cada feature como raster GeoTIFF.
    """
    import os
    
    meta = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': np.nan,
        'width': features_dict['freq'].shape[1],
        'height': features_dict['freq'].shape[0],
        'count': 1,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }
    
    for name, data in features_dict.items():
        output_path = os.path.join(output_dir, f"feature_{name}.tif")
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(data.astype('float32'), 1)
        print(f"[OK] Feature guardado: {output_path}")