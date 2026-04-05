"""
Este módulo identifica qué píxeles han experimentado algún cambio entre bosque y no bosque
durante toda la serie temporal.
"""
import numpy as np
import rasterio
from rasterio.windows import Window

def detectar_cambios_por_tiles(rutas_raster, tamanio_tile=5000):
    """
    Detecta píxeles que cambiaron (bosque <=> no bosque) en algún momento de la serie.
    
    Procesa por tiles espaciales para evitar problemas de memoria.
    
    Parameters:
    -----------
    rutas_raster : list
        Lista de rutas a rasters de bosque/no bosque
    tamanio_tile : int
        Tamaño del tile en píxeles (default: 5000x5000)
    
    Returns:
    --------
    np.ndarray : Mapa de cambios (1=cambió, 0=sin cambio, 255=nodata)
    transform : Transformación geoespacial
    crs : Sistema de coordenadas
    dict : Estadísticas de cambios
    """
    print("\n" + "="*70)
    print("DETECCIÓN DE CAMBIOS BOSQUE <=> NO BOSQUE")
    print("="*70 + "\n")
    
    # Leer metadata del primer raster
    with rasterio.open(rutas_raster[0]) as src:
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs
    
    cantidad_anios = len(rutas_raster)
    print(f"[INFO] Dimensiones de ráster bosque - no bosque: {width} x {height} píxeles")
    
    # Calcular número de tiles
    n_tiles_x = (width + tamanio_tile - 1) // tamanio_tile
    n_tiles_y = (height + tamanio_tile - 1) // tamanio_tile
    total_tiles = n_tiles_x * n_tiles_y
    
    print(f"[INFO] Procesando en {n_tiles_x} x {n_tiles_y} = {total_tiles} tiles")
    print(f"[INFO] Tamaño de tile: {tamanio_tile} x {tamanio_tile} píxeles\n")
    
    # Inicializar mapa de cambios
    mapa_cambios = np.zeros((height, width), dtype=np.uint8)
    
    # Estadísticas globales
    total_pixeles_validos = 0
    total_pixeles_con_cambio = 0
    
    # Procesar tile por tile
    indice_tile = 0
    for tile_y in range(n_tiles_y):
        for tile_x in range(n_tiles_x):
            indice_tile += 1
            
            # Definir ventana del tile
            col_off = tile_x * tamanio_tile
            row_off = tile_y * tamanio_tile
            
            width_tile = min(tamanio_tile, width - col_off)
            height_tile = min(tamanio_tile, height - row_off)
            
            window = Window(col_off, row_off, width_tile, height_tile)
            
            print(f"  Tile {indice_tile}/{total_tiles}: "
                  f"row={row_off}, col={col_off}, size={width_tile}x{height_tile}")
            
            # Cargar datos del tile para todos los años
            conjunto_tiles = np.zeros((cantidad_anios, height_tile, width_tile), dtype=np.uint8)
            
            for t, path in enumerate(rutas_raster):
                with rasterio.open(path) as src:
                    conjunto_tiles[t] = src.read(1, window=window)
            
            # Detectar cambios en este tile
            cambios_tile, stats_tile = detectar_cambios_tile(conjunto_tiles)
            
            # Escribir resultados en mapa global
            mapa_cambios[row_off:row_off+height_tile, col_off:col_off+width_tile] = cambios_tile
            
            # Acumular estadísticas
            total_pixeles_validos += stats_tile['pixeles_validos']
            total_pixeles_con_cambio += stats_tile['pixeles_con_cambio']
            
            # Liberar memoria
            del conjunto_tiles, cambios_tile
    
    # Estadísticas finales
    pct_cambio = (total_pixeles_con_cambio / total_pixeles_validos * 100) if total_pixeles_validos > 0 else 0
    
    print("\n" + "="*70)
    print("ESTADÍSTICAS DE CAMBIOS")
    print("="*70)
    print(f"  Píxeles válidos:      {total_pixeles_validos:,}")
    print(f"  Píxeles con cambio:   {total_pixeles_con_cambio:,} ({pct_cambio:.2f}%)")
    print(f"  Píxeles sin cambio:   {total_pixeles_validos - total_pixeles_con_cambio:,} ({100-pct_cambio:.2f}%)")
    print("="*70 + "\n")
    
    stats = {
        'pixeles_validos': total_pixeles_validos,
        'pixeles_con_cambio': total_pixeles_con_cambio,
        'pixeles_sin_cambio': total_pixeles_validos - total_pixeles_con_cambio,
        'porcentaje_cambio': pct_cambio
    }
    
    return mapa_cambios, transform, crs, stats


def detectar_cambios_tile(conjunto_tiles):
    """
    Detecta cambios en un tile espacial a lo largo de la serie temporal.
    
    Parameters:
    -----------
    conjunto_tiles : np.ndarray
        Stack 3D (tiempo, y, x) con valores bosque/no bosque
        1 = bosque, 0 = no bosque, 255 = nodata
    
    Returns:
    --------
    cambios : np.ndarray
        Mapa binario (1=cambió, 0=sin cambio, 255=nodata)
    stats : dict
        Estadísticas del tile
    """
    cantidad_anios, height, width = conjunto_tiles.shape
    
    # Inicializar mapa de cambios
    cambios = np.zeros((height, width), dtype=np.uint8)
    
    # Máscara de píxeles válidos (excluir nodata)
    # Un píxel es válido si NO es nodata en NINGÚN año
    mask_nodata = np.any(conjunto_tiles == 255, axis=0)
    mask_valido = ~mask_nodata
    
    # Para píxeles válidos, detectar si hubo algún cambio
    for y in range(height):
        for x in range(width):
            if mask_valido[y, x]:
                # Extraer serie temporal del píxel
                serie = conjunto_tiles[:, y, x]
                
                # Detectar si hubo algún cambio (transición 0→1 o 1→0)
                if hubo_cambio(serie):
                    cambios[y, x] = 1  # Cambió
                else:
                    cambios[y, x] = 0  # No cambió
            else:
                cambios[y, x] = 255  # Nodata
    
    # Estadísticas del tile
    pixeles_validos = np.sum(mask_valido)
    pixeles_con_cambio = np.sum(cambios == 1)
    
    stats = {
        'pixeles_validos': int(pixeles_validos),
        'pixeles_con_cambio': int(pixeles_con_cambio)
    }
    
    return cambios, stats


def hubo_cambio(serie):
    """
    Detecta si hubo algún cambio en la serie temporal de un píxel.
    
    Un cambio se define como una transición entre estados distintos:
    - 0 → 1 (no bosque → bosque)
    - 1 → 0 (bosque → no bosque)
    
    Parameters:
    -----------
    serie : np.ndarray
        Serie temporal de valores (0 o 1)
    
    Returns:
    --------
    bool : True si hubo al menos un cambio, False si no
    """
    # Detectar transiciones comparando año t con año t+1
    for t in range(len(serie) - 1):
        if serie[t] != serie[t+1]:
            return True  # Hubo un cambio
    
    return False  # No hubo cambios


def guardar_mapa_cambios(mapa_cambios, transform, crs, output_path):
    """
    Guarda el mapa de cambios como raster GeoTIFF.
    
    Parameters:
    -----------
    mapa_cambios : np.ndarray
        Mapa de cambios (1=cambió, 0=sin cambio, 255=nodata)
    transform : affine.Affine
        Transformación geoespacial
    crs : CRS
        Sistema de coordenadas
    output_path : str
        Ruta de salida
    """
    meta = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'nodata': 255,
        'width': mapa_cambios.shape[1],
        'height': mapa_cambios.shape[0],
        'count': 1,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(mapa_cambios, 1)
    
    print(f"[OK] Mapa de cambios guardado: {output_path}")


def exportar_estadisticas_cambios(stats, output_path):
    """
    Exporta estadísticas de cambios a archivo de texto.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("ESTADÍSTICAS DE CAMBIOS BOSQUE ↔ NO BOSQUE\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Píxeles válidos:      {stats['pixeles_validos']:,}\n")
        f.write(f"Píxeles con cambio:   {stats['pixeles_con_cambio']:,} ({stats['porcentaje_cambio']:.2f}%)\n")
        f.write(f"Píxeles sin cambio:   {stats['pixeles_sin_cambio']:,} ({100 - stats['porcentaje_cambio']:.2f}%)\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"[OK] Estadísticas exportadas: {output_path}")
