"""
Este módulo identifica qué píxeles han experimentado algún cambio entre bosque y no bosque
durante toda la serie temporal.
"""
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from O1.config import NODATA

def detectar_cambios_por_tiles(rutas_mapas_reclasificados, tamanio_tile=5000):
    """
    Detecta píxeles que cambiaron (bosque <=> no bosque) en algún momento de la serie del tile.
    
    Parameters:
    -----------
    rutas_mapas_reclasificados : list
        Lista de rutas a mapas reclasificados de bosque/no bosque
    tamanio_tile : int
        Tamaño del tile en píxeles (default: 5000x5000)
    
    Returns:
    --------
    np.ndarray : Mapa de cambios (1=cambió, 0=sin cambio, 255=nodata)
    meta : Metadata del mapa (transformación geoespacial)
    """
    print("\n" + "="*70)
    print("DETECCIÓN DE CAMBIOS BOSQUE <=> NO BOSQUE")
    print("="*70 + "\n")
    
    # Leer metadata del primer raster
    with rasterio.open(rutas_mapas_reclasificados[0]) as src:
        meta = src.meta.copy()
    
    cantidad_anios = len(rutas_mapas_reclasificados)
    width = meta['width']
    height = meta['height']

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
            
            for t, path in enumerate(rutas_mapas_reclasificados):
                with rasterio.open(path) as src:
                    conjunto_tiles[t] = src.read(1, window=window)
            
            tile_cambios, estadisticas_tile = detectar_cambios_tile(conjunto_tiles)
            total_pixeles_validos += estadisticas_tile['pixeles_validos']
            total_pixeles_con_cambio += estadisticas_tile['pixeles_con_cambio']
            
            # Escribir resultados en mapa global
            mapa_cambios[row_off:row_off+height_tile, col_off:col_off+width_tile] = tile_cambios
            
            # Liberar memoria
            del conjunto_tiles, tile_cambios
    
    # Estadísticas finales
    porcentaje_cambio = (total_pixeles_con_cambio / total_pixeles_validos * 100) if total_pixeles_validos > 0 else 0
    
    print("\n" + "="*70)
    print("ESTADÍSTICAS DE CAMBIOS")
    print("="*70)
    print(f"  Píxeles válidos:      {total_pixeles_validos:,}")
    print(f"  Píxeles con cambio:   {total_pixeles_con_cambio:,} ({porcentaje_cambio:.2f}%)")
    print(f"  Píxeles sin cambio:   {total_pixeles_validos - total_pixeles_con_cambio:,} ({100-porcentaje_cambio:.2f}%)")
    print("="*70 + "\n")
    
    return mapa_cambios, meta


def detectar_cambios_tile(conjunto_tiles):
    """
    Detecta cambios en un tile espacial a lo largo de la serie temporal.
    
    Parameters:
    -----------
    conjunto_tiles : np.ndarray
        Stack 3D (tiempo, y, x) con valores bosque/no bosque (1/0) y nodata (255)
    
    Returns:
    --------
    tile_cambios : np.ndarray
        Mapa binario (1=cambió, 0=sin cambio, 255=nodata)
    estadisticas_tile : dict
        Estadísticas del tile
    """
    _, height, width = conjunto_tiles.shape
    
    # Inicializar tile de cambios
    tile_cambios = np.zeros((height, width), dtype=np.uint8)
    
    # Máscara de píxeles válidos (excluir nodata)
    mascara_nodata = np.any(conjunto_tiles == NODATA, axis=0)
    mascara_valido = ~mascara_nodata
    
    # Para píxeles válidos, detectar si hubo algún cambio
    for y in range(height):
        for x in range(width):
            if mascara_valido[y, x]:
                # Extraer serie temporal del píxel
                serie = conjunto_tiles[:, y, x]
                
                if hubo_cambio(serie):
                    tile_cambios[y, x] = 1  # Cambió
                else:
                    tile_cambios[y, x] = 0  # No cambió
            else:
                tile_cambios[y, x] = NODATA  # Nodata
    
    estadisticas_tile = {
        'pixeles_validos': int(np.count_nonzero(mascara_valido)),
        'pixeles_con_cambio': int(np.count_nonzero(tile_cambios == 1))
    }
    
    return tile_cambios, estadisticas_tile


def hubo_cambio(serie):
    """
    Detecta si hubo algún cambio en la serie temporal de un píxel.
    
    Un cambio se define como una transición entre estados distintos:
    - 0 → 1 (no bosque → bosque) o 1 → 0 (bosque → no bosque)
    
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
            return True
    
    return False


def guardar_mapa_cambios(mapa_cambios, meta, ruta_mapa_cambios):
    """
    Guarda el mapa de cambios como raster GeoTIFF.
    
    Parameters:
    -----------
    mapa_cambios : np.ndarray
        Mapa de cambios (1=cambió, 0=sin cambio, 255=nodata)
    meta : dict
        Metadatos del raster (transformación, CRS, etc.)
    ruta_mapa_cambios : str
        Ruta de salida
    """

    meta_out = meta.copy()
    meta_out.update(dtype="uint8", count=1, nodata=NODATA, compress="lzw")
    
    with rasterio.open(ruta_mapa_cambios, 'w', **meta_out) as dst:
        dst.write(mapa_cambios, 1)
    
    print(f"[OK] Mapa de cambios guardado: {ruta_mapa_cambios}")


def exportar_estadisticas_cambios(ruta_mapa_cambios, ruta_estadisticas_cambios):
    """
    Exporta estadísticas del mapa de cambios a un archivo csv.
    """

    with rasterio.open(ruta_mapa_cambios) as src:
        mapa = src.read(1)
        meta = src.meta

    mascara_valido = (mapa != meta['nodata'])
    mascara_cambio = (mapa == 1)

    total_pixeles = mapa.size
    pixeles_validos = np.count_nonzero(mascara_valido)
    pixeles_con_cambio = np.count_nonzero(mascara_cambio)
    pixeles_sin_cambio = pixeles_validos - pixeles_con_cambio

    porcentaje_cambio = (pixeles_con_cambio / pixeles_validos * 100) if pixeles_validos > 0 else 0
    porcentaje_sin_cambio = 100 - porcentaje_cambio

    df = pd.DataFrame([{
        "total_pixeles": int(total_pixeles),
        "pixeles_validos": int(pixeles_validos),
        "pixeles_con_cambio": int(pixeles_con_cambio),
        "pixeles_sin_cambio": int(pixeles_sin_cambio),
        "porcentaje_cambio": round(porcentaje_cambio, 4),
        "porcentaje_sin_cambio": round(porcentaje_sin_cambio, 4),
    }])

    df.to_csv(ruta_estadisticas_cambios, index=False)

    print(f"[OK] Estadísticas exportadas: {ruta_estadisticas_cambios}")
