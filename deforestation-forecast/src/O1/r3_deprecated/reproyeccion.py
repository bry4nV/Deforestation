import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from O1.config import CRS_PROYECTADO


def reproyectar_a_metrico(raster_path, raster_proyectado_path, dst_crs, resolucion_m=None):
    """
    Reproyecta un raster a sistema de coordenadas métrico.
    
    Parameters:
    -----------
    raster_path : str
        Ruta del raster original (EPSG:4326)
    raster_proyectado_path : str
        Ruta del raster reproyectado (sistema métrico)
    dst_crs : str
        Sistema de coordenadas de destino
    resolucion_m : float or None
        Resolución en metros.
    """
    
    with rasterio.open(raster_path) as src:
        
        # Calcular transformación y dimensiones para el CRS destino
        transform, width, height = calculate_default_transform(
            src.crs, 
            dst_crs, 
            src.width, 
            src.height, 
            *src.bounds,
            resolution=resolucion_m  # None = automático
        )
        
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        src_data = src.read(1)
        
        # Crear raster destino y reproyectar
        with rasterio.open(raster_proyectado_path, 'w', **kwargs) as dst:
            reproject(
                source=src_data,
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest  # Nearest para datos categóricos
            )
    
    print(f"[OK] Reproyectado {dst_crs}: {raster_proyectado_path}")


def reproyectar_con_referencia(raster_path, raster_proyectado_path, transform_ref, crs_ref, width_ref, height_ref):
    """
    Reproyecta un raster usando una grilla de referencia EXACTA.
    Garantiza que TODOS los años tengan exactamente la misma malla.
    
    Parameters:
    -----------
    raster_path : str
        Ruta del raster a reproyectar
    raster_proyectado_path : str
        Ruta del raster reproyectado
    transform_ref : affine.Affine
        Transformación de referencia
    crs_ref : CRS
        Sistema de coordenadas de referencia
    width_ref : int
        Ancho en píxeles de referencia
    height_ref : int
        Alto en píxeles de referencia
    """
    
    with rasterio.open(raster_path) as src:
        
        # Preparar metadata con grilla de referencia
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': crs_ref,
            'transform': transform_ref,
            'width': width_ref,
            'height': height_ref
        })
        
        # Crear raster destino y reproyectar
        with rasterio.open(raster_proyectado_path, 'w', **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform_ref,
                dst_crs=crs_ref,
                resampling=Resampling.nearest
            )
    
    print(f"[OK] Reproyectado (grilla fija): {raster_proyectado_path}")