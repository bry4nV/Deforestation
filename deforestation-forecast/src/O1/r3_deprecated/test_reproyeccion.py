"""
Script de prueba para verificar que la reproyección funciona correctamente.
"""
import os
import rasterio
from O1.config import MAPAS_RECLAS_DIR, MAPAS_REPROYECTADOS_DIR, CRS_PROYECTADO, ANIOS
from O1.r3.reproyeccion import reproyectar_a_metrico

def test_reproyeccion():
    """
    Prueba la reproyección del primer año y verifica el resultado.
    """
    print("\n" + "="*70)
    print("TEST DE REPROYECCIÓN")
    print("="*70 + "\n")
    
    # Raster original
    raster_original = os.path.join(
        MAPAS_RECLAS_DIR,
        f"bosque_nobosque_amazonia_{ANIOS[0]}.tif"
    )
    
    # Raster reproyectado
    raster_proyectado = os.path.join(
        MAPAS_REPROYECTADOS_DIR,
        f"test_bosque_{ANIOS[0]}_reproyectado.tif"
    )
    
    # Verificar que existe el original
    if not os.path.exists(raster_original):
        print(f"[ERROR] No existe raster original: {raster_original}")
        return
    
    print(f"[INFO] Raster original: {raster_original}")
    
    # Leer información del original
    with rasterio.open(raster_original) as src:
        print(f"\n  ORIGINAL:")
        print(f"    CRS:         {src.crs}")
        print(f"    Dimensiones: {src.width} × {src.height} píxeles")
        print(f"    Resolución:  {abs(src.transform[0]):.6f}° × {abs(src.transform[4]):.6f}°")
        print(f"    Bounds:      {src.bounds}")
    
    # Reproyectar
    print(f"\n[INFO] Reproyectando a {CRS_PROYECTADO}...")
    reproyectar_a_metrico(raster_original, raster_proyectado)
    
    # Verificar resultado
    with rasterio.open(raster_proyectado) as dst:
        print(f"\n  REPROYECTADO:")
        print(f"    CRS:         {dst.crs}")
        print(f"    Dimensiones: {dst.width} × {dst.height} píxeles")
        print(f"    Resolución:  {abs(dst.transform[0]):.2f} m × {abs(dst.transform[4]):.2f} m")
        print(f"    Área píxel:  {(abs(dst.transform[0]) * abs(dst.transform[4])) / 1e6:.6f} km²")
        print(f"    Bounds:      {dst.bounds}")
        
        # Verificar que efectivamente cambió el CRS
        if str(dst.crs) == CRS_PROYECTADO:
            print(f"\n  ✓ CRS correcto: {dst.crs}")
        else:
            print(f"\n  ✗ ERROR: CRS esperado {CRS_PROYECTADO}, obtenido {dst.crs}")
    
    print("\n" + "="*70 + "\n")
    print(f"[OK] Archivo de prueba generado: {raster_proyectado}")
    print("     Puedes abrirlo en QGIS para verificar visualmente.")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    test_reproyeccion()
