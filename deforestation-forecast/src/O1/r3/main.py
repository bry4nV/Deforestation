"""
Pipeline simplificado para detección de cambios y zonificación.

ENFOQUE NUEVO:
- NO reproyección (trabajamos en sistema original EPSG:4326)
- Detección simple de píxeles que cambiaron (bosque ↔ no bosque)
- Procesamiento por tiles para manejar memoria
- Generación de mapa de cambios como base para zonificación futura
"""
import os
import numpy as np
import rasterio
from O1.config import (
    ANIOS, MAPAS_RECLAS_DIR, O1_INTERIM_DIR
)
from O1.r3.deteccion_cambios import (
    detectar_cambios_por_tiles, 
    guardar_mapa_cambios, 
    exportar_estadisticas_cambios
)


def main():
    """
    Pipeline simplificado de detección de cambios.
    
    Pasos:
    1. Identificar píxeles que cambiaron (bosque ↔ no bosque) en algún momento
    2. Guardar mapa de cambios
    3. Exportar estadísticas
    """
    
    print("\n" + "="*70)
    print(" PIPELINE DE DETECCIÓN DE CAMBIOS - R3 (VERSIÓN SIMPLIFICADA)")
    print("="*70)
    print(f"  Años: {ANIOS[0]} - {ANIOS[-1]} ({len(ANIOS)} años)")
    print(f"  Sistema: EPSG:4326 (sin reproyección)")
    print("="*70 + "\n")
    
    # Crear directorio de salida
    CAMBIOS_DIR = os.path.join(O1_INTERIM_DIR, "mapas-cambios")
    os.makedirs(CAMBIOS_DIR, exist_ok=True)
    
    # ========================================================================
    # PASO 1: DETECCIÓN DE CAMBIOS
    # ========================================================================
    
    # Preparar rutas de rasters originales (bosque/no bosque)
    raster_paths = [
        os.path.join(MAPAS_RECLAS_DIR, f"bosque_nobosque_amazonia_{anio}.tif")
        for anio in ANIOS
    ]
    
    # Verificar que existen los archivos
    print("[INFO] Verificando archivos de entrada...")
    archivos_faltantes = [p for p in raster_paths if not os.path.exists(p)]
    
    if archivos_faltantes:
        print("\n[ERROR] Archivos faltantes:")
        for f in archivos_faltantes[:5]:  # Mostrar primeros 5
            print(f"  - {f}")
        if len(archivos_faltantes) > 5:
            print(f"  ... y {len(archivos_faltantes) - 5} más")
        print("\n[INFO] Asegúrate de ejecutar primero el pipeline R1/R2 para generar estos archivos.")
        return
    
    print(f"[OK] {len(raster_paths)} archivos encontrados\n")
    
    # Detectar cambios por tiles
    mapa_cambios, transform, crs, stats = detectar_cambios_por_tiles(
        raster_paths,
        tile_size=5000  # Procesar en tiles de 5000x5000 píxeles
    )
    
    # ========================================================================
    # PASO 2: GUARDAR RESULTADOS
    # ========================================================================
    
    # Guardar mapa de cambios
    output_cambios = os.path.join(CAMBIOS_DIR, "mapa_cambios_1985_2024.tif")
    guardar_mapa_cambios(mapa_cambios, transform, crs, output_cambios)
    
    # Exportar estadísticas
    output_stats = os.path.join(CAMBIOS_DIR, "estadisticas_cambios.txt")
    exportar_estadisticas_cambios(stats, output_stats)
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    
    print("\n" + "="*70)
    print(" DETECCIÓN DE CAMBIOS COMPLETADA ✓")
    print("="*70)
    print(f"\n  Outputs generados:")
    print(f"    - Mapa de cambios: {output_cambios}")
    print(f"    - Estadísticas:    {output_stats}")
    print(f"\n  Resumen:")
    print(f"    - Píxeles con cambio: {stats['pixeles_con_cambio']:,} ({stats['porcentaje_cambio']:.2f}%)")
    print(f"    - Píxeles estables:   {stats['pixeles_sin_cambio']:,}")
    print("\n" + "="*70 + "\n")
    
    print("[INFO] Próximos pasos:")
    print("  1. Visualizar mapa_cambios_1985_2024.tif en QGIS")
    print("  2. Usar este mapa como base para identificar zonas de cambio")
    print("  3. Aplicar clustering espacial sobre píxeles con cambio=1")
    print()


if __name__ == "__main__":
    main()
