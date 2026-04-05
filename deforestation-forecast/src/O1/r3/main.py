import os
from O1.config import (
    ANIOS, MAPAS_RECLAS_DIR, MAPAS_CAMBIOS_DIR, ZONAS_DIR
)
from O1.r3.deteccion_cambios import (
    detectar_cambios_por_tiles, 
    guardar_mapa_cambios, 
    exportar_estadisticas_cambios
)
from O1.r3.zonificacion import pipeline_zonificacion


def main():
    """
    Pipeline completo: detección de cambios + zonificación.
    
    Pasos:
    1. Identificar píxeles que cambiaron (bosque <=> no bosque) en algún momento
    2. Guardar mapa de cambios y estadísticas
    3. Identificar zonas de deforestación mediante componentes conectados
    4. Exportar mapa de zonas, estadísticas y visualizaciones
    """
    
    print("\n" + "="*70)
    print(" PIPELINE DE DETECCIÓN DE CAMBIOS - R3")
    print("="*70)
    
    # ========================================================================
    # PASO 1: DETECCIÓN DE CAMBIOS
    # ========================================================================
    
    rutas_raster = [
        os.path.join(MAPAS_RECLAS_DIR, f"bosque_nobosque_amazonia_{anio}.tif")
        for anio in ANIOS
    ]
    
    output_cambios = os.path.join(MAPAS_CAMBIOS_DIR, "mapa_cambios_1985_2024.tif")
    output_stats = os.path.join(MAPAS_CAMBIOS_DIR, "estadisticas_cambios.txt")
    
    # Verificar que existen los archivos
    print("[INFO] Verificando archivos de entrada...")
    archivos_faltantes = [p for p in rutas_raster if not os.path.exists(p)]
    
    if archivos_faltantes:
        print("\n[ERROR] Faltan archivos:")
        for f in archivos_faltantes:
            print(f" - {f}")
        
        print("\nEjecuta primero el pipeline R1/R2.")
        raise FileNotFoundError("Archivos de entrada faltantes")

    print(f"[OK] {len(rutas_raster)} archivos encontrados\n")

    # Verificar si no existe el mapa de cambios
    if os.path.exists(output_cambios):
        print(f"[INFO] El mapa de cambios ya existe: {output_cambios}, no se volverá a generar.")
    else:
        mapa_cambios, transform, crs, stats = detectar_cambios_por_tiles(
            rutas_raster,
            tamanio_tile=5000
        )
        guardar_mapa_cambios(mapa_cambios, transform, crs, output_cambios)
        exportar_estadisticas_cambios(stats, output_stats)
    
    # ========================================================================
    # PASO 2: ZONIFICACIÓN
    # ========================================================================
    
    print("\n" + "="*70)
    print(" INICIANDO ZONIFICACIÓN...")
    print("="*70 + "\n")
    
    resumen_zonas = pipeline_zonificacion(
        mapa_cambios_path=output_cambios,
        output_dir=ZONAS_DIR,
        pixeles_min=20,    # Mínimo 20 píxeles por zona para considerar válida
    )
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    
    print("\n" + "="*70)
    print(" PIPELINE COMPLETO FINALIZADO ✓")
    print("="*70)
    
    print(f"\n  📊 DETECCIÓN DE CAMBIOS:")
    print(f"    - Mapa de cambios: {output_cambios}")
    print(f"    - Píxeles con cambio: {stats['pixeles_con_cambio']:,} ({stats['porcentaje_cambio']:.2f}%)")
    print(f"    - Píxeles estables:   {stats['pixeles_sin_cambio']:,}")
    
    print(f"\n  🗺️  ZONIFICACIÓN:")
    print(f"    - Zonas identificadas: {resumen_zonas['n_zonas']:,}")
    print(f"    - Área total:          {resumen_zonas['area_total_km2']:.2f} km²")
    print(f"    - Área media por zona: {resumen_zonas['area_media_km2']:.2f} km²")
    print(f"    - Mapa de zonas:       {resumen_zonas['output_raster']}")
    print(f"    - Estadísticas CSV:    {resumen_zonas['output_csv']}")
    print(f"    - Histograma:          {resumen_zonas['output_grafico']}")
    
    print("\n  📝 PRÓXIMOS PASOS:")
    print("    1. Visualizar zonas en QGIS")
    print("    2. Revisar estadisticas_zonas.csv")
    print("    3. Calcular series temporales por zona")
    print("    4. Construir panel zona-año para modelado")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
