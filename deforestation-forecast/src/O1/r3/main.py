import os
from O1.config import (
    ANIOS, BIOMAS_PERU_DIR, DISTRITOS_PERU_DIR, 
    MAPAS_RECLAS_DIR, DISTRITOS_AMAZONIA_DIR, MAPAS_CAMBIOS_DIR,
    METRICAS_DISTRITOS_DIR, DISTRITOS_ALTO_CAMBIO_DIR, 
    SERIES_ENTRENAMIENTO_DIR, SERIES_GENERALIZACION_ESPACIAL_DIR
)

from O1.r3.delimitacion_distritos_amazonas import pipeline_delimitacion_distritos_amazonia;

from O1.r3.deteccion_cambios import (
    detectar_cambios_por_tiles, 
    guardar_mapa_cambios, 
    exportar_estadisticas_cambios
)
from O1.r3.zonificacion_distrito import pipeline_zonificacion_distrito
from O1.r3.distritos_alto_cambio import pipeline_seleccion_distritos_alto_cambio
from O1.r3.series_temporales import pipeline_extraer_series_temporales

def main():
    """
    Pipeline completo: detección de cambios + zonificación por distrito.
    
    Pasos:
    1. Identificar píxeles que cambiaron (bosque <=> no bosque) en algún momento
    2. Guardar mapa de cambios y estadísticas
    3. Intersectar cambios con distritos amazónicos
    4. Generar shapefile con densidad de cambios y Excel con estadísticas
    """
    
    print("\n" + "="*70)
    print(" PIPELINE DE DETECCIÓN DE CAMBIOS + ZONIFICACIÓN ")
    print("="*70)

    # ========================================================================
    # PASO 1: GENERACIÓN DE DISTRITOS VÁLIDOS
    # ========================================================================

    print("\n" + "="*70)
    print(" INICIANDO GENERACIÓN DE DISTRITOS VÁLIDOS...")
    print("="*70)

    rutas_mapas_reclasificados = [
        os.path.join(MAPAS_RECLAS_DIR, f"bosque_nobosque_amazonia_{anio}.tif")
        for anio in ANIOS
    ]

    # Verificar que existen los archivos
    print("[INFO] Verificando archivos de entrada...")
    archivos_faltantes = [p for p in rutas_mapas_reclasificados if not os.path.exists(p)]
    
    if archivos_faltantes:
        print("\n[ERROR] Faltan archivos:")
        for f in archivos_faltantes:
            print(f" - {f}")
        
        print("\nEjecuta primero el pipeline R1/R2.")
        raise FileNotFoundError("Archivos de entrada faltantes")
    
    print(f"[OK] {len(rutas_mapas_reclasificados)} archivos encontrados\n")

    ruta_biomas_peru = os.path.join(BIOMAS_PERU_DIR, "BIOMES_v1.shp")
    ruta_distritos_peru = os.path.join(DISTRITOS_PERU_DIR, "POLITICAL_LEVEL_4_v1.shp")

    ruta_distritos_amazonia_delimitados = os.path.join(DISTRITOS_AMAZONIA_DIR, "distritos_bosque_minimo.gpkg")

    if os.path.exists(ruta_distritos_amazonia_delimitados):
        print(f"[INFO] El mapa de distritos ya existe: {ruta_distritos_amazonia_delimitados}.")
    else:
        pipeline_delimitacion_distritos_amazonia(ruta_biomas_peru, ruta_distritos_peru, ruta_distritos_amazonia_delimitados)

    # ========================================================================
    # PASO 2: DETECCIÓN DE CAMBIOS
    # ========================================================================

    print("\n" + "="*70)
    print(" INICIANDO DETECCIÓN DE CAMBIOS...")
    print("="*70)
    
    ruta_mapa_cambios = os.path.join(MAPAS_CAMBIOS_DIR, "mapa_cambios_1985_2024.tif")
    ruta_estadisticas_cambios = os.path.join(MAPAS_CAMBIOS_DIR, "estadisticas_cambios.csv")

    # Verificar si no existe el mapa de cambios
    if os.path.exists(ruta_mapa_cambios):
        print(f"[INFO] El mapa de cambios ya existe: {ruta_mapa_cambios}.")
    else:
        mapa_cambios, meta = detectar_cambios_por_tiles(
            rutas_mapas_reclasificados,
            tamanio_tile=5000
        )
        guardar_mapa_cambios(mapa_cambios, meta, ruta_mapa_cambios)
        exportar_estadisticas_cambios(ruta_mapa_cambios, ruta_estadisticas_cambios)

    # ========================================================================
    # PASO 3: ZONIFICACIÓN POR DISTRITO AMAZÓNICO
    # ========================================================================
    
    print("\n" + "="*70)
    print(" INICIANDO ZONIFICACIÓN POR DISTRITO...")
    print("="*70 + "\n")

    ruta_mapa_cambios_distrito = os.path.join(METRICAS_DISTRITOS_DIR, "mapa_cambios_distrito_1985_2024.gpkg")
    ruta_estadisticas_cambios_distrito = os.path.join(METRICAS_DISTRITOS_DIR, "estadisticas_cambios_distritos.csv")

    if os.path.exists(ruta_mapa_cambios_distrito):
        print(f"[INFO] El mapa de cambios por densidad en distritos ya existe: {ruta_mapa_cambios_distrito}.")
    else:
        pipeline_zonificacion_distrito(
            ruta_mapa_cambios,
            ruta_distritos_amazonia_delimitados,
            ruta_mapa_cambios_distrito,
            ruta_estadisticas_cambios_distrito
        )
    
    # ========================================================================
    # PASO 4: SELECCIONAR DISTRITOS
    # ========================================================================

    print("\n" + "="*70)
    print(" SELECCIÓN DE DISTRITOS PARA ENTRENAMIENTO...")
    print("="*70 + "\n")

    ruta_distritos_alto_cambio = os.path.join(DISTRITOS_ALTO_CAMBIO_DIR, "distritos_alto_cambio.gpkg")

    pipeline_seleccion_distritos_alto_cambio(
        ruta_mapa_cambios_distrito,
        ruta_distritos_alto_cambio
    )

    # ========================================================================
    # PASO 5: OBTENCIÓN DE SERIES TEMPORALES POR DISTRITOS
    # ========================================================================
    
    print("\n" + "="*70)
    print(" INICIANDO OBTENCIÓN DE SERIES TEMPORALES POR ZONAS...")
    print("="*70 + "\n")
    
    print(f"[INFO] Rango de años: {min(ANIOS)} - {max(ANIOS)}\n")
    
    ruta_series_entrenamiento = os.path.join(SERIES_ENTRENAMIENTO_DIR, "distritos_entrenamiento.csv")
    ruta_estadisticas_series_entrenamiento = os.path.join(SERIES_ENTRENAMIENTO_DIR, "estadisticas_distritos_entrenamiento.csv")

    ruta_series_generalizacion = os.path.join(SERIES_GENERALIZACION_ESPACIAL_DIR, "distritos_generalizacion_espacial.csv")
    ruta_estadisticas_series_generalizacion = os.path.join(SERIES_GENERALIZACION_ESPACIAL_DIR, "estadisticas_distritos_generalizacion_espacial.csv")
    
    # Verificar si ya existe el panel
    if os.path.exists(ruta_series_entrenamiento) and os.path.exists(ruta_series_generalizacion):
        print(f"[INFO] El panel de series temporales ya existe: {ruta_series_entrenamiento}, {ruta_series_generalizacion}")
    else:    
        pipeline_extraer_series_temporales(
            rutas_mapas_reclasificados,
            ruta_distritos_alto_cambio,
            ruta_series_entrenamiento,
            ruta_estadisticas_series_entrenamiento,
            ruta_series_generalizacion,
            ruta_estadisticas_series_generalizacion
        )

    return
    
if __name__ == "__main__":
    main()
