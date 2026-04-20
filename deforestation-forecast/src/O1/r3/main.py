import os
from O1.config import (
    ANIOS, MAPAS_RECLAS_DIR, 
    BIOMAS_PERU_DIR, DISTRITOS_PERU_DIR, BOSQUE_MINIMO_DIR,
    CAMBIOS_DIR, DEFORESTACION_DIR, REFORESTACION_DIR,
    METRICAS_CAMBIOS_DIR, METRICAS_DEFORESTACION_DIR, METRICAS_REFORESTACION_DIR,
    DISTRITOS_AMAZONIA_DIR, ZONAS_DIR
)

from O1.r3.delimitacion_distritos_amazonas import pipeline_delimitacion_distritos_amazonia;

from O1.r3.deteccion_cambios import (
    detectar_cambios_por_tiles, 
    guardar_mapa_cambios, 
    exportar_estadisticas_cambios
)

from O1.r3.zonificacion_distrito import pipeline_zonificacion_distrito

from O1.r3.seleccion_distritos import seleccionar_distritos_entrenamiento;

# from O1.r3.zonificacion import pipeline_zonificacion
# from O1.r3.series_temporales import (
#     extraer_series_temporales_por_zona,
#     generar_estadisticas_series_temporales,
#     visualizar_series_temporales_muestra,
#     validar_series_temporales
# )


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

    anio_inicial = min(ANIOS)

    ruta_mapa_inicial = os.path.join(MAPAS_RECLAS_DIR, f"bosque_nobosque_amazonia_{anio_inicial}.tif")

    ruta_biomas_peru = os.path.join(BIOMAS_PERU_DIR, "BIOMES_v1.shp")
    ruta_distritos_peru = os.path.join(DISTRITOS_PERU_DIR, "POLITICAL_LEVEL_4_v1.shp")

    ruta_distritos_amazonia_delimitados = os.path.join(BOSQUE_MINIMO_DIR, "distritos_bosque_minimo.gpkg")

    if os.path.exists(ruta_distritos_amazonia_delimitados):
        print(f"[INFO] El mapa de distritos ya existe: {ruta_distritos_amazonia_delimitados}.")
    else:
        pipeline_delimitacion_distritos_amazonia(ruta_biomas_peru, ruta_distritos_peru, ruta_mapa_inicial, ruta_distritos_amazonia_delimitados)

    # ========================================================================
    # PASO 2: DETECCIÓN DE CAMBIOS
    # ========================================================================

    print("\n" + "="*70)
    print(" INICIANDO DETECCIÓN DE CAMBIOS...")
    print("="*70)
    
    ruta_mapa_cambios = os.path.join(CAMBIOS_DIR, "mapa_cambios_1985_2024.tif")
    ruta_estadisticas_cambios = os.path.join(CAMBIOS_DIR, "estadisticas_cambios.csv")

    ruta_mapa_cambios_deforestacion = os.path.join(DEFORESTACION_DIR, "mapa_deforestacion_1985_2024.tif")
    ruta_estadisticas_cambios_deforestacion = os.path.join(DEFORESTACION_DIR, "estadisticas_deforestacion.csv")

    ruta_mapa_cambios_reforestacion = os.path.join(REFORESTACION_DIR, "mapa_reforestacion_1985_2024.tif")
    ruta_estadisticas_cambios_reforestacion = os.path.join(REFORESTACION_DIR, "estadisticas_reforestacion.csv")

    # Verificar si no existe el mapa de cambios
    if os.path.exists(ruta_mapa_cambios):
        print(f"[INFO] El mapa de cambios ya existe: {ruta_mapa_cambios}.")
    else:
        mapa_cambios, meta = detectar_cambios_por_tiles(
            rutas_mapas_reclasificados,
            tamanio_tile=5000,
            tipo_cambio="cambios"
        )
        guardar_mapa_cambios(mapa_cambios, meta, ruta_mapa_cambios)
        exportar_estadisticas_cambios(ruta_mapa_cambios, ruta_estadisticas_cambios)

    # Verificar si no existe el mapa de cambios de deforestación
    if os.path.exists(ruta_mapa_cambios_deforestacion):
        print(f"[INFO] El mapa de cambios de deforestación ya existe: {ruta_mapa_cambios_deforestacion}.")
    else:
        mapa_cambios, meta = detectar_cambios_por_tiles(
            rutas_mapas_reclasificados,
            tamanio_tile=5000,
            tipo_cambio="deforestacion"

        )
        guardar_mapa_cambios(mapa_cambios, meta, ruta_mapa_cambios_deforestacion)
        exportar_estadisticas_cambios(ruta_mapa_cambios_deforestacion, ruta_estadisticas_cambios_deforestacion)
    
    # Verificar si no existe el mapa de cambios de reforestación
    if os.path.exists(ruta_mapa_cambios_reforestacion):
        print(f"[INFO] El mapa de cambios de reforestación ya existe: {ruta_mapa_cambios_reforestacion}.")
    else:
        mapa_cambios, meta = detectar_cambios_por_tiles(
            rutas_mapas_reclasificados,
            tamanio_tile=5000,
            tipo_cambio="reforestacion"
        )
        guardar_mapa_cambios(mapa_cambios, meta, ruta_mapa_cambios_reforestacion)
        exportar_estadisticas_cambios(ruta_mapa_cambios_reforestacion, ruta_estadisticas_cambios_reforestacion)

    # ========================================================================
    # PASO 3: ZONIFICACIÓN POR DISTRITO AMAZÓNICO
    # ========================================================================
    
    print("\n" + "="*70)
    print(" INICIANDO ZONIFICACIÓN POR DISTRITO...")
    print("="*70 + "\n")

    ruta_mapa_cambios_distrito = os.path.join(METRICAS_CAMBIOS_DIR, "mapa_cambios_distrito_1985_2024.gpkg")
    ruta_estadisticas_cambios_distrito = os.path.join(METRICAS_CAMBIOS_DIR, "estadisticas_cambios_distrito.csv")

    ruta_mapa_cambios_deforestacion_distrito = os.path.join(METRICAS_DEFORESTACION_DIR, "mapa_cambios_deforestacion_distrito_1985_2024.gpkg")
    ruta_estadisticas_cambios_deforestacion_distrito = os.path.join(METRICAS_DEFORESTACION_DIR, "estadisticas_cambios_deforestacion_distrito.csv")

    ruta_mapa_cambios_reforestacion_distrito = os.path.join(METRICAS_REFORESTACION_DIR, "mapa_cambios_reforestacion_distrito_1985_2024.gpkg")
    ruta_estadisticas_cambios_reforestacion_distrito = os.path.join(METRICAS_REFORESTACION_DIR, "estadisticas_cambios_reforestacion_distrito.csv")

    if os.path.exists(ruta_mapa_cambios_distrito):
        print(f"[INFO] El mapa de cambios por densidad en distritos ya existe: {ruta_mapa_cambios_distrito}.")
    else:
        pipeline_zonificacion_distrito(
            ruta_mapa_cambios,
            ruta_distritos_amazonia_delimitados,
            ruta_mapa_cambios_distrito,
            ruta_estadisticas_cambios_distrito
        )
    
    if os.path.exists(ruta_mapa_cambios_deforestacion_distrito):
        print(f"[INFO] El mapa de cambios por densidad en distritos ya existe: {ruta_mapa_cambios_deforestacion_distrito}.")
    else:
        pipeline_zonificacion_distrito(
            ruta_mapa_cambios_deforestacion,
            ruta_distritos_amazonia_delimitados,
            ruta_mapa_cambios_deforestacion_distrito,
            ruta_estadisticas_cambios_deforestacion_distrito
        )
    
    if os.path.exists(ruta_mapa_cambios_reforestacion_distrito):
        print(f"[INFO] El mapa de cambios por densidad en distritos ya existe: {ruta_mapa_cambios_reforestacion_distrito}.")
    else:
        pipeline_zonificacion_distrito(
            ruta_mapa_cambios_reforestacion,
            ruta_distritos_amazonia_delimitados,
            ruta_mapa_cambios_reforestacion_distrito,
            ruta_estadisticas_cambios_reforestacion_distrito
        )
    
    # ========================================================================
    # PASO 4: SELECCIONAR DISTRITOS
    # ========================================================================

    print("\n" + "="*70)
    print(" SELECCIÓN DE DISTRITOS PARA ENTRENAMIENTO...")
    print("="*70 + "\n")

    gdf_total, gdf_entrenamiento = seleccionar_distritos_entrenamiento(
        ruta_mapa_cambios_distrito,
        ruta_mapa_cambios_deforestacion_distrito,
        ruta_mapa_cambios_reforestacion_distrito
    )

    # ========================================================================
    # PASOS ANTERIORES (COMENTADOS - ENFOQUE ANTIGUO)
    # ========================================================================
    
    # PASO 2 (ANTIGUO): ZONIFICACIÓN POR CONECTIVIDAD
    # print("\n" + "="*70)
    # print(" INICIANDO ZONIFICACIÓN...")
    # print("="*70 + "\n")
    # 
    # ruta_mapa_zonas = os.path.join(ZONAS_DIR, "zonas_cambios_conectividad_8.tif")
    # 
    # pipeline_zonificacion(
    #     ruta_mapa_cambios,
    #     ZONAS_DIR,
    #     pixeles_min=1000,
    # )
    #
    # if os.path.exists(ruta_mapa_zonas):
    #     print(f"[INFO] El mapa de zonas ya existe: {ruta_mapa_zonas}")
    #     print("[INFO] Saltando zonificación.\n")
    # else:
    #     pipeline_zonificacion(
    #         ruta_mapa_cambios,
    #         ZONAS_DIR,
    #         pixeles_min=1000,
    #     )
    #
    # # ========================================================================
    # # PASO 3 (ANTIGUO): OBTENCIÓN DE SERIES TEMPORALES POR ZONAS
    # # ========================================================================
    # 
    # print("\n" + "="*70)
    # print(" INICIANDO OBTENCIÓN DE SERIES TEMPORALES POR ZONAS...")
    # print("="*70 + "\n")
    # 
    # # Usar mapas de bosque/no bosque (1985-2024)
    # # La pérdida se calcula internamente comparando años consecutivos
    # print(f"[INFO] Mapas de bosque/no bosque: {len(rutas_raster)}")
    # print(f"[INFO] Rango de años: {min(ANIOS)} - {max(ANIOS)}\n")
    # 
    # # Rutas de salida
    # ruta_panel_csv = os.path.join(ZONAS_DIR, "panel_series_temporales_zonas.csv")
    # ruta_stats_csv = os.path.join(ZONAS_DIR, "estadisticas_series_temporales_zonas.csv")
    # ruta_grafico = os.path.join(ZONAS_DIR, "series_temporales_top_zonas.png")
    # 
    # # Verificar que existe el mapa de zonas
    # if not os.path.exists(ruta_mapa_zonas):
    #     print(f"[ERROR] Mapa de zonas no encontrado: {ruta_mapa_zonas}")
    #     print("[ERROR] Ejecuta primero el paso de zonificación.")
    #     return
    # 
    # # Verificar si ya existe el panel
    # if os.path.exists(ruta_panel_csv):
    #     print(f"[INFO] El panel de series temporales ya existe: {ruta_panel_csv}")
    #     print("[INFO] Saltando extracción de series temporales.\n")
    #     return
    # 
    # # Extraer series temporales
    # df_panel = extraer_series_temporales_por_zona(
    #     ruta_mapa_zonas=ruta_mapa_zonas,
    #     rutas_mapas_bosque=rutas_raster,
    #     anios=ANIOS,
    #     ruta_salida_csv=ruta_panel_csv
    # )
    # 
    # # Validar
    # validar_series_temporales(df_panel)
    # 
    # # Generar estadísticas
    # generar_estadisticas_series_temporales(
    #     df_panel=df_panel,
    #     ruta_salida_stats=ruta_stats_csv
    # )
    # 
    # # Visualizar
    # visualizar_series_temporales_muestra(
    #     df_panel=df_panel,
    #     ruta_salida_grafico=ruta_grafico,
    #     n_zonas=10
    # )
    
if __name__ == "__main__":
    main()
