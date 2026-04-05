import os
import numpy as np
import rasterio
from O1.config import (
    ANIOS, MAPAS_RECLAS_DIR, MAPAS_PERDIDA_DIR, MAPAS_REPROYECTADOS_DIR, CRS_PROYECTADO,
    FEATURES_DIR, NUCLEOS_DIR, ZONAS_DIR, PANEL_DIR, PARAMS_ZONIFICACION,
)
from O1.r3.reproyeccion import reproyectar_a_metrico,reproyectar_con_referencia
from O1.r3.eventos import calcular_perdida_anual
from O1.r3.features import calcular_features_por_tiles, guardar_features
from O1.r3.score import calcular_score, suavizar_score, guardar_score, estadisticas_score
from O1.r3.nucleos import identificar_nucleos, guardar_nucleos
from O1.r3.expansion import calcular_bosque_remanente_actual, expandir_nucleos_sobre_bosque, estadisticas_expansion, guardar_zonas
from O1.r3.regularizacion import regularizar_zonas, guardar_zonas_regularizadas
from O1.r3.zonas import seleccionar_zonas_finales, guardar_zonas_finales, exportar_estadisticas_csv
from O1.r3.panel import construir_panel_zona_anio, agregar_flags_panel, generar_estadisticas_panel, exportar_panel


def main():
    """
    Pipeline completo de zonificación para predicción de deforestación.
    
    Fases:
    0. Reproyección a sistema métrico
    1. Detección de pérdida anual
    2. Cálculo de features por píxel
    3. Score de potencial histórico
    4. Identificación de núcleos
    5. Expansión a zonas candidatas
    6. Regularización jerárquica
    7. Selección de zonas finales
    11. Construcción de panel zona-año
    """
    
    print("\n" + "="*70)
    print(" PIPELINE DE ZONIFICACIÓN - R3")
    print("="*70)
    print(f"  Años: {ANIOS[0]} - {ANIOS[-1]} ({len(ANIOS)} años)")
    print(f"  Parámetros: {PARAMS_ZONIFICACION}")
    print("="*70 + "\n")
    
    # ========================================================================
    # FASE 0: REPROYECCIÓN
    # ========================================================================
    print("\n" + "="*70)
    print("FASE 0: REPROYECCIÓN A SISTEMA MÉTRICO")
    print("="*70 + "\n")
    
    # Paso 1: Reproyectar el primer año para establecer la grilla de referencia
    raster_primer_anio = os.path.join(
        MAPAS_RECLAS_DIR,
        f"bosque_nobosque_amazonia_{ANIOS[0]}.tif"
    )
    
    raster_primer_anio_proyectado = os.path.join(
        MAPAS_REPROYECTADOS_DIR,
        f"bosque_nobosque_{ANIOS[0]}_reproyectado.tif"
    )
    
    if not os.path.exists(raster_primer_anio_proyectado):
        print(f"[INFO] Reproyectando año base {ANIOS[0]} para establecer grilla de referencia...")
        reproyectar_a_metrico(raster_primer_anio, raster_primer_anio_proyectado, CRS_PROYECTADO)
    else:
        print(f"[SKIP] Grilla de referencia ya existe: {raster_primer_anio_proyectado}")
    
    # Leer grilla de referencia
    with rasterio.open(raster_primer_anio_proyectado) as ref:
        transform_ref = ref.transform
        crs_ref = ref.crs
        width_ref = ref.width
        height_ref = ref.height
        
        print(f"\n[INFO] Grilla de referencia:")
        print(f"  CRS: {crs_ref}")
        print(f"  Dimensiones: {width_ref} × {height_ref} píxeles")
        print(f"  Resolución: {abs(transform_ref[0]):.2f} m")
        print(f"  Área píxel: {(abs(transform_ref[0]) * abs(transform_ref[4])) / 1e6:.6f} km²\n")
    
    # Paso 2: Reproyectar el resto de años usando la MISMA grilla
    for anio in ANIOS[1:]:  # Desde el segundo año en adelante
        raster_path = os.path.join(
            MAPAS_RECLAS_DIR,
            f"bosque_nobosque_amazonia_{anio}.tif"
        )
        
        raster_proyectado_path = os.path.join(
            MAPAS_REPROYECTADOS_DIR,
            f"bosque_nobosque_{anio}_reproyectado.tif"
        )
        
        if os.path.exists(raster_proyectado_path):
            print(f"[SKIP] Ya existe: {raster_proyectado_path}")
            continue
        
        reproyectar_con_referencia(
            raster_path, 
            raster_proyectado_path, 
            transform_ref, 
            crs_ref, 
            width_ref, 
            height_ref
        )
    
    # ========================================================================
    # FASE 1: DETECCIÓN DE PÉRDIDA ANUAL
    # ========================================================================
    print("\n" + "="*70)
    print("FASE 1: DETECCIÓN DE PÉRDIDA ANUAL")
    print("="*70 + "\n")
    
    for i in range(len(ANIOS) - 1):
        anio_t = ANIOS[i]
        anio_t1 = ANIOS[i+1]
        
        raster_t = os.path.join(MAPAS_REPROYECTADOS_DIR, f"bosque_nobosque_{anio_t}_reproyectado.tif")
        raster_t1 = os.path.join(MAPAS_REPROYECTADOS_DIR, f"bosque_nobosque_{anio_t1}_reproyectado.tif")
        
        raster_perdida_path = os.path.join(MAPAS_PERDIDA_DIR, f"mapa_perdida_{anio_t1}.tif")
        
        if os.path.exists(raster_perdida_path):
            print(f"[SKIP] Ya existe: {raster_perdida_path}")
            continue
        
        calcular_perdida_anual(raster_t, raster_t1, raster_perdida_path)
    
    # ========================================================================
    # FASE 2: CÁLCULO DE FEATURES
    # ========================================================================
    print("\n" + "="*70)
    print("FASE 2: CÁLCULO DE FEATURES POR PÍXEL")
    print("="*70 + "\n")
    
    # Preparar rutas de rasters
    bosque_paths = [os.path.join(MAPAS_REPROYECTADOS_DIR, f"bosque_nobosque_{a}_reproyectado.tif") for a in ANIOS]
    perdida_paths = [os.path.join(MAPAS_PERDIDA_DIR, f"mapa_perdida_{a}.tif") for a in ANIOS[1:]]
    
    features, transform, crs = calcular_features_por_tiles(
        perdida_paths,
        bosque_paths,
        radio_densidad=PARAMS_ZONIFICACION['radio_densidad'],
        tile_size=5000
    )
    
    print(f"transform: {transform}")
    print(f"crs: {crs}")
    
    # Guardar features
    guardar_features(features, transform, crs, FEATURES_DIR)
    
    print("\n" + "="*70)
    print(" PIPELINE COMPLETADO ✓")
    print("="*70)
    print(f"\n  Outputs generados:")
    print(f"    - Features:           {FEATURES_DIR}")
    print(f"    - Núcleos:            {NUCLEOS_DIR}")
    print(f"    - Zonas:              {ZONAS_DIR}")
    print(f"    - Panel zona-año:     {PANEL_DIR}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()