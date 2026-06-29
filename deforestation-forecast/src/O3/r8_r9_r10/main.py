import logging
from datetime import datetime

import geopandas as gpd

from O3.config import DISTRITOS_ALTO_CAMBIO_GPKG, PANEL_REPORTE_EJECUCION_CSV
from O3.utils import guardar_metadatos, iniciar_log_archivo, log_config, validar_fuentes

from O3.r8_r9_r10.metadata_fuentes import generar_metadata_fuentes
from O3.r8_r9_r10.construir_agropecuaria import construir_agropecuaria
from O3.r8_r9_r10.construir_rios_lagos   import construir_rios_lagos
from O3.r8_r9_r10.construir_urbano       import construir_urbano
from O3.r8_r9_r10.construir_carreteras   import construir_carreteras
from O3.r8_r9_r10.construir_rios         import construir_rios
from O3.r8_r9_r10.construir_anp          import construir_anp
from O3.r8_r9_r10.construir_elevacion    import construir_elevacion
from O3.r8_r9_r10.construir_pendiente    import construir_pendiente
from O3.r8_r9_r10.integrar_panel         import integrar_panel

logger = logging.getLogger(__name__)


def main():
    print("\n" + "=" * 70)
    print(" O3 — PIPELINE DE VARIABLES LOCALES (R8–R10)")
    print("=" * 70)

    log_config()
    validar_fuentes()

    # Metadata de la corrida completa (Pasos 0-4) — evidencia de ejecución
    # reproducible y sin errores para el IOV de R9. Se escribe en el `finally`
    # para dejar constancia incluso si el pipeline falla a mitad de camino.
    inicio = datetime.now()
    ultimo_paso_completado = None
    estado = "OK"
    error_msg = None

    try:
        # =================================================================
        # PASO 0: VALIDACIÓN INICIAL — metadata de fuentes + carga geometrías
        # =================================================================

        print("\n[PASO 0] Metadata de fuentes RAW")
        generar_metadata_fuentes()

        logger.info("Cargando distritos_alto_cambio.gpkg...")
        distritos_gdf = gpd.read_file(DISTRITOS_ALTO_CAMBIO_GPKG)
        logger.info(f"  {len(distritos_gdf)} distritos cargados")
        ultimo_paso_completado = "PASO 0"

        # =================================================================
        # PASO 1: VARIABLES MAPBIOMAS — cobertura temporal 1985-2024
        # =================================================================

        print("\n[PASO 1] Variables MapBiomas — cobertura temporal 1985-2024")

        construir_agropecuaria(distritos_gdf)   # pct_agropecuario — clases {9,15,21,35,40}
        construir_rios_lagos(distritos_gdf)     # pct_rios_lagos   — clase {33}
        construir_urbano(distritos_gdf)         # pct_urbano       — clase {24}
        ultimo_paso_completado = "PASO 1"

        # =================================================================
        # PASO 2: VARIABLES VECTORIALES
        # =================================================================

        print("\n[PASO 2] Variables vectoriales")

        construir_carreteras(distritos_gdf)     # km_carreteras — red vial MTC dic-2018 (estática)
        construir_rios(distritos_gdf)           # km_rios       — red hidrográfica ANA (estática)
        construir_anp(distritos_gdf)            # pct_anp       — ANP acumuladas año a año (temporal)
        ultimo_paso_completado = "PASO 2"

        # =================================================================
        # PASO 3: VARIABLES RASTER ESTÁTICAS — SRTM
        # =================================================================

        print("\n[PASO 3] Variables SRTM")

        construir_elevacion(distritos_gdf)      # elev_media_m         — DEM SRTM 30m (estática)
        construir_pendiente(distritos_gdf)      # pendiente_media_deg  — derivada del DEM (estática)
        ultimo_paso_completado = "PASO 3"

        # =================================================================
        # PASO 4: PANEL INTEGRADO
        # =================================================================

        print("\n[PASO 4] Panel integrado")

        integrar_panel(distritos_gdf)
        ultimo_paso_completado = "PASO 4"
    except Exception as exc:
        estado = "ERROR"
        error_msg = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        fin = datetime.now()
        guardar_metadatos(
            {
                "fecha_inicio":            inicio.isoformat(timespec="seconds"),
                "fecha_fin":               fin.isoformat(timespec="seconds"),
                "duracion_segundos":       round((fin - inicio).total_seconds(), 1),
                "pasos_esperados":         "PASO 0 a PASO 4",
                "ultimo_paso_completado":  ultimo_paso_completado,
                "estado":                  estado,
                "error":                   error_msg,
            },
            PANEL_REPORTE_EJECUCION_CSV,
        )
        logger.info(f"[OK] Metadata de ejecución: {PANEL_REPORTE_EJECUCION_CSV} (estado={estado})")

    print("\n" + "=" * 70)
    print(" O3 — Pipeline completo (Pasos 0–4)")
    print("=" * 70)


if __name__ == "__main__":
    iniciar_log_archivo("r8_r9_r10")
    main()
