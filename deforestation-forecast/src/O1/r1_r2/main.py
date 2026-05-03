from O1.config import (
    ANIOS, BIOMAS_PERU_DIR, DISTRITOS_AMAZONIA_DIR, DISTRITOS_PERU_DIR, 
    MAPAS_RAW_DIR, MAPAS_AMAZONIA_DIR, MAPAS_RECLAS_DIR
)
from O1.r1_r2.pipeline import ejecutar_pipeline_anio
from O1.r1_r2.delimitacion_mapa_amazonas import pipeline_delimitacion_amazonia;
import pandas as pd
import os

if __name__ == "__main__":

    # Delimitacion mapa amazonas.
    ruta_biomas_peru = os.path.join(BIOMAS_PERU_DIR, "BIOMES_v1.shp")
    ruta_distritos_peru = os.path.join(DISTRITOS_PERU_DIR, "POLITICAL_LEVEL_4_v1.shp")

    ruta_distritos_amazonia = os.path.join(DISTRITOS_AMAZONIA_DIR, "distritos_amazonia.gpkg")
    carpeta_mapas_raw = MAPAS_RAW_DIR
    carpeta_mapas_amazonia = MAPAS_AMAZONIA_DIR

    mapas_generados = [
        f for f in os.listdir(carpeta_mapas_amazonia)
        if f.endswith(".tif")
    ] if os.path.exists(carpeta_mapas_amazonia) else []

    if (not os.path.exists(ruta_distritos_amazonia)) or (len(mapas_generados) == 0):
        print("[INFO] Ejecutando pipeline Amazonía") 
        pipeline_delimitacion_amazonia(
            ruta_biomas_peru,
            ruta_distritos_peru,
            ruta_distritos_amazonia,
            carpeta_mapas_raw,
            carpeta_mapas_amazonia
        )
    else:
        print("[INFO] Todo ya está generado. No se ejecuta pipeline.")

    registros_raw = []
    registros_reclasificados = []

    for anio in ANIOS:
        info = ejecutar_pipeline_anio(anio)

        if info is not None:
            raw = {"anio": anio, **info.get("raw", {})}
            recl = {"anio": anio, **info.get("reclasificado", {})}

            registros_raw.append(raw)
            registros_reclasificados.append(recl)

    if registros_raw:
        df_raw = pd.DataFrame(registros_raw)
        ruta_raw = os.path.join(MAPAS_AMAZONIA_DIR, "metadatos_mapas_amazonia.csv")
        df_raw.to_csv(ruta_raw, index=False)
        print(f"[OK] Metadatos RAW guardados en {ruta_raw}")

    if registros_reclasificados:
        df_recl = pd.DataFrame(registros_reclasificados)
        ruta_recl = os.path.join(MAPAS_RECLAS_DIR, "metadatos_mapas_reclasificados_amazonia.csv")
        df_recl.to_csv(ruta_recl, index=False)
        print(f"[OK] Metadatos reclasificados guardados en {ruta_recl}")