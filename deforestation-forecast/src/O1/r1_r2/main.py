from O1.config import ANIOS, METADATOS_DIR
from O1.r1_r2.pipeline import ejecutar_pipeline_anio
import pandas as pd
import os

if __name__ == "__main__":
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
        ruta_raw = os.path.join(METADATOS_DIR, "metadatos_raw_mapbiomas_amazonia.csv")
        df_raw.to_csv(ruta_raw, index=False)
        print(f"[OK] Metadatos RAW guardados en {ruta_raw}")

    if registros_reclasificados:
        df_recl = pd.DataFrame(registros_reclasificados)
        ruta_recl = os.path.join(METADATOS_DIR, "metadatos_reclasificados_mapbiomas_amazonia.csv")
        df_recl.to_csv(ruta_recl, index=False)
        print(f"[OK] Metadatos reclasificados guardados en {ruta_recl}")