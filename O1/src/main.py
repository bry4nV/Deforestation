from config import ANIOS, CLASES_BOSQUE, METADATOS_DIR
from pipeline import ejecutar_pipeline_anio
import pandas as pd
import os

if __name__ == "__main__":
    registros = []

    for anio in ANIOS:
        info = ejecutar_pipeline_anio(anio)
        if info is not None:
            registros.append(info)

    if registros:
        df = pd.DataFrame(registros)
        ruta_meta = os.path.join(METADATOS_DIR, "metadatos_mapbiomas_amazonia.csv")
        df.to_csv(ruta_meta, index=False)
        print(f"[OK] Metadatos guardados en {ruta_meta}")