import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats
from O1.config import NODATA, ANIOS


def pipeline_extraer_series_temporales(
    rutas_mapas_reclasificados,
    ruta_distritos_clasificados,
    ruta_series_temporales,
    ruta_estadisticas_series_temporales
):

    print("\n" + "="*70)
    print(" SERIES TEMPORALES (ESTADO BOSQUE / NO BOSQUE)")
    print("="*70 + "\n")

    rutas_mapas_reclasificados = sorted(rutas_mapas_reclasificados)

    if len(rutas_mapas_reclasificados) != len(ANIOS):
        raise ValueError("Número de rutas y años no coincide")

    gdf = gpd.read_file(ruta_distritos_clasificados)
    print(f"[INFO] Distritos: {len(gdf)}")

    registros = []

    for ruta, anio in zip(rutas_mapas_reclasificados, ANIOS):

        print(f"[INFO] Año {anio}")

        stats = zonal_stats(
            gdf.geometry,
            ruta,
            categorical=True,
            nodata=NODATA
        )

        for idx, s in enumerate(stats):

            s = s or {}
            fila = gdf.iloc[idx]

            pix_bosque = int(s.get(1, 0))
            pix_no_bosque = int(s.get(0, 0))
            pix_total = pix_bosque + pix_no_bosque

            registros.append({
                "geocode": fila["GEOCODE"],
                "departamento": fila["LEVEL_2"],
                "distrito": fila["LEVEL_4"],
                "score": fila.get("score"),
                "dataset": fila.get("dataset"),

                "pix_cambio": fila.get("pixeles_cambiados_cambio"),
                "pix_no_cambio": fila.get("pixeles_no_cambiados_cambio"),

                "pix_ref": fila.get("pixeles_cambiados_ref"),
                "pix_no_ref": fila.get("pixeles_no_cambiados_ref"),

                "pix_def": fila.get("pixeles_cambiados_def"),
                "pix_no_def": fila.get("pixeles_no_cambiados_def"),

                "anio": anio,
                "pix_total": pix_total,
                "pix_bosque": pix_bosque,
                "pix_no_bosque": pix_no_bosque,
                "pct_bosque": pix_bosque / pix_total if pix_total > 0 else 0,
                "pct_no_bosque": pix_no_bosque / pix_total if pix_total > 0 else 0
            })

    df = pd.DataFrame(registros)
    df = df.sort_values(["geocode", "anio"]).reset_index(drop=True)

    print(f"[OK] Registros: {len(df)}")

    df.to_csv(ruta_series_temporales, index=False)
    print(f"[OK] Guardado: {ruta_series_temporales}")

    if ruta_estadisticas_series_temporales:
        stats = (
            df.groupby("geocode")[["pct_bosque", "pct_no_bosque"]]
            .agg(["mean", "min", "max"])
            .reset_index()
        )

        stats.columns = [
            "geocode",
            "pct_bosque_mean",
            "pct_bosque_min",
            "pct_bosque_max",
            "pct_no_bosque_mean",
            "pct_no_bosque_min",
            "pct_no_bosque_max"
        ]

        stats.to_csv(ruta_estadisticas_series_temporales, index=False)
        print(f"[OK] Stats: {ruta_estadisticas_series_temporales}")

    return df