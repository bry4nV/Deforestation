import geopandas as gpd
import pandas as pd
import numpy as np

def pipeline_seleccion_distritos_alto_cambio(
    ruta_mapa_cambios_distrito,
    ruta_distritos_alto_cambio,
):

    gdf_cambio = gpd.read_file(ruta_mapa_cambios_distrito)

    columnas_base = [
        "GEOCODE",
        "CATEG_ID",
        "LEVEL_1",
        "LEVEL_2",
        "LEVEL_3",
        "LEVEL_4",
        "NAME_STD",
        "SOURCE",
        "VERSION",
        "FEATURE_ID",
        "NAME",
        "CATEG_NAME",
        "geometry"
    ]

    gdf = gdf_cambio[columnas_base].copy()

    gdf = gdf.merge(
        gdf_cambio[[
            "GEOCODE",
            "pixeles_validos",
            "pixeles_cambiados",
            "pixeles_no_cambiados",
            "porcentaje_cambio"
        ]],
        on="GEOCODE"
    )


    gdf = gdf.sort_values("porcentaje_cambio", ascending=False)

    gdf_seleccionados = gdf.head(200).copy()    
    gdf_seleccionados.to_file(ruta_distritos_alto_cambio, driver="GPKG", encoding="utf-8")

    print(f"[OK] Distritos seleccionados guardados: {ruta_distritos_alto_cambio}")

    ruta_csv = ruta_distritos_alto_cambio.replace(".gpkg", ".csv")
    df_csv = gdf_seleccionados.drop(columns="geometry").copy()
    df_csv = df_csv.sort_values("porcentaje_cambio", ascending=False)

    df_csv = df_csv.rename(columns={
        "LEVEL_2": "Departamento",
        "LEVEL_4": "Distrito",
        "porcentaje_cambio": "% Cambio"
    })

    df_csv.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] CSV guardado: {ruta_csv}")