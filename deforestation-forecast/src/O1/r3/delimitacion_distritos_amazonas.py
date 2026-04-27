import os
import geopandas as gpd

def guardar_csv(gdf, ruta_csv):
    df = gdf.copy()
    if "geometry" in df.columns:
        df = df.drop(columns="geometry")
    
    df.to_csv(ruta_csv, index=False)

def identificar_distritos_amazonia_interseccion(ruta_biomas_peru, ruta_distritos_peru, ruta_distritos_amazonia_delimitados):
    
    biomas = gpd.read_file(ruta_biomas_peru)
    distritos = gpd.read_file(ruta_distritos_peru)

    if biomas.crs != distritos.crs:
        raise ValueError("CRS diferentes entre biomas y distritos")
    
    biomas["geometry"] = biomas.geometry.buffer(0)
    distritos["geometry"] = distritos.geometry.buffer(0)

    # Filtrar Amazonía
    amazonia = biomas.loc[biomas["NAME"] == "[Amazonía]"].copy()

    amazonia_union = amazonia.geometry.union_all()

    # Quedarse con los que intersectan
    distritos_intersectados = distritos[distritos.intersects(amazonia_union)].copy()

    distritos_intersectados.to_file(ruta_distritos_amazonia_delimitados, driver="GPKG", encoding="utf-8")
    ruta_estadisticas_csv = ruta_distritos_amazonia_delimitados.replace(".gpkg", ".csv")
    guardar_csv(distritos_intersectados, ruta_estadisticas_csv)

    print(f"Total distritos amazónicos: {len(distritos_intersectados)}")
    print(f"Shapefile guardado en: {ruta_distritos_amazonia_delimitados}")

def pipeline_delimitacion_distritos_amazonia(ruta_biomas_peru, ruta_distritos_peru, ruta_distritos_amazonia_delimitados):
    print("\n" + "="*70)
    print(" INICIANDO DELIMITACIÓN DE DISTRITOS AMAZÓNICOS")
    print("="*70)
    
    identificar_distritos_amazonia_interseccion(ruta_biomas_peru, ruta_distritos_peru, ruta_distritos_amazonia_delimitados)

    print("\n" + "="*70)
    print(" DELIMITACIÓN DE DISTRITOS AMAZÓNICOS SELECCIONADOS COMPLETADA")
    print("="*70)