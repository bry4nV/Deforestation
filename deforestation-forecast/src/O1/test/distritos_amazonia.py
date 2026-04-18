import os
import geopandas as gpd

def identificar_distritos_amazonia():
    biomas_peru = gpd.read_file(r'D:\VisualCode\gvSigTesis\BIOMES_v1\BIOMES_v1.shp')
    distritos_peru = gpd.read_file(r'D:\VisualCode\gvSigTesis\POLITICAL_LEVEL_4_v1\POLITICAL_LEVEL_4_v1.shp')

    # Asegurar mismo CRS original
    if biomas_peru.crs != distritos_peru.crs:
        distritos_peru = distritos_peru.to_crs(biomas_peru.crs)

    # Limpiar geometrías
    biomas_peru["geometry"] = biomas_peru.buffer(0)
    distritos_peru["geometry"] = distritos_peru.buffer(0)

    # Filtrar Amazonía
    amazonia = biomas_peru[biomas_peru["NAME"] == "[Amazonía]"].copy()

    # Reproyectar a CRS proyectado para calcular áreas en Perú: UTM zona 18S (EPSG:32718)
    crs_area = "EPSG:32718"

    amazonia = amazonia.to_crs(crs_area)
    distritos_peru = distritos_peru.to_crs(crs_area)

    # Unión geométrica
    amazonia_union = amazonia.geometry.union_all()

    # Quedarse con los que intersectan
    distritos_intersect = distritos_peru[
        distritos_peru.intersects(amazonia_union)
    ].copy()

    # Calcular áreas correctamente
    distritos_intersect["area_total"] = distritos_intersect.geometry.area
    distritos_intersect["area_inter"] = (
        distritos_intersect.geometry.intersection(amazonia_union).area
    )
    distritos_intersect["pct_amazon"] = (
        distritos_intersect["area_inter"] / distritos_intersect["area_total"]
    )

    # Filtrar
    distritos_amazonia = distritos_intersect[
        distritos_intersect["pct_amazon"] >= 0.95
    ].copy()

    print(f"Total distritos amazónicos: {len(distritos_amazonia)}")

    # Volver a 4326 si quieres guardar para visores web/GIS
    distritos_amazonia = distritos_amazonia.to_crs("EPSG:4326")

    carpeta_salida = r'D:\VisualCode\gvSigTesis\POLITICAL_LEVEL_4_AMAZON'
    os.makedirs(carpeta_salida, exist_ok=True)

    ruta_salida = os.path.join(carpeta_salida, 'political_level_4_amazon_95.shp')
    distritos_amazonia.to_file(ruta_salida, encoding='utf-8')

    print(f"Shapefile guardado en: {ruta_salida}")


if __name__ == "__main__":
    identificar_distritos_amazonia()