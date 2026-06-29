"""
Metadata estructural de las fuentes RAW — primer paso de PASO 0, antes de
construir cualquier variable.

Diferencia con `validar_fuentes()` (O3/utils.py): `validar_fuentes()` solo
verifica que cada ruta exista (`os.path.exists`), como chequeo rápido que
falla temprano. Este módulo abre cada fuente que ya pasó esa verificación y
extrae sus atributos estructurales (CRS, tipo de geometría/resolución,
dimensiones, etc.), dejando un registro trazable de "con qué datos exactos
se corrió esta versión del pipeline" -- útil para detectar, por ejemplo, un
CRS distinto al esperado o un tile SRTM con nodata distinto antes de que ese
problema se propague silenciosamente a una variable derivada.

No es idempotente a propósito (a diferencia de los `construir_*`): leer
~175 cabeceras de archivo cuesta menos de un segundo, así que no hay
beneficio de cachear, y SIEMPRE regenerar da una garantía más fuerte de
auditoría (refleja el estado actual de las fuentes RAW en cada corrida, no
una foto vieja de la primera vez que se ejecutó).

No hay un CSV consolidado con todas las fuentes mezcladas: cada copia se
escribe directamente en la carpeta de la variable que la consume, junto a
su `_metadatos.csv` ya procesado, con solo las columnas que aplican a su
tipo (vectorial o raster) -- como cada archivo resultante es homogéneo en
tipo, ya no hace falta el esquema con columnas vacías que un consolidado
heterogéneo sí necesitaría. La única fuente sin variable propia es
`distritos_alto_cambio` (los 8 construir_*.py la reciben por igual vía
distritos_gdf, ninguno es su "dueño"); por eso tiene su propio archivo
pequeño en vez de vivir en alguna carpeta de variable.

Este reparto se hace deliberadamente DESPUÉS del cálculo central -- no
dentro de cada `construir_*.py` -- porque esos scripts retornan temprano
cuando su CSV ya existe (ver `construir_anp.py`), y la auditoría cruda nunca
debe depender de esa guarda de idempotencia o quedaría tan congelada como la
metadata que reemplaza.
"""

import logging
import os

import geopandas as gpd
import pandas as pd
import rasterio

from O3.config import (
    ANIOS,
    ANP_ACP_SHP,
    ANP_ACR_SHP,
    ANP_SHP,
    ANP_ZR_SHP,
    CARRETERAS_DEPARTAMENTAL_SHP,
    CARRETERAS_NACIONAL_SHP,
    CARRETERAS_VECINAL_SHP,
    DISTRITOS_ALTO_CAMBIO_GPKG,
    DISTRITOS_METADATA_RAW_CSV,
    MAPBIOMAS_AMAZONIA_PATRON,
    RIOS_SHP,
    SRTM_TILES,
    VAR_AGROPECUARIA_DIR,
    VAR_ANP_DIR,
    VAR_CARRETERAS_DIR,
    VAR_ELEVACION_DIR,
    VAR_PENDIENTE_DIR,
    VAR_RIOS_DIR,
    VAR_RIOS_LAGOS_DIR,
    VAR_URBANO_DIR,
)
from O3.utils import guardar_csv

logger = logging.getLogger(__name__)

# Fuentes vectoriales a auditar — todas, una fila por archivo (son solo 9).
FUENTES_VECTORIALES = {
    "distritos_alto_cambio": DISTRITOS_ALTO_CAMBIO_GPKG,
    "carreteras_nacional":   CARRETERAS_NACIONAL_SHP,
    "carreteras_departamental": CARRETERAS_DEPARTAMENTAL_SHP,
    "carreteras_vecinal":    CARRETERAS_VECINAL_SHP,
    "rios":                  RIOS_SHP,
    "anp_nacional":          ANP_SHP,
    "anp_zona_reservada":    ANP_ZR_SHP,
    "anp_acr":               ANP_ACR_SHP,
    "anp_acp":               ANP_ACP_SHP,
}

# Columnas que aplican a cada tipo de fuente. Cada archivo de salida es
# homogéneo en tipo (una variable nunca mezcla fuentes vectoriales y
# raster), así que no hace falta cargar columnas vacías del otro tipo.
COLUMNAS_VECTORIAL = [
    "categoria", "tipo_fuente", "archivo", "ruta", "crs",
    "geom_type", "n_features", "bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy",
    "file_size_mb",
]
COLUMNAS_RASTER = [
    "categoria", "tipo_fuente", "archivo", "ruta", "anio", "crs",
    "bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy",
    "res_x", "res_y", "width", "height", "n_bandas", "dtype", "nodata",
    "file_size_mb",
]

# Carpetas de variable donde escribir la metadata cruda, junto a la
# metadata ya procesada de esa variable (p. ej. anp_metadatos_raw.csv al
# lado de anp_metadatos.csv). Una fuente compartida por varias variables
# (MapBiomas, SRTM) se copia en cada una de las carpetas que la consumen.
DESTINOS_POR_CATEGORIA = {
    "carreteras_nacional":      [VAR_CARRETERAS_DIR],
    "carreteras_departamental": [VAR_CARRETERAS_DIR],
    "carreteras_vecinal":       [VAR_CARRETERAS_DIR],
    "rios":                     [VAR_RIOS_DIR],
    "anp_nacional":             [VAR_ANP_DIR],
    "anp_zona_reservada":       [VAR_ANP_DIR],
    "anp_acr":                  [VAR_ANP_DIR],
    "anp_acp":                  [VAR_ANP_DIR],
    "mapbiomas":                [VAR_AGROPECUARIA_DIR, VAR_RIOS_LAGOS_DIR, VAR_URBANO_DIR],
    "srtm":                     [VAR_ELEVACION_DIR, VAR_PENDIENTE_DIR],
}

NOMBRE_ARCHIVO_POR_CARPETA = {
    VAR_AGROPECUARIA_DIR: "agropecuaria_metadatos_raw.csv",
    VAR_CARRETERAS_DIR:   "carreteras_metadatos_raw.csv",
    VAR_RIOS_DIR:         "rios_metadatos_raw.csv",
    VAR_ANP_DIR:          "anp_metadatos_raw.csv",
    VAR_ELEVACION_DIR:    "elevacion_metadatos_raw.csv",
    VAR_PENDIENTE_DIR:    "pendiente_metadatos_raw.csv",
    VAR_RIOS_LAGOS_DIR:   "rios_lagos_metadatos_raw.csv",
    VAR_URBANO_DIR:       "urbano_metadatos_raw.csv",
}


def _metadata_vectorial(categoria: str, ruta: str) -> dict:
    gdf = gpd.read_file(ruta)
    minx, miny, maxx, maxy = gdf.total_bounds
    tipos_geom = sorted(gdf.geom_type.dropna().unique().tolist())
    return {
        "categoria":    categoria,
        "tipo_fuente":  "vectorial",
        "archivo":      os.path.basename(ruta),
        "ruta":         ruta,
        "crs":          str(gdf.crs),
        "geom_type":    ", ".join(tipos_geom),
        "n_features":   len(gdf),
        "bbox_minx":    round(float(minx), 6),
        "bbox_miny":    round(float(miny), 6),
        "bbox_maxx":    round(float(maxx), 6),
        "bbox_maxy":    round(float(maxy), 6),
        "file_size_mb": round(os.path.getsize(ruta) / (1024 * 1024), 2),
    }


def _metadata_raster(categoria: str, ruta: str, anio: int = None) -> dict:
    with rasterio.open(ruta) as src:
        return {
            "categoria":    categoria,
            "tipo_fuente":  "raster",
            "archivo":      os.path.basename(ruta),
            "ruta":         ruta,
            "anio":         anio,
            "crs":          str(src.crs),
            "bbox_minx":    round(src.bounds.left, 6),
            "bbox_miny":    round(src.bounds.bottom, 6),
            "bbox_maxx":    round(src.bounds.right, 6),
            "bbox_maxy":    round(src.bounds.top, 6),
            "res_x":        round(src.res[0], 8),
            "res_y":        round(src.res[1], 8),
            "width":        src.width,
            "height":       src.height,
            "n_bandas":     src.count,
            "dtype":        src.dtypes[0],
            "nodata":       src.nodata,
            "file_size_mb": round(os.path.getsize(ruta) / (1024 * 1024), 2),
        }


def _columnas_aplicables(subset: pd.DataFrame) -> pd.DataFrame:
    """Recorta a las columnas del tipo de `subset` (homogéneo: vectorial o raster)."""
    tipo = subset["tipo_fuente"].iloc[0]
    columnas = COLUMNAS_VECTORIAL if tipo == "vectorial" else COLUMNAS_RASTER
    return subset[columnas]


def _escribir_copias_por_variable(df: pd.DataFrame) -> None:
    """
    Escribe la metadata cruda en la carpeta de cada variable, junto a su
    `_metadatos.csv` ya procesado. Una fuente compartida (MapBiomas, SRTM)
    se duplica en cada carpeta que la consume.
    """
    filas_por_carpeta: dict = {}
    for categoria, carpetas in DESTINOS_POR_CATEGORIA.items():
        subset = df[df["categoria"] == categoria]
        if subset.empty:
            continue
        for carpeta in carpetas:
            filas_por_carpeta.setdefault(carpeta, []).append(subset)

    for carpeta, partes in filas_por_carpeta.items():
        df_carpeta = _columnas_aplicables(pd.concat(partes, ignore_index=True))
        ruta = os.path.join(carpeta, NOMBRE_ARCHIVO_POR_CARPETA[carpeta])
        guardar_csv(df_carpeta, ruta)
        logger.info(f"  [OK] Metadata cruda: {ruta} ({len(df_carpeta)} filas)")


def generar_metadata_fuentes(ruta_distritos: str = DISTRITOS_METADATA_RAW_CSV) -> pd.DataFrame:
    """
    Recorre todas las fuentes RAW configuradas (vectoriales + raster),
    extrae su metadata estructural y la escribe en la carpeta de cada
    variable que la consume (`_escribir_copias_por_variable`), junto a su
    `_metadatos.csv` ya procesado. `distritos_alto_cambio` -- la única
    fuente sin carpeta de variable propia -- se escribe en `ruta_distritos`.

    MapBiomas y SRTM se auditan exhaustivamente, no muestreados: cada
    archivo de metadata cruda vive dentro de la carpeta de la variable que
    efectivamente abre esos rásters uno por uno (`construir_agropecuaria`,
    `construir_rios_lagos` y `construir_urbano` procesan los 40 años, no
    solo 1985/2024), así que una muestra ahí sería engañosa -- parecería
    una auditoría completa sin serlo. `O1/mapas-amazonia/metadatos_mapas_amazonia.csv`
    ya documenta estos mismos 40 archivos exhaustivamente, pero vive en O1,
    no junto a `agropecuaria/`; el propósito aquí es trazabilidad local por
    variable, no evitar duplicar ese archivo.

    Devuelve el DataFrame completo (todas las fuentes, todas las columnas)
    por si se necesita para un análisis puntual -- no se guarda como tal.
    """
    logger.info("Generando metadata de fuentes RAW (vectoriales + raster)...")
    filas = []

    for categoria, ruta in FUENTES_VECTORIALES.items():
        logger.info(f"  [vectorial] {categoria}")
        filas.append(_metadata_vectorial(categoria, ruta))

    logger.info(f"  [raster] mapbiomas: auditando los {len(ANIOS)} años ({ANIOS[0]}-{ANIOS[-1]})...")
    for anio in ANIOS:
        ruta = MAPBIOMAS_AMAZONIA_PATRON.format(anio=anio)
        filas.append(_metadata_raster("mapbiomas", ruta, anio=anio))

    logger.info(f"  [raster] srtm: auditando {len(SRTM_TILES)} tiles...")
    for ruta in SRTM_TILES:
        filas.append(_metadata_raster("srtm", ruta))

    df = pd.DataFrame(filas)

    fila_distritos = _columnas_aplicables(df[df["categoria"] == "distritos_alto_cambio"])
    guardar_csv(fila_distritos, ruta_distritos)
    logger.info(f"  [OK] Metadata cruda: {ruta_distritos} ({len(fila_distritos)} filas)")

    _escribir_copias_por_variable(df)

    logger.info(f"[OK] Metadata de fuentes generada ({len(df)} archivos auditados)")
    return df
