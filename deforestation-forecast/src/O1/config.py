import os

# =============================
# R1 y R2 - PIPELINE
# =============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
MAPAS_RAW_DIR = os.path.join(RAW_DIR, "mapbiomas-peru")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
O1_INTERIM_DIR = os.path.join(INTERIM_DIR, "O1")

MAPAS_RECLAS_DIR = os.path.join(O1_INTERIM_DIR, "mapas-reclasificados")
METADATOS_DIR = os.path.join(O1_INTERIM_DIR, "metadatos")

for d in [O1_INTERIM_DIR, MAPAS_RECLAS_DIR, METADATOS_DIR, OUTPUTS_DIR]:
    os.makedirs(d, exist_ok=True)

ANIOS = list(range(1985, 2025))

# Clases de bosque (ATBD – MapBiomas Perú – C3)
CLASES_BOSQUE = {3, 4, 5, 6}

# Clases válidas para Amazonía Perú (C3)
CLASES_VALIDAS = {
    3,4,5,6,              # Bosques
    11,12,29,66,70,13,    # Naturales no boscosas
    15,18,35,40,72,9,21,  # Agropecuario
    23,24,30,32,61,68,25, # Sin vegetación
    33,31,34              # Agua
}

CLASE_NOBSERVADO = 27

NODATA = 255

# =============================
# R3 - ZONIFICACIÓN
# =============================

MAPAS_CAMBIOS_DIR = os.path.join(O1_INTERIM_DIR, "mapas-cambios")
METRICAS_DISTRITOS_DIR = os.path.join(O1_INTERIM_DIR, "metricas-distritos")

SERIES_TEMPORALES_DIR = os.path.join(O1_INTERIM_DIR, "series-temporales")
SERIES_ENTRENAMIENTO_DIR = os.path.join(SERIES_TEMPORALES_DIR, "entrenamiento")
SERIES_GENERALIZACION_ESPACIAL_DIR = os.path.join(SERIES_TEMPORALES_DIR, "generalizacion-espacial")

for d in [MAPAS_CAMBIOS_DIR, METRICAS_DISTRITOS_DIR, 
          SERIES_TEMPORALES_DIR, SERIES_ENTRENAMIENTO_DIR, SERIES_GENERALIZACION_ESPACIAL_DIR]:
    os.makedirs(d, exist_ok=True)

CRS_PROYECTADO = "EPSG:32718"  # UTM zona 18 sur
CRS_GEOG = "EPSG:4326"  # WGS84 para distritos
TAMANIO_ENTRENAMIENTO = 0.9
SEMILLA_SPLIT = 42

# =============================
# RUTAS SHAPEFILES
# =============================
DISTRITOS_PERU_DIR = os.path.join(RAW_DIR, "distritos-peru")
BIOMAS_PERU_DIR = os.path.join(RAW_DIR, "biomas-peru")
DISTRITOS_AMAZONIA_DIR = os.path.join(O1_INTERIM_DIR, "distritos-amazonas")

DISTRITOS_ALTO_CAMBIO_DIR = os.path.join(O1_INTERIM_DIR, "distritos-alto-cambio")

for d in [DISTRITOS_AMAZONIA_DIR, DISTRITOS_ALTO_CAMBIO_DIR]:
    os.makedirs(d, exist_ok=True)