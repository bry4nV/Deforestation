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

MAPAS_REPROYECTADOS_DIR = os.path.join(O1_INTERIM_DIR, "mapas-reproyectados")

MAPAS_CAMBIOS_DIR = os.path.join(O1_INTERIM_DIR, "mapas-cambios")

CAMBIOS_DIR = os.path.join(MAPAS_CAMBIOS_DIR, "cambios")
DEFORESTACION_DIR = os.path.join(MAPAS_CAMBIOS_DIR, "deforestacion")
REFORESTACION_DIR = os.path.join(MAPAS_CAMBIOS_DIR, "reforestacion")

METRICAS_DISTRITOS_DIR = os.path.join(O1_INTERIM_DIR, "metricas-distritos")

METRICAS_CAMBIOS_DIR = os.path.join(METRICAS_DISTRITOS_DIR, "cambios")
METRICAS_DEFORESTACION_DIR = os.path.join(METRICAS_DISTRITOS_DIR, "deforestacion")
METRICAS_REFORESTACION_DIR = os.path.join(METRICAS_DISTRITOS_DIR, "reforestacion")

ZONAS_DIR = os.path.join(O1_INTERIM_DIR, "zonas")

for d in [MAPAS_CAMBIOS_DIR, MAPAS_REPROYECTADOS_DIR, ZONAS_DIR, 
          MAPAS_CAMBIOS_DIR, CAMBIOS_DIR, DEFORESTACION_DIR, 
          REFORESTACION_DIR, METRICAS_CAMBIOS_DIR, METRICAS_DEFORESTACION_DIR, 
          METRICAS_REFORESTACION_DIR]:
    os.makedirs(d, exist_ok=True)

CRS_PROYECTADO = "EPSG:32718"  # UTM zona 18 sur
CRS_GEOG = "EPSG:4326"  # WGS84 para distritos

# =============================
# RUTAS SHAPEFILES
# =============================
DISTRITOS_PERU_DIR = os.path.join(RAW_DIR, "distritos-peru")
BIOMAS_PERU_DIR = os.path.join(RAW_DIR, "biomas-peru")
DISTRITOS_AMAZONIA_DIR = os.path.join(O1_INTERIM_DIR, "distritos-amazonas")

INICIALES_DIR = os.path.join(DISTRITOS_AMAZONIA_DIR, "iniciales")
COBERTURA_MIN95_DIR = os.path.join(DISTRITOS_AMAZONIA_DIR, "cobertura_min95")
AREA_MINIMA_DIR = os.path.join(DISTRITOS_AMAZONIA_DIR, "area_minima")
BOSQUE_MINIMO_DIR = os.path.join(DISTRITOS_AMAZONIA_DIR, "bosque_minimo")

for d in [INICIALES_DIR, COBERTURA_MIN95_DIR, 
          AREA_MINIMA_DIR, BOSQUE_MINIMO_DIR]:
    os.makedirs(d, exist_ok=True)

'''
# Parámetros de zonificación
PARAMS_ZONIFICACION = {
    # Expansión
    'distancia_max_expansion_km': 5.0,
    'area_objetivo_zona_km2': 500,  # tamaño objetivo de zona
    
    # Regularización
    'area_min_zona_km2': 50,
    'area_max_zona_km2': 2000,
    'bosque_remanente_min_pct': 20,
    'anios_activos_min': 5,
    'cv_max_heterogeneidad': 2.0,  # coeficiente de variación máximo
}'''