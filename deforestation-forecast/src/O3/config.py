import glob
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ============================================================
# RUTAS BASE
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR  = os.path.join(DATA_DIR, "raw")

VARIABLES_LOCALES_RAW_DIR = os.path.join(RAW_DIR, "variables-locales")

O3_INTERIM_DIR = os.path.join(DATA_DIR, "interim", "O3")

# ============================================================
# RUTAS — FUENTES RAW POR VARIABLE
# ============================================================

# --- Carreteras (red vial MTC — diciembre 2018) ---
# Tres capas separadas; se concatenan en construir_carreteras.py
CARRETERAS_NACIONAL_SHP      = os.path.join(
    VARIABLES_LOCALES_RAW_DIR, "redes-viales", "nacional",
    "red_vial_nacional_dic18.shp",
)
CARRETERAS_DEPARTAMENTAL_SHP = os.path.join(
    VARIABLES_LOCALES_RAW_DIR, "redes-viales", "departamental",
    "red_vial_departamental_dic18.shp",
)
CARRETERAS_VECINAL_SHP       = os.path.join(
    VARIABLES_LOCALES_RAW_DIR, "redes-viales", "vecinal",
    "red_vial_vecinal_dic18.shp",
)

# --- Rios (red hidrografica ANA) ---
RIOS_SHP = os.path.join(VARIABLES_LOCALES_RAW_DIR, "rios", "Rios.shp")

# --- ANP (SERNANP) — cuatro categorías a unificar en construir_anp.py ---
# Rutas a cada capa shapefile
ANP_SHP     = os.path.join(
    VARIABLES_LOCALES_RAW_DIR, "anp", "ANP Nacional Definitivas",
    "ANPNacionalDefinitivas.shp",
)
ANP_ZR_SHP  = os.path.join(
    VARIABLES_LOCALES_RAW_DIR, "anp", "Zonas Reservadas",
    "ZonasReservadas.shp",
)
ANP_ACR_SHP = os.path.join(
    VARIABLES_LOCALES_RAW_DIR, "anp", "Áreas de Conservación Regional",
    "AreasdeConservacionRegional.shp",
)
ANP_ACP_SHP = os.path.join(
    VARIABLES_LOCALES_RAW_DIR, "anp", "Áreas de Conservación Privada",
    "AreasdeConservacionPrivada.shp",
)

# Campo de fecha de establecimiento en cada shapefile original (prefijo distinto por capa)
ANP_COL_FECHA     = "anp_felec"    # ANP Nacional Definitivas
ANP_ZR_COL_FECHA  = "zr_felec"     # Zonas Reservadas
ANP_ACR_COL_FECHA = "acr_felec"    # Áreas de Conservación Regional
ANP_ACP_COL_FECHA = "acp_felec"    # Áreas de Conservación Privada
# ACP tiene además acp_fecad (caducidad), ignorada — se asume renovación continua

# Nombre de columna unificado tras estandarización en construir_anp.py
ANP_COL_FECHA_STD = "felec"

# Campos de código, nombre y categoría solo en ANP Nacional (referencia descriptiva)
ANP_COL_CODIGO    = "anp_codi"
ANP_COL_NOMBRE    = "anp_nomb"
ANP_COL_CATEGORIA = "anp_cate"

# --- Elevacion y pendiente (SRTM 1 arc-second / 30 m) ---
# Cada tile vive dentro de su propio subdirectorio: elevacion/S##W###.SRTMGL1.hgt/S##W###.hgt
SRTM_TILES = sorted(
    glob.glob(
        os.path.join(VARIABLES_LOCALES_RAW_DIR, "elevacion", "**", "*.hgt"),
        recursive=True,
    )
)

# ============================================================
# RUTAS — OUTPUTS DE O1 CONSUMIDOS POR O3
# ============================================================

O1_INTERIM_DIR        = os.path.join(DATA_DIR, "interim", "O1")
O1_MAPAS_AMAZONIA_DIR = os.path.join(O1_INTERIM_DIR, "mapas-amazonia")

# Rasters anuales MapBiomas C3 ya recortados a Amazonia (salida de O1 R1).
# Usados por: construir_agropecuaria, construir_rios_lagos, construir_urbano.
MAPBIOMAS_AMAZONIA_PATRON = os.path.join(
    O1_MAPAS_AMAZONIA_DIR, "peru_amazonia_{anio}.tif"
)

DISTRITOS_ENTRENAMIENTO_CSV  = os.path.join(
    O1_INTERIM_DIR, "series-temporales", "entrenamiento",
    "distritos_entrenamiento.csv",
)
DISTRITOS_GENERALIZACION_CSV = os.path.join(
    O1_INTERIM_DIR, "series-temporales", "generalizacion-espacial",
    "distritos_generalizacion_espacial.csv",
)
DISTRITOS_ALTO_CAMBIO_GPKG   = os.path.join(
    O1_INTERIM_DIR, "distritos-alto-cambio", "distritos_alto_cambio.gpkg",
)

# Columnas del GeoDataFrame distritos_alto_cambio.gpkg (GADM)
GPKG_COL_GEOCODE      = "GEOCODE"
GPKG_COL_DEPARTAMENTO = "LEVEL_2"
GPKG_COL_DISTRITO     = "LEVEL_4"

# ============================================================
# RUTAS — SALIDAS INTERMEDIAS O3
# ============================================================

# Variables principales
VAR_AGROPECUARIA_DIR = os.path.join(O3_INTERIM_DIR, "variables", "agropecuaria")
VAR_CARRETERAS_DIR   = os.path.join(O3_INTERIM_DIR, "variables", "carreteras")
VAR_RIOS_DIR         = os.path.join(O3_INTERIM_DIR, "variables", "rios")
VAR_ANP_DIR          = os.path.join(O3_INTERIM_DIR, "variables", "anp")
VAR_ELEVACION_DIR    = os.path.join(O3_INTERIM_DIR, "variables", "elevacion")
VAR_PENDIENTE_DIR    = os.path.join(O3_INTERIM_DIR, "variables", "pendiente")

# Variables de respaldo
VAR_RIOS_LAGOS_DIR   = os.path.join(O3_INTERIM_DIR, "variables-respaldo", "rios_lagos")
VAR_URBANO_DIR       = os.path.join(O3_INTERIM_DIR, "variables-respaldo", "urbano")

# Panel integrado
PANEL_DIR = os.path.join(O3_INTERIM_DIR, "panel-integrado")

# Archivos de salida — agropecuaria
AGROPECUARIA_CSV  = os.path.join(VAR_AGROPECUARIA_DIR, "agropecuaria_por_distrito.csv")
AGROPECUARIA_META = os.path.join(VAR_AGROPECUARIA_DIR, "agropecuaria_metadatos.csv")

# Archivos de salida — carreteras
CARRETERAS_CSV    = os.path.join(VAR_CARRETERAS_DIR, "carreteras_por_distrito.csv")
CARRETERAS_META   = os.path.join(VAR_CARRETERAS_DIR, "carreteras_metadatos.csv")

# Archivos de salida — rios
RIOS_CSV          = os.path.join(VAR_RIOS_DIR, "rios_por_distrito.csv")
RIOS_META         = os.path.join(VAR_RIOS_DIR, "rios_metadatos.csv")

# Archivos de salida — ANP
ANP_CSV                = os.path.join(VAR_ANP_DIR, "anp_por_distrito.csv")
ANP_META               = os.path.join(VAR_ANP_DIR, "anp_metadatos.csv")
ANP_RESUMEN_ANUAL_CSV  = os.path.join(VAR_ANP_DIR, "anp_resumen_anual.csv")

# Archivos de salida — elevacion
ELEVACION_CSV     = os.path.join(VAR_ELEVACION_DIR, "elevacion_por_distrito.csv")
ELEVACION_META    = os.path.join(VAR_ELEVACION_DIR, "elevacion_metadatos.csv")
ELEVACION_MOSAIC  = os.path.join(VAR_ELEVACION_DIR, "dem_mosaico.tif")

# Archivos de salida — pendiente (depende del mosaico DEM de elevacion)
PENDIENTE_CSV     = os.path.join(VAR_PENDIENTE_DIR, "pendiente_por_distrito.csv")
PENDIENTE_META    = os.path.join(VAR_PENDIENTE_DIR, "pendiente_metadatos.csv")
PENDIENTE_RASTER  = os.path.join(VAR_PENDIENTE_DIR, "pendiente.tif")

# Archivos de salida — rios y lagos (respaldo)
RIOS_LAGOS_CSV  = os.path.join(VAR_RIOS_LAGOS_DIR, "rios_lagos_por_distrito.csv")
RIOS_LAGOS_META = os.path.join(VAR_RIOS_LAGOS_DIR, "rios_lagos_metadatos.csv")

# Archivos de salida — urbano (respaldo)
URBANO_CSV        = os.path.join(VAR_URBANO_DIR, "urbano_por_distrito.csv")
URBANO_META       = os.path.join(VAR_URBANO_DIR, "urbano_metadatos.csv")

# Archivos de salida — panel integrado
PANEL_CSV                = os.path.join(PANEL_DIR, "panel_integrado.csv")
PANEL_LIGHT_CSV          = os.path.join(PANEL_DIR, "panel_integrado_light.csv")
PANEL_ENTRENAMIENTO_CSV  = os.path.join(PANEL_DIR, "panel_integrado_entrenamiento.csv")
PANEL_GENERALIZACION_CSV = os.path.join(PANEL_DIR, "panel_integrado_generalizacion.csv")
PANEL_REPORTE_CSV        = os.path.join(PANEL_DIR, "reporte_integracion.csv")

# ============================================================
# CONSTANTES — CLASES MAPBIOMAS C3
# ============================================================

# Clases objetivo por variable (verificadas contra ATBD MapBiomas Peru C3)
CLASES_AGROPECUARIA = {9, 15, 21, 35, 40}
CLASES_RIOS_LAGOS   = {33}
CLASES_URBANO       = {24}

# Excluidos del denominador en todos los calculos de fraccion
CLASE_NO_OBSERVADO = 27
CLASE_FONDO        = 0

# ============================================================
# CONSTANTES — SRTM
# ============================================================

SRTM_NODATA  = -32768    # valor nodata estándar de SRTM SRTMGL1 (int16)
SLOPE_NODATA = -9999.0   # valor nodata del raster de pendiente derivado

# ============================================================
# CONSTANTES — CRS
# ============================================================

CRS_PROYECTADO = "EPSG:32718"   # UTM Zona 18S — calculos de area y longitud
CRS_GEOG       = "EPSG:4326"    # WGS84 — almacenamiento de geometrias y rasters

# ============================================================
# CONSTANTES — TEMPORAL
# ============================================================

ANIOS = list(range(1985, 2025))   # 40 anos, alineado con O1

# ============================================================
# CREACION IDEMPOTENTE DE DIRECTORIOS DE SALIDA
# ============================================================

for _dir in [
    VAR_AGROPECUARIA_DIR,
    VAR_CARRETERAS_DIR,
    VAR_RIOS_DIR,
    VAR_ANP_DIR,
    VAR_ELEVACION_DIR,
    VAR_PENDIENTE_DIR,
    VAR_RIOS_LAGOS_DIR,
    VAR_URBANO_DIR,
    PANEL_DIR,
]:
    os.makedirs(_dir, exist_ok=True)

# ============================================================
# R11 — Modelo de pronóstico extendido (multivariable)
# ============================================================

# Orden canónico de canales — todos los módulos de r11/ referencian esta lista
COLUMNAS_PREDICTORAS = [
    "pct_bosque",
    "pct_agropecuario",
    "pct_anp",
    "densidad_carreteras_km_km2",
    "densidad_rios_km_km2",
    "elev_media_m",
    "pendiente_media_deg",
]

# Canales con distribución cero-inflada/sesgada (ver EDA de O3, sección 8)
# a los que se aplica log1p ANTES de ajustar_y_escalar(). El resto (skew bajo
# segun 02_asimetria_kurtosis.csv) se deja sin transformar.
VARIABLES_LOG1P = [
    "pct_anp",
    "densidad_carreteras_km_km2",
    "densidad_rios_km_km2",
]

# Rutas de salida R11
R11_MODELOS_DIR     = os.path.join(O3_INTERIM_DIR, "modelos")
R11_ESCALADOR_DIR   = os.path.join(R11_MODELOS_DIR, "escalador")
R11_MLP_DIR         = os.path.join(R11_MODELOS_DIR, "mlp")
R11_LSTM_DIR        = os.path.join(R11_MODELOS_DIR, "lstm")
R11_CNN_DIR         = os.path.join(R11_MODELOS_DIR, "cnn")
R11_COMPARACION_DIR = os.path.join(R11_MODELOS_DIR, "comparacion")

R11_ESCALADOR_PKL  = os.path.join(R11_ESCALADOR_DIR, "escalador_standard.pkl")
R11_ESCALADOR_META = os.path.join(R11_ESCALADOR_DIR, "escalador_metadatos.json")

# Rutas base O2 para comparación (se leen sin re-ejecutar O2)
O2_MLP_GLOBAL_CSV  = os.path.join(DATA_DIR, "interim", "O2", "modelos", "mlp",  "mlp_final_global.csv")
O2_LSTM_GLOBAL_CSV = os.path.join(DATA_DIR, "interim", "O2", "modelos", "lstm", "lstm_final_global.csv")
O2_CNN_GLOBAL_CSV  = os.path.join(DATA_DIR, "interim", "O2", "modelos", "cnn",  "cnn_final_global.csv")

# Constantes temporales (alineadas con O1/O2)
SEMILLA               = 42
ANIO_INICIO           = 1985
TAMANIO_ENTRENAMIENTO = 35    # años 1985-2019
HORIZONTE             = 5     # años 2020-2024

# Ventanas deslizantes DL
DL_VENTANAS = [3, 4, 5, 6, 7]

# Grids de hiperparámetros — MLP
MLP_CAPAS_OCULTAS_VALORES = [[32, 16], [64, 32], [128, 64, 32]]
MLP_ACTIVACION_VALORES    = ["relu", "leaky_relu"]
MLP_DROPOUT_VALORES       = [0.0, 0.1]
MLP_EPOCAS_VALORES        = [50]
MLP_LR_VALORES            = [0.001, 0.0005]
MLP_LOTE_VALORES          = [8, 16]

# Grids de hiperparámetros — LSTM
LSTM_UNIDADES_OCULTAS_VALORES = [16, 32, 64]
LSTM_NUM_CAPAS_VALORES        = [1, 2]
LSTM_DROPOUT_VALORES          = [0.0, 0.1]
LSTM_EPOCAS_VALORES           = [50]
LSTM_LR_VALORES               = [0.001, 0.0005]
LSTM_LOTE_VALORES             = [8, 16]

# Grids de hiperparámetros — CNN1D
CNN_CANALES_CONV_VALORES  = [[16], [32], [16, 32]]
CNN_KERNEL_VALORES        = [2, 3]
CNN_DROPOUT_VALORES       = [0.0, 0.1]
CNN_ACTIVACION_VALORES    = ["relu", "leaky_relu"]
CNN_TAMANIO_DENSO_VALORES = [16, 32]
CNN_EPOCAS_VALORES        = [50]
CNN_LR_VALORES            = [0.001, 0.0005]
CNN_LOTE_VALORES          = [8, 16]

for _dir in [
    R11_ESCALADOR_DIR,
    R11_MLP_DIR,
    R11_LSTM_DIR,
    R11_CNN_DIR,
    R11_COMPARACION_DIR,
]:
    os.makedirs(_dir, exist_ok=True)
