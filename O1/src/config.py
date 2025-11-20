import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "Output")
MAPAS_RECLAS_DIR = os.path.join(OUTPUT_DIR, "MapasReclasificados")
METADATOS_DIR = os.path.join(OUTPUT_DIR, "Metadatos")

for d in [OUTPUT_DIR, MAPAS_RECLAS_DIR, METADATOS_DIR]:
    os.makedirs(d, exist_ok=True)

ANIOS = list(range(2020, 2024))

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

# Valor nodata para uint8
NODATA_OUT = 255
