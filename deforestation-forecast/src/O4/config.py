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
O4_INTERIM_DIR = os.path.join(DATA_DIR, "interim", "O4")

# ============================================================
# RUTAS — INSUMOS YA EXISTENTES DE O1/O2/O3 (reusadas, no recalculadas)
# ============================================================
# Importadas desde los config.py de O1/O2/O3 en vez de reconstruir las
# mismas rutas aquí — si esos módulos cambian de estructura, O4 no se
# desincroniza silenciosamente.

from O1.config import SERIES_GENERALIZACION_ESPACIAL_DIR  # noqa: E402

DISTRITOS_GENERALIZACION_GPKG = os.path.join(
    SERIES_GENERALIZACION_ESPACIAL_DIR, "distritos_generalizacion_espacial.gpkg",
)

from O3.config import (  # noqa: E402
    ANIO_INICIO,
    PANEL_ENTRENAMIENTO_CSV,
    PANEL_GENERALIZACION_CSV,
    R11_CNN_DIR,
    R11_ESCALADOR_PKL,
    TAMANIO_ENTRENAMIENTO,
)

R11_CNN_MODEL_PTH = os.path.join(R11_CNN_DIR, "cnn_final_model.pth")
R11_CNN_GLOBAL_CSV = os.path.join(R11_CNN_DIR, "cnn_final_global.csv")

# ============================================================
# RUTAS — SALIDAS DE O4
# ============================================================

R12_DIR = os.path.join(O4_INTERIM_DIR, "r12_dataset_generalizacion")
R13_DIR = os.path.join(O4_INTERIM_DIR, "r13_pronosticos")
R14_DIR = os.path.join(O4_INTERIM_DIR, "r14_informe")

REPORTE_R12_CSV = os.path.join(R12_DIR, "reporte_r12.csv")

ANIO_FIN = 2024  # ANIO_INICIO y TAMANIO_ENTRENAMIENTO se reusan de O3.config

for _dir in [O4_INTERIM_DIR, R12_DIR, R13_DIR, R14_DIR]:
    os.makedirs(_dir, exist_ok=True)
