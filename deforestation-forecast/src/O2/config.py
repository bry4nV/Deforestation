import os
from O1.config import (
    INTERIM_DIR,
    O1_INTERIM_DIR
)

# =============================
# R4, R5, R6 y R7
# =============================

O2_INTERIM_DIR = os.path.join(INTERIM_DIR, "O2")
MODELOS_DIR = os.path.join(O2_INTERIM_DIR, "modelos")
PERSISTENCIA_DIR = os.path.join(MODELOS_DIR, "persistencia")
ARIMA_DIR = os.path.join(MODELOS_DIR, "arima")

for d in [O2_INTERIM_DIR, MODELOS_DIR, 
          PERSISTENCIA_DIR, ARIMA_DIR]:
    os.makedirs(d, exist_ok=True)