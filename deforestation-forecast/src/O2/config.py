import os
from O1.config import (
    INTERIM_DIR
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

TAMANIO_ENTRENAMIENTO = 35
HORIZONTE = 5

# =============================
# HIPERPARÁMETROS - MLP
# =============================
# Rangos para grid search y pruebas
MLP_EPOCHS = [20, 50, 100]
MLP_LEARNING_RATES = [0.001, 0.01, 0.05]
MLP_BATCH_SIZES = [16, 32, 64]

# Valores por defecto para pruebas rápidas
MLP_DEFAULT_EPOCHS = 50
MLP_DEFAULT_LR = 0.01
MLP_DEFAULT_BATCH_SIZE = 32
MLP_HIDDEN_SIZES = [64, 32]  # Arquitectura de capas ocultas

# Directorio para modelos MLP
MLP_DIR = os.path.join(MODELOS_DIR, "mlp")
os.makedirs(MLP_DIR, exist_ok=True)