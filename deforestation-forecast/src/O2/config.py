import os
from O1.config import (
    INTERIM_DIR
)

# =============================
# R4, R5, R6 y R7
# =============================

O2_INTERIM_DIR = os.path.join(INTERIM_DIR, "O2")
MODELOS_DIR = os.path.join(O2_INTERIM_DIR, "modelos")
PERSISTENCIA_DIR    = os.path.join(MODELOS_DIR, "persistencia")
ARIMA_DIR           = os.path.join(MODELOS_DIR, "arima")
ANALISIS_ARIMA_DIR  = os.path.join(ARIMA_DIR,   "analisis_arima")
MLP_DIR             = os.path.join(MODELOS_DIR, "mlp")
LSTM_DIR            = os.path.join(MODELOS_DIR, "lstm")
COMPARACION_DIR     = os.path.join(MODELOS_DIR, "comparacion")

for d in [O2_INTERIM_DIR, MODELOS_DIR,
          PERSISTENCIA_DIR, ARIMA_DIR, ANALISIS_ARIMA_DIR, MLP_DIR, LSTM_DIR,
          COMPARACION_DIR]:
    os.makedirs(d, exist_ok=True)

ANIO_INICIO = 1985

TAMANIO_ENTRENAMIENTO = 35
HORIZONTE = 5

# =============================
# Hiperparámetros ARIMA
# =============================

ARIMA_P_VALUES = [0, 1, 2]
ARIMA_D_VALUES = [1]
ARIMA_Q_VALUES = [0, 1, 2]
ARIMA_WINDOW_VALUES = [3, 4, 5, 6, 7, 10, 15, 20, 25, 30, 35, None]

# =============================
# Reproducibilidad
# =============================

SEMILLA = 42

# =============================
# Hiperparámetros MLP
# =============================

MLP_ACTIVATION_VALUES   = ["relu", "leaky_relu"]
MLP_HIDDEN_SIZES_VALUES = [[32, 16], [64, 32], [128, 64, 32]]
MLP_DROPOUT_VALUES      = [0.0, 0.1]
MLP_EPOCHS_VALUES       = [50, 100]
MLP_LR_VALUES           = [0.001, 0.0005]
MLP_BATCH_SIZE_VALUES   = [8, 16]

# =============================
# Hiperparámetros LSTM
# =============================

LSTM_HIDDEN_SIZE_VALUES = [16, 32, 64]
LSTM_NUM_LAYERS_VALUES  = [1, 2]
LSTM_DROPOUT_VALUES     = [0.0]
LSTM_EPOCHS_VALUES      = [50, 100]
LSTM_LR_VALUES          = [0.001, 0.0005]
LSTM_BATCH_SIZE_VALUES  = [8, 16]

# =============================
# Ventanas Deep Learning
# =============================

DL_WINDOW_VALUES = [3, 4, 5, 6, 7]