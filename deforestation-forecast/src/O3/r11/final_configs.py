# =====================================================================
# CONFIGURACIONES FINALES — R11 MODELOS MULTIVARIABLES
#
# Completar después de revisar los resultados de Fase 1:
#   mlp_resultados.csv  /  mlp_top5_configuraciones.csv  /  mlp_mejores_por_ventana.csv
#   lstm_resultados.csv /  lstm_top5_configuraciones.csv /  lstm_mejores_por_ventana.csv
#   cnn_resultados.csv  /  cnn_top5_configuraciones.csv  /  cnn_mejores_por_ventana.csv
#
# Mientras el valor sea None, Fase 2 queda pendiente para ese modelo.
# =====================================================================

FINAL_CONFIG_MLP = {
    "window_size":    6,
    "capas_ocultas":  [64, 32],
    "activacion":     "leaky_relu",
    "dropout":        0.0,
    "epocas":         50,
    "lr":             0.001,
    "lote":           16,
}

FINAL_CONFIG_LSTM = {
    "window_size":       6,
    "unidades_ocultas":  16,
    "num_capas":         2,
    "dropout":           0.0,
    "epocas":            50,
    "lr":                0.001,
    "lote":              8,
}

FINAL_CONFIG_CNN = {
    "window_size":    6,
    "canales_conv":   [32],
    "kernel_size":    2,
    "activacion":     "relu",
    "dropout":        0.0,
    "tamanio_denso":  16,
    "epocas":         50,
    "lr":             0.001,
    "lote":           8,
}
