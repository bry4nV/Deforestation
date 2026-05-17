# =====================================================================
# CONFIGURACIONES FINALES — ARIMA y DEEP LEARNING
#
# Completar después de revisar los resultados de Fase 1:
#   mlp_resultados.csv  /  mlp_top5_configuraciones.csv  /  mlp_mejores_por_ventana.csv
#   lstm_resultados.csv /  lstm_top5_configuraciones.csv / lstm_mejores_por_ventana.csv
#   cnn_resultados.csv  /  cnn_top5_configuraciones.csv  /  cnn_mejores_por_ventana.csv
#   tcn_resultados.csv  /  tcn_top5_configuraciones.csv  /  tcn_mejores_por_ventana.csv
#
# Mientras el valor sea None, Fase 2 queda pendiente para ese modelo.
# =====================================================================

FINAL_CONFIG_ARIMA = {
    "window": 30,
    "p": 1,
    "d": 1,
    "q": 0,
}

FINAL_CONFIG_MLP = {
    "window_size":  3,
    "hidden_sizes": [128, 64, 32],
    "activation":   "leaky_relu",
    "dropout":      0.0,
    "epochs":       50,
    "lr":           0.001,
    "batch_size":   16,
}

FINAL_CONFIG_LSTM = {
    "window_size": 6,
    "hidden_size": 64,
    "num_layers":  1,
    "dropout":     0.0,
    "epochs":      50,
    "lr":          0.001,
    "batch_size":  16,
}

FINAL_CONFIG_CNN = {
    "window_size":   5,
    "conv_channels": [16, 32],
    "kernel_size":   3,
    "activation":    "relu",
    "dropout":       0.0,
    "dense_size":    32,
    "epochs":        50,
    "lr":            0.001,
    "batch_size":    8,
}

FINAL_CONFIG_TCN = None
