# =====================================================================
# CONFIGURACIONES FINALES — ARIMA y DEEP LEARNING
#
# Completar después de revisar los resultados de Fase 1:
#   mlp_resultados.csv  /  mlp_top5_configuraciones.csv  /  mlp_mejores_por_ventana.csv
#   lstm_resultados.csv /  lstm_top5_configuraciones.csv / lstm_mejores_por_ventana.csv
#   cnn_resultados.csv  /  cnn_top5_configuraciones.csv  /  cnn_mejores_por_ventana.csv
#
# Mientras el valor sea None, Fase 2 queda pendiente para ese modelo.
# =====================================================================

FINAL_CONFIG_ARIMA = None
# FINAL_CONFIG_ARIMA = {
#     "window": "full",   # "full" o un entero (ej: 10)
#     "p": 1,
#     "d": 1,
#     "q": 1,
# }

FINAL_CONFIG_MLP = None
# FINAL_CONFIG_MLP = {
#     "window_size":  5,
#     "hidden_sizes": [64, 32],
#     "activation":   "relu",
#     "dropout":      0.0,
#     "epochs":       50,
#     "lr":           0.001,
#     "batch_size":   16,
# }

FINAL_CONFIG_LSTM = None
# FINAL_CONFIG_LSTM = {
#     "window_size": 5,
#     "hidden_size": 32,
#     "num_layers":  1,
#     "dropout":     0.0,
#     "epochs":      50,
#     "lr":          0.001,
#     "batch_size":  16,
# }

FINAL_CONFIG_CNN = None
# FINAL_CONFIG_CNN = {
#     "window_size":   5,
#     "conv_channels": [16, 32],
#     "kernel_size":   3,
#     "activation":    "relu",
#     "dropout":       0.0,
#     "dense_size":    32,
#     "epochs":        50,
#     "lr":            0.001,
#     "batch_size":    16,
# }
