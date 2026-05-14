import os

import numpy as np
import pandas as pd

from O1.config import SERIES_ENTRENAMIENTO_DIR

from O2.config import (
    PERSISTENCIA_DIR, ARIMA_DIR, ANALISIS_ARIMA_DIR, MLP_DIR, LSTM_DIR, CNN_DIR, COMPARACION_DIR,
    ANIO_INICIO, TAMANIO_ENTRENAMIENTO, HORIZONTE,
    ARIMA_P_VALUES, ARIMA_D_VALUES, ARIMA_Q_VALUES, ARIMA_WINDOW_VALUES,
    MLP_HIDDEN_SIZES_VALUES, MLP_DROPOUT_VALUES, MLP_ACTIVATION_VALUES,
    MLP_EPOCHS_VALUES, MLP_LR_VALUES, MLP_BATCH_SIZE_VALUES,
    LSTM_HIDDEN_SIZE_VALUES, LSTM_NUM_LAYERS_VALUES, LSTM_DROPOUT_VALUES,
    LSTM_EPOCHS_VALUES, LSTM_LR_VALUES, LSTM_BATCH_SIZE_VALUES,
    CNN_CONV_CHANNELS_VALUES, CNN_KERNEL_SIZE_VALUES, CNN_DROPOUT_VALUES,
    CNN_ACTIVATION_VALUES, CNN_DENSE_SIZE_VALUES,
    CNN_EPOCHS_VALUES, CNN_LR_VALUES, CNN_BATCH_SIZE_VALUES,
    DL_WINDOW_VALUES,
)

from O2.r4_r5.final_configs import FINAL_CONFIG_ARIMA, FINAL_CONFIG_MLP, FINAL_CONFIG_LSTM, FINAL_CONFIG_CNN
from O2.r4_r5.analisis_arima import generar_analisis_arima
from O2.r4_r5.construir_dataset import cargar_series, construir_dataset_estadistico, construir_dataset_dl
from O2.r4_r5.pipeline_persistencia import pipeline_persistencia
from O2.r4_r5.pipeline_arima import pipeline_arima, evaluar_config_final_arima
from O2.r4_r5.pipeline_mlp import pipeline_mlp, entrenar_config_final_mlp
from O2.r4_r5.pipeline_lstm import pipeline_lstm, entrenar_config_final_lstm
from O2.r4_r5.pipeline_cnn import pipeline_cnn, entrenar_config_final_cnn
from O2.r4_r5.pipeline_comparacion import pipeline_comparacion


def main():

    print("\n" + "=" * 70)
    print(" PIPELINE DE MODELOS DE PRONÓSTICO ")
    print("=" * 70)

    # =====================================================================
    # PASO 1: CONSTRUCCIÓN DEL DATASET
    # =====================================================================

    ruta_series = os.path.join(SERIES_ENTRENAMIENTO_DIR, "distritos_entrenamiento.csv")
    series, df_distritos_info = cargar_series(ruta_series)

    anios = list(range(ANIO_INICIO, ANIO_INICIO + series.shape[1]))

    X_train_stat, y_train_stat = construir_dataset_estadistico(
        series, TAMANIO_ENTRENAMIENTO, HORIZONTE
    )

    dataset_dl = construir_dataset_dl(series, DL_WINDOW_VALUES, TAMANIO_ENTRENAMIENTO)

    # =====================================================================
    # PASO 2: MODELOS ESTADÍSTICOS
    # =====================================================================

    resultados = []

    # — Persistencia
    ruta_persistencia = os.path.join(PERSISTENCIA_DIR, "persistencia_resultados.csv")
    ruta_pers_global  = ruta_persistencia.replace(".csv", "_global.csv")
    ruta_pers_ypred   = ruta_persistencia.replace(".csv", "_ypred.npy")

    if os.path.exists(ruta_pers_ypred):
        print("\n[SKIP] Persistencia — cargando resultados existentes.")
        row = pd.read_csv(ruta_pers_global).iloc[0]
        resultados.append({
            "modelo": row["modelo"],
            "rmse":   float(row["rmse"]),
            "mae":    float(row["mae"]),
            "y_pred": np.load(ruta_pers_ypred),
        })
    else:
        resultados.append(pipeline_persistencia(
            X_train_stat, y_train_stat, df_distritos_info, ruta_persistencia,
            anios=anios[-HORIZONTE:],
        ))

    # — Diagnóstico ARIMA
    ruta_estadisticas = os.path.join(
        SERIES_ENTRENAMIENTO_DIR, "estadisticas_distritos_entrenamiento.csv"
    )
    generar_analisis_arima(ruta_estadisticas, ruta_series, ANALISIS_ARIMA_DIR)

    # — ARIMA
    ruta_base_arima  = os.path.join(ARIMA_DIR, "arima.csv")
    ruta_arima_fase1 = ruta_base_arima.replace(".csv", "_resultados.csv")
    ruta_arima_gbl   = ruta_base_arima.replace(".csv", "_final_global.csv")
    ruta_arima_npy   = ruta_base_arima.replace(".csv", "_final_ypred.npy")

    res_arima = None
    res_mlp   = None
    res_lstm  = None
    res_cnn   = None

    if not os.path.exists(ruta_arima_fase1):
        print("\n[INFO] ARIMA — Fase 1: búsqueda exploratoria...")
        pipeline_arima(
            X_train_stat, y_train_stat, df_distritos_info, ruta_base_arima,
            ARIMA_P_VALUES, ARIMA_D_VALUES, ARIMA_Q_VALUES, ARIMA_WINDOW_VALUES,
        )
    else:
        print("\n[SKIP] ARIMA Fase 1 — ya existe. Revisar arima_resultados.csv.")

    if FINAL_CONFIG_ARIMA is not None:
        if os.path.exists(ruta_arima_npy):
            print("\n[SKIP] ARIMA Fase 2 — cargando resultados existentes.")
            row = pd.read_csv(ruta_arima_gbl).iloc[0]
            res_arima = {
                "modelo": row["modelo"],
                "rmse":   float(row["rmse"]),
                "mae":    float(row["mae"]),
                "y_pred": np.load(ruta_arima_npy),
            }
        else:
            print("\n[INFO] ARIMA — Fase 2: evaluación final...")
            res_arima = evaluar_config_final_arima(
                X_train_stat, y_train_stat, df_distritos_info,
                FINAL_CONFIG_ARIMA, ruta_base_arima,
                anios=anios[-HORIZONTE:],
            )
        resultados.append(res_arima)
    else:
        print("\n[PENDIENTE] ARIMA Fase 2 — configura FINAL_CONFIG_ARIMA en final_configs.py.")

    # =====================================================================
    # PASO 3: MODELOS DEEP LEARNING
    # =====================================================================

    # — MLP
    ruta_mlp       = os.path.join(MLP_DIR, "mlp.csv")
    ruta_mlp_fase1 = ruta_mlp.replace(".csv", "_resultados.csv")
    ruta_mlp_gbl   = ruta_mlp.replace(".csv", "_final_global.csv")
    ruta_mlp_npy   = ruta_mlp.replace(".csv", "_final_ypred.npy")

    if not os.path.exists(ruta_mlp_fase1):
        print("\n[INFO] MLP — Fase 1: búsqueda exploratoria...")
        pipeline_mlp(
            dataset_dl, ruta_mlp,
            epochs_values=MLP_EPOCHS_VALUES,
            lr_values=MLP_LR_VALUES,
            batch_size_values=MLP_BATCH_SIZE_VALUES,
            hidden_sizes_values=MLP_HIDDEN_SIZES_VALUES,
            dropout_values=MLP_DROPOUT_VALUES,
            activation_values=MLP_ACTIVATION_VALUES,
        )
    else:
        print("\n[SKIP] MLP Fase 1 — ya existe. Revisar mlp_resultados.csv.")

    if FINAL_CONFIG_MLP is not None:
        if os.path.exists(ruta_mlp_npy):
            print("\n[SKIP] MLP Fase 2 — cargando resultados existentes.")
            row = pd.read_csv(ruta_mlp_gbl).iloc[0]
            res_mlp = {
                "modelo": row["modelo"],
                "rmse":   float(row["rmse"]),
                "mae":    float(row["mae"]),
                "y_pred": np.load(ruta_mlp_npy),
            }
        else:
            print("\n[INFO] MLP — Fase 2: entrenamiento final...")
            res_mlp = entrenar_config_final_mlp(
                dataset_dl, FINAL_CONFIG_MLP, ruta_mlp,
                series, df_distritos_info, TAMANIO_ENTRENAMIENTO, anios=anios,
            )
        resultados.append(res_mlp)
    else:
        print("\n[PENDIENTE] MLP Fase 2 — configura FINAL_CONFIG_MLP en final_configs.py.")

    # — LSTM
    ruta_lstm       = os.path.join(LSTM_DIR, "lstm.csv")
    ruta_lstm_fase1 = ruta_lstm.replace(".csv", "_resultados.csv")
    ruta_lstm_gbl   = ruta_lstm.replace(".csv", "_final_global.csv")
    ruta_lstm_npy   = ruta_lstm.replace(".csv", "_final_ypred.npy")

    if not os.path.exists(ruta_lstm_fase1):
        print("\n[INFO] LSTM — Fase 1: búsqueda exploratoria...")
        pipeline_lstm(
            dataset_dl, ruta_lstm,
            epochs_values=LSTM_EPOCHS_VALUES,
            lr_values=LSTM_LR_VALUES,
            batch_size_values=LSTM_BATCH_SIZE_VALUES,
            hidden_size_values=LSTM_HIDDEN_SIZE_VALUES,
            num_layers_values=LSTM_NUM_LAYERS_VALUES,
            dropout_values=LSTM_DROPOUT_VALUES,
        )
    else:
        print("\n[SKIP] LSTM Fase 1 — ya existe. Revisar lstm_resultados.csv.")

    if FINAL_CONFIG_LSTM is not None:
        if os.path.exists(ruta_lstm_npy):
            print("\n[SKIP] LSTM Fase 2 — cargando resultados existentes.")
            row = pd.read_csv(ruta_lstm_gbl).iloc[0]
            res_lstm = {
                "modelo": row["modelo"],
                "rmse":   float(row["rmse"]),
                "mae":    float(row["mae"]),
                "y_pred": np.load(ruta_lstm_npy),
            }
        else:
            print("\n[INFO] LSTM — Fase 2: entrenamiento final...")
            res_lstm = entrenar_config_final_lstm(
                dataset_dl, FINAL_CONFIG_LSTM, ruta_lstm,
                series, df_distritos_info, TAMANIO_ENTRENAMIENTO, anios=anios,
            )
        resultados.append(res_lstm)
    else:
        print("\n[PENDIENTE] LSTM Fase 2 — configura FINAL_CONFIG_LSTM en final_configs.py.")

    # — CNN
    ruta_cnn       = os.path.join(CNN_DIR, "cnn.csv")
    ruta_cnn_fase1 = ruta_cnn.replace(".csv", "_resultados.csv")
    ruta_cnn_gbl   = ruta_cnn.replace(".csv", "_final_global.csv")
    ruta_cnn_npy   = ruta_cnn.replace(".csv", "_final_ypred.npy")

    if not os.path.exists(ruta_cnn_fase1):
        print("\n[INFO] CNN — Fase 1: búsqueda exploratoria...")
        pipeline_cnn(
            dataset_dl, ruta_cnn,
            epochs_values=CNN_EPOCHS_VALUES,
            lr_values=CNN_LR_VALUES,
            batch_size_values=CNN_BATCH_SIZE_VALUES,
            conv_channels_values=CNN_CONV_CHANNELS_VALUES,
            kernel_size_values=CNN_KERNEL_SIZE_VALUES,
            dropout_values=CNN_DROPOUT_VALUES,
            activation_values=CNN_ACTIVATION_VALUES,
            dense_size_values=CNN_DENSE_SIZE_VALUES,
        )
    else:
        print("\n[SKIP] CNN Fase 1 — ya existe. Revisar cnn_resultados.csv.")

    if FINAL_CONFIG_CNN is not None:
        if os.path.exists(ruta_cnn_npy):
            print("\n[SKIP] CNN Fase 2 — cargando resultados existentes.")
            row = pd.read_csv(ruta_cnn_gbl).iloc[0]
            res_cnn = {
                "modelo": row["modelo"],
                "rmse":   float(row["rmse"]),
                "mae":    float(row["mae"]),
                "y_pred": np.load(ruta_cnn_npy),
            }
        else:
            print("\n[INFO] CNN — Fase 2: entrenamiento final...")
            res_cnn = entrenar_config_final_cnn(
                dataset_dl, FINAL_CONFIG_CNN, ruta_cnn,
                series, df_distritos_info, TAMANIO_ENTRENAMIENTO, anios=anios,
            )
        resultados.append(res_cnn)
    else:
        print("\n[PENDIENTE] CNN Fase 2 — configura FINAL_CONFIG_CNN en final_configs.py.")

    # =====================================================================
    # PASO 4: COMPARACIÓN FINAL
    # =====================================================================

    pendientes = [
        nombre for nombre, res in [
            ("ARIMA", res_arima), ("MLP", res_mlp), ("LSTM", res_lstm), ("CNN", res_cnn)
        ]
        if res is None
    ]

    if pendientes:
        print(f"\n[PENDIENTE] Comparación — faltan Fase 2 de: {', '.join(pendientes)}")
        print("            Completa las configs en final_configs.py y vuelve a ejecutar.")
    else:
        pipeline_comparacion(
            resultados, series, df_distritos_info, TAMANIO_ENTRENAMIENTO,
            COMPARACION_DIR, anio_inicio=ANIO_INICIO,
        )


if __name__ == "__main__":
    main()
