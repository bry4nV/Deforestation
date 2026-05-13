import glob
import os

import numpy as np
import pandas as pd

from O1.config import (
    SERIES_ENTRENAMIENTO_DIR
)

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
    DL_WINDOW_VALUES
)

from O2.r4_r5.analisis_arima import generar_analisis_arima
from O2.r4_r5.construir_dataset import cargar_series, construir_dataset_estadistico, construir_dataset_dl
from O2.r4_r5.pipeline_persistencia import pipeline_persistencia
from O2.r4_r5.pipeline_arima import pipeline_arima
from O2.r4_r5.pipeline_mlp import pipeline_mlp
from O2.r4_r5.pipeline_lstm import pipeline_lstm
from O2.r4_r5.pipeline_cnn import pipeline_cnn
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

        res_persistencia = {
            "modelo": row["modelo"],
            "rmse": row["rmse"],
            "mae": row["mae"],
            "y_pred": np.load(ruta_pers_ypred),
        }
    else:
        print("\n[INFO] Ejecutando Persistencia...")
        res_persistencia = pipeline_persistencia(
            X_train_stat,
            y_train_stat,
            df_distritos_info,
            ruta_persistencia
        )

    resultados.append(res_persistencia)

    # — Diagnóstico ARIMA (ACF/PACF)
    ruta_estadisticas = os.path.join(SERIES_ENTRENAMIENTO_DIR, "estadisticas_distritos_entrenamiento.csv")
    generar_analisis_arima(ruta_estadisticas, ruta_series, ANALISIS_ARIMA_DIR)

    # — ARIMA
    ruta_base_arima = os.path.join(ARIMA_DIR, "arima.csv")
    ruta_arima_global = ruta_base_arima.replace(".csv", "_global.csv")
    ruta_arima_ypred = ruta_base_arima.replace(".csv", "_mejor_ypred.npy")

    if os.path.exists(ruta_arima_ypred):
        print("\n[SKIP] ARIMA — cargando resultados existentes.")

        row = pd.read_csv(ruta_arima_global).iloc[0]

        res_arima = {
            "modelo": row["modelo"],
            "rmse": row["rmse"],
            "mae": row["mae"],
            "y_pred": np.load(ruta_arima_ypred),
        }
    else:
        print("\n[INFO] Ejecutando ARIMA...")
        res_arima = pipeline_arima(
            X_train_stat,
            y_train_stat,
            df_distritos_info,
            ruta_base_arima,
            ARIMA_P_VALUES,
            ARIMA_D_VALUES,
            ARIMA_Q_VALUES,
            ARIMA_WINDOW_VALUES,
        )

    resultados.append(res_arima)

    # =====================================================================
    # PASO 3: MODELOS DEEP LEARNING
    # =====================================================================

    # — MLP
    ruta_mlp       = os.path.join(MLP_DIR, "mlp.csv")
    ruta_mlp_res   = ruta_mlp.replace(".csv", "_resultados.csv")
    ruta_mlp_ypred = ruta_mlp.replace(".csv", "_mejor_ypred.npy")

    if os.path.exists(ruta_mlp_ypred):
        print("\n[SKIP] MLP — cargando resultados existentes.")
        row = pd.read_csv(ruta_mlp_res).iloc[0]

        res_mlp = {
            "modelo": row["modelo"],
            "rmse": row["rmse_test"],
            "mae": row["mae_test"],
            "y_pred": np.load(ruta_mlp_ypred),
        }
    else:
        print("\n[INFO] Ejecutando MLP...")

        res_mlp = pipeline_mlp(
            dataset_dl,
            ruta_mlp,
            MLP_EPOCHS_VALUES,
            MLP_LR_VALUES,
            MLP_BATCH_SIZE_VALUES,
            MLP_HIDDEN_SIZES_VALUES,
            MLP_DROPOUT_VALUES,
            MLP_ACTIVATION_VALUES,
            series,
            df_distritos_info,
            TAMANIO_ENTRENAMIENTO,
        )

    resultados.append(res_mlp)

    # — LSTM
    ruta_lstm = os.path.join(LSTM_DIR, "lstm.csv")
    ruta_lstm_res = ruta_lstm.replace(".csv", "_resultados.csv")
    ruta_lstm_ypred = ruta_lstm.replace(".csv", "_mejor_ypred.npy")

    if os.path.exists(ruta_lstm_ypred):
        print("\n[SKIP] LSTM — cargando resultados existentes.")

        row = pd.read_csv(ruta_lstm_res).iloc[0]

        res_lstm = {
            "modelo": row["modelo"],
            "rmse": row["rmse_test"],
            "mae": row["mae_test"],
            "y_pred": np.load(ruta_lstm_ypred),
        }
    else:
        print("\n[INFO] Ejecutando LSTM...")

        res_lstm = pipeline_lstm(
            dataset_dl,
            ruta_lstm,
            LSTM_EPOCHS_VALUES,
            LSTM_LR_VALUES,
            LSTM_BATCH_SIZE_VALUES,
            LSTM_HIDDEN_SIZE_VALUES,
            LSTM_NUM_LAYERS_VALUES,
            LSTM_DROPOUT_VALUES,
            series,
            df_distritos_info,
            TAMANIO_ENTRENAMIENTO,
        )

    resultados.append(res_lstm)

    # — CNN
    ruta_cnn       = os.path.join(CNN_DIR, "cnn.csv")
    ruta_cnn_res   = ruta_cnn.replace(".csv", "_resultados.csv")
    ruta_cnn_ypred = ruta_cnn.replace(".csv", "_mejor_ypred.npy")

    if os.path.exists(ruta_cnn_ypred):
        print("\n[SKIP] CNN — cargando resultados existentes.")
        row = pd.read_csv(ruta_cnn_res).iloc[0]

        res_cnn = {
            "modelo": row["modelo"],
            "rmse":   row["rmse_test"],
            "mae":    row["mae_test"],
            "y_pred": np.load(ruta_cnn_ypred),
        }
    else:
        print("\n[INFO] Ejecutando CNN...")

        res_cnn = pipeline_cnn(
            dataset_dl,
            ruta_cnn,
            CNN_EPOCHS_VALUES,
            CNN_LR_VALUES,
            CNN_BATCH_SIZE_VALUES,
            CNN_CONV_CHANNELS_VALUES,
            CNN_KERNEL_SIZE_VALUES,
            CNN_DROPOUT_VALUES,
            CNN_ACTIVATION_VALUES,
            CNN_DENSE_SIZE_VALUES,
            series,
            df_distritos_info,
            TAMANIO_ENTRENAMIENTO,
        )

    resultados.append(res_cnn)

    # =====================================================================
    # PASO 4: COMPARACIÓN FINAL
    # =====================================================================

    pipeline_comparacion(
        resultados, series, df_distritos_info, TAMANIO_ENTRENAMIENTO,
        COMPARACION_DIR, anio_inicio=ANIO_INICIO,
    )


if __name__ == "__main__":
    main()
