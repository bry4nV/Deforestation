import glob
import os
import pandas as pd

from O1.config import (
    SERIES_ENTRENAMIENTO_DIR
)

from O2.config import (
    PERSISTENCIA_DIR, ARIMA_DIR, ANALISIS_ARIMA_DIR, MLP_DIR, LSTM_DIR,
    ARIMA_P_VALUES, ARIMA_D_VALUES, ARIMA_Q_VALUES, ARIMA_WINDOW_VALUES,
    MLP_HIDDEN_SIZES_VALUES, MLP_DROPOUT_VALUES,
    MLP_EPOCHS_VALUES, MLP_LR_VALUES, MLP_BATCH_SIZE_VALUES,
    LSTM_HIDDEN_SIZE_VALUES, LSTM_NUM_LAYERS_VALUES, LSTM_DROPOUT_VALUES,
    LSTM_EPOCHS_VALUES, LSTM_LR_VALUES, LSTM_BATCH_SIZE_VALUES,
    DL_WINDOW_VALUES,
    TAMANIO_ENTRENAMIENTO, HORIZONTE
)

from O2.r4_r5.analisis_arima import generar_analisis_arima
from O2.r4_r5.construir_dataset import cargar_series, construir_dataset_estadistico, construir_dataset_dl
from O2.r4_r5.pipeline_persistencia import pipeline_persistencia
from O2.r4_r5.pipeline_arima import pipeline_arima
from O2.r4_r5.pipeline_mlp import pipeline_mlp
from O2.r4_r5.pipeline_lstm import pipeline_lstm


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
    if os.path.exists(ruta_pers_global):
        print("\n[SKIP] Persistencia — cargando resultados existentes.")
        row = pd.read_csv(ruta_pers_global).iloc[0]
        res_persistencia = {"modelo": row["modelo"], "rmse": row["rmse"], "mae": row["mae"]}
    else:
        print("\n[INFO] Ejecutando Persistencia...")
        res_persistencia = pipeline_persistencia(
            X_train_stat, y_train_stat, df_distritos_info, ruta_persistencia
        )
    resultados.append(res_persistencia)

    # — Diagnóstico ARIMA (ACF/PACF)
    ruta_estadisticas = os.path.join(SERIES_ENTRENAMIENTO_DIR, "estadisticas_distritos_entrenamiento.csv")
    if glob.glob(os.path.join(ANALISIS_ARIMA_DIR, "*.png")):
        print("[SKIP] Análisis ARIMA — gráficos ya generados.")
    else:
        generar_analisis_arima(ruta_estadisticas, ruta_series, ANALISIS_ARIMA_DIR)

    # — ARIMA
    ruta_base_arima    = os.path.join(ARIMA_DIR, "arima.csv")
    ruta_arima_mejores = ruta_base_arima.replace(".csv", "_mejores_por_ventana.csv")
    if os.path.exists(ruta_arima_mejores):
        print("[SKIP] ARIMA — cargando resultados existentes.")
        best = pd.read_csv(ruta_arima_mejores).sort_values("rmse").iloc[0]
        res_arima = {
            "modelo": f"ARIMA_WF_w{int(best['window'])}",
            "rmse":   best["rmse"],
            "mae":    best["mae"],
        }
    else:
        res_arima = pipeline_arima(
            X_train_stat, y_train_stat, df_distritos_info,
            ruta_base_arima,
            ARIMA_P_VALUES, ARIMA_D_VALUES, ARIMA_Q_VALUES, ARIMA_WINDOW_VALUES,
        )
    resultados.append(res_arima)

    # =====================================================================
    # PASO 3: MODELOS DEEP LEARNING
    # =====================================================================

    # — MLP
    ruta_mlp     = os.path.join(MLP_DIR, "mlp.csv")
    ruta_mlp_res = ruta_mlp.replace(".csv", "_resultados.csv")
    if os.path.exists(ruta_mlp_res):
        print("\n[SKIP] MLP — cargando resultados existentes.")
        row = pd.read_csv(ruta_mlp_res).iloc[0]
        res_mlp = {"modelo": row["modelo"], "rmse": row["rmse_test"], "mae": row["mae_test"]}
    else:
        print("\n[INFO] Ejecutando MLP...")
        res_mlp = pipeline_mlp(
            dataset_dl, ruta_mlp,
            MLP_EPOCHS_VALUES, MLP_LR_VALUES, MLP_BATCH_SIZE_VALUES,
            MLP_HIDDEN_SIZES_VALUES, MLP_DROPOUT_VALUES,
            series, df_distritos_info, TAMANIO_ENTRENAMIENTO,
        )
    resultados.append(res_mlp)

    # — LSTM
    ruta_lstm     = os.path.join(LSTM_DIR, "lstm.csv")
    ruta_lstm_res = ruta_lstm.replace(".csv", "_resultados.csv")
    if os.path.exists(ruta_lstm_res):
        print("\n[SKIP] LSTM — cargando resultados existentes.")
        row = pd.read_csv(ruta_lstm_res).iloc[0]
        res_lstm = {"modelo": row["modelo"], "rmse": row["rmse_test"], "mae": row["mae_test"]}
    else:
        print("\n[INFO] Ejecutando LSTM...")
        res_lstm = pipeline_lstm(
            dataset_dl, ruta_lstm,
            LSTM_EPOCHS_VALUES, LSTM_LR_VALUES, LSTM_BATCH_SIZE_VALUES,
            LSTM_HIDDEN_SIZE_VALUES, LSTM_NUM_LAYERS_VALUES, LSTM_DROPOUT_VALUES,
            series, df_distritos_info, TAMANIO_ENTRENAMIENTO,
        )
    resultados.append(res_lstm)

    # =====================================================================
    # PASO 4: COMPARACIÓN FINAL
    # =====================================================================

    print("\n" + "=" * 70)
    print(" COMPARACIÓN DE MODELOS ")
    print("=" * 70)

    df_comp = (
        pd.DataFrame([
            {"modelo": r["modelo"], "rmse": r["rmse"], "mae": r["mae"]}
            for r in resultados
        ])
        .sort_values("rmse")
        .reset_index(drop=True)
    )

    print(df_comp.to_string(index=False))

    ruta_comp = os.path.join(
        os.path.dirname(PERSISTENCIA_DIR), "comparacion_modelos.csv"
    )
    df_comp.to_csv(ruta_comp, index=False)
    print(f"\n[OK] Comparación guardada: {ruta_comp}")

    mejor = df_comp.iloc[0]
    print(f"\n[GANADOR] {mejor['modelo']}  RMSE={mejor['rmse']:.4f}  MAE={mejor['mae']:.4f}")


if __name__ == "__main__":
    main()
