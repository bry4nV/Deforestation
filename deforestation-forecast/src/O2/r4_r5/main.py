import os
from O1.config import (
    SERIES_ENTRENAMIENTO_DIR, TAMANIO_ENTRENAMIENTO, HORIZONTE
)

from O2.config import (
    PERSISTENCIA_DIR, ARIMA_DIR, MLP_DIR,
    MLP_DEFAULT_EPOCHS, MLP_DEFAULT_LR, MLP_DEFAULT_BATCH_SIZE,
    MLP_HIDDEN_SIZES
)

from O2.r4_r5.analisis_arima import generar_analisis_arima
from O2.r4_r5.construir_dataset import cargar_series, construir_dataset_estadistico, construir_dataset_dl
from O2.r4_r5.pipeline_persistencia import pipeline_persistencia
from O2.r4_r5.pipeline_arima import grid_search_arima, pipeline_arima
from O2.r4_r5.pipeline_mlp import pipeline_mlp

def main():

    print("\n" + "="*70)
    print(" PIPELINE DE MODELOS DE PRONÓSTICO ")
    print("="*70)

    # =====================================================================
    # PASO 1: CONSTRUCCIÓN DEL DATASET
    # =====================================================================

    print("\n[INFO] Construyendo dataset...")

    ruta_series_entrenamiento = os.path.join(SERIES_ENTRENAMIENTO_DIR, "distritos_entrenamiento.csv")
    series, df_distritos_info = cargar_series(ruta_series_entrenamiento)

    X_train_stat, y_train_stat = construir_dataset_estadistico(series, TAMANIO_ENTRENAMIENTO, HORIZONTE)

    window_sizes = [3, 4, 5]

    dataset_dl = construir_dataset_dl(series, window_sizes, TAMANIO_ENTRENAMIENTO)

    # =====================================================================
    # PASO 2: MODELOS ESTADÍSTICOS
    # =====================================================================

    resultados = []

    print("\n[INFO] Ejecutando modelo de persistencia...")
    ruta_modelo_persistencia = os.path.join(PERSISTENCIA_DIR, "persistencia_resultados.csv")
    res_persistencia = pipeline_persistencia(X_train_stat, y_train_stat, df_distritos_info, ruta_modelo_persistencia)
    resultados.append(res_persistencia)

    ruta_estadisticas = os.path.join(SERIES_ENTRENAMIENTO_DIR, "estadisticas_distritos_entrenamiento.csv")
    ruta_analisis_arima = os.path.join(ARIMA_DIR, "analisis_arima")

    generar_analisis_arima(ruta_estadisticas, ruta_series_entrenamiento, ruta_analisis_arima)

    print("\n[INFO] Ejecutando modelo ARIMA...")

    best_cfg, best_score = grid_search_arima(
        X_train_stat,
        y_train_stat,
        df_distritos_info,
        p_values=[1, 2],
        d_values=[1],
        q_values=[0, 1],
        window_values=[3, 4, 5],
        ruta_base=os.path.join(ARIMA_DIR, "arima.csv")
    )

    best_w, best_p, best_d, best_q = best_cfg

    ruta_modelo_arima = os.path.join(ARIMA_DIR, f"arima_best_w{best_w}_p{best_p}_d{best_d}_q{best_q}.csv")

    res = pipeline_arima(
        X_train_stat,
        y_train_stat,
        df_distritos_info,
        ruta_modelo_arima,
        best_w,
        order=(best_p, best_d, best_q)
    )

    resultados.append(res)

    # =====================================================================
    # PASO 3: MODELOS DEEP LEARNING (MLP)
    # =====================================================================

    for w in window_sizes:

        print(f"\n[INFO] Ejecutando pipeline MLP con ventana={w}...")

        ruta_modelo_mlp = os.path.join(MLP_DIR, f"mlp_w{w}_resultados.csv")

        res_mlp = pipeline_mlp(
            datasets_dl=dataset_dl,
            df_distritos_info=df_distritos_info,
            window_size=w,
            epochs=MLP_DEFAULT_EPOCHS,
            lr=MLP_DEFAULT_LR,
            batch_size=MLP_DEFAULT_BATCH_SIZE,
            hidden_sizes=MLP_HIDDEN_SIZES,
            ruta_modelo_mlp=ruta_modelo_mlp,
            exportar=True
        )

        resultados.append(res_mlp)

    # =====================================================================
    # PASO 4: COMPARACIÓN DE RESULTADOS
    # =====================================================================

    print("\n" + "="*70)
    print(" RESUMEN DE RESULTADOS ")
    print("="*70)

    for i, res in enumerate(resultados, 1):
        print(f"\n[{i}] {res['modelo']}")
        if "rmse" in res:
            print(f"    RMSE: {res['rmse']:.4f}")
        if "rmse_test" in res:
            print(f"    RMSE (Test): {res['rmse_test']:.4f}")

    print("\n[OK] Pipeline finalizado")

    return

if __name__ == "__main__":
    main()
