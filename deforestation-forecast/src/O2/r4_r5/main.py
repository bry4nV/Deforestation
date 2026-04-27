import os
from O1.config import (
    SERIES_ENTRENAMIENTO_DIR
)

from O2.config import (
    PERSISTENCIA_DIR, ARIMA_DIR
)

from O2.r4_r5.construir_dataset import construir_dataset
from O2.r4_r5.pipeline_lstm import pipeline_lstm
from O2.r4_r5.pipeline_cnn import pipeline_cnn
from O2.r4_r5.pipeline_arima import pipeline_arima
from O2.r4_r5.pipeline_persistencia import pipeline_persistencia

def main():

    print("\n" + "="*70)
    print(" PIPELINE DE MODELOS DE PRONÓSTICO ")
    print("="*70)

    # =====================================================================
    # PASO 1: CONSTRUCCIÓN DEL DATASET
    # =====================================================================

    print("\n[INFO] Construyendo dataset...")

    ruta_series_entrenamiento = os.path.join(SERIES_ENTRENAMIENTO_DIR, "distritos_entrenamiento.csv")

    X_train, y_train, df_distritos_info = construir_dataset(
        ruta_series_entrenamiento
    )

    # =====================================================================
    # PASO 2: MODELOS
    # =====================================================================

    resultados = []

    print("\n[INFO] Ejecutando modelo de persistencia...")
    ruta_modelo_persistencia = os.path.join(PERSISTENCIA_DIR, "persistencia_resultados.csv")
    res_persistencia = pipeline_persistencia(X_train, y_train, df_distritos_info, ruta_modelo_persistencia)
    resultados.append(res_persistencia)

    print("\n[INFO] Ejecutando modelo ARIMA...")
    ruta_modelo_arima = os.path.join(ARIMA_DIR, "arima_resultados.csv")
    res_arima = pipeline_arima(X_train, y_train, df_distritos_info, ruta_modelo_arima)
    resultados.append(res_arima)

    #print("\n[INFO] Ejecutando modelo LSTM...")
    #res_lstm = pipeline_lstm(X_train, y_train)
    #resultados.append(res_lstm)
#
    #print("\n[INFO] Ejecutando modelo CNN...")
    #res_cnn = pipeline_cnn(X_train, y_train)
    #resultados.append(res_cnn)
#
    ## =====================================================================
    ## PASO 3: COMPARACIÓN
    ## =====================================================================
#
    #print("\n[INFO] Comparando modelos...")
#
    ## ordenar por RMSE (ejemplo)
    #resultados = sorted(resultados, key=lambda x: x["rmse"])
#
    #mejor = resultados[0]
#
    #print(f"\n[OK] Mejor modelo: {mejor['modelo']}")
    #print(f"RMSE: {mejor['rmse']}")

    return

if __name__ == "__main__":
    main()