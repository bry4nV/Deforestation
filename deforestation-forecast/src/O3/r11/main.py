"""
Orquestador de R11 — Modelos de pronóstico multivariables.

Protocolo de ejecución:
  1. Carga panel multivariable (180 distritos × 40 años × 7 canales)
  2. Ajusta StandardScaler (solo sobre 1985–2019) y escala el panel completo
  3. Construye datasets de ventanas deslizantes para todas las ventanas DL
  4. Fase 1 (grid search) para MLP, LSTM, CNN — idempotente por modelo
  5. Análisis visual de Fase 1
  6. Fase 2 (config final) para MLP, LSTM, CNN — solo cuando FINAL_CONFIG_* ≠ None
  7. Comparación base (O2) vs extendido (R11), por departamento y boxplot distrital
  8. Pronóstico 2025 (sin reentrenamiento, ancla = pct_bosque_real_2024)

Uso:
    python -m O3.r11.main
"""

import logging
import os

import numpy as np
import pandas as pd

from O3.config import (
    ANIO_INICIO,
    DL_VENTANAS,
    R11_CNN_DIR,
    R11_COMPARACION_DIR,
    R11_LSTM_DIR,
    R11_MLP_DIR,
    TAMANIO_ENTRENAMIENTO,
)
from O3.r11.seleccion_fase1 import generar_seleccion_final
from O3.r11.cargar_panel import cargar_panel
from O3.r11.construir_dataset import construir_datasets
from O3.r11.escalador import ajustar_y_escalar
from O3.r11.pronostico_2025 import generar_pronostico_2025
from O3.r11.transformaciones import aplicar_transformaciones
from O3.utils import iniciar_log_archivo
from O3.r11.final_configs import FINAL_CONFIG_CNN, FINAL_CONFIG_LSTM, FINAL_CONFIG_MLP
from O3.r11.pipeline_cnn import entrenar_config_final_cnn, pipeline_cnn
from O3.r11.pipeline_comparacion import pipeline_comparacion
from O3.r11.pipeline_lstm import entrenar_config_final_lstm, pipeline_lstm
from O3.r11.pipeline_mlp import entrenar_config_final_mlp, pipeline_mlp
from O3.config import (
    CNN_ACTIVACION_VALORES,
    CNN_CANALES_CONV_VALORES,
    CNN_DROPOUT_VALORES,
    CNN_EPOCAS_VALORES,
    CNN_KERNEL_VALORES,
    CNN_LR_VALORES,
    CNN_LOTE_VALORES,
    CNN_TAMANIO_DENSO_VALORES,
    LSTM_DROPOUT_VALORES,
    LSTM_EPOCAS_VALORES,
    LSTM_LR_VALORES,
    LSTM_LOTE_VALORES,
    LSTM_NUM_CAPAS_VALORES,
    LSTM_UNIDADES_OCULTAS_VALORES,
    MLP_ACTIVACION_VALORES,
    MLP_CAPAS_OCULTAS_VALORES,
    MLP_DROPOUT_VALORES,
    MLP_EPOCAS_VALORES,
    MLP_LR_VALORES,
    MLP_LOTE_VALORES,
)

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=" * 70)
    logger.info("R11 — PIPELINE DE MODELOS MULTIVARIABLES")
    logger.info("=" * 70)

    # =========================================================================
    # PASO 1: Panel multivariable y escalado
    # =========================================================================

    panel, df_distritos_info = cargar_panel(anio_inicio=1985, anio_fin=2024)

    logger.info("Aplicando transformaciones (log1p) antes de escalar...")
    panel_transformado = aplicar_transformaciones(panel)

    panel_escalado, escalador = ajustar_y_escalar(panel_transformado)

    anios = list(range(ANIO_INICIO, ANIO_INICIO + panel.shape[1]))

    # =========================================================================
    # PASO 2: Datasets por ventana
    # =========================================================================

    datasets = construir_datasets(panel_escalado, DL_VENTANAS, TAMANIO_ENTRENAMIENTO)

    # =========================================================================
    # PASO 3: MLP
    # =========================================================================

    res_mlp = None
    ruta_mlp       = os.path.join(R11_MLP_DIR,  "mlp.csv")
    ruta_mlp_fase1 = ruta_mlp.replace(".csv", "_resultados.csv")
    ruta_mlp_gbl   = ruta_mlp.replace(".csv", "_final_global.csv")
    ruta_mlp_npy   = ruta_mlp.replace(".csv", "_final_ypred.npy")

    if not os.path.exists(ruta_mlp_fase1):
        logger.info("MLP R11 — Fase 1: búsqueda exploratoria...")
        pipeline_mlp(
            datasets, ruta_mlp, escalador,
            epocas_valores=MLP_EPOCAS_VALORES,
            lr_valores=MLP_LR_VALORES,
            lote_valores=MLP_LOTE_VALORES,
            capas_ocultas_valores=MLP_CAPAS_OCULTAS_VALORES,
            dropout_valores=MLP_DROPOUT_VALORES,
            activacion_valores=MLP_ACTIVACION_VALORES,
        )
    else:
        logger.info("[SKIP] MLP Fase 1 — ya existe. Revisar mlp_resultados.csv.")

    if FINAL_CONFIG_MLP is not None:
        if os.path.exists(ruta_mlp_npy) and os.path.exists(ruta_mlp_gbl):
            logger.info("[SKIP] MLP Fase 2 — cargando resultados existentes.")
            df_gbl = pd.read_csv(ruta_mlp_gbl)
            if df_gbl.empty:
                raise RuntimeError(f"Archivo vacío: {ruta_mlp_gbl}")
            row = df_gbl.iloc[0]
            res_mlp = {"modelo": row["modelo"], "rmse": float(row["rmse"]),
                       "mae": float(row["mae"]), "y_pred": np.load(ruta_mlp_npy),
                       "etiqueta": "R11_MLP"}
        else:
            logger.info("MLP R11 — Fase 2: entrenamiento final...")
            res_mlp = entrenar_config_final_mlp(
                datasets, FINAL_CONFIG_MLP, ruta_mlp,
                panel_escalado, df_distritos_info, escalador,
                TAMANIO_ENTRENAMIENTO, anios=anios,
            )
            res_mlp["etiqueta"] = "R11_MLP"
    else:
        logger.info("[PENDIENTE] MLP Fase 2 — configura FINAL_CONFIG_MLP en final_configs.py.")

    # =========================================================================
    # PASO 4: LSTM
    # =========================================================================

    res_lstm = None
    ruta_lstm       = os.path.join(R11_LSTM_DIR, "lstm.csv")
    ruta_lstm_fase1 = ruta_lstm.replace(".csv", "_resultados.csv")
    ruta_lstm_gbl   = ruta_lstm.replace(".csv", "_final_global.csv")
    ruta_lstm_npy   = ruta_lstm.replace(".csv", "_final_ypred.npy")

    if not os.path.exists(ruta_lstm_fase1):
        logger.info("LSTM R11 — Fase 1: búsqueda exploratoria...")
        pipeline_lstm(
            datasets, ruta_lstm, escalador,
            epocas_valores=LSTM_EPOCAS_VALORES,
            lr_valores=LSTM_LR_VALORES,
            lote_valores=LSTM_LOTE_VALORES,
            unidades_ocultas_valores=LSTM_UNIDADES_OCULTAS_VALORES,
            num_capas_valores=LSTM_NUM_CAPAS_VALORES,
            dropout_valores=LSTM_DROPOUT_VALORES,
        )
    else:
        logger.info("[SKIP] LSTM Fase 1 — ya existe. Revisar lstm_resultados.csv.")

    if FINAL_CONFIG_LSTM is not None:
        if os.path.exists(ruta_lstm_npy) and os.path.exists(ruta_lstm_gbl):
            logger.info("[SKIP] LSTM Fase 2 — cargando resultados existentes.")
            df_gbl = pd.read_csv(ruta_lstm_gbl)
            if df_gbl.empty:
                raise RuntimeError(f"Archivo vacío: {ruta_lstm_gbl}")
            row = df_gbl.iloc[0]
            res_lstm = {"modelo": row["modelo"], "rmse": float(row["rmse"]),
                        "mae": float(row["mae"]), "y_pred": np.load(ruta_lstm_npy),
                        "etiqueta": "R11_LSTM"}
        else:
            logger.info("LSTM R11 — Fase 2: entrenamiento final...")
            res_lstm = entrenar_config_final_lstm(
                datasets, FINAL_CONFIG_LSTM, ruta_lstm,
                panel_escalado, df_distritos_info, escalador,
                TAMANIO_ENTRENAMIENTO, anios=anios,
            )
            res_lstm["etiqueta"] = "R11_LSTM"
    else:
        logger.info("[PENDIENTE] LSTM Fase 2 — configura FINAL_CONFIG_LSTM en final_configs.py.")
    
    # =========================================================================
    # PASO 5: CNN
    # =========================================================================

    res_cnn = None
    ruta_cnn       = os.path.join(R11_CNN_DIR,  "cnn.csv")
    ruta_cnn_fase1 = ruta_cnn.replace(".csv", "_resultados.csv")
    ruta_cnn_gbl   = ruta_cnn.replace(".csv", "_final_global.csv")
    ruta_cnn_npy   = ruta_cnn.replace(".csv", "_final_ypred.npy")

    if not os.path.exists(ruta_cnn_fase1):
        logger.info("CNN R11 — Fase 1: búsqueda exploratoria...")
        pipeline_cnn(
            datasets, ruta_cnn, escalador,
            epocas_valores=CNN_EPOCAS_VALORES,
            lr_valores=CNN_LR_VALORES,
            lote_valores=CNN_LOTE_VALORES,
            canales_conv_valores=CNN_CANALES_CONV_VALORES,
            kernel_valores=CNN_KERNEL_VALORES,
            dropout_valores=CNN_DROPOUT_VALORES,
            activacion_valores=CNN_ACTIVACION_VALORES,
            tamanio_denso_valores=CNN_TAMANIO_DENSO_VALORES,
        )
    else:
        logger.info("[SKIP] CNN Fase 1 — ya existe. Revisar cnn_resultados.csv.")

    if FINAL_CONFIG_CNN is not None:
        if os.path.exists(ruta_cnn_npy) and os.path.exists(ruta_cnn_gbl):
            logger.info("[SKIP] CNN Fase 2 — cargando resultados existentes.")
            df_gbl = pd.read_csv(ruta_cnn_gbl)
            if df_gbl.empty:
                raise RuntimeError(f"Archivo vacío: {ruta_cnn_gbl}")
            row = df_gbl.iloc[0]
            res_cnn = {"modelo": row["modelo"], "rmse": float(row["rmse"]),
                       "mae": float(row["mae"]), "y_pred": np.load(ruta_cnn_npy),
                       "etiqueta": "R11_CNN"}
        else:
            logger.info("CNN R11 — Fase 2: entrenamiento final...")
            res_cnn = entrenar_config_final_cnn(
                datasets, FINAL_CONFIG_CNN, ruta_cnn,
                panel_escalado, df_distritos_info, escalador,
                TAMANIO_ENTRENAMIENTO, anios=anios,
            )
            res_cnn["etiqueta"] = "R11_CNN"
    else:
        logger.info("[PENDIENTE] CNN Fase 2 — configura FINAL_CONFIG_CNN en final_configs.py.")

    # =========================================================================
    # PASO 6: Verificación de configuraciones finales
    # =========================================================================

    logger.info("=" * 70)
    logger.info("VERIFICACIÓN DE CONFIGURACIONES FINALES")
    logger.info("=" * 70)
    generar_seleccion_final(
        final_configs={"mlp": FINAL_CONFIG_MLP, "lstm": FINAL_CONFIG_LSTM, "cnn": FINAL_CONFIG_CNN},
        ruta_salida=os.path.join(R11_COMPARACION_DIR, "seleccion_configuraciones_finales.csv"),
    )

    # =========================================================================
    # PASO 7: Comparación O2 vs R11
    # =========================================================================

    pendientes = [
        nombre for nombre, res in [("MLP", res_mlp), ("LSTM", res_lstm), ("CNN", res_cnn)]
        if res is None
    ]
    if pendientes:
        logger.info(
            f"[PENDIENTE] Comparación — faltan Fase 2 de: {', '.join(pendientes)}\n"
            "            Completa las configs en final_configs.py y vuelve a ejecutar."
        )
    else:
        rutas_departamento = {
            "MLP":  os.path.join(R11_MLP_DIR,  "mlp_final_departamento.csv"),
            "LSTM": os.path.join(R11_LSTM_DIR, "lstm_final_departamento.csv"),
            "CNN":  os.path.join(R11_CNN_DIR,  "cnn_final_departamento.csv"),
        }
        rutas_distrito_dl = {
            "MLP":  os.path.join(R11_MLP_DIR,  "mlp_final_distrito.csv"),
            "LSTM": os.path.join(R11_LSTM_DIR, "lstm_final_distrito.csv"),
            "CNN":  os.path.join(R11_CNN_DIR,  "cnn_final_distrito.csv"),
        }
        pipeline_comparacion(
            panel_original=panel,
            df_distritos_info=df_distritos_info,
            tamanio_entrenamiento=TAMANIO_ENTRENAMIENTO,
            anio_inicio=ANIO_INICIO,
            rutas_departamento=rutas_departamento,
            rutas_distrito_dl=rutas_distrito_dl,
        )

        # =====================================================================
        # PASO 8: PRONÓSTICO 2025 (sin reentrenamiento, ancla = pct_bosque_real_2024)
        # =====================================================================

        ruta_pronostico_2025 = os.path.join(R11_COMPARACION_DIR, "pronostico_2025.csv")
        if os.path.exists(ruta_pronostico_2025):
            logger.info(f"[SKIP] Pronóstico 2025 — ya existe {ruta_pronostico_2025}.")
        else:
            logger.info("=" * 70)
            logger.info("PRONÓSTICO 2025 — MLP/LSTM/CNN multivariables")
            logger.info("=" * 70)
            rutas_modelo_2025 = {
                "mlp":  os.path.join(R11_MLP_DIR,  "mlp_final_model.pth"),
                "lstm": os.path.join(R11_LSTM_DIR, "lstm_final_model.pth"),
                "cnn":  os.path.join(R11_CNN_DIR,  "cnn_final_model.pth"),
            }
            df_pronostico_2025 = generar_pronostico_2025(
                panel_escalado, panel, df_distritos_info, rutas_modelo_2025, escalador,
                anio_anchor=anios[-1],
            )
            df_pronostico_2025.to_csv(ruta_pronostico_2025, index=False)
            logger.info(f"[OK] {ruta_pronostico_2025}")
            logger.info("\n" + df_pronostico_2025.head().to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    iniciar_log_archivo("r11")
    main()
