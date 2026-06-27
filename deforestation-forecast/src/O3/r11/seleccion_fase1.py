"""
Verificación de la configuración final seleccionada por modelo de R11 (Fase 1 -> Fase 2).

No decide la configuración final: esa decisión es manual e interpretativa (RMSE
como criterio principal, MAE y complejidad como respaldo) y se registra en
final_configs.py. Este módulo solo confirma que la configuración elegida existe
en el grid search de Fase 1, recupera su RMSE/MAE exacto sin transcripción manual,
y reporta en qué posición del ranking por RMSE quedó -- para que cualquier
desviación del mínimo absoluto quede visible y pueda justificarse en el anexo.
"""

import logging
import os

import pandas as pd

from O3.config import R11_CNN_DIR, R11_LSTM_DIR, R11_MLP_DIR

logger = logging.getLogger(__name__)


MODELOS = {
    "mlp":  {"archivo": "mlp_resultados.csv",  "rmse_col": "rmse_test", "mae_col": "mae_test"},
    "lstm": {"archivo": "lstm_resultados.csv", "rmse_col": "rmse_test", "mae_col": "mae_test"},
    "cnn":  {"archivo": "cnn_resultados.csv",  "rmse_col": "rmse_test", "mae_col": "mae_test"},
}


def formatear_config(config: dict) -> str:
    return ", ".join(f"{clave}={valor}" for clave, valor in config.items())


def verificar_configuracion(df: pd.DataFrame, config: dict, rmse_col: str, mae_col: str):
    df = df.sort_values(rmse_col).reset_index(drop=True)

    mask = pd.Series(True, index=df.index)
    for clave, valor in config.items():
        if clave in df.columns:
            mask &= df[clave].astype(str) == str(valor)

    candidatos = df[mask]
    if candidatos.empty:
        return None

    fila = candidatos.iloc[0]
    return {
        "rmse":     float(fila[rmse_col]),
        "mae":      float(fila[mae_col]),
        "posicion": int(candidatos.index[0]) + 1,
        "total":    len(df),
    }


def generar_seleccion_final(
    mlp_dir: str = None,
    lstm_dir: str = None,
    cnn_dir: str = None,
    final_configs: dict = None,
    ruta_salida: str = None,
) -> pd.DataFrame:
    """
    Para cada modelo de R11 con configuración final definida, busca esa
    configuración exacta en su CSV de resultados de Fase 1 y exporta una tabla
    consolidada (modelo, configuración, RMSE/MAE de Fase 1, posición en el
    ranking) lista para usarse directamente en el Anexo E.
    """
    dirs = {
        "mlp":  mlp_dir  or R11_MLP_DIR,
        "lstm": lstm_dir or R11_LSTM_DIR,
        "cnn":  cnn_dir  or R11_CNN_DIR,
    }
    final_configs = final_configs or {}

    registros = []
    for nombre, spec in MODELOS.items():
        config = final_configs.get(nombre)
        carpeta = dirs.get(nombre)
        if config is None:
            logger.info(f"[SKIP] {nombre.upper()}: sin configuración final definida.")
            continue

        ruta = os.path.join(carpeta, spec["archivo"])
        if not os.path.exists(ruta):
            logger.info(f"[SKIP] {nombre.upper()}: no existe {ruta}")
            continue

        df = pd.read_csv(ruta)
        resultado = verificar_configuracion(df, config, spec["rmse_col"], spec["mae_col"])

        if resultado is None:
            logger.warning(f"[WARN] {nombre.upper()}: la configuración de final_configs.py no "
                            f"aparece en {ruta}. Revisar manualmente.")
            continue

        if resultado["posicion"] != 1:
            logger.info(f"[INFO] {nombre.upper()}: configuración elegida en posición "
                        f"{resultado['posicion']}/{resultado['total']} por RMSE -- "
                        f"confirmar que el anexo justifica la desviación del mínimo absoluto.")

        registros.append({
            "modelo":                nombre.upper(),
            "configuracion":         formatear_config(config),
            "rmse_fase1":            resultado["rmse"],
            "mae_fase1":             resultado["mae"],
            "posicion_ranking":      resultado["posicion"],
            "total_configuraciones": resultado["total"],
        })

    df_seleccion = pd.DataFrame(registros)
    if ruta_salida:
        df_seleccion.to_csv(ruta_salida, index=False)
        logger.info(f"[OK] Selección final verificada: {ruta_salida}")
    logger.info("\n" + df_seleccion.to_string(index=False))
    return df_seleccion
