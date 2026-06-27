"""
Verificación de la configuración final seleccionada por modelo (Fase 1 -> Fase 2).

No decide la configuración final: esa decisión es manual e interpretativa (RMSE
como criterio principal, MAE y complejidad como respaldo) y se registra en
final_configs.py. Este módulo solo confirma que la configuración elegida existe
en el grid search de Fase 1, recupera su RMSE/MAE exacto sin transcripción manual,
y reporta en qué posición del ranking por RMSE quedó -- para que cualquier
desviación del mínimo absoluto quede visible y pueda justificarse en el anexo.
"""

import os

import pandas as pd

from O2.config import ARIMA_DIR, CNN_DIR, LSTM_DIR, MLP_DIR


MODELOS = {
    "arima": {"archivo": "arima_resultados.csv", "rmse_col": "rmse",      "mae_col": "mae"},
    "mlp":   {"archivo": "mlp_resultados.csv",   "rmse_col": "rmse_test", "mae_col": "mae_test"},
    "lstm":  {"archivo": "lstm_resultados.csv",  "rmse_col": "rmse_test", "mae_col": "mae_test"},
    "cnn":   {"archivo": "cnn_resultados.csv",   "rmse_col": "rmse_test", "mae_col": "mae_test"},
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


def generar_seleccion_final(dirs: dict, final_configs: dict, ruta_salida: str) -> pd.DataFrame:
    """
    Para cada modelo con configuración final definida, busca esa configuración
    exacta en su CSV de resultados de Fase 1 y exporta una tabla consolidada
    (modelo, configuración, RMSE/MAE de Fase 1, posición en el ranking) lista
    para usarse directamente en el Anexo E (búsqueda y selección de configuraciones).
    """
    registros = []

    for nombre, spec in MODELOS.items():
        config = final_configs.get(nombre)
        carpeta = dirs.get(nombre)
        if config is None or carpeta is None:
            print(f"[SKIP] {nombre.upper()}: sin configuración final definida.")
            continue

        ruta = os.path.join(carpeta, spec["archivo"])
        if not os.path.exists(ruta):
            print(f"[SKIP] {nombre.upper()}: no existe {ruta}")
            continue

        df = pd.read_csv(ruta)
        resultado = verificar_configuracion(df, config, spec["rmse_col"], spec["mae_col"])

        if resultado is None:
            print(f"[WARN] {nombre.upper()}: la configuración de final_configs.py no aparece "
                  f"en {ruta}. Revisar manualmente.")
            continue

        if resultado["posicion"] != 1:
            print(f"[INFO] {nombre.upper()}: configuración elegida en posición "
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
    df_seleccion.to_csv(ruta_salida, index=False)
    print(f"[OK] Selección final verificada: {ruta_salida}")
    print(df_seleccion.to_string(index=False))
    return df_seleccion
