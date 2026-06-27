"""
R7: pronóstico del año siguiente (2025) para los tres modelos candidatos de R6
(MLP, LSTM, CNN1D).

No reentrena ningún modelo -- carga el mismo archivo de modelo final ya
entrenado y validado en R6 (mismos pesos que generaron la Tabla 25 / Tabla 27),
para preservar la trazabilidad entre la tasa de error reportada y el modelo
que efectivamente produce el pronóstico.

El ancla de cada predicción es siempre pct_bosque_real_2024: el último valor
observado en el dataset, nunca una predicción del propio modelo. Esto replica
para el pronóstico final la misma condición de "ancla real" ya verificada
para la evaluación walk-forward de R6 (ver verificacion_deforestacion.py).
No hay tasa de error asociada a este pronóstico porque 2025 no tiene valor
observado contra el cual calcularla.
"""

import os

import numpy as np
import pandas as pd
import torch

from O2.r4_r5.pipeline_cnn import CNN1D, parse_conv_channels, preparar_X_cnn
from O2.r4_r5.pipeline_lstm import LSTM, preparar_X_lstm
from O2.r4_r5.pipeline_mlp import MLP, parse_hidden_sizes, preparar_X_mlp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cargar_mlp(checkpoint):
    config = checkpoint["config"]
    modelo = MLP(
        input_size=int(config["window_size"]),
        hidden_sizes=parse_hidden_sizes(config["hidden_sizes"]),
        dropout=float(config["dropout"]),
        activation=config["activation"],
    )
    modelo.load_state_dict(checkpoint["model_state_dict"])
    return modelo.to(DEVICE).eval(), int(config["window_size"]), preparar_X_mlp


def cargar_lstm(checkpoint):
    config = checkpoint["config"]
    modelo = LSTM(
        input_size=1,
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
    )
    modelo.load_state_dict(checkpoint["model_state_dict"])
    return modelo.to(DEVICE).eval(), int(config["window_size"]), preparar_X_lstm


def cargar_cnn(checkpoint):
    config = checkpoint["config"]
    modelo = CNN1D(
        input_channels=1,
        window_size=int(config["window_size"]),
        conv_channels=parse_conv_channels(config["conv_channels"]),
        kernel_size=int(config["kernel_size"]),
        dropout=float(config["dropout"]),
        activation=config["activation"],
        dense_size=int(config["dense_size"]),
    )
    modelo.load_state_dict(checkpoint["model_state_dict"])
    return modelo.to(DEVICE).eval(), int(config["window_size"]), preparar_X_cnn


CARGADORES = {
    "MLP": cargar_mlp,
    "LSTM": cargar_lstm,
    "CNN1D": cargar_cnn,
}


def generar_pronostico_2025(series, df_distritos_info, rutas_modelo, anio_anchor=2024):
    """
    series: ndarray (n_distritos, n_anios) -- histórico real completo, 1985-2024.
    rutas_modelo: dict {"MLP": ruta_pth, "LSTM": ruta_pth, "CNN1D": ruta_pth}

    Devuelve un DataFrame con geocode, departamento, distrito,
    pct_bosque_real_2024 (ancla explícita, trazable) y <modelo>_pred_2025
    para cada arquitectura.
    """
    # Ancla real explícita: último valor observado del dataset (2024).
    # Nunca se sustituye por una predicción propia del modelo.
    pct_bosque_real_2024 = series[:, -1].copy()

    df_pronostico = pd.DataFrame({
        "geocode": df_distritos_info["geocode"].values,
        "departamento": df_distritos_info["departamento"].values,
        "distrito": df_distritos_info["distrito"].values,
        "pct_bosque_real_2024": pct_bosque_real_2024,
    })

    for nombre, ruta in rutas_modelo.items():
        print(f"[INFO] Pronóstico 2025 -- {nombre}: cargando {ruta}")
        checkpoint = torch.load(ruta, map_location=DEVICE)
        modelo, window_size, preparar_fn = CARGADORES[nombre](checkpoint)

        # Ventana de entrada: los últimos window_size valores REALES de la
        # serie (su último elemento es, precisamente, pct_bosque_real_2024).
        ventana = series[:, -window_size:]
        x = torch.tensor(ventana, dtype=torch.float32)
        x = preparar_fn(x).to(DEVICE)

        with torch.no_grad():
            preds = modelo(x).cpu().numpy().reshape(-1)

        df_pronostico[f"{nombre.lower()}_pred_{anio_anchor + 1}"] = preds
        print(f"[OK] {nombre}: pronóstico generado para {len(preds)} distritos (window_size={window_size})")

    return df_pronostico
