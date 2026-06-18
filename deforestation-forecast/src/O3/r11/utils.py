"""
Utilidades compartidas de R11 — autónomas respecto a O2.

Diferencias clave respecto a O2/r4_r5/utils.py:
  - logging en lugar de print
  - DEVICE definido aquí (no en cada pipeline)
  - inversa_pct_bosque para reescalar predicciones al espacio original
"""

import logging
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

from O3.config import SEMILLA

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fijar_semilla(seed: int = SEMILLA) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return rmse, mae


def diagnosticar_ajuste(
    rmse_train: float,
    mae_train: float,
    rmse_test: float,
    mae_test: float,
) -> dict:
    gap_rmse = rmse_test - rmse_train
    gap_mae = mae_test - mae_train
    ratio_rmse = rmse_test / rmse_train if rmse_train > 0 else float("nan")
    ratio_mae = mae_test / mae_train if mae_train > 0 else float("nan")
    return {
        "gap_rmse":   round(float(gap_rmse), 6),
        "gap_mae":    round(float(gap_mae), 6),
        "ratio_rmse": round(float(ratio_rmse), 6),
        "ratio_mae":  round(float(ratio_mae), 6),
    }


def obtener_activacion(nombre: str) -> nn.Module:
    nombre = nombre.lower()
    if nombre == "relu":
        return nn.ReLU()
    elif nombre == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    elif nombre == "tanh":
        return nn.Tanh()
    elif nombre == "elu":
        return nn.ELU()
    elif nombre == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Activación no soportada: {nombre}")


def inversa_pct_bosque(valores_escalados: np.ndarray, escalador) -> np.ndarray:
    """
    Invierte el escalado de pct_bosque (canal 0) usando mean_ y scale_ del StandardScaler.
    Trabaja directamente sobre los coeficientes del canal 0 sin construir filas artificiales.
    """
    return np.asarray(valores_escalados) * escalador.scale_[0] + escalador.mean_[0]


def construir_df_predicciones(
    modelo_nombre: str,
    y_true_total: np.ndarray,
    y_pred_total: np.ndarray,
    df_distritos_info: pd.DataFrame,
    anios_test: list = None,
) -> pd.DataFrame:
    """
    Construye DataFrame de predicciones por distrito y paso temporal.

    y_true_total, y_pred_total: (n_distritos, horizonte) en escala original.
    """
    registros = []
    for i in range(y_true_total.shape[0]):
        info = df_distritos_info.iloc[i]
        for j in range(y_true_total.shape[1]):
            y_true_val = float(y_true_total[i, j])
            y_pred_val = float(y_pred_total[i, j])
            error = y_pred_val - y_true_val
            registro = {
                "modelo":       modelo_nombre,
                "geocode":      info["geocode"],
                "departamento": info["departamento"],
                "distrito":     info["distrito"],
                "horizonte":    j + 1,
                "y_true":       y_true_val,
                "y_pred":       y_pred_val,
                "error":        error,
                "abs_error":    abs(error),
                "squared_error": error ** 2,
            }
            if anios_test is not None:
                registro["anio"] = anios_test[j]
            registros.append(registro)
    return pd.DataFrame(registros)


def graficar_curva(train_losses: list, nombre: str, ruta_png: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(train_losses) + 1), train_losses, linewidth=1.5, label="Train MSE")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
    ax.set_title(f"Curva de aprendizaje — {nombre}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=120)
    plt.close(fig)
    logger.info(f"[OK] Curva guardada: {ruta_png}")
