import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

from O2.config import SEMILLA


def fijar_semilla(seed=SEMILLA):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calcular_metricas(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    return rmse, mae


def diagnosticar_ajuste(rmse_train, mae_train, rmse_test, mae_test):
    gap_rmse = rmse_test - rmse_train
    gap_mae = mae_test - mae_train

    ratio_rmse = rmse_test / rmse_train if rmse_train > 0 else np.nan
    ratio_mae = mae_test / mae_train if mae_train > 0 else np.nan

    return {
        "gap_rmse": round(float(gap_rmse), 6),
        "gap_mae": round(float(gap_mae), 6),
        "ratio_rmse": round(float(ratio_rmse), 6),
        "ratio_mae": round(float(ratio_mae), 6),
    }


def obtener_activacion(nombre):
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
        raise ValueError(f"Función de activación no soportada: {nombre}")


def construir_df_predicciones(
    modelo_nombre,
    y_true_total,
    y_pred_total,
    df_distritos_info,
    anios_test=None,
):
    registros = []

    for i in range(y_true_total.shape[0]):
        info = df_distritos_info.iloc[i]

        for j in range(y_true_total.shape[1]):
            y_true = float(y_true_total[i, j])
            y_pred = float(y_pred_total[i, j])
            error = y_pred - y_true

            registro = {
                "modelo": modelo_nombre,
                "geocode": info["geocode"],
                "departamento": info["departamento"],
                "distrito": info["distrito"],
                "horizonte": j + 1,
                "y_true": y_true,
                "y_pred": y_pred,
                "error": error,
                "abs_error": abs(error),
                "squared_error": error ** 2,
            }

            if anios_test is not None:
                registro["anio"] = anios_test[j]

            registros.append(registro)

    return pd.DataFrame(registros)


def graficar_curva(train_losses, nombre, ruta_png):
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        range(1, len(train_losses) + 1),
        train_losses,
        label="Train MSE",
        linewidth=1.5,
    )

    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
    ax.set_title(f"Curva de aprendizaje - {nombre}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(ruta_png, dpi=120)
    plt.close(fig)
