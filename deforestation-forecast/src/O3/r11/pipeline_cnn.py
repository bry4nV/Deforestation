"""
Pipeline CNN1D multivariable para R11.

Input shape: (n, window_size, 7) → permutado a (n, 7, window_size) para Conv1d.
input_channels = 7 (número de variables).
Output: pct_bosque escalado (1 valor).
Métricas reportadas en escala original.
"""

import ast
import json
import logging
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from O3.config import SEMILLA, TAMANIO_ENTRENAMIENTO
from O3.r11.utils import (
    DEVICE,
    calcular_metricas,
    construir_df_predicciones,
    diagnosticar_ajuste,
    fijar_semilla,
    graficar_curva,
    inversa_pct_bosque,
    obtener_activacion,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Arquitectura
# ─────────────────────────────────────────────────────────────────────────────

class CNN1D(nn.Module):
    def __init__(
        self,
        canales_entrada: int,
        window_size: int,
        canales_conv: list,
        kernel_size: int,
        dropout: float,
        activacion: str,
        tamanio_denso: int,
    ):
        super().__init__()

        if kernel_size > window_size:
            raise ValueError(
                f"kernel_size={kernel_size} no puede ser mayor que window_size={window_size}"
            )

        capas_conv = []
        prev = canales_entrada
        for canales_out in canales_conv:
            capas_conv.append(nn.Conv1d(prev, canales_out, kernel_size=kernel_size, padding="same"))
            capas_conv.append(obtener_activacion(activacion))
            capas_conv.append(nn.Dropout(dropout))
            prev = canales_out

        self.conv = nn.Sequential(*capas_conv)

        conv_output_size = canales_conv[-1] * window_size

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, tamanio_denso),
            obtener_activacion(activacion),
            nn.Dropout(dropout),
            nn.Linear(tamanio_denso, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.fc(x)


def _parsear_canales(canales_conv):
    if isinstance(canales_conv, str):
        return ast.literal_eval(canales_conv)
    return list(canales_conv)


def _preparar_X(X_tensor: torch.Tensor) -> torch.Tensor:
    # (n, window_size, 7) → (n, 7, window_size)
    return X_tensor.permute(0, 2, 1).float()


# ─────────────────────────────────────────────────────────────────────────────
# Entrenamiento / evaluación
# ─────────────────────────────────────────────────────────────────────────────

def _entrenar(
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    canales_conv: list,
    kernel_size: int,
    dropout: float,
    activacion: str,
    tamanio_denso: int,
    epocas: int,
    lote: int,
    lr: float,
    seed: int = SEMILLA,
) -> tuple:
    fijar_semilla(seed)

    X = _preparar_X(X_train_t)
    y = y_train_t.float()

    generador = torch.Generator()
    generador.manual_seed(seed)

    cargador = DataLoader(
        TensorDataset(X, y),
        batch_size=lote,
        shuffle=True,
        generator=generador,
        num_workers=0,
    )

    modelo = CNN1D(
        canales_entrada=X.shape[1],
        window_size=X.shape[2],
        canales_conv=canales_conv,
        kernel_size=kernel_size,
        dropout=dropout,
        activacion=activacion,
        tamanio_denso=tamanio_denso,
    ).to(DEVICE)

    criterio = nn.MSELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    perdidas = []

    for epoca in range(epocas):
        modelo.train()
        perdida_epoca = 0.0
        for X_lote, y_lote in cargador:
            X_lote, y_lote = X_lote.to(DEVICE), y_lote.to(DEVICE)
            optimizador.zero_grad()
            pred = modelo(X_lote)
            perdida = criterio(pred, y_lote)
            perdida.backward()
            optimizador.step()
            perdida_epoca += perdida.item() * X_lote.size(0)
        perdidas.append(perdida_epoca / len(cargador.dataset))

        if (epoca + 1) % 10 == 0:
            logger.info(f"    Época {epoca + 1}/{epocas} | Loss={perdidas[-1]:.6f}")

    return modelo, perdidas


def _evaluar(modelo, X_tensor: torch.Tensor, y_tensor: torch.Tensor, escalador) -> tuple:
    modelo.eval()
    X = _preparar_X(X_tensor).to(DEVICE)
    y_esc = y_tensor.detach().cpu().numpy().reshape(-1)
    with torch.no_grad():
        pred_esc = modelo(X).detach().cpu().numpy().reshape(-1)
    return calcular_metricas(inversa_pct_bosque(y_esc, escalador), inversa_pct_bosque(pred_esc, escalador))


def _evaluar_geografico(
    modelo,
    panel_escalado: np.ndarray,
    df_distritos_info: pd.DataFrame,
    window_size: int,
    tamanio_entrenamiento: int,
    escalador,
    anios_test: list = None,
    modelo_nombre: str = "",
) -> tuple:
    modelo.eval()
    n_distritos, n_anios, _ = panel_escalado.shape
    horizonte = n_anios - tamanio_entrenamiento

    predicciones_originales = []
    registros = []

    for i in range(n_distritos):
        preds_esc = []
        for t in range(horizonte):
            inicio = tamanio_entrenamiento + t - window_size
            fin    = tamanio_entrenamiento + t
            ventana = panel_escalado[i, inicio:fin, :]  # (window_size, 7)

            x_t = torch.tensor(ventana[np.newaxis, ...], dtype=torch.float32)
            x_t = _preparar_X(x_t).to(DEVICE)

            with torch.no_grad():
                preds_esc.append(modelo(x_t).item())

        preds_orig  = inversa_pct_bosque(np.array(preds_esc), escalador)
        y_true_orig = inversa_pct_bosque(panel_escalado[i, tamanio_entrenamiento:, 0], escalador)
        predicciones_originales.append(preds_orig)

        rmse_i, mae_i = calcular_metricas(y_true_orig, preds_orig)
        info = df_distritos_info.iloc[i]
        registros.append({
            "geocode": info["geocode"], "departamento": info["departamento"],
            "distrito": info["distrito"], "rmse": round(rmse_i, 6), "mae": round(mae_i, 6),
        })

    y_pred_total = np.array(predicciones_originales)
    y_true_total = inversa_pct_bosque(panel_escalado[:, tamanio_entrenamiento:, 0], escalador)
    rmse_global, mae_global = calcular_metricas(y_true_total, y_pred_total)

    df_distrito = (
        pd.DataFrame(registros)
        .sort_values(["mae", "rmse"], ascending=False)
        .reset_index(drop=True)
    )

    departamentos = df_distritos_info["departamento"].values
    registros_dep = [
        {"departamento": dep,
         "rmse": round(calcular_metricas(y_true_total[departamentos == dep], y_pred_total[departamentos == dep])[0], 6),
         "mae":  round(calcular_metricas(y_true_total[departamentos == dep], y_pred_total[departamentos == dep])[1], 6)}
        for dep in np.unique(departamentos)
    ]
    df_departamento = (
        pd.DataFrame(registros_dep)
        .sort_values(["mae", "rmse"], ascending=False)
        .reset_index(drop=True)
    )

    df_predicciones = construir_df_predicciones(
        modelo_nombre=modelo_nombre,
        y_true_total=y_true_total,
        y_pred_total=y_pred_total,
        df_distritos_info=df_distritos_info,
        anios_test=anios_test,
    )

    return df_distrito, df_departamento, rmse_global, mae_global, y_pred_total, df_predicciones


# ─────────────────────────────────────────────────────────────────────────────
# Fase 1
# ─────────────────────────────────────────────────────────────────────────────

def pipeline_cnn(
    datasets: dict,
    ruta_base: str,
    escalador,
    epocas_valores: list,
    lr_valores: list,
    lote_valores: list,
    canales_conv_valores: list,
    kernel_valores: list,
    dropout_valores: list,
    activacion_valores: list,
    tamanio_denso_valores: list,
) -> dict:
    logger.info("=" * 70)
    logger.info("CNN R11 — Fase 1: búsqueda exploratoria de hiperparámetros")
    logger.info(f"Device: {DEVICE}")

    if not datasets:
        raise RuntimeError("pipeline_cnn: datasets vacío.")

    combinaciones = list(product(
        datasets.items(),
        canales_conv_valores,
        kernel_valores,
        dropout_valores,
        activacion_valores,
        tamanio_denso_valores,
        epocas_valores,
        lr_valores,
        lote_valores,
    ))
    logger.info(f"Combinaciones totales: {len(combinaciones)}")

    resultados = []
    for idx, ((w, data), canales, kernel, dropout, activacion, denso, epocas, lr, lote) in enumerate(combinaciones, 1):
        if kernel > int(w):
            continue

        X_train, y_train = data["train"]
        X_test,  y_test  = data["test"]
        canales = _parsear_canales(canales)

        modelo, _ = _entrenar(X_train, y_train, canales, kernel, dropout, activacion, denso, epocas, lote, lr)
        rmse_train, mae_train = _evaluar(modelo, X_train, y_train, escalador)
        rmse_test,  mae_test  = _evaluar(modelo, X_test,  y_test,  escalador)
        diag = diagnosticar_ajuste(rmse_train, mae_train, rmse_test, mae_test)

        c_str  = "x".join(map(str, canales))
        nombre = f"CNN_w{w}_c{c_str}_k{kernel}_act{activacion}_d{dropout}_dense{denso}_e{epocas}_lr{lr}_b{lote}"
        logger.info(f"  [{idx}/{len(combinaciones)}] {nombre}  RMSE_test={rmse_test:.4f}  MAE_test={mae_test:.4f}")

        resultados.append({
            "modelo": nombre, "window_size": int(w),
            "canales_conv": str(canales), "kernel_size": int(kernel),
            "activacion": activacion, "dropout": dropout,
            "tamanio_denso": int(denso), "epocas": epocas, "lr": lr, "lote": lote,
            "rmse_train": round(rmse_train, 6), "mae_train": round(mae_train, 6),
            "rmse_test":  round(rmse_test, 6),  "mae_test":  round(mae_test, 6),
            **{k: diag[k] for k in diag},
        })

    if not resultados:
        raise RuntimeError("pipeline_cnn: ninguna combinación válida (kernel > window_size en todos los casos).")

    df = (
        pd.DataFrame(resultados)
        .sort_values(["rmse_test", "mae_test", "gap_rmse"])
        .reset_index(drop=True)
    )
    df.to_csv(ruta_base.replace(".csv", "_resultados.csv"), index=False)
    df.head(5).to_csv(ruta_base.replace(".csv", "_top5_configuraciones.csv"), index=False)
    df.groupby("window_size", sort=True).first().reset_index().to_csv(
        ruta_base.replace(".csv", "_mejores_por_ventana.csv"), index=False
    )

    top1 = df.iloc[0]
    logger.info(f"[OK] Top 1 — {top1['modelo']}  RMSE_test={top1['rmse_test']}  gap_rmse={top1['gap_rmse']}")
    return {"grid_resultados": df, "top5": df.head(5), "mejores_por_ventana": df.groupby("window_size").first().reset_index()}


# ─────────────────────────────────────────────────────────────────────────────
# Fase 2
# ─────────────────────────────────────────────────────────────────────────────

def entrenar_config_final_cnn(
    datasets: dict,
    config_final: dict,
    ruta_base: str,
    panel_escalado: np.ndarray,
    df_distritos_info: pd.DataFrame,
    escalador,
    tamanio_entrenamiento: int = TAMANIO_ENTRENAMIENTO,
    anios: list = None,
) -> dict:
    logger.info("=" * 70)
    logger.info("CNN R11 — Fase 2: entrenamiento final")
    logger.info(f"Device: {DEVICE}")

    config_final   = dict(config_final)
    window_size    = int(config_final["window_size"])
    canales_conv   = _parsear_canales(config_final["canales_conv"])
    kernel_size    = int(config_final["kernel_size"])
    activacion     = config_final["activacion"]
    dropout        = float(config_final["dropout"])
    tamanio_denso  = int(config_final["tamanio_denso"])
    epocas         = int(config_final["epocas"])
    lr             = float(config_final["lr"])
    lote           = int(config_final["lote"])

    if window_size not in datasets:
        raise ValueError(f"window_size={window_size} no existe en datasets.")
    if kernel_size > window_size:
        raise ValueError(f"kernel_size={kernel_size} > window_size={window_size}.")

    X_train, y_train = datasets[window_size]["train"]
    X_test,  y_test  = datasets[window_size]["test"]

    c_str  = "x".join(map(str, canales_conv))
    nombre = f"CNN_FINAL_w{window_size}_c{c_str}_k{kernel_size}_act{activacion}_d{dropout}_dense{tamanio_denso}_e{epocas}_lr{lr}_b{lote}"

    modelo, perdidas = _entrenar(X_train, y_train, canales_conv, kernel_size, dropout, activacion, tamanio_denso, epocas, lote, lr)
    rmse_train, mae_train = _evaluar(modelo, X_train, y_train, escalador)
    rmse_test,  mae_test  = _evaluar(modelo, X_test,  y_test,  escalador)
    diag = diagnosticar_ajuste(rmse_train, mae_train, rmse_test, mae_test)

    fila_config = {
        "modelo": nombre, "window_size": window_size,
        "canales_conv": str(canales_conv), "kernel_size": kernel_size,
        "activacion": activacion, "dropout": dropout, "tamanio_denso": tamanio_denso,
        "epocas": epocas, "lr": lr, "lote": lote, "semilla": SEMILLA,
        "rmse_train": round(rmse_train, 6), "mae_train": round(mae_train, 6),
        "rmse_test":  round(rmse_test, 6),  "mae_test":  round(mae_test, 6),
        **{k: diag[k] for k in diag},
    }

    anios_test = anios[tamanio_entrenamiento:] if anios is not None else None

    df_distrito, df_departamento, rmse_wf, mae_wf, y_pred_wf, df_predicciones = _evaluar_geografico(
        modelo, panel_escalado, df_distritos_info,
        window_size, tamanio_entrenamiento, escalador,
        anios_test=anios_test, modelo_nombre=nombre,
    )

    ruta_model = ruta_base.replace(".csv", "_final_model.pth")
    torch.save({"model_state_dict": modelo.state_dict(), "config": fila_config,
                "train_losses": perdidas, "seed": SEMILLA, "model_type": "CNN",
                "device_entrenamiento": str(DEVICE)}, ruta_model)

    graficar_curva(perdidas, nombre, ruta_base.replace(".csv", "_final_curva.png"))

    with open(ruta_base.replace(".csv", "_final_config.json"), "w", encoding="utf-8") as f:
        json.dump(fila_config, f, indent=4, ensure_ascii=False)

    pd.DataFrame([{
        "modelo": nombre, "rmse": round(rmse_wf, 6), "mae": round(mae_wf, 6),
        "rmse_train": round(rmse_train, 6), "mae_train": round(mae_train, 6),
        "rmse_test_directo": round(rmse_test, 6), "mae_test_directo": round(mae_test, 6),
        **{k: diag[k] for k in diag},
    }]).to_csv(ruta_base.replace(".csv", "_final_global.csv"), index=False)

    df_distrito.to_csv(ruta_base.replace(".csv", "_final_distrito.csv"), index=False)
    df_departamento.to_csv(ruta_base.replace(".csv", "_final_departamento.csv"), index=False)
    df_predicciones.to_csv(ruta_base.replace(".csv", "_final_predicciones.csv"), index=False)
    np.save(ruta_base.replace(".csv", "_final_ypred.npy"), y_pred_wf)

    logger.info(f"[OK] CNN FINAL — RMSE_wf={rmse_wf:.6f}  MAE_wf={mae_wf:.6f}")

    return {
        "modelo": nombre, "rmse": rmse_wf, "mae": mae_wf,
        "y_pred": y_pred_wf, "config": fila_config,
        "df_predicciones": df_predicciones,
        "df_departamento": df_departamento,
        "df_distrito": df_distrito,
    }
