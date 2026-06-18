"""
Pipeline LSTM multivariable para R11.

Input shape: (n, window_size, 7) — 7 canales, input_size=7.
Output: pct_bosque escalado (1 valor).
Métricas reportadas en escala original.
"""

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
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Arquitectura
# ─────────────────────────────────────────────────────────────────────────────

class LSTM(nn.Module):
    def __init__(self, input_size: int, unidades_ocultas: int, num_capas: int, dropout: float):
        super().__init__()
        lstm_dropout = dropout if num_capas > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=unidades_ocultas,
            num_layers=num_capas,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(unidades_ocultas, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        salida, _ = self.lstm(x)
        salida = salida[:, -1, :]
        salida = self.dropout(salida)
        return self.fc(salida)


def _preparar_X(X_tensor: torch.Tensor) -> torch.Tensor:
    return X_tensor.float()  # ya tiene shape (n, window_size, 7)


# ─────────────────────────────────────────────────────────────────────────────
# Entrenamiento / evaluación
# ─────────────────────────────────────────────────────────────────────────────

def _entrenar(
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    unidades_ocultas: int,
    num_capas: int,
    dropout: float,
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

    modelo = LSTM(
        input_size=X.shape[2],
        unidades_ocultas=unidades_ocultas,
        num_capas=num_capas,
        dropout=dropout,
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

def pipeline_lstm(
    datasets: dict,
    ruta_base: str,
    escalador,
    epocas_valores: list,
    lr_valores: list,
    lote_valores: list,
    unidades_ocultas_valores: list,
    num_capas_valores: list,
    dropout_valores: list,
) -> dict:
    logger.info("=" * 70)
    logger.info("LSTM R11 — Fase 1: búsqueda exploratoria de hiperparámetros")
    logger.info(f"Device: {DEVICE}")

    if not datasets:
        raise RuntimeError("pipeline_lstm: datasets vacío.")

    combinaciones = list(product(
        datasets.items(),
        unidades_ocultas_valores,
        num_capas_valores,
        dropout_valores,
        epocas_valores,
        lr_valores,
        lote_valores,
    ))
    logger.info(f"Combinaciones totales: {len(combinaciones)}")

    resultados = []
    for idx, ((w, data), unidades, num_capas, dropout, epocas, lr, lote) in enumerate(combinaciones, 1):
        X_train, y_train = data["train"]
        X_test,  y_test  = data["test"]

        modelo, _ = _entrenar(X_train, y_train, unidades, num_capas, dropout, epocas, lote, lr)
        rmse_train, mae_train = _evaluar(modelo, X_train, y_train, escalador)
        rmse_test,  mae_test  = _evaluar(modelo, X_test,  y_test,  escalador)
        diag = diagnosticar_ajuste(rmse_train, mae_train, rmse_test, mae_test)

        nombre = f"LSTM_w{w}_h{unidades}_l{num_capas}_d{dropout}_e{epocas}_lr{lr}_b{lote}"
        logger.info(f"  [{idx}/{len(combinaciones)}] {nombre}  RMSE_test={rmse_test:.4f}  MAE_test={mae_test:.4f}")

        resultados.append({
            "modelo": nombre, "window_size": int(w),
            "unidades_ocultas": int(unidades), "num_capas": int(num_capas),
            "dropout": dropout, "epocas": epocas, "lr": lr, "lote": lote,
            "rmse_train": round(rmse_train, 6), "mae_train": round(mae_train, 6),
            "rmse_test":  round(rmse_test, 6),  "mae_test":  round(mae_test, 6),
            **{k: diag[k] for k in diag},
        })

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

def entrenar_config_final_lstm(
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
    logger.info("LSTM R11 — Fase 2: entrenamiento final")
    logger.info(f"Device: {DEVICE}")

    config_final   = dict(config_final)
    window_size    = int(config_final["window_size"])
    unidades       = int(config_final["unidades_ocultas"])
    num_capas      = int(config_final["num_capas"])
    dropout        = float(config_final["dropout"])
    epocas         = int(config_final["epocas"])
    lr             = float(config_final["lr"])
    lote           = int(config_final["lote"])

    if window_size not in datasets:
        raise ValueError(f"window_size={window_size} no existe en datasets.")

    X_train, y_train = datasets[window_size]["train"]
    X_test,  y_test  = datasets[window_size]["test"]

    nombre = f"LSTM_FINAL_w{window_size}_h{unidades}_l{num_capas}_d{dropout}_e{epocas}_lr{lr}_b{lote}"

    modelo, perdidas = _entrenar(X_train, y_train, unidades, num_capas, dropout, epocas, lote, lr)
    rmse_train, mae_train = _evaluar(modelo, X_train, y_train, escalador)
    rmse_test,  mae_test  = _evaluar(modelo, X_test,  y_test,  escalador)
    diag = diagnosticar_ajuste(rmse_train, mae_train, rmse_test, mae_test)

    fila_config = {
        "modelo": nombre, "window_size": window_size,
        "unidades_ocultas": unidades, "num_capas": num_capas,
        "dropout": dropout, "epocas": epocas, "lr": lr, "lote": lote, "semilla": SEMILLA,
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
                "train_losses": perdidas, "seed": SEMILLA, "model_type": "LSTM",
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

    logger.info(f"[OK] LSTM FINAL — RMSE_wf={rmse_wf:.6f}  MAE_wf={mae_wf:.6f}")

    return {
        "modelo": nombre, "rmse": rmse_wf, "mae": mae_wf,
        "y_pred": y_pred_wf, "config": fila_config,
        "df_predicciones": df_predicciones,
        "df_departamento": df_departamento,
        "df_distrito": df_distrito,
    }
