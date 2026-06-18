"""
Análisis visual de los resultados de Fase 1 (grid search) de R11.

Para cada modelo (mlp, lstm, cnn) genera en su directorio de salida:
  1. <modelo>_analisis_ventanas.png
       Barras: RMSE de la mejor configuración por ventana.
       La barra de la mejor ventana se marca en naranja.

  2. <modelo>_analisis_top5_w<mejor_ventana>.png
       Tabla visual: Top-5 configuraciones de la mejor ventana.

Los modelos cuyo CSV de Fase 1 no existe se omiten con [SKIP].
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from O3.config import R11_CNN_DIR, R11_LSTM_DIR, R11_MLP_DIR

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Carga
# ─────────────────────────────────────────────────────────────────────────────

def _cargar_resultados_dl(modelo_dir: str, prefijo: str) -> pd.DataFrame:
    ruta = os.path.join(modelo_dir, f"{prefijo}_resultados.csv")
    if not os.path.exists(ruta):
        raise FileNotFoundError(ruta)
    df = pd.read_csv(ruta)
    df = df.rename(columns={"window_size": "window_tag", "rmse_test": "rmse_eval", "mae_test": "mae_eval"})
    df["window_tag"] = df["window_tag"].astype(str)
    df["window_order"] = df["window_tag"].astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Etiquetas compactas
# ─────────────────────────────────────────────────────────────────────────────

def _label_mlp(row: pd.Series) -> str:
    return (
        f"h={row['capas_ocultas']}\n"
        f"act={row['activacion']}  d={row['dropout']}\n"
        f"lr={row['lr']}  bs={int(row['lote'])}"
    )


def _label_lstm(row: pd.Series) -> str:
    return (
        f"hid={int(row['unidades_ocultas'])}  L={int(row['num_capas'])}\n"
        f"d={row['dropout']}\n"
        f"lr={row['lr']}  bs={int(row['lote'])}"
    )


def _label_cnn(row: pd.Series) -> str:
    return (
        f"ch={row['canales_conv']}  k={int(row['kernel_size'])}\n"
        f"act={row['activacion']}  d={row['dropout']}\n"
        f"lr={row['lr']}  bs={int(row['lote'])}"
    )


_LABEL_FN = {"mlp": _label_mlp, "lstm": _label_lstm, "cnn": _label_cnn}


# ─────────────────────────────────────────────────────────────────────────────
# Gráfico 1: mejor RMSE por ventana
# ─────────────────────────────────────────────────────────────────────────────

def _grafico_ventanas(df: pd.DataFrame, modelo_nombre: str, label_fn, ruta_salida: str) -> str:
    df_mejor = (
        df.sort_values("rmse_eval")
        .groupby("window_tag", sort=False)
        .first()
        .reset_index()
        .sort_values("window_order")
        .reset_index(drop=True)
    )
    n = len(df_mejor)
    idx_min = int(df_mejor["rmse_eval"].idxmin())
    colores = ["darkorange" if i == idx_min else "steelblue" for i in range(n)]

    etiquetas = [f"w={row['window_tag']}\n{label_fn(row)}" for _, row in df_mejor.iterrows()]

    fig, ax = plt.subplots(figsize=(max(7, n * 2.2), 7))
    bars = ax.bar(np.arange(n), df_mejor["rmse_eval"], width=0.5,
                  color=colores, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.5f", padding=3, fontsize=8)

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(etiquetas, fontsize=8, ha="center")
    ax.set_xlabel("Ventana y configuración")
    ax.set_ylabel("RMSE (mejor config por ventana, escala original)")
    ax.set_title(f"{modelo_nombre.upper()} R11 — Mejor RMSE por ventana (Fase 1)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    mejor_tag = df_mejor.loc[idx_min, "window_tag"]
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="darkorange", label=f"mejor (w={mejor_tag})"),
        Patch(facecolor="steelblue",  label="otras ventanas"),
    ], loc="upper right", fontsize=8, framealpha=0.85)

    fig.subplots_adjust(bottom=0.38)
    fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[OK] {ruta_salida}")
    return mejor_tag


# ─────────────────────────────────────────────────────────────────────────────
# Gráfico 2: tabla top-5
# ─────────────────────────────────────────────────────────────────────────────

def _tabla_top5(df: pd.DataFrame, mejor_ventana: str, modelo_nombre: str, label_fn, ruta_salida: str) -> None:
    df_ventana = (
        df[df["window_tag"] == mejor_ventana]
        .sort_values("rmse_eval")
        .head(5)
        .reset_index(drop=True)
    )
    n = len(df_ventana)
    filas = [
        [f"#{i+1}", label_fn(row).replace("\n", "  |  "), f"{row['rmse_eval']:.5f}", f"{row['mae_eval']:.5f}"]
        for i, (_, row) in enumerate(df_ventana.iterrows())
    ]

    row_height = 0.45
    fig, ax = plt.subplots(figsize=(14, (n + 1) * row_height + 0.7))
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    tabla = ax.table(
        cellText=filas, colLabels=["#", "Configuración", "RMSE", "MAE"],
        cellLoc="left", loc="center", bbox=[0.01, 0.01, 0.98, 0.82],
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.auto_set_column_width([0, 1, 2, 3])

    for j in range(4):
        tabla[(0, j)].set_facecolor("#2c5f8a")
        tabla[(0, j)].set_text_props(color="white", fontweight="bold")
    for j in range(4):
        tabla[(1, j)].set_facecolor("#ffe0b2")
    for i in range(2, n + 1):
        color = "#f0f4f8" if i % 2 == 0 else "white"
        for j in range(4):
            tabla[(i, j)].set_facecolor(color)

    ax.set_title(
        f"{modelo_nombre.upper()} R11 — Top {n} configuraciones  |  ventana = {mejor_ventana}",
        fontsize=11, fontweight="bold", pad=14,
    )
    fig.savefig(ruta_salida, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    logger.info(f"[OK] {ruta_salida}")


# ─────────────────────────────────────────────────────────────────────────────
# Función pública
# ─────────────────────────────────────────────────────────────────────────────

def analizar_fase1(
    mlp_dir: str = None,
    lstm_dir: str = None,
    cnn_dir: str = None,
) -> dict:
    """
    Genera gráficos de ventanas y top-5 para cada modelo de R11.

    Devuelve dict con la mejor ventana por modelo:
        {"mlp": "4", "lstm": "5", "cnn": "3"}
    """
    dirs = {
        "mlp":  mlp_dir  or R11_MLP_DIR,
        "lstm": lstm_dir or R11_LSTM_DIR,
        "cnn":  cnn_dir  or R11_CNN_DIR,
    }

    resultados = {}
    for nombre, carpeta in dirs.items():
        logger.info(f"Analizando {nombre.upper()} R11...")
        try:
            df = _cargar_resultados_dl(carpeta, nombre)
            label_fn = _LABEL_FN[nombre]

            ruta_v = os.path.join(carpeta, f"{nombre}_analisis_ventanas.png")
            mejor_ventana = _grafico_ventanas(df, nombre, label_fn, ruta_v)

            ruta_t = os.path.join(carpeta, f"{nombre}_analisis_top5_w{mejor_ventana}.png")
            _tabla_top5(df, mejor_ventana, nombre, label_fn, ruta_t)

            resultados[nombre] = mejor_ventana
            logger.info(f"  Mejor ventana: {mejor_ventana}")
        except FileNotFoundError as e:
            logger.warning(f"[SKIP] {nombre.upper()}: CSV no encontrado — {e}")

    return resultados
