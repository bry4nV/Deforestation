"""
Análisis visual de los resultados de Fase 1 (grid search).

Para cada modelo genera dos gráficos en su propio directorio de salida:
  1. <modelo>_analisis_ventanas.png
       Barras: RMSE de la mejor configuración por ventana temporal.
       La barra de la mejor ventana se marca en naranja.

  2. <modelo>_analisis_top5_w<mejor_ventana>.png
       Barras: Top-5 configuraciones de la mejor ventana, con
       los hiperparámetros clave en la etiqueta de cada barra.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from O2.config import ARIMA_DIR, CNN_DIR, LSTM_DIR, MLP_DIR, TCN_DIR


# ─────────────────────────────────────────────────────────────
# Carga y normalización de columnas
# ─────────────────────────────────────────────────────────────

def _orden_ventana(tag: str) -> int:
    """Convierte el tag de ventana a entero para ordenar. 'full' va al final."""
    return 9999 if tag == "full" else int(tag)


def _cargar_arima(arima_dir: str) -> pd.DataFrame:
    ruta = os.path.join(arima_dir, "arima_resultados.csv")
    if not os.path.exists(ruta):
        raise FileNotFoundError(ruta)
    df = pd.read_csv(ruta)
    df = df.rename(columns={"window": "window_tag", "rmse": "rmse_eval", "mae": "mae_eval"})
    df["window_tag"] = df["window_tag"].astype(str)
    df["window_order"] = df["window_tag"].apply(_orden_ventana)
    return df


def _cargar_dl(modelo_dir: str, prefijo: str) -> pd.DataFrame:
    ruta = os.path.join(modelo_dir, f"{prefijo}_resultados.csv")
    if not os.path.exists(ruta):
        raise FileNotFoundError(ruta)
    df = pd.read_csv(ruta)
    df = df.rename(columns={"window_size": "window_tag", "rmse_test": "rmse_eval", "mae_test": "mae_eval"})
    df["window_tag"] = df["window_tag"].astype(str)
    df["window_order"] = df["window_tag"].astype(int)
    return df


# ─────────────────────────────────────────────────────────────
# Etiquetas compactas para el gráfico top-5
# ─────────────────────────────────────────────────────────────

def _label_arima(row: pd.Series) -> str:
    return f"ARIMA({int(row['p'])},{int(row['d'])},{int(row['q'])})"


def _label_mlp(row: pd.Series) -> str:
    return (
        f"h={row['hidden_sizes']}\n"
        f"act={row['activation']}  d={row['dropout']}\n"
        f"lr={row['lr']}  bs={int(row['batch_size'])}"
    )


def _label_lstm(row: pd.Series) -> str:
    return (
        f"hid={int(row['hidden_size'])}  L={int(row['num_layers'])}\n"
        f"d={row['dropout']}\n"
        f"lr={row['lr']}  bs={int(row['batch_size'])}"
    )


def _label_cnn(row: pd.Series) -> str:
    return (
        f"ch={row['conv_channels']}  k={int(row['kernel_size'])}\n"
        f"act={row['activation']}  d={row['dropout']}\n"
        f"lr={row['lr']}  bs={int(row['batch_size'])}"
    )


def _label_tcn(row: pd.Series) -> str:
    return (
        f"ch={row['num_channels']}  k={int(row['kernel_size'])}\n"
        f"act={row['activation']}  d={row['dropout']}\n"
        f"lr={row['lr']}  bs={int(row['batch_size'])}"
    )


_LABEL_FN = {
    "arima": _label_arima,
    "mlp":   _label_mlp,
    "lstm":  _label_lstm,
    "cnn":   _label_cnn,
    "tcn":   _label_tcn,
}


# ─────────────────────────────────────────────────────────────
# Gráfico 1: mejor RMSE por ventana
# ─────────────────────────────────────────────────────────────

def _grafico_ventanas(
    df: pd.DataFrame,
    modelo_nombre: str,
    label_fn,
    ruta_salida: str,
) -> str:
    """
    Genera el gráfico de barras con el mejor RMSE por ventana.
    Cada barra muestra en el eje X: ventana + misma configuración que el top-5.
    Devuelve el window_tag de la mejor ventana (barra naranja).
    """
    df_mejor = (
        df.sort_values("rmse_eval")
        .groupby("window_tag", sort=False)
        .first()
        .reset_index()
        .sort_values("window_order")
        .reset_index(drop=True)
    )
    n = len(df_mejor)

    color_default = "steelblue"
    color_mejor   = "darkorange"
    idx_min = int(df_mejor["rmse_eval"].idxmin())
    colores = [color_mejor if i == idx_min else color_default for i in range(n)]

    # Misma función de etiqueta que el top-5, con el prefijo de ventana
    etiquetas = [
        f"w={row['window_tag']}\n{label_fn(row)}"
        for _, row in df_mejor.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(max(7, n * 2.2), 7))
    bars = ax.bar(
        np.arange(n),
        df_mejor["rmse_eval"],
        width=0.5,
        color=colores,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar_label(bars, fmt="%.5f", padding=3, fontsize=8)

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(etiquetas, fontsize=8, ha="center")
    ax.set_xlabel("Ventana y configuración")
    ax.set_ylabel("RMSE  (mejor config por ventana)")
    ax.set_title(f"{modelo_nombre.upper()} — Mejor RMSE por ventana (Fase 1)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0.011, 0.012)

    mejor_tag = df_mejor.loc[idx_min, "window_tag"]

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor=color_mejor, label=f"mejor  (w={mejor_tag})"),
            Patch(facecolor=color_default, label="otras ventanas"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        bbox_transform=ax.transAxes,
        fontsize=8,
        framealpha=0.85,
    )

    fig.subplots_adjust(bottom=0.38)
    fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {ruta_salida}")

    return mejor_tag


# ─────────────────────────────────────────────────────────────
# Tabla 2: ranking top-5 configs de la mejor ventana
# ─────────────────────────────────────────────────────────────

def _tabla_top5(
    df: pd.DataFrame,
    mejor_ventana: str,
    modelo_nombre: str,
    label_fn,
    ruta_salida: str,
) -> None:
    """
    Genera una tabla con el ranking de las top-5 configuraciones de la mejor ventana.
    Columnas: #, Configuración, RMSE, MAE.
    """
    df_ventana = (
        df[df["window_tag"] == mejor_ventana]
        .sort_values("rmse_eval")
        .head(5)
        .reset_index(drop=True)
    )
    n = len(df_ventana)

    col_labels = ["#", "Configuración", "RMSE", "MAE"]
    filas = []
    for i, (_, row) in enumerate(df_ventana.iterrows()):
        config = label_fn(row).replace("\n", "  |  ")
        filas.append([
            f"#{i + 1}",
            config,
            f"{row['rmse_eval']:.5f}",
            f"{row['mae_eval']:.5f}",
        ])

    row_height = 0.45
    fig_h = (n + 1) * row_height + 0.7   # filas + encabezado + título
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    tabla = ax.table(
        cellText=filas,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        bbox=[0.01, 0.01, 0.98, 0.82],   # [x0, y0, width, height] en coordenadas de ejes
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.auto_set_column_width(col=[0, 1, 2, 3])

    # Encabezado
    for j in range(len(col_labels)):
        cell = tabla[(0, j)]
        cell.set_facecolor("#2c5f8a")
        cell.set_text_props(color="white", fontweight="bold")

    # Fila 1 (mejor) en naranja claro
    for j in range(len(col_labels)):
        tabla[(1, j)].set_facecolor("#ffe0b2")

    # Filas restantes alternadas
    for i in range(2, n + 1):
        color = "#f0f4f8" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tabla[(i, j)].set_facecolor(color)

    ax.set_title(
        f"{modelo_nombre.upper()} — Top {n} configuraciones  |  ventana = {mejor_ventana}",
        fontsize=11,
        fontweight="bold",
        pad=14,
    )

    fig.savefig(ruta_salida, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"[OK] {ruta_salida}")


# ─────────────────────────────────────────────────────────────
# Orquestador por modelo
# ─────────────────────────────────────────────────────────────

def _analizar_modelo(
    df: pd.DataFrame,
    nombre: str,
    output_dir: str,
) -> str:
    label_fn = _LABEL_FN[nombre]

    ruta_v = os.path.join(output_dir, f"{nombre}_analisis_ventanas.png")
    mejor_ventana = _grafico_ventanas(df, nombre, label_fn, ruta_v)

    ruta_t = os.path.join(output_dir, f"{nombre}_analisis_top5_w{mejor_ventana}.png")
    _tabla_top5(df, mejor_ventana, nombre, label_fn, ruta_t)

    return mejor_ventana


# ─────────────────────────────────────────────────────────────
# Función pública
# ─────────────────────────────────────────────────────────────

def analizar_fase1(
    arima_dir: str = None,
    mlp_dir: str = None,
    lstm_dir: str = None,
    cnn_dir: str = None,
    tcn_dir: str = None,
) -> dict:
    """
    Carga los resultados de Fase 1 de cada modelo y genera los dos gráficos
    de análisis de ventanas e hiperparámetros.

    Devuelve un dict con la mejor ventana por modelo:
      {"arima": "full", "mlp": "5", "lstm": "5", "cnn": "4", "tcn": "5"}

    Los modelos cuyo CSV de Fase 1 no existe se omiten con [SKIP].
    """
    dirs = {
        "arima": arima_dir or ARIMA_DIR,
        "mlp":   mlp_dir   or MLP_DIR,
        "lstm":  lstm_dir  or LSTM_DIR,
        "cnn":   cnn_dir   or CNN_DIR,
        "tcn":   tcn_dir   or TCN_DIR,
    }

    resultados = {}

    for nombre, carpeta in dirs.items():
        print(f"\n[INFO] Analizando {nombre.upper()}...")
        try:
            df = (
                _cargar_arima(carpeta)
                if nombre == "arima"
                else _cargar_dl(carpeta, nombre)   # mlp, lstm, cnn, tcn
            )
            mejor = _analizar_modelo(df, nombre, carpeta)
            resultados[nombre] = mejor
            print(f"     Mejor ventana: {mejor}")

        except FileNotFoundError as e:
            print(f"[SKIP] {nombre.upper()}: CSV no encontrado — {e}")

    return resultados
