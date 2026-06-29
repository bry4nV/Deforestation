"""
Pronóstico 2025 para los tres modelos multivariables de R11 (MLP, LSTM, CNN).

No reentrena: reutiliza los pesos ya validados en Fase 2, para que la tasa de
error reportada siga correspondiendo al modelo que genera el pronóstico.
El ancla de cada predicción es siempre pct_bosque_real_2024 en escala
ORIGINAL (nunca una predicción propia), igual que en la evaluación
walk-forward de Fase 2 (`_evaluar_geografico`). No hay tasa de error para
2025 porque no existe valor observado de ese año.

Análogo multivariable de O2/r4_r5/pronostico_r7.py — ver ese módulo para el
mismo diseño aplicado al caso univariable.
"""

import ast
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from O1.config import PIXEL_AREA_KM2

from O3.config import COLUMNAS_PREDICTORAS, NOMBRES_DEPARTAMENTO_DISPLAY, PANEL_ENTRENAMIENTO_CSV
from O3.r11.pipeline_cnn import CNN1D
from O3.r11.pipeline_lstm import LSTM
from O3.r11.pipeline_mlp import MLP
from O3.r11.utils import DEVICE, inversa_pct_bosque

logger = logging.getLogger(__name__)

MODELOS_CANDIDATOS = ("mlp", "lstm", "cnn")
N_CANALES = len(COLUMNAS_PREDICTORAS)


def _parsear_lista(valor):
    return ast.literal_eval(valor) if isinstance(valor, str) else list(valor)


# ─────────────────────────────────────────────────────────────────────────────
# Carga de modelos finales
# ─────────────────────────────────────────────────────────────────────────────

def _cargar_mlp(checkpoint):
    config = checkpoint["config"]
    window_size = int(config["window_size"])
    modelo = MLP(
        input_size=window_size * N_CANALES,
        capas_ocultas=_parsear_lista(config["capas_ocultas"]),
        dropout=float(config["dropout"]),
        activacion=config["activacion"],
    )
    modelo.load_state_dict(checkpoint["model_state_dict"])
    preparar_fn = lambda x: x.reshape(x.shape[0], -1).float()
    return modelo.to(DEVICE).eval(), window_size, preparar_fn


def _cargar_lstm(checkpoint):
    config = checkpoint["config"]
    modelo = LSTM(
        input_size=N_CANALES,
        unidades_ocultas=int(config["unidades_ocultas"]),
        num_capas=int(config["num_capas"]),
        dropout=float(config["dropout"]),
    )
    modelo.load_state_dict(checkpoint["model_state_dict"])
    preparar_fn = lambda x: x.float()
    return modelo.to(DEVICE).eval(), int(config["window_size"]), preparar_fn


def _cargar_cnn(checkpoint):
    config = checkpoint["config"]
    modelo = CNN1D(
        canales_entrada=N_CANALES,
        window_size=int(config["window_size"]),
        canales_conv=_parsear_lista(config["canales_conv"]),
        kernel_size=int(config["kernel_size"]),
        dropout=float(config["dropout"]),
        activacion=config["activacion"],
        tamanio_denso=int(config["tamanio_denso"]),
    )
    modelo.load_state_dict(checkpoint["model_state_dict"])
    preparar_fn = lambda x: x.permute(0, 2, 1).float()
    return modelo.to(DEVICE).eval(), int(config["window_size"]), preparar_fn


CARGADORES = {
    "mlp": _cargar_mlp,
    "lstm": _cargar_lstm,
    "cnn": _cargar_cnn,
}


# ─────────────────────────────────────────────────────────────────────────────
# Pronóstico 2025
# ─────────────────────────────────────────────────────────────────────────────

def generar_pronostico_2025(
    panel_escalado: np.ndarray,
    panel_original: np.ndarray,
    df_distritos_info: pd.DataFrame,
    rutas_modelo: dict,
    escalador,
    anio_anchor: int = 2024,
) -> pd.DataFrame:
    """
    panel_escalado: (n_distritos, n_anios, n_canales) — transformado (log1p) y escalado.
    panel_original: (n_distritos, n_anios, n_canales) — escala original, canal 0 = pct_bosque.
    rutas_modelo: dict {"mlp": ruta_pth, "lstm": ruta_pth, "cnn": ruta_pth}.

    Devuelve un DataFrame con geocode, departamento, distrito,
    pct_bosque_real_2024 (ancla explícita, trazable) y <modelo>_pred_2025
    para cada arquitectura.
    """
    pct_bosque_real_2024 = panel_original[:, -1, 0].copy()

    df_pronostico = pd.DataFrame({
        "geocode": df_distritos_info["geocode"].values,
        "departamento": df_distritos_info["departamento"].values,
        "distrito": df_distritos_info["distrito"].values,
        "pct_bosque_real_2024": pct_bosque_real_2024,
    })

    for nombre, ruta in rutas_modelo.items():
        logger.info(f"Pronóstico 2025 — {nombre.upper()}: cargando {ruta}")
        checkpoint = torch.load(ruta, map_location=DEVICE)
        modelo, window_size, preparar_fn = CARGADORES[nombre](checkpoint)

        ventana = panel_escalado[:, -window_size:, :]  # (n_distritos, window_size, n_canales)
        x = torch.tensor(ventana, dtype=torch.float32)
        x = preparar_fn(x).to(DEVICE)

        with torch.no_grad():
            preds_escaladas = modelo(x).cpu().numpy().reshape(-1)

        preds = inversa_pct_bosque(preds_escaladas, escalador)
        df_pronostico[f"{nombre}_pred_{anio_anchor + 1}"] = preds
        logger.info(f"[OK] {nombre.upper()}: pronóstico generado para {len(preds)} distritos (window_size={window_size})")

    return df_pronostico


# ─────────────────────────────────────────────────────────────────────────────
# Deforestación en km² (uso manual/anexo — no se invoca desde main.py)
# ─────────────────────────────────────────────────────────────────────────────

def calcular_deforestacion_km2(
    ruta_deforestacion_2025: str,
    ruta_panel_origen: str = PANEL_ENTRENAMIENTO_CSV,
    pixel_area_km2: float = PIXEL_AREA_KM2,
) -> pd.DataFrame:
    """
    Agrega a deforestacion_2025.csv el área de cada distrito en km² y la
    deforestación estimada 2025 en km² por modelo (sobrescribe si ya existen).

    Espera que deforestacion_2025.csv ya tenga las columnas
    deforestacion_2025_{mlp,lstm,cnn} (fracción) -- igual que en
    O2/r4_r5/pronostico_r7.py, esa tabla de fracciones es un insumo externo
    al pipeline, no generado por ningún script de este repositorio.

    Usa pix_total de 2024 del panel de O3 (misma fuente que ya usó O1 para
    calcular pct_bosque), no el shapefile.
    """
    df = pd.read_csv(ruta_deforestacion_2025, dtype={"geocode": str})

    panel_origen = pd.read_csv(ruta_panel_origen, dtype={"geocode": str})
    pix_2024 = (
        panel_origen[panel_origen["anio"] == 2024][["geocode", "pix_total"]]
        .rename(columns={"pix_total": "pix_total_2024"})
    )

    df = df.drop(columns=[c for c in df.columns if c.endswith("_km2")], errors="ignore")
    df = df.merge(pix_2024, on="geocode", how="left")

    if df["pix_total_2024"].isna().any():
        faltantes = df.loc[df["pix_total_2024"].isna(), "geocode"].tolist()
        raise ValueError(f"No se encontró pix_total 2024 para: {faltantes}")

    df["area_km2_2024"] = df["pix_total_2024"] * pixel_area_km2
    for modelo in MODELOS_CANDIDATOS:
        df[f"deforestacion_2025_{modelo}_km2"] = (
            df["area_km2_2024"] * df[f"deforestacion_2025_{modelo}"]
        )

    df = df.drop(columns=["pix_total_2024"])
    df.to_csv(ruta_deforestacion_2025, index=False)
    logger.info(f"[OK] {ruta_deforestacion_2025} actualizado con columnas de área y deforestación en km²")

    return df


def graficar_deforestacion_departamento_km2(
    df_deforestacion_km2: pd.DataFrame,
    comparacion_dir: str,
    departamento_resaltado: str = "Cajamarca",
    departamentos_destacados: tuple = ("San Martin", "Huanuco", "Ucayali"),
) -> str:
    """
    Barras horizontales agrupadas: deforestación estimada 2025 en km² por
    departamento y modelo R11, ordenadas de mayor a menor por el promedio de
    los 3. Idéntico en diseño a O2/r4_r5/pronostico_r7.py.
    """
    cols_km2 = [f"deforestacion_2025_{m}_km2" for m in MODELOS_CANDIDATOS]
    agregado = df_deforestacion_km2.groupby("departamento")[cols_km2].sum()
    agregado["promedio"] = agregado[cols_km2].mean(axis=1)
    agregado = agregado.sort_values("promedio", ascending=True)

    color_eje = "#444444"
    color_acento = "#F4A261"
    colores_modelo = {"mlp": "#264653", "lstm": "#2A9D8F", "cnn": "#8AB4C4"}

    y = np.arange(len(agregado))
    alto_barra = 0.25

    fig, ax = plt.subplots(figsize=(8, 6.5))

    if departamento_resaltado in agregado.index:
        y_resaltado = list(agregado.index).index(departamento_resaltado)
        ax.axhspan(y_resaltado - 0.4, y_resaltado + 0.4, color=color_acento, alpha=0.25, zorder=0)

    for j, modelo in enumerate(MODELOS_CANDIDATOS):
        ax.barh(
            y + (j - 1) * alto_barra, agregado[f"deforestacion_2025_{modelo}_km2"],
            height=alto_barra, color=colores_modelo[modelo], label=modelo.upper(), zorder=2,
        )

    etiquetas_departamento = [
        NOMBRES_DEPARTAMENTO_DISPLAY.get(nombre, nombre) for nombre in agregado.index
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(etiquetas_departamento, fontsize=11, fontweight="normal")
    for tick_label, nombre_interno in zip(ax.get_yticklabels(), agregado.index):
        if nombre_interno in departamentos_destacados:
            tick_label.set_fontweight("medium")
            tick_label.set_color("#1A1A1A")
        else:
            tick_label.set_color(color_eje)

    ax.set_xlabel("Deforestación estimada (km²)", color=color_eje)
    ax.tick_params(axis="x", colors=color_eje)
    ax.tick_params(axis="y", length=0)
    ax.spines["bottom"].set_color(color_eje)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.grid(axis="x", linestyle="--", linewidth=0.8, color="#E0E0E0", zorder=1)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="lower right", fontsize=8.5, frameon=False)

    fig.tight_layout()
    ruta_fig = os.path.join(comparacion_dir, "deforestacion_2025_departamento_km2.png")
    fig.savefig(ruta_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[OK] {ruta_fig}")

    return ruta_fig


def graficar_correlacion_rmse_divergencia_2025(
    rutas_distrito_dl: dict,
    ruta_deforestacion_2025: str,
    comparacion_dir: str,
    departamento_resaltado: str = "Cajamarca",
) -> str:
    """
    Dispersión por distrito: RMSE promedio de validación (MLP/LSTM/CNN) vs.
    divergencia entre modelos en la deforestación 2025 (desviación estándar
    de deforestacion_2025_mlp/lstm/cnn, en fracción). Idéntico en diseño a
    O2/r4_r5/pronostico_r7.py.
    """
    rmse_por_modelo = {}
    for nombre, ruta in rutas_distrito_dl.items():
        df = pd.read_csv(ruta, dtype={"geocode": str})
        df["geocode"] = df["geocode"].str.zfill(6)
        rmse_por_modelo[nombre] = df.set_index("geocode")["rmse"]

    rmse_promedio = pd.DataFrame(rmse_por_modelo).mean(axis=1).rename("rmse_promedio")

    defo = pd.read_csv(ruta_deforestacion_2025, dtype={"geocode": str})
    defo["geocode"] = defo["geocode"].str.zfill(6)
    cols_frac = [f"deforestacion_2025_{m}" for m in MODELOS_CANDIDATOS]
    defo = defo.set_index("geocode")
    divergencia = defo[cols_frac].std(axis=1).rename("divergencia_2025")

    datos = pd.concat([rmse_promedio, divergencia, defo["departamento"]], axis=1).dropna()

    x = datos["rmse_promedio"].values
    y = datos["divergencia_2025"].values
    es_resaltado = (datos["departamento"] == departamento_resaltado).values

    color_resto, color_acento, color_eje = "#A9B7C0", "#F4A261", "#444444"

    fig, ax = plt.subplots(figsize=(7.5, 6))

    pendiente, intercepto = np.polyfit(x, y, 1)
    x_linea = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_linea, pendiente * x_linea + intercepto, color="#AAAAAA", linewidth=1.3, zorder=1)

    ax.scatter(x[~es_resaltado], y[~es_resaltado], color=color_resto, alpha=0.65, s=40,
               edgecolor="white", linewidth=0.3, label="Resto del territorio", zorder=2)
    ax.scatter(x[es_resaltado], y[es_resaltado], color=color_acento, alpha=0.85, s=46,
               edgecolor="white", linewidth=0.4, label=departamento_resaltado, zorder=3)

    ax.set_xlabel("RMSE promedio de validación (fracción)", color=color_eje)
    ax.set_ylabel("Desviación estándar entre modelos en el\npronóstico de deforestación 2025 (fracción)", color=color_eje)
    ax.tick_params(colors=color_eje)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(color_eje)
    ax.spines["left"].set_color(color_eje)
    ax.grid(linestyle="--", linewidth=0.7, color="#E0E0E0", zorder=0)
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    fig.tight_layout()
    ruta_fig = os.path.join(comparacion_dir, "dispersion_rmse_divergencia_2025.png")
    fig.savefig(ruta_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[OK] {ruta_fig}")

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    logger.info(f"n={len(datos)} distritos | pendiente={pendiente:.4f}")
    logger.info(f"Pearson  r={pearson_r:.4f}  p={pearson_p:.4g}")
    logger.info(f"Spearman r={spearman_r:.4f}  p={spearman_p:.4g}")
    logger.info("\n" + str(datos.assign(resaltado=es_resaltado).groupby("resaltado")[["rmse_promedio", "divergencia_2025"]].mean()))

    return ruta_fig
