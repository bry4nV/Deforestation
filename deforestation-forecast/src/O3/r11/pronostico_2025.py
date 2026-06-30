"""
R11: pronóstico 2025 para MLP, LSTM y CNN multivariables sin reentrenamiento.
Ancla = pct_bosque_real_2024 en escala original. Análogo multivariable de
O2/r4_r5/pronostico_r7.py.
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

from O3.config import COLUMNAS_PREDICTORAS, NOMBRES_DEPARTAMENTO_DISPLAY
from O3.r11.pipeline_cnn import CNN1D
from O3.r11.pipeline_lstm import LSTM
from O3.r11.pipeline_mlp import MLP
from O3.r11.utils import DEVICE, inversa_pct_bosque

logger = logging.getLogger(__name__)

MODELOS_CANDIDATOS = ("mlp", "lstm", "cnn")
N_CANALES = len(COLUMNAS_PREDICTORAS)


def _parsear_lista(valor):
    return ast.literal_eval(valor) if isinstance(valor, str) else list(valor)


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


def generar_predicciones(
    panel_escalado: np.ndarray,
    panel_original: np.ndarray,
    df_distritos_info: pd.DataFrame,
    rutas_modelo: dict,
    escalador,
    anio_anchor: int = 2024,
) -> pd.DataFrame:
    """
    Genera predicciones 2025 para cada modelo en rutas_modelo.
    Devuelve DataFrame con geocode, departamento, distrito, pct_bosque_real_2024
    y {modelo}_pred_{anio_anchor+1} por arquitectura.
    """
    pct_bosque_real_2024 = panel_original[:, -1, 0].copy()

    df = pd.DataFrame({
        "geocode": df_distritos_info["geocode"].values,
        "departamento": df_distritos_info["departamento"].values,
        "distrito": df_distritos_info["distrito"].values,
        "pct_bosque_real_2024": pct_bosque_real_2024,
    })

    for nombre, ruta in rutas_modelo.items():
        logger.info(f"{nombre.upper()}: cargando {ruta}")
        checkpoint = torch.load(ruta, map_location=DEVICE)
        modelo, window_size, preparar_fn = CARGADORES[nombre](checkpoint)
        ventana = panel_escalado[:, -window_size:, :]
        x = preparar_fn(torch.tensor(ventana, dtype=torch.float32)).to(DEVICE)
        with torch.no_grad():
            preds_escaladas = modelo(x).cpu().numpy().reshape(-1)
        preds = inversa_pct_bosque(preds_escaladas, escalador)
        df[f"{nombre}_pred_{anio_anchor + 1}"] = preds
        logger.info(f"[OK] {nombre.upper()}: {len(preds)} distritos (window_size={window_size})")

    return df


def _graficar_deforestacion_departamento_km2(
    df, comparacion_dir, departamento_resaltado="Cajamarca",
    departamentos_destacados=("San Martin", "Huanuco", "Ucayali"),
):
    cols_km2 = [f"deforestacion_2025_{m}_km2" for m in MODELOS_CANDIDATOS]
    agregado = df.groupby("departamento")[cols_km2].sum()
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


def _graficar_correlacion_rmse_divergencia_2025(
    rutas_distrito_dl, df, comparacion_dir, departamento_resaltado="Cajamarca",
):
    rmse_por_modelo = {}
    for nombre, ruta in rutas_distrito_dl.items():
        dfl = pd.read_csv(ruta, dtype={"geocode": str})
        dfl["geocode"] = dfl["geocode"].str.zfill(6)
        rmse_por_modelo[nombre] = dfl.set_index("geocode")["rmse"]

    rmse_promedio = pd.DataFrame(rmse_por_modelo).mean(axis=1).rename("rmse_promedio")

    defo = df.copy()
    defo["geocode"] = defo["geocode"].astype(str).str.zfill(6)
    defo = defo.set_index("geocode")
    cols_frac = [f"deforestacion_2025_{m}" for m in MODELOS_CANDIDATOS]
    divergencia = defo[cols_frac].std(axis=1).rename("divergencia_2025")

    datos = pd.concat([rmse_promedio, divergencia, defo["departamento"]], axis=1).dropna()

    x = datos["rmse_promedio"].values
    y = datos["divergencia_2025"].values
    es_resaltado = (datos["departamento"] == departamento_resaltado).values

    pearson_r,  pearson_p  = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    x_excl, y_excl = x[~es_resaltado], y[~es_resaltado]
    pearson_r_excl,  pearson_p_excl  = stats.pearsonr(x_excl, y_excl)
    spearman_r_excl, spearman_p_excl = stats.spearmanr(x_excl, y_excl)

    color_resto, color_acento, color_eje = "#A9B7C0", "#F4A261", "#444444"

    fig, ax = plt.subplots(figsize=(7.5, 6))

    pendiente, intercepto = np.polyfit(x, y, 1)
    pendiente_excl, _ = np.polyfit(x_excl, y_excl, 1)
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

    p_str = f"{pearson_p:.2e}" if pearson_p < 0.001 else f"{pearson_p:.3f}"
    ax.text(
        0.97, 0.05,
        f"Pearson r = {pearson_r:.3f}  (p = {p_str})",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=color_eje,
    )

    fig.tight_layout()
    ruta_fig = os.path.join(comparacion_dir, "deforestacion_2025_dispersion_rmse_divergencia.png")
    fig.savefig(ruta_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[OK] {ruta_fig}")

    medias = datos.assign(resaltado=es_resaltado).groupby("resaltado")[["rmse_promedio", "divergencia_2025"]].mean()
    rmse_resaltado = medias.loc[True,  "rmse_promedio"]
    rmse_resto     = medias.loc[False, "rmse_promedio"]
    div_resaltado  = medias.loc[True,  "divergencia_2025"]
    div_resto      = medias.loc[False, "divergencia_2025"]

    df_stats = pd.DataFrame([
        {
            "grupo": "global",
            "n": len(datos),
            "pearson_r": pearson_r, "pearson_p": pearson_p,
            "spearman_r": spearman_r, "spearman_p": spearman_p,
            "pendiente": pendiente,
            "rmse_promedio_media": None, "divergencia_media": None,
        },
        {
            "grupo": departamento_resaltado,
            "n": int(es_resaltado.sum()),
            "pearson_r": None, "pearson_p": None,
            "spearman_r": None, "spearman_p": None,
            "pendiente": None,
            "rmse_promedio_media": rmse_resaltado, "divergencia_media": div_resaltado,
        },
        {
            "grupo": f"excl_{departamento_resaltado}",
            "n": int((~es_resaltado).sum()),
            "pearson_r": pearson_r_excl, "pearson_p": pearson_p_excl,
            "spearman_r": spearman_r_excl, "spearman_p": spearman_p_excl,
            "pendiente": pendiente_excl,
            "rmse_promedio_media": rmse_resto, "divergencia_media": div_resto,
        },
    ])
    ruta_stats = os.path.join(comparacion_dir, "deforestacion_2025_dispersion_stats.csv")
    df_stats.to_csv(ruta_stats, index=False)
    logger.info(f"[OK] {ruta_stats}")

    logger.info(f"n={len(datos)} | pendiente={pendiente:.4f}")
    logger.info(f"Pearson  r={pearson_r:.4f}  p={pearson_p:.4g}")
    logger.info(f"Spearman r={spearman_r:.4f}  p={spearman_p:.4g}")
    logger.info("\n" + str(medias))


def pipeline_r11(
    panel_escalado, panel_original, df_distritos_info, rutas_modelo, escalador,
    ruta_panel_origen, comparacion_dir, rutas_distrito_dl, anio_anchor=2024,
    departamento_resaltado="Cajamarca",
    departamentos_destacados=("San Martin", "Huanuco", "Ucayali"),
):
    """
    Genera pronóstico 2025, deforestación en fracción y km², y todos los gráficos
    del R11. Produce un único archivo deforestacion_2025.csv.
    """
    logger.info("=" * 70)
    logger.info("R11: PRONÓSTICO 2025 (ancla = pct_bosque_real_2024)")
    logger.info("=" * 70)

    df = generar_predicciones(
        panel_escalado, panel_original, df_distritos_info, rutas_modelo, escalador, anio_anchor,
    )

    # Deforestación como fracción de cobertura perdida
    for modelo in MODELOS_CANDIDATOS:
        df[f"deforestacion_2025_{modelo}"] = (
            df["pct_bosque_real_2024"] - df[f"{modelo}_pred_{anio_anchor + 1}"]
        )

    # Área en km² y deforestación absoluta
    panel_csv = pd.read_csv(ruta_panel_origen, dtype={"geocode": str})
    pix_2024 = (
        panel_csv[panel_csv["anio"] == anio_anchor][["geocode", "pix_total"]]
        .rename(columns={"pix_total": "pix_total_2024"})
    )
    df = df.merge(pix_2024, on="geocode", how="left")
    if df["pix_total_2024"].isna().any():
        faltantes = df.loc[df["pix_total_2024"].isna(), "geocode"].tolist()
        raise ValueError(f"No se encontró pix_total {anio_anchor} para: {faltantes}")
    df["area_km2_2024"] = df["pix_total_2024"] * PIXEL_AREA_KM2
    for modelo in MODELOS_CANDIDATOS:
        df[f"deforestacion_2025_{modelo}_km2"] = (
            df["area_km2_2024"] * df[f"deforestacion_2025_{modelo}"]
        )
    df = df.drop(columns=["pix_total_2024"])

    ruta_csv = os.path.join(comparacion_dir, "deforestacion_2025.csv")
    df.to_csv(ruta_csv, index=False)
    logger.info(f"[OK] {ruta_csv}")

    _graficar_deforestacion_departamento_km2(
        df, comparacion_dir, departamento_resaltado, departamentos_destacados,
    )
    _graficar_correlacion_rmse_divergencia_2025(
        rutas_distrito_dl, df, comparacion_dir, departamento_resaltado,
    )

    return df
