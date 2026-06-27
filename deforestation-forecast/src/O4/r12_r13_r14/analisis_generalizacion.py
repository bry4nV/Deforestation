"""
R14 — Informe de análisis explicativo de las causas de la generalización
espacial.

Combina los RMSE/MAE por zona de R13 con factores territoriales ya
conocidos por el EDA de O3 (pendiente y elevación se relacionan con la
*velocidad* de cambio de bosque, no solo el nivel — ver eda_panel.py
Sección 6), para ver si también explican *dónde* los modelos generalizan
peor. Genera la tabla de métricas (≥2: RMSE y MAE, en muestra vs.
generalización) y los gráficos para el informe técnico.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from O3.utils import guardar_csv

from O4.config import (
    O2_COMPARACION_MODELOS_CSV,
    PANEL_GENERALIZACION_CSV,
    R11_CNN_GLOBAL_CSV,
    R13_DIR,
    R14_DIR,
)

logger = logging.getLogger(__name__)

FACTORES_TERRITORIALES = ["pendiente_media_deg", "elev_media_m", "pct_agropecuario"]

INFORME_CSV = os.path.join(R14_DIR, "informe_generalizacion.csv")
FACTORES_CSV = os.path.join(R14_DIR, "factores_generalizacion.csv")
INFORME_MD = os.path.join(R14_DIR, "informe_generalizacion.md")


# ─────────────────────────────────────────────────────────────────────────────
# Carga
# ─────────────────────────────────────────────────────────────────────────────

def _cargar_distrito(modelo: str) -> pd.DataFrame:
    ruta = os.path.join(R13_DIR, f"{modelo}_generalizacion_distrito.csv")
    df = pd.read_csv(ruta, dtype={"geocode": str})
    df["modelo"] = modelo.upper()
    return df


def _cargar_factores_territoriales() -> pd.DataFrame:
    """Un registro por distrito: factores estáticos (first) + dinámicos (mean)."""
    df = pd.read_csv(PANEL_GENERALIZACION_CSV, dtype={"geocode": str})
    factores = df.groupby("geocode").agg(
        pendiente_media_deg=("pendiente_media_deg", "first"),
        elev_media_m=("elev_media_m", "first"),
        pct_agropecuario=("pct_agropecuario", "mean"),
    ).reset_index()
    return factores


def _rmse_en_muestra() -> dict:
    """RMSE/MAE 'en muestra' de ARIMA (O2) y CNN (R11), para comparar contra generalización."""
    df_o2 = pd.read_csv(O2_COMPARACION_MODELOS_CSV)
    fila_arima = df_o2[df_o2["modelo"].str.startswith("ARIMA")].iloc[0]

    df_cnn = pd.read_csv(R11_CNN_GLOBAL_CSV)
    fila_cnn = df_cnn.iloc[0]

    return {
        "ARIMA": {"rmse": float(fila_arima["rmse"]), "mae": float(fila_arima["mae"])},
        "CNN":   {"rmse": float(fila_cnn["rmse"]),    "mae": float(fila_cnn["mae"])},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tabla de métricas e indicadores
# ─────────────────────────────────────────────────────────────────────────────

def _rmse_generalizacion_global(modelo: str) -> tuple:
    """RMSE/MAE global de generalización (pooled sobre todas las zonas-año),
    ya calculado por R13 — NO se recalcula como promedio de RMSE por zona,
    que es una agregación distinta (y daría un número distinto)."""
    df = pd.read_csv(os.path.join(R13_DIR, f"{modelo}_generalizacion_global.csv"))
    fila = df.iloc[0]
    return float(fila["rmse"]), float(fila["mae"])


def construir_tabla_metricas(df_arima: pd.DataFrame, df_cnn: pd.DataFrame) -> pd.DataFrame:
    en_muestra = _rmse_en_muestra()
    filas = []
    for nombre, df in [("arima", df_arima), ("cnn", df_cnn)]:
        rmse_gen, mae_gen = _rmse_generalizacion_global(nombre)
        nombre = nombre.upper()
        rmse_m, mae_m = en_muestra[nombre]["rmse"], en_muestra[nombre]["mae"]
        filas.append({
            "modelo": nombre,
            "rmse_en_muestra": round(rmse_m, 6),
            "mae_en_muestra": round(mae_m, 6),
            "rmse_generalizacion": round(rmse_gen, 6),
            "mae_generalizacion": round(mae_gen, 6),
            "diff_rmse_pct": round((rmse_gen - rmse_m) / rmse_m * 100, 3),
            "diff_mae_pct": round((mae_gen - mae_m) / mae_m * 100, 3),
        })
    return pd.DataFrame(filas)


def calcular_factores_generalizacion(df_distrito_modelo: pd.DataFrame, factores: pd.DataFrame) -> pd.DataFrame:
    base = df_distrito_modelo.merge(factores, on="geocode", how="left")
    if base[FACTORES_TERRITORIALES].isna().any().any():
        raise RuntimeError("Merge con factores territoriales dejó NaN — revisar geocodes.")

    modelo = base["modelo"].iloc[0]
    registros = []
    for factor in FACTORES_TERRITORIALES:
        r_p, p_p = scipy_stats.pearsonr(base[factor], base["rmse"])
        r_s, p_s = scipy_stats.spearmanr(base[factor], base["rmse"])
        registros.append({
            "modelo": modelo, "factor": factor, "n_zonas": len(base),
            "r_pearson": round(r_p, 4), "p_pearson": round(p_p, 4),
            "r_spearman": round(r_s, 4), "p_spearman": round(p_s, 4),
        })
    return pd.DataFrame(registros)


# ─────────────────────────────────────────────────────────────────────────────
# Gráficos
# ─────────────────────────────────────────────────────────────────────────────

def graficar_en_muestra_vs_generalizacion(tabla_metricas: pd.DataFrame, ruta: str) -> None:
    modelos = tabla_metricas["modelo"].tolist()
    x = np.arange(len(modelos))
    ancho = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - ancho / 2, tabla_metricas["rmse_en_muestra"], ancho, label="En muestra", color="steelblue")
    ax.bar(x + ancho / 2, tabla_metricas["rmse_generalizacion"], ancho, label="Generalización (20 zonas)", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(modelos)
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE en muestra vs. generalización espacial")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (v1, v2) in enumerate(zip(tabla_metricas["rmse_en_muestra"], tabla_metricas["rmse_generalizacion"])):
        ax.text(i - ancho / 2, v1, f"{v1:.4f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + ancho / 2, v2, f"{v2:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    logger.info(f"[OK] {ruta}")


def graficar_factores(df_arima: pd.DataFrame, df_cnn: pd.DataFrame, factores: pd.DataFrame, ruta: str) -> None:
    base_arima = df_arima.merge(factores, on="geocode")
    base_cnn = df_cnn.merge(factores, on="geocode")

    fig, axes = plt.subplots(1, len(FACTORES_TERRITORIALES), figsize=(5 * len(FACTORES_TERRITORIALES), 4.5))
    for ax, factor in zip(axes, FACTORES_TERRITORIALES):
        ax.scatter(base_arima[factor], base_arima["rmse"], color="#e41a1c", label="ARIMA", alpha=0.8)
        ax.scatter(base_cnn[factor], base_cnn["rmse"], color="#377eb8", label="CNN", alpha=0.8)
        ax.set_xlabel(factor)
        ax.set_ylabel("RMSE por zona")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("RMSE de generalización por zona vs. factores territoriales")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    logger.info(f"[OK] {ruta}")


def graficar_boxplot_departamento(df_arima: pd.DataFrame, df_cnn: pd.DataFrame, ruta: str) -> None:
    todos = pd.concat([df_arima, df_cnn], ignore_index=True)
    departamentos = sorted(todos["departamento"].unique())

    fig, ax = plt.subplots(figsize=(max(7, len(departamentos) * 1.2), 5))
    posiciones = np.arange(len(departamentos))
    for offset, (modelo, color) in zip([-0.2, 0.2], [("ARIMA", "#e41a1c"), ("CNN", "#377eb8")]):
        datos = [todos[(todos["departamento"] == d) & (todos["modelo"] == modelo)]["rmse"].values for d in departamentos]
        bp = ax.boxplot(datos, positions=posiciones + offset, widths=0.35, patch_artist=True)
        for box in bp["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.6)
    ax.set_xticks(posiciones)
    ax.set_xticklabels(departamentos, rotation=30, ha="right")
    ax.set_ylabel("RMSE por zona")
    ax.set_title("RMSE de generalización por departamento (rojo=ARIMA, azul=CNN)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    logger.info(f"[OK] {ruta}")


def graficar_mejor_peor_zona(df_arima_dist: pd.DataFrame, df_cnn_dist: pd.DataFrame, ruta: str) -> None:
    """Series reales-vs-predicho de la zona donde CNN generaliza mejor y peor."""
    pred_arima = pd.read_csv(os.path.join(R13_DIR, "arima_generalizacion_predicciones.csv"), dtype={"geocode": str})
    pred_cnn = pd.read_csv(os.path.join(R13_DIR, "cnn_generalizacion_predicciones.csv"), dtype={"geocode": str})

    orden = df_cnn_dist.sort_values("rmse")
    geocode_mejor = orden.iloc[0]["geocode"]
    geocode_peor = orden.iloc[-1]["geocode"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, geocode, etiqueta in [(axes[0], geocode_mejor, "mejor"), (axes[1], geocode_peor, "peor")]:
        pa = pred_arima[pred_arima["geocode"] == geocode].sort_values("anio")
        pc = pred_cnn[pred_cnn["geocode"] == geocode].sort_values("anio")
        info = df_cnn_dist[df_cnn_dist["geocode"] == geocode].iloc[0]

        ax.plot(pa["anio"], pa["y_true"], color="black", marker="o", linewidth=2, label="Real")
        ax.plot(pa["anio"], pa["y_pred"], color="#e41a1c", marker="s", linestyle="--", label="ARIMA")
        ax.plot(pc["anio"], pc["y_pred"], color="#377eb8", marker="^", linestyle="--", label="CNN")
        ax.set_title(f"{etiqueta.capitalize()} generalización (CNN) — {info['distrito']} ({info['departamento']})")
        ax.set_xlabel("Año")
        ax.set_ylabel("% cobertura boscosa")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    logger.info(f"[OK] {ruta}")


# ─────────────────────────────────────────────────────────────────────────────
# Informe narrativo
# ─────────────────────────────────────────────────────────────────────────────

def _tabla_markdown(df: pd.DataFrame) -> str:
    """Tabla markdown sin depender de tabulate (no instalado en el venv)."""
    encabezado = "| " + " | ".join(df.columns) + " |"
    separador = "| " + " | ".join("---" for _ in df.columns) + " |"
    filas = ["| " + " | ".join(str(v) for v in fila) + " |" for fila in df.values]
    return "\n".join([encabezado, separador] + filas)


def escribir_informe_md(tabla_metricas: pd.DataFrame, tabla_factores: pd.DataFrame, ruta: str) -> None:
    lineas = [
        "# Informe de generalización espacial — O4 / R14\n",
        "## Métricas por modelo (en muestra vs. 20 zonas nuevas)\n",
        _tabla_markdown(tabla_metricas), "\n",
        "## Factores territoriales y su relación con el error de generalización\n",
        _tabla_markdown(tabla_factores), "\n",
        "## Lectura\n",
    ]
    for _, fila in tabla_metricas.iterrows():
        signo = "mejora" if fila["diff_rmse_pct"] < 0 else "empeora"
        lineas.append(
            f"- **{fila['modelo']}**: el RMSE en las 20 zonas nuevas {signo} un "
            f"{abs(fila['diff_rmse_pct']):.1f}% respecto al RMSE en muestra "
            f"({fila['rmse_en_muestra']:.6f} → {fila['rmse_generalizacion']:.6f})."
        )
    for _, fila in tabla_factores.iterrows():
        if fila["p_pearson"] < 0.05:
            lineas.append(
                f"- Para **{fila['modelo']}**, `{fila['factor']}` se correlaciona "
                f"significativamente con el error por zona (r={fila['r_pearson']}, "
                f"p={fila['p_pearson']})."
            )
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    logger.info(f"[OK] {ruta}")


# ─────────────────────────────────────────────────────────────────────────────
# Orquestador
# ─────────────────────────────────────────────────────────────────────────────

def analizar_generalizacion() -> pd.DataFrame:
    if os.path.exists(INFORME_CSV):
        logger.info("[SKIP] R14 ya generado — cargando informe existente")
        return pd.read_csv(INFORME_CSV)

    logger.info("=" * 70)
    logger.info("R14 — Análisis de generalización espacial")

    df_arima = _cargar_distrito("arima")
    df_cnn = _cargar_distrito("cnn")
    factores = _cargar_factores_territoriales()

    tabla_metricas = construir_tabla_metricas(df_arima, df_cnn)
    guardar_csv(tabla_metricas, INFORME_CSV)
    logger.info("\n" + tabla_metricas.to_string(index=False))

    tabla_factores = pd.concat([
        calcular_factores_generalizacion(df_arima, factores),
        calcular_factores_generalizacion(df_cnn, factores),
    ], ignore_index=True)
    guardar_csv(tabla_factores, FACTORES_CSV)
    logger.info("\n" + tabla_factores.to_string(index=False))

    graficar_en_muestra_vs_generalizacion(tabla_metricas, os.path.join(R14_DIR, "grafico_en_muestra_vs_generalizacion.png"))
    graficar_factores(df_arima, df_cnn, factores, os.path.join(R14_DIR, "grafico_factores_territoriales.png"))
    graficar_boxplot_departamento(df_arima, df_cnn, os.path.join(R14_DIR, "grafico_boxplot_departamento.png"))
    graficar_mejor_peor_zona(df_arima, df_cnn, os.path.join(R14_DIR, "grafico_mejor_peor_zona.png"))

    escribir_informe_md(tabla_metricas, tabla_factores, INFORME_MD)

    logger.info("[OK] R14 completado.")
    return tabla_metricas
