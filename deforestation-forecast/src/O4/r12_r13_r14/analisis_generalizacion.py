"""
R14 — Informe de análisis explicativo de las causas de la generalización
espacial, para el CNN1D extendido (O3/R11) — el modelo final del proyecto.

Combina el RMSE/MAE por zona de R13 con factores territoriales ya conocidos
por el EDA de O3 (pendiente y elevación se relacionan con la *velocidad* de
cambio de bosque, no solo el nivel — ver eda_panel.py Sección 6), para ver
si también explican *dónde* el modelo generaliza peor. Genera la tabla de
métricas (RMSE y MAE, en muestra vs. generalización) y los gráficos para el
informe técnico.
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
    PANEL_GENERALIZACION_CSV,
    R11_CNN_GLOBAL_CSV,
    R13_DIR,
    R14_DIR,
)

logger = logging.getLogger(__name__)

# Las 6 variables locales de O3 (ver eda_panel.py sección 7 para la
# clasificación dinámica/estática empírica). pct_agropecuario y pct_anp son
# DINAMICAS (se agregan por media sobre el panel); el resto son ESTATICAS
# (atributos territoriales fijos, se toma el primer valor).
FACTORES_DINAMICOS = ["pct_agropecuario", "pct_anp"]
FACTORES_ESTATICOS = ["densidad_carreteras_km_km2", "densidad_rios_km_km2", "elev_media_m", "pendiente_media_deg"]
FACTORES_TERRITORIALES = FACTORES_DINAMICOS + FACTORES_ESTATICOS

# Departamentos amazónicos limítrofes con Cajamarca (ya identificado como el
# caso estructuralmente difícil en R6/R11) presentes en las 20 zonas nuevas.
DEPARTAMENTOS_VECINOS_CAJAMARCA = ["Amazonas", "San Martin"]

# n de casos extremos a reportar (mismo criterio 3+3 que el resto del proyecto,
# ver pipeline_comparacion.py de O2/R11).
N_CASOS_EXTREMOS = 3

INFORME_CSV = os.path.join(R14_DIR, "informe_generalizacion.csv")
FACTORES_CSV = os.path.join(R14_DIR, "factores_generalizacion.csv")
TABLA_DISTRITAL_CSV = os.path.join(R14_DIR, "tabla_distrital_completa.csv")
CASOS_EXTREMOS_CSV = os.path.join(R14_DIR, "casos_extremos.csv")
# Salidas adicionales del análisis explicativo de factores (R14):
#  - factores_por_grupo: bloque 2 (perfil territorial promedio por tercil de error)
#  - casos_extremos_detalle: bloque 3 (3+3 distritos con sus 6 variables locales)
#  - recomendacion_factores: bloque 4 (ranking combinado de evidencia)
FACTORES_POR_GRUPO_CSV = os.path.join(R14_DIR, "factores_por_grupo.csv")
CASOS_EXTREMOS_DETALLE_CSV = os.path.join(R14_DIR, "casos_extremos_detalle.csv")
RECOMENDACION_FACTORES_CSV = os.path.join(R14_DIR, "recomendacion_factores.csv")
INFORME_MD = os.path.join(R14_DIR, "informe_generalizacion.md")

# Entregables finales del R14 — versiones limpias (CSV + Markdown + figuras)
# de los cuadros y gráficos que entran al cuerpo del informe. Se guardan
# aparte para no mezclar artefactos intermedios con los que cita la tesis.
FINAL_DIR = os.path.join(R14_DIR, "final")
INTERPRETACION_FACTORES = {
    "pct_agropecuario":           "presión antrópica",
    "elev_media_m":               "condición biofísica",
    "pendiente_media_deg":        "relieve",
    "pct_anp":                    "protección territorial",
    "densidad_rios_km_km2":       "condición hidrográfica",
    "densidad_carreteras_km_km2": "accesibilidad antrópica",
}
# Orden de las 6 variables en las tablas finales — biofísicas y antrópicas
# alternadas, igual que el orden de discusión en el informe.
ORDEN_FACTORES_FINAL = [
    "pct_agropecuario", "elev_media_m", "pendiente_media_deg",
    "pct_anp", "densidad_rios_km_km2", "densidad_carreteras_km_km2",
]
DECIMALES_FACTOR_FINAL = {
    "pct_agropecuario": 4, "pct_anp": 4,
    "densidad_carreteras_km_km2": 4, "densidad_rios_km_km2": 4,
    "elev_media_m": 2, "pendiente_media_deg": 2,
}

# Umbral para considerar significativo el ranking de un factor: mismo p<0.05
# del resto del proyecto (ver pipeline_comparacion.py de O2/R5).
P_SIGNIFICATIVO = 0.05
# n de factores territoriales principales a recomendar como "factores
# explicativos del error" en el informe (bloque 4 de R14).
N_FACTORES_RECOMENDADOS = 3


# ─────────────────────────────────────────────────────────────────────────────
# Carga
# ─────────────────────────────────────────────────────────────────────────────

def _cargar_distrito(modelo: str = "cnn") -> pd.DataFrame:
    ruta = os.path.join(R13_DIR, f"{modelo}_generalizacion_distrito.csv")
    df = pd.read_csv(ruta, dtype={"geocode": str})
    df["modelo"] = modelo.upper()
    return df


def _anios_evaluacion() -> list:
    """Años del periodo de prueba — los 5 años (2020-2024) sobre los que se
    mide el RMSE de generalización en R13. Se infieren del CSV de
    predicciones para no acoplar este análisis a una constante hardcoded."""
    df = pd.read_csv(
        os.path.join(R13_DIR, "cnn_generalizacion_predicciones.csv"),
        usecols=["anio"],
    )
    return sorted(int(a) for a in df["anio"].unique())


def _cargar_factores_territoriales(anios_evaluacion: list = None) -> pd.DataFrame:
    """Un registro por distrito: factores estáticos (first) + dinámicos (mean
    sobre `anios_evaluacion`).

    Para las dinámicas (pct_agropecuario, pct_anp) se promedia solo sobre los
    5 años del periodo de prueba, no sobre los 40 del panel completo —
    coherente con la sección 6.5.5 del informe (factores territoriales del
    análisis explicativo de O3/R11), donde el "valor del factor" para cada
    distrito es su promedio en la ventana donde el modelo es evaluado, no
    en toda su historia."""
    df = pd.read_csv(PANEL_GENERALIZACION_CSV, dtype={"geocode": str})
    if anios_evaluacion is None:
        anios_evaluacion = _anios_evaluacion()
    df_din = df[df["anio"].isin(anios_evaluacion)]
    if df_din.empty:
        raise RuntimeError(
            f"Panel vacío tras filtrar dinámicas por años {anios_evaluacion}."
        )
    dinamicos = df_din.groupby("geocode")[FACTORES_DINAMICOS].mean().reset_index()
    estaticos = df.groupby("geocode")[FACTORES_ESTATICOS].first().reset_index()
    return dinamicos.merge(estaticos, on="geocode", how="inner")


def _cargar_sesgo_distrital(modelo: str = "cnn") -> pd.DataFrame:
    """Sesgo por distrito = media(predicho - observado) sobre los 5 años de
    prueba. La columna 'error' de *_predicciones.csv ya está en esa
    convención (predicho - observado), ver pipeline_cnn._evaluar_geografico /
    O3.r11.utils.construir_df_predicciones."""
    ruta = os.path.join(R13_DIR, f"{modelo}_generalizacion_predicciones.csv")
    df = pd.read_csv(ruta, dtype={"geocode": str})
    return df.groupby("geocode")["error"].mean().rename("sesgo").reset_index()


def _rmse_en_muestra() -> tuple:
    """RMSE/MAE 'en muestra' del CNN (R11), para comparar contra generalización."""
    df_cnn = pd.read_csv(R11_CNN_GLOBAL_CSV)
    fila_cnn = df_cnn.iloc[0]
    return float(fila_cnn["rmse"]), float(fila_cnn["mae"])


# ─────────────────────────────────────────────────────────────────────────────
# Tabla de métricas e indicadores
# ─────────────────────────────────────────────────────────────────────────────

def _rmse_generalizacion_global(modelo: str = "cnn") -> tuple:
    """RMSE/MAE global de generalización (pooled sobre todas las zonas-año),
    ya calculado por R13 — NO se recalcula como promedio de RMSE por zona,
    que es una agregación distinta (y daría un número distinto)."""
    df = pd.read_csv(os.path.join(R13_DIR, f"{modelo}_generalizacion_global.csv"))
    fila = df.iloc[0]
    return float(fila["rmse"]), float(fila["mae"])


def construir_tabla_metricas(df_cnn: pd.DataFrame, n_distritos_en_muestra: int = 180) -> pd.DataFrame:
    rmse_m, mae_m = _rmse_en_muestra()
    rmse_gen, mae_gen = _rmse_generalizacion_global("cnn")
    n_zonas_gen = df_cnn["geocode"].nunique()
    n_anios_gen = pd.read_csv(os.path.join(R13_DIR, "cnn_generalizacion_predicciones.csv"))["anio"].nunique()
    fila = {
        "modelo": "CNN",
        "n_distritos_en_muestra": n_distritos_en_muestra,
        "n_predicciones_en_muestra": n_distritos_en_muestra * n_anios_gen,
        "rmse_en_muestra": round(rmse_m, 6),
        "mae_en_muestra": round(mae_m, 6),
        "n_distritos_generalizacion": n_zonas_gen,
        "n_predicciones_generalizacion": n_zonas_gen * n_anios_gen,
        "rmse_generalizacion": round(rmse_gen, 6),
        "mae_generalizacion": round(mae_gen, 6),
        "diff_rmse_pct": round((rmse_gen - rmse_m) / rmse_m * 100, 3),
        "diff_mae_pct": round((mae_gen - mae_m) / mae_m * 100, 3),
    }
    return pd.DataFrame([fila])


def construir_tabla_distrital_completa(df_cnn: pd.DataFrame) -> pd.DataFrame:
    """Tabla distrital completa (punto 2 de R14): geocode, departamento,
    distrito, rmse, mae, sesgo (predicho-observado) y clasificación por
    terciles de RMSE dentro de las 20 zonas."""
    sesgo = _cargar_sesgo_distrital("cnn")
    tabla = df_cnn.merge(sesgo, on="geocode", how="left").sort_values("rmse").reset_index(drop=True)
    tabla["clasificacion"] = pd.qcut(
        tabla["rmse"], 3, labels=["menor error", "error medio", "mayor error"]
    )
    columnas = ["geocode", "departamento", "distrito", "rmse", "mae", "sesgo", "clasificacion"]
    return tabla[columnas]


def identificar_casos_extremos(tabla_distrital: pd.DataFrame, n: int = N_CASOS_EXTREMOS) -> pd.DataFrame:
    """n distritos de menor error + n de mayor error (mismo formato y criterio
    3+3 de los casos extremos de R6/R11: geocode, departamento, distrito,
    rmse, mae)."""
    columnas = ["geocode", "departamento", "distrito", "rmse", "mae"]
    mejores = tabla_distrital.nsmallest(n, "rmse")[columnas].copy()
    mejores["grupo"] = "menor error"
    peores = tabla_distrital.nlargest(n, "rmse")[columnas].copy()
    peores["grupo"] = "mayor error"
    return pd.concat([mejores, peores], ignore_index=True)


def construir_tabla_factores_por_grupo(
    tabla_distrital: pd.DataFrame, factores: pd.DataFrame,
) -> pd.DataFrame:
    """Bloque 2 del análisis explicativo: perfil territorial promedio por
    grupo de error (terciles de RMSE ya definidos en
    `tabla_distrital_completa.csv` — menor / medio / mayor, ~7 distritos por
    grupo). Reporta n, RMSE promedio del grupo y media de cada una de las 6
    variables locales. Con n≈7 por grupo se usa la media — los efectos de
    outliers se discuten desde la tabla de casos extremos (bloque 3) y
    desde Spearman (bloque 1), no introduciendo una segunda agregación que
    duplicaría columnas sin agregar señal."""
    base = tabla_distrital.merge(factores, on="geocode", how="left")
    if base[FACTORES_TERRITORIALES].isna().any().any():
        raise RuntimeError("Merge tabla distrital × factores dejó NaN — revisar geocodes.")
    agg = {"geocode": "count", "rmse": "mean"}
    agg.update({f: "mean" for f in FACTORES_TERRITORIALES})
    resumen = (
        base.groupby("clasificacion", observed=False)
        .agg(agg)
        .rename(columns={"geocode": "n_distritos", "rmse": "rmse_promedio"})
        .reindex(["menor error", "error medio", "mayor error"])
        .reset_index()
        .rename(columns={"clasificacion": "grupo_error"})
    )
    resumen["rmse_promedio"] = resumen["rmse_promedio"].round(6)
    for f in FACTORES_TERRITORIALES:
        resumen[f] = resumen[f].round(5)
    return resumen


def construir_casos_extremos_detalle(
    casos_extremos: pd.DataFrame, factores: pd.DataFrame,
) -> pd.DataFrame:
    """Bloque 3 del análisis explicativo: 3+3 casos extremos enriquecidos
    con sus valores en las 6 variables locales — la tabla operativa para
    leer qué perfil territorial acompaña a los mejores y peores RMSE."""
    base = casos_extremos.merge(factores, on="geocode", how="left")
    if base[FACTORES_TERRITORIALES].isna().any().any():
        raise RuntimeError("Merge casos extremos × factores dejó NaN — revisar geocodes.")
    columnas = (
        ["geocode", "departamento", "distrito", "grupo", "rmse"]
        + FACTORES_TERRITORIALES
    )
    detalle = base[columnas].copy()
    for f in FACTORES_TERRITORIALES:
        detalle[f] = detalle[f].round(5)
    return detalle


def recomendar_factores_principales(
    tabla_factores: pd.DataFrame, tabla_por_grupo: pd.DataFrame,
    factores: pd.DataFrame, top_n: int = N_FACTORES_RECOMENDADOS,
) -> pd.DataFrame:
    """Bloque 4 del análisis explicativo: ranking combinado de las 6
    variables locales por evidencia de relación con el error de
    generalización. Tres ejes que se promedian por ranking (1 = peor
    evidencia, n = mejor) para no mezclar unidades:

      - **correlación**: max(|r_pearson|, |r_spearman|) — usar el mayor de
        los dos absorbe relaciones lineales y monótonas no lineales sin
        privilegiar a Pearson, que es frágil con n=20.
      - **significancia**: min(p_pearson, p_spearman) — coherente con
        cómo se reporta cada factor como significativo en el informe.
      - **separación entre extremos**: |media_mayor − media_menor| /
        std_global (efecto tipo *Cohen-d* simplificado, usando la
        desviación entre las 20 zonas como referencia común que
        normaliza el escalón natural de cada variable).

    Con n=20 ninguno de los tres por sí solo es definitivo (ver nota en
    `calcular_factores_generalizacion`), así que la recomendación se
    construye sobre el promedio de los tres rankings — los top_n son los
    "factores territoriales principales" para el bloque narrativo del
    informe."""
    rec = tabla_factores[["factor", "r_pearson", "p_pearson", "r_spearman", "p_spearman"]].copy()
    rec["max_abs_r"] = rec[["r_pearson", "r_spearman"]].abs().max(axis=1).round(4)
    rec["min_p"] = rec[["p_pearson", "p_spearman"]].min(axis=1).round(4)

    medias_menor = tabla_por_grupo.loc[tabla_por_grupo["grupo_error"] == "menor error"].iloc[0]
    medias_mayor = tabla_por_grupo.loc[tabla_por_grupo["grupo_error"] == "mayor error"].iloc[0]
    stds = factores[FACTORES_TERRITORIALES].std(ddof=1)
    rec["diff_estandarizada"] = [
        round(float(abs(medias_mayor[f] - medias_menor[f]) / stds[f]), 4)
        for f in rec["factor"]
    ]

    # Rankings (ascending=False salvo en p, donde menor es mejor).
    rec["rank_correlacion"] = rec["max_abs_r"].rank(method="min", ascending=True)
    rec["rank_significancia"] = rec["min_p"].rank(method="min", ascending=False)
    rec["rank_grupos"] = rec["diff_estandarizada"].rank(method="min", ascending=True)
    rec["score_combinado"] = (
        rec[["rank_correlacion", "rank_significancia", "rank_grupos"]].mean(axis=1).round(2)
    )
    rec = rec.sort_values("score_combinado", ascending=False).reset_index(drop=True)
    rec["recomendado_top"] = rec.index < top_n
    return rec[
        ["factor", "max_abs_r", "min_p", "diff_estandarizada",
         "rank_correlacion", "rank_significancia", "rank_grupos",
         "score_combinado", "recomendado_top"]
    ]


def calcular_factores_generalizacion(
    df_distrito_modelo: pd.DataFrame, factores: pd.DataFrame, casos_extremos: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Para cada una de las 6 variables locales de O3: correlación (Pearson y
    Spearman, con p-valor) sobre las 20 zonas + comparación directa de
    grupos (media de la variable en los casos extremos de mayor error vs.
    menor error, point (3)) — con n=20 la correlación sola puede no alcanzar
    significancia aunque exista una diferencia clara entre los extremos, o
    viceversa (una correlación significativa arrastrada por 1-2 outliers
    puede no reflejarse en una diferencia de grupo limpia). Reportar ambas
    evita forzar una lectura que ninguna de las dos sostiene por sí sola.
    """
    base = df_distrito_modelo.merge(factores, on="geocode", how="left")
    if base[FACTORES_TERRITORIALES].isna().any().any():
        raise RuntimeError("Merge con factores territoriales dejó NaN — revisar geocodes.")

    base_factores = factores.set_index("geocode")
    if casos_extremos is not None:
        menor = casos_extremos.loc[casos_extremos["grupo"] == "menor error", "geocode"]
        mayor = casos_extremos.loc[casos_extremos["grupo"] == "mayor error", "geocode"]

    modelo = base["modelo"].iloc[0]
    registros = []
    for factor in FACTORES_TERRITORIALES:
        r_p, p_p = scipy_stats.pearsonr(base[factor], base["rmse"])
        r_s, p_s = scipy_stats.spearmanr(base[factor], base["rmse"])
        fila = {
            "modelo": modelo, "factor": factor, "n_zonas": len(base),
            "r_pearson": round(r_p, 4), "p_pearson": round(p_p, 4),
            "r_spearman": round(r_s, 4), "p_spearman": round(p_s, 4),
        }
        if casos_extremos is not None:
            media_menor = base_factores.loc[menor, factor].mean()
            media_mayor = base_factores.loc[mayor, factor].mean()
            fila["media_grupo_menor_error"] = round(media_menor, 5)
            fila["media_grupo_mayor_error"] = round(media_mayor, 5)
            fila["diferencia_grupos"] = round(media_mayor - media_menor, 5)
        registros.append(fila)
    return pd.DataFrame(registros)


# ─────────────────────────────────────────────────────────────────────────────
# Gráficos
# ─────────────────────────────────────────────────────────────────────────────

def graficar_en_muestra_vs_generalizacion(tabla_metricas: pd.DataFrame, ruta: str) -> None:
    modelos = tabla_metricas["modelo"].tolist()
    x = np.arange(len(modelos))
    ancho = 0.35

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(x - ancho / 2, tabla_metricas["rmse_en_muestra"], ancho, label="En muestra", color="steelblue")
    ax.bar(x + ancho / 2, tabla_metricas["rmse_generalizacion"], ancho, label="Generalización (20 zonas)", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(modelos)
    ax.set_ylabel("RMSE")
    ax.set_title("CNN1D extendido — RMSE en muestra vs. generalización espacial")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (v1, v2) in enumerate(zip(tabla_metricas["rmse_en_muestra"], tabla_metricas["rmse_generalizacion"])):
        ax.text(i - ancho / 2, v1, f"{v1:.4f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + ancho / 2, v2, f"{v2:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    logger.info(f"[OK] {ruta}")


def graficar_factores(df_cnn: pd.DataFrame, factores: pd.DataFrame, ruta: str) -> None:
    base_cnn = df_cnn.merge(factores, on="geocode")

    n = len(FACTORES_TERRITORIALES)
    ncols = 3
    nrows = -(-n // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = axes.flatten()
    for ax, factor in zip(axes, FACTORES_TERRITORIALES):
        ax.scatter(base_cnn[factor], base_cnn["rmse"], color="#377eb8", alpha=0.8)
        ax.set_xlabel(factor)
        ax.set_ylabel("RMSE por zona")
        ax.grid(alpha=0.3)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("CNN1D extendido — RMSE de generalización por zona vs. factores territoriales")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    logger.info(f"[OK] {ruta}")


def graficar_boxplot_departamento(df_cnn: pd.DataFrame, ruta: str) -> None:
    departamentos = sorted(df_cnn["departamento"].unique())

    fig, ax = plt.subplots(figsize=(max(7, len(departamentos) * 1.0), 5))
    posiciones = np.arange(len(departamentos))
    datos = [df_cnn[df_cnn["departamento"] == d]["rmse"].values for d in departamentos]
    bp = ax.boxplot(datos, positions=posiciones, widths=0.5, patch_artist=True)
    for box in bp["boxes"]:
        box.set_facecolor("#377eb8")
        box.set_alpha(0.6)
    ax.set_xticks(posiciones)
    ax.set_xticklabels(departamentos, rotation=30, ha="right")
    ax.set_ylabel("RMSE por zona")
    ax.set_title("CNN1D extendido — RMSE de generalización por departamento")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    logger.info(f"[OK] {ruta}")


def graficar_pronostico_2025_departamento(ruta: str) -> None:
    """Barras horizontales de pérdida neta estimada de bosque (2025) en las
    20 zonas de generalización, agrupadas por departamento. Sin título."""
    csv_path = os.path.join(R13_DIR, "cnn_generalizacion_deforestacion_2025.csv")
    if not os.path.exists(csv_path):
        logger.warning(f"[SKIP] {csv_path} no existe — omitiendo figura pronóstico 2025.")
        return

    df = pd.read_csv(csv_path, dtype={"geocode": str})
    agg = (
        df.groupby("departamento")
        .agg(
            perdida_neta_km2=("deforestacion_2025_cnn_km2", "sum"),
            n_distritos=("geocode", "count"),
        )
        .reset_index()
    )
    agg["participacion_pct"] = agg["perdida_neta_km2"] / agg["perdida_neta_km2"].sum() * 100
    agg = agg.sort_values("perdida_neta_km2", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(agg))
    ax.barh(y, agg["perdida_neta_km2"], color="#2C7BB6", height=0.6, zorder=2)

    max_val = agg["perdida_neta_km2"].max()
    offset = max_val * 0.025
    for i, fila in agg.iterrows():
        etiqueta = (
            f"{fila['perdida_neta_km2']:.2f} km²"
            f"  ({fila['participacion_pct']:.1f} %,  n={int(fila['n_distritos'])})"
        )
        ax.text(fila["perdida_neta_km2"] + offset, i, etiqueta,
                va="center", ha="left", fontsize=8.5, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(agg["departamento"], fontsize=10)
    ax.set_xlabel("Pérdida neta estimada de bosque (km²)", fontsize=10)
    ax.set_xlim(0, max_val * 1.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", linestyle="--", linewidth=0.7, color="#E0E0E0", alpha=0.8, zorder=0)
    fig.tight_layout()
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[OK] {ruta}")


def graficar_mejor_peor_zona(df_cnn_dist: pd.DataFrame, ruta: str) -> None:
    """Series real-vs-predicho de la zona donde CNN generaliza mejor y peor."""
    pred_cnn = pd.read_csv(os.path.join(R13_DIR, "cnn_generalizacion_predicciones.csv"), dtype={"geocode": str})

    orden = df_cnn_dist.sort_values("rmse")
    geocode_mejor = orden.iloc[0]["geocode"]
    geocode_peor = orden.iloc[-1]["geocode"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, geocode, etiqueta in [(axes[0], geocode_mejor, "mejor"), (axes[1], geocode_peor, "peor")]:
        pc = pred_cnn[pred_cnn["geocode"] == geocode].sort_values("anio")
        info = df_cnn_dist[df_cnn_dist["geocode"] == geocode].iloc[0]

        ax.plot(pc["anio"], pc["y_true"], color="black", marker="o", linewidth=2, label="Real")
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
# Entregables finales (carpeta final/) — los cuadros y figuras que entran
# al cuerpo del informe, con redondeos, columnas y orden consistentes.
# ─────────────────────────────────────────────────────────────────────────────

def _tabla_md(df: pd.DataFrame) -> str:
    """Markdown sin tabulate (mismo formato que el resto del proyecto)."""
    encabezado = "| " + " | ".join(df.columns) + " |"
    separador  = "| " + " | ".join("---" for _ in df.columns) + " |"
    filas      = ["| " + " | ".join(str(v) for v in fila) + " |" for fila in df.values]
    return "\n".join([encabezado, separador] + filas) + "\n"


def _guardar_csv_md(df: pd.DataFrame, nombre_base: str) -> tuple:
    csv_path = os.path.join(FINAL_DIR, f"{nombre_base}.csv")
    md_path  = os.path.join(FINAL_DIR, f"{nombre_base}.md")
    df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_tabla_md(df))
    logger.info(f"[OK] {csv_path}")
    logger.info(f"[OK] {md_path}")
    return csv_path, md_path


def _formatear(df: pd.DataFrame, mapa_decimales: dict) -> pd.DataFrame:
    """Formatea columnas como strings con n decimales fijos — necesario
    porque `df.to_csv()` y la serialización por defecto recortan ceros
    finales (p.ej. 0.011290 → '0.01129'), y la regla de R14 pide
    decimales fijos para RMSE/MAE/sesgo, correlaciones, p-valores y
    variables territoriales."""
    df = df.copy()
    for col, n in mapa_decimales.items():
        if col in df.columns:
            df[col] = df[col].map(lambda x, n=n: f"{x:.{n}f}")
    return df


def _redondear_factores(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    mapa = {c: DECIMALES_FACTOR_FINAL.get(c, 4) for c in cols if c in df.columns}
    return _formatear(df, mapa)


def construir_tabla_final_comparacion(tabla_metricas: pd.DataFrame) -> pd.DataFrame:
    """Entregable 1: comparativa global desarrollo experimental vs.
    generalización espacial (un renglón por conjunto evaluado)."""
    m = tabla_metricas.iloc[0]
    df = pd.DataFrame([
        {
            "conjunto_evaluado": "Desarrollo experimental",
            "n_distritos":       int(m["n_distritos_en_muestra"]),
            "n_predicciones":    int(m["n_predicciones_en_muestra"]),
            "rmse":              float(m["rmse_en_muestra"]),
            "mae":               float(m["mae_en_muestra"]),
        },
        {
            "conjunto_evaluado": "Generalización espacial",
            "n_distritos":       int(m["n_distritos_generalizacion"]),
            "n_predicciones":    int(m["n_predicciones_generalizacion"]),
            "rmse":              float(m["rmse_generalizacion"]),
            "mae":               float(m["mae_generalizacion"]),
        },
    ])
    return _formatear(df, {"rmse": 6, "mae": 6})


def construir_tabla_final_distritales(tabla_distrital: pd.DataFrame) -> pd.DataFrame:
    """Entregable 2: 20 distritos × (departamento, distrito, rmse, mae,
    sesgo, clasificacion), ordenados de menor a mayor RMSE."""
    mt = tabla_distrital[
        ["departamento", "distrito", "rmse", "mae", "sesgo", "clasificacion"]
    ].copy()
    mt = mt.sort_values("rmse").reset_index(drop=True)
    return _formatear(mt, {"rmse": 6, "mae": 6, "sesgo": 6})


def construir_tabla_final_correlaciones(tabla_factores: pd.DataFrame) -> pd.DataFrame:
    """Entregable 3: correlaciones Pearson y Spearman entre RMSE distrital
    y cada una de las 6 variables locales (n = 20), con su interpretación
    territorial. Ordenada según ORDEN_FACTORES_FINAL para alinear discusión."""
    cor = tabla_factores[
        ["factor", "r_pearson", "p_pearson", "r_spearman", "p_spearman"]
    ].copy()
    cor.columns = [
        "variable_local", "pearson_r", "p_valor_pearson",
        "spearman_rho", "p_valor_spearman",
    ]
    cor["interpretacion_territorial"] = cor["variable_local"].map(INTERPRETACION_FACTORES)
    cor["orden"] = cor["variable_local"].map({f: i for i, f in enumerate(ORDEN_FACTORES_FINAL)})
    cor = cor.sort_values("orden").drop(columns="orden").reset_index(drop=True)
    return _formatear(cor, {
        "pearson_r": 4, "spearman_rho": 4,
        "p_valor_pearson": 4, "p_valor_spearman": 4,
    })


def construir_tabla_final_perfil_grupos(tabla_por_grupo: pd.DataFrame) -> pd.DataFrame:
    """Entregable 4: perfil territorial promedio por tercil de RMSE."""
    cols = ["grupo_error", "n_distritos", "rmse_promedio"] + ORDEN_FACTORES_FINAL
    perfil = tabla_por_grupo[cols].copy()
    perfil = _formatear(perfil, {"rmse_promedio": 6})
    return _redondear_factores(perfil, ORDEN_FACTORES_FINAL)


def construir_tabla_final_casos_extremos(
    casos_extremos: pd.DataFrame, factores: pd.DataFrame,
) -> pd.DataFrame:
    """Entregable 5: 3 mejores + 3 peores casos con sus 6 variables locales."""
    cex = casos_extremos.merge(factores, on="geocode", how="left")
    cols = ["grupo", "departamento", "distrito", "rmse"] + ORDEN_FACTORES_FINAL
    cex = cex[cols].copy()
    orden_grupo = {"menor error": 0, "mayor error": 1}
    cex["_g"] = cex["grupo"].map(orden_grupo)
    cex = cex.sort_values(["_g", "rmse"]).drop(columns="_g").reset_index(drop=True)
    cex = _formatear(cex, {"rmse": 6})
    return _redondear_factores(cex, ORDEN_FACTORES_FINAL)


def graficar_final_rmse_factores(
    tabla_distrital: pd.DataFrame, factores: pd.DataFrame,
    casos_extremos: pd.DataFrame, ruta: str,
) -> None:
    """Figura 1 del informe final: dispersión RMSE × (elev_media, pct_agro),
    con línea de tendencia, casos extremos resaltados y etiquetas, y caja
    con Pearson y Spearman por panel."""
    base = tabla_distrital[["geocode", "distrito", "rmse"]].merge(factores, on="geocode")
    geo_extremos = set(casos_extremos["geocode"])

    factores_panel = [
        ("elev_media_m",     "Elevación media (m)"),
        ("pct_agropecuario", "Proporción agropecuaria"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    for ax, (factor, etiqueta) in zip(axes, factores_panel):
        x = base[factor].values
        y = base["rmse"].values
        ax.scatter(x, y, color="#377eb8", alpha=0.7, s=55, edgecolor="white", linewidth=0.6)
        # Casos extremos sobre la misma capa
        sub = base[base["geocode"].isin(geo_extremos)]
        ax.scatter(sub[factor], sub["rmse"], color="#e41a1c", s=95,
                   edgecolor="white", linewidth=1.2, zorder=3)
        for _, fila in sub.iterrows():
            ax.annotate(
                fila["distrito"], (fila[factor], fila["rmse"]),
                xytext=(6, 5), textcoords="offset points", fontsize=8.5, color="#333333",
            )
        # Línea de tendencia (OLS simple, solo descriptiva)
        if np.std(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, slope * xs + intercept,
                    linestyle="--", color="#666666", linewidth=1.4, zorder=1)
        # Caja con Pearson y Spearman
        rp, pp = scipy_stats.pearsonr(x, y)
        rs, ps = scipy_stats.spearmanr(x, y)
        ax.text(
            0.04, 0.96,
            f"Pearson r = {rp:+.4f} (p = {pp:.4f})\n"
            f"Spearman ρ = {rs:+.4f} (p = {ps:.4f})",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#cccccc", alpha=0.95),
        )
        ax.set_xlabel(etiqueta)
        ax.set_ylabel("RMSE por distrito")
        ax.grid(alpha=0.3)
    fig.suptitle(
        "Relación entre RMSE de generalización y factores territoriales principales",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[OK] {ruta}")


def graficar_final_observado_predicho_extremos(
    tabla_distrital: pd.DataFrame, ruta: str,
) -> None:
    """Figura 2 del informe final: series observada vs. pronóstico para el
    mejor caso (Mariscal Castilla) y el peor (Shanao), con RMSE en cada
    subtítulo."""
    pred = pd.read_csv(
        os.path.join(R13_DIR, "cnn_generalizacion_predicciones.csv"),
        dtype={"geocode": str},
    )
    orden = tabla_distrital.sort_values("rmse").reset_index(drop=True)
    mejor, peor = orden.iloc[0], orden.iloc[-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    for ax, fila, etq in [(axes[0], mejor, "Mejor caso"), (axes[1], peor, "Peor caso")]:
        sub = pred[pred["geocode"] == fila["geocode"]].sort_values("anio")
        ax.plot(sub["anio"], sub["y_true"], color="black",
                marker="o", linewidth=2.2, label="Observado")
        ax.plot(sub["anio"], sub["y_pred"], color="#377eb8",
                marker="^", linestyle="--", linewidth=2.2, label="Pronosticado (CNN)")
        ax.set_title(
            f"{etq}: {fila['distrito']} ({fila['departamento']}) — RMSE = {fila['rmse']:.4f}",
            fontsize=10.5,
        )
        ax.set_xlabel("Año")
        ax.set_ylabel("Proporción de cobertura boscosa")
        ax.set_xticks(sorted(sub["anio"].unique()))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="best")
    fig.suptitle(
        "Comparación entre valores observados y pronosticados en casos extremos de generalización",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[OK] {ruta}")


def generar_entregables_finales(
    tabla_metricas: pd.DataFrame, tabla_distrital: pd.DataFrame,
    tabla_factores: pd.DataFrame, tabla_por_grupo: pd.DataFrame,
    casos_extremos: pd.DataFrame, factores: pd.DataFrame,
) -> None:
    """Empaqueta los 5 cuadros y las 2 figuras del informe final en
    `r14_informe/final/`. Cada salida ya viene con la convención de
    redondeo (RMSE/MAE/sesgo a 6 dec, r/ρ y p a 4 dec, variables
    territoriales a 4 o 2 según escala) y orden coherente entre todas."""
    os.makedirs(FINAL_DIR, exist_ok=True)

    _guardar_csv_md(construir_tabla_final_comparacion(tabla_metricas), "tabla_r14_comparacion_global")
    _guardar_csv_md(construir_tabla_final_distritales(tabla_distrital), "tabla_r14_metricas_distritales")
    _guardar_csv_md(construir_tabla_final_correlaciones(tabla_factores), "tabla_r14_correlaciones_factores")
    _guardar_csv_md(construir_tabla_final_perfil_grupos(tabla_por_grupo), "tabla_r14_perfil_grupos_error")
    _guardar_csv_md(construir_tabla_final_casos_extremos(casos_extremos, factores), "tabla_r14_casos_extremos")

    graficar_final_rmse_factores(
        tabla_distrital, factores, casos_extremos,
        os.path.join(FINAL_DIR, "figura_r14_rmse_factores_territoriales.png"),
    )
    graficar_final_observado_predicho_extremos(
        tabla_distrital,
        os.path.join(FINAL_DIR, "figura_r14_observado_predicho_extremos.png"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Informe narrativo
# ─────────────────────────────────────────────────────────────────────────────

def _tabla_markdown(df: pd.DataFrame) -> str:
    """Tabla markdown sin depender de tabulate (no instalado en el venv)."""
    encabezado = "| " + " | ".join(df.columns) + " |"
    separador = "| " + " | ".join("---" for _ in df.columns) + " |"
    filas = ["| " + " | ".join(str(v) for v in fila) + " |" for fila in df.values]
    return "\n".join([encabezado, separador] + filas)


def _nota_cajamarca_y_vecinos(tabla_distrital: pd.DataFrame) -> list:
    """Punto 5 de R14: distritos de Cajamarca y de departamentos amazónicos
    limítrofes presentes en las 20 zonas, y si repiten el nivel de error
    relativo ya identificado en R6/R11."""
    deptos = ["Cajamarca"] + DEPARTAMENTOS_VECINOS_CAJAMARCA
    presentes = tabla_distrital[tabla_distrital["departamento"].isin(deptos)]
    lineas = []
    for depto in deptos:
        sub = presentes[presentes["departamento"] == depto].sort_values("rmse")
        if sub.empty:
            lineas.append(f"- **{depto}**: ningún distrito en las 20 zonas de generalización.")
            continue
        detalle = "; ".join(f"{r.distrito} (RMSE={r.rmse:.4f}, {r.clasificacion})" for r in sub.itertuples())
        lineas.append(f"- **{depto}** ({len(sub)} distrito(s)): {detalle}.")
    return lineas


def escribir_informe_md(
    tabla_metricas: pd.DataFrame, tabla_factores: pd.DataFrame,
    tabla_distrital: pd.DataFrame, casos_extremos: pd.DataFrame,
    tabla_por_grupo: pd.DataFrame, casos_extremos_detalle: pd.DataFrame,
    recomendacion: pd.DataFrame, anios_evaluacion: list, ruta: str,
) -> None:
    rango_anios = f"{anios_evaluacion[0]}–{anios_evaluacion[-1]}"
    lineas = [
        "# Informe de generalización espacial — O4 / R14 (CNN1D extendido)\n",
        "## 1. Métricas globales (en muestra vs. 20 zonas nuevas)\n",
        _tabla_markdown(tabla_metricas), "\n",
        "## 2. Tabla distrital completa (20 zonas)\n",
        _tabla_markdown(tabla_distrital), "\n",
        "## 3. Casos extremos\n",
        _tabla_markdown(casos_extremos), "\n",
        "## 4. Factores territoriales — análisis explicativo\n",
        f"_Dinámicas (pct_agropecuario, pct_anp) promediadas sobre el periodo "
        f"de evaluación {rango_anios}; estáticas con su valor por distrito. "
        f"Mismo protocolo que la sección 6.5.5 del informe._\n",
        "### 4.1 Correlaciones RMSE × factor (Pearson y Spearman)\n",
        _tabla_markdown(tabla_factores), "\n",
        "### 4.2 Perfil territorial promedio por grupo de error (terciles de RMSE)\n",
        _tabla_markdown(tabla_por_grupo), "\n",
        "### 4.3 Casos extremos con sus 6 variables locales\n",
        _tabla_markdown(casos_extremos_detalle), "\n",
        "### 4.4 Recomendación — factores principales\n",
        _tabla_markdown(recomendacion), "\n",
        "## 5. Cajamarca y departamentos amazónicos limítrofes\n",
    ]
    lineas += _nota_cajamarca_y_vecinos(tabla_distrital)
    lineas.append("\n## Lectura\n")
    for _, fila in tabla_metricas.iterrows():
        signo = "mejora" if fila["diff_rmse_pct"] < 0 else "empeora"
        lineas.append(
            f"- **{fila['modelo']}**: el RMSE en las {fila['n_predicciones_generalizacion']} predicciones de "
            f"generalización ({fila['n_distritos_generalizacion']} zonas × 5 años) {signo} un "
            f"{abs(fila['diff_rmse_pct']):.1f}% respecto al RMSE en muestra "
            f"({fila['rmse_en_muestra']:.6f} → {fila['rmse_generalizacion']:.6f})."
        )
    # Un factor se reporta como significativo si Pearson O Spearman lo es —
    # Pearson exige linealidad y es sensible a outliers con n=20, así que un
    # factor robusto puede pasar Spearman sin pasar Pearson (y viceversa).
    for _, fila in tabla_factores.iterrows():
        if min(fila["p_pearson"], fila["p_spearman"]) < P_SIGNIFICATIVO:
            lineas.append(
                f"- `{fila['factor']}` se correlaciona significativamente con el error por zona "
                f"(r_pearson={fila['r_pearson']} p={fila['p_pearson']}, "
                f"r_spearman={fila['r_spearman']} p={fila['p_spearman']}); "
                f"media en grupo de mayor error={fila['media_grupo_mayor_error']} vs. "
                f"menor error={fila['media_grupo_menor_error']}."
            )
    top_factores = recomendacion.loc[recomendacion["recomendado_top"], "factor"].tolist()
    lineas.append(
        f"- **Factores territoriales principales** (score combinado, top {len(top_factores)}): "
        + ", ".join(f"`{f}`" for f in top_factores)
        + " — son los que muestran mayor evidencia conjunta de correlación, significancia y "
          "separación entre los grupos de menor y mayor error."
    )
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    logger.info(f"[OK] {ruta}")


# ─────────────────────────────────────────────────────────────────────────────
# Orquestador
# ─────────────────────────────────────────────────────────────────────────────

def analizar_generalizacion() -> pd.DataFrame:
    _ruta_fig_2025 = os.path.join(R13_DIR, "grafico_pronostico_2025_departamento.png")

    if os.path.exists(INFORME_CSV):
        logger.info("[SKIP] R14 ya generado — cargando informe existente")
        if not os.path.exists(_ruta_fig_2025):
            graficar_pronostico_2025_departamento(_ruta_fig_2025)
        return pd.read_csv(INFORME_CSV)

    logger.info("=" * 70)
    logger.info("R14 — Análisis de generalización espacial")

    df_cnn = _cargar_distrito("cnn")
    anios_evaluacion = _anios_evaluacion()
    factores = _cargar_factores_territoriales(anios_evaluacion)
    logger.info(
        f"Factores dinámicos promediados sobre {len(anios_evaluacion)} años "
        f"de evaluación: {anios_evaluacion[0]}–{anios_evaluacion[-1]}."
    )

    tabla_metricas = construir_tabla_metricas(df_cnn)
    guardar_csv(tabla_metricas, INFORME_CSV)
    logger.info("\n" + tabla_metricas.to_string(index=False))

    tabla_distrital = construir_tabla_distrital_completa(df_cnn)
    guardar_csv(tabla_distrital, TABLA_DISTRITAL_CSV)
    logger.info("\n" + tabla_distrital.to_string(index=False))

    casos_extremos = identificar_casos_extremos(tabla_distrital)
    guardar_csv(casos_extremos, CASOS_EXTREMOS_CSV)
    logger.info("\n" + casos_extremos.to_string(index=False))

    tabla_factores = calcular_factores_generalizacion(df_cnn, factores, casos_extremos)
    guardar_csv(tabla_factores, FACTORES_CSV)
    logger.info("\n" + tabla_factores.to_string(index=False))

    # Bloques adicionales del análisis explicativo (perfil por grupo,
    # casos extremos con detalle, recomendación combinada).
    tabla_por_grupo = construir_tabla_factores_por_grupo(tabla_distrital, factores)
    guardar_csv(tabla_por_grupo, FACTORES_POR_GRUPO_CSV)
    logger.info("\n" + tabla_por_grupo.to_string(index=False))

    casos_extremos_detalle = construir_casos_extremos_detalle(casos_extremos, factores)
    guardar_csv(casos_extremos_detalle, CASOS_EXTREMOS_DETALLE_CSV)
    logger.info("\n" + casos_extremos_detalle.to_string(index=False))

    recomendacion = recomendar_factores_principales(tabla_factores, tabla_por_grupo, factores)
    guardar_csv(recomendacion, RECOMENDACION_FACTORES_CSV)
    logger.info("\n" + recomendacion.to_string(index=False))

    graficar_en_muestra_vs_generalizacion(tabla_metricas, os.path.join(R14_DIR, "grafico_en_muestra_vs_generalizacion.png"))
    graficar_factores(df_cnn, factores, os.path.join(R14_DIR, "grafico_factores_territoriales.png"))
    graficar_boxplot_departamento(df_cnn, os.path.join(R14_DIR, "grafico_boxplot_departamento.png"))
    graficar_mejor_peor_zona(df_cnn, os.path.join(R14_DIR, "grafico_mejor_peor_zona.png"))
    graficar_pronostico_2025_departamento(_ruta_fig_2025)

    escribir_informe_md(
        tabla_metricas, tabla_factores, tabla_distrital, casos_extremos,
        tabla_por_grupo, casos_extremos_detalle, recomendacion,
        anios_evaluacion, INFORME_MD,
    )

    generar_entregables_finales(
        tabla_metricas, tabla_distrital, tabla_factores, tabla_por_grupo,
        casos_extremos, factores,
    )

    logger.info("[OK] R14 completado.")
    return tabla_metricas
