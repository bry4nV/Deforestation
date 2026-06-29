"""
Comparación entre modelos base (O2 — univariable) y extendidos (R11 — multivariable).

Lee los CSVs de O2 ya existentes sin re-ejecutar O2.
Lee los CSVs de R11 desde los directorios de salida configurados en config.py.

Genera:
  comparacion_base_vs_extendido.csv   — hasta 8 filas: 5 O2 + 3 R11
  mejores_01–03_<geocode>.png         — 3 distritos con mejor RMSE en R11
  peores_01–03_<geocode>.png          — 3 distritos con peor RMSE en R11
  comparacion_departamentos.csv       — RMSE/MAE de los 3 modelos R11 por departamento
  heatmap_departamentos.png           — heatmap de la tabla anterior
  boxplot_rmse_distrital_3candidatos.png — distribución de RMSE distrital de los 3 modelos R11
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from O3.config import (
    ANIO_INICIO,
    O2_ARIMA_GLOBAL_CSV,
    O2_CNN_GLOBAL_CSV,
    O2_LSTM_GLOBAL_CSV,
    O2_MLP_GLOBAL_CSV,
    O2_PERSISTENCIA_GLOBAL_CSV,
    R11_CNN_DIR,
    R11_COMPARACION_DIR,
    R11_LSTM_DIR,
    R11_MLP_DIR,
    TAMANIO_ENTRENAMIENTO,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Carga de resultados
# ─────────────────────────────────────────────────────────────────────────────

def _cargar_global(ruta_csv: str, etiqueta: str) -> dict | None:
    if not os.path.exists(ruta_csv):
        logger.warning(f"[SKIP] No encontrado: {ruta_csv}")
        return None
    df = pd.read_csv(ruta_csv)
    if df.empty:
        logger.warning(f"[SKIP] Vacío: {ruta_csv}")
        return None
    row = df.iloc[0]
    return {
        "etiqueta": etiqueta,
        "modelo":   row["modelo"],
        "rmse":     float(row["rmse"]),
        "mae":      float(row["mae"]),
    }


def _cargar_ypred(ruta_npy: str) -> np.ndarray | None:
    if not os.path.exists(ruta_npy):
        return None
    return np.load(ruta_npy)


def cargar_resultados_r11() -> list:
    """Carga métricas globales y predicciones de los 3 modelos R11."""
    items = []
    for nombre, carpeta in [("mlp", R11_MLP_DIR), ("lstm", R11_LSTM_DIR), ("cnn", R11_CNN_DIR)]:
        ruta_global = os.path.join(carpeta, f"{nombre}_final_global.csv")
        ruta_npy    = os.path.join(carpeta, f"{nombre}_final_ypred.npy")
        resultado = _cargar_global(ruta_global, f"R11_{nombre.upper()}")
        if resultado is not None:
            resultado["y_pred"] = _cargar_ypred(ruta_npy)
            items.append(resultado)
    return items


def cargar_resultados_o2() -> list:
    """Carga métricas globales de los 5 modelos de O2 (sin y_pred).

    Incluye ARIMA y Persistencia, no solo los 3 que comparten arquitectura
    con R11 — sin ellos, la comparación nunca incluye al modelo que
    realmente ganó en O2 (ARIMA).
    """
    items = []
    for nombre, ruta in [
        ("persistencia", O2_PERSISTENCIA_GLOBAL_CSV),
        ("arima", O2_ARIMA_GLOBAL_CSV),
        ("mlp",  O2_MLP_GLOBAL_CSV),
        ("lstm", O2_LSTM_GLOBAL_CSV),
        ("cnn",  O2_CNN_GLOBAL_CSV),
    ]:
        resultado = _cargar_global(ruta, f"O2_{nombre.upper()}")
        if resultado is not None:
            resultado["y_pred"] = None
            items.append(resultado)
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Tabla de comparación
# ─────────────────────────────────────────────────────────────────────────────

def exportar_tabla_comparacion(
    resultados_o2: list,
    resultados_r11: list,
    ruta_csv: str,
) -> pd.DataFrame:
    filas = []
    for r in resultados_o2:
        filas.append({"etiqueta": r["etiqueta"], "modelo": r["modelo"],
                      "rmse": r["rmse"], "mae": r["mae"], "conjunto": "base_O2"})
    for r in resultados_r11:
        filas.append({"etiqueta": r["etiqueta"], "modelo": r["modelo"],
                      "rmse": r["rmse"], "mae": r["mae"], "conjunto": "extendido_R11"})

    df = pd.DataFrame(filas).sort_values("rmse").reset_index(drop=True)
    df.to_csv(ruta_csv, index=False)
    logger.info(f"[OK] Tabla de comparación guardada: {ruta_csv}")
    logger.info("\n" + df[["etiqueta", "rmse", "mae", "conjunto"]].to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Gráficos de predicciones por distrito
# ─────────────────────────────────────────────────────────────────────────────

def graficar_predicciones_por_distrito(
    resultados_r11: list,
    panel_original: np.ndarray,
    df_distritos_info: pd.DataFrame,
    tamanio_entrenamiento: int,
    comparacion_dir: str,
    n: int = 5,
    anio_inicio: int = ANIO_INICIO,
) -> None:
    """
    Genera n gráficos de mejores + n de peores distritos usando los modelos R11.

    panel_original: (n_distritos, n_anios, n_canales) en escala ORIGINAL (sin escalar).
    Canal 0 = pct_bosque.
    """
    modelos_con_pred = [r for r in resultados_r11 if r.get("y_pred") is not None]
    if not modelos_con_pred:
        logger.warning("[SKIP] Ningún modelo R11 tiene y_pred disponible.")
        return

    y_true_test = panel_original[:, tamanio_entrenamiento:, 0]  # (n_dist, horizonte)
    horizonte   = y_true_test.shape[1]

    rmse_matrix = np.stack([
        np.sqrt(np.mean((y_true_test - r["y_pred"]) ** 2, axis=1))
        for r in modelos_con_pred
    ], axis=0)

    max_rmse = rmse_matrix.max(axis=0)
    min_rmse = rmse_matrix.min(axis=0)
    idx_best  = np.argsort(max_rmse)[:n]
    idx_worst = np.argsort(min_rmse)[-n:][::-1]

    colores = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    anio_inicio_plot = 2000
    offset  = anio_inicio_plot - anio_inicio
    x_train = list(range(anio_inicio_plot, anio_inicio + tamanio_entrenamiento))
    x_test  = list(range(
        anio_inicio + tamanio_entrenamiento,
        anio_inicio + tamanio_entrenamiento + horizonte,
    ))

    for grupo, indices in [("mejores", idx_best), ("peores", idx_worst)]:
        for rank, i in enumerate(indices, 1):
            info = df_distritos_info.iloc[i]
            dep, dist, geo = info["departamento"], info["distrito"], info["geocode"]

            y_train_i = panel_original[i, offset:tamanio_entrenamiento, 0]
            y_test_i  = y_true_test[i]

            fig = plt.figure(figsize=(10, 4))
            gs  = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.02)
            ax_tr = fig.add_subplot(gs[0])
            ax_te = fig.add_subplot(gs[1], sharey=ax_tr)

            ax_tr.plot(x_train, y_train_i, color="black", linewidth=1.5, label="Real (entrenamiento)")
            ax_tr.set_xlabel("Año")
            ax_tr.set_ylabel("% Cobertura boscosa")
            ax_tr.grid(True, alpha=0.3)

            ax_te.plot(x_test, y_test_i, color="black", linewidth=2, linestyle="--", label="Real (test)")
            for j, r in enumerate(modelos_con_pred):
                y_pred_i  = r["y_pred"][i]
                rmse_mod  = float(np.sqrt(np.mean((y_test_i - y_pred_i) ** 2)))
                etiqueta  = r["etiqueta"]
                ax_te.plot(x_test, y_pred_i, color=colores[j % len(colores)],
                           linewidth=1.5, marker="o", markersize=5,
                           label=f"{etiqueta} (RMSE={rmse_mod:.4f})")

            ax_te.set_xlabel("Año")
            ax_te.grid(True, alpha=0.3)
            ax_tr.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax_te.set_xticks(x_test)
            ax_tr.spines["right"].set_visible(False)
            ax_te.spines["left"].set_visible(False)
            ax_te.tick_params(axis="y", left=False, labelleft=False)

            h_tr, l_tr = ax_tr.get_legend_handles_labels()
            h_te, l_te = ax_te.get_legend_handles_labels()
            ax_te.legend(h_tr + h_te, l_tr + l_te, fontsize=8,
                         loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

            metrica_titulo = (
                f"max RMSE entre modelos R11: {max_rmse[i]:.4f}"
                if grupo == "mejores"
                else f"min RMSE entre modelos R11: {min_rmse[i]:.4f}"
            )
            fig.suptitle(f"{dep} — {dist} (geocode: {geo})\n{metrica_titulo}", y=1.02)
            fig.tight_layout()

            nombre_png = f"{grupo}_{rank:02d}_{geo}.png"
            fig.savefig(os.path.join(comparacion_dir, nombre_png), dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"[OK] {nombre_png}")


# ─────────────────────────────────────────────────────────────────────────────
# Comparación por departamento (3 modelos propios de R11)
# ─────────────────────────────────────────────────────────────────────────────

def comparar_departamentos(
    rutas_departamento: dict, df_distritos_info: pd.DataFrame, comparacion_dir: str,
) -> pd.DataFrame | None:
    """
    Tabla consolidada de RMSE/MAE por departamento para los modelos R11 en
    rutas_departamento (típicamente MLP/LSTM/CNN), número de distritos por
    departamento (para juzgar robustez) y mejor modelo por RMSE y por MAE.
    También genera el heatmap de RMSE, resaltando en negrita el mejor modelo
    de cada fila. Idéntico en diseño a O2/r4_r5/pipeline_comparacion.py.

    rutas_departamento: dict {nombre_modelo: ruta_csv}, en el orden en que se
    quiere que aparezcan en las columnas.
    """
    dfs = []
    for nombre, ruta in rutas_departamento.items():
        if not os.path.exists(ruta):
            logger.info(f"[SKIP] {nombre}: no existe {ruta}")
            continue
        df = pd.read_csv(ruta)[["departamento", "rmse", "mae"]].copy()
        df["modelo"] = nombre
        dfs.append(df)

    if not dfs:
        logger.warning("[WARN] comparar_departamentos: no hay archivos disponibles, se omite.")
        return None

    todos = pd.concat(dfs, ignore_index=True)
    n_distritos = df_distritos_info.groupby("departamento").size().rename("n_distritos")
    orden_modelos = [n for n in rutas_departamento if n in todos["modelo"].unique()]

    pivot_rmse = todos.pivot_table(index="departamento", columns="modelo", values="rmse")[orden_modelos]
    pivot_mae = todos.pivot_table(index="departamento", columns="modelo", values="mae")[orden_modelos]

    consolidado = pd.DataFrame(index=pivot_rmse.index)
    consolidado["n_distritos"] = n_distritos
    for m in orden_modelos:
        consolidado[f"rmse_{m.lower()}"] = pivot_rmse[m]
    for m in orden_modelos:
        consolidado[f"mae_{m.lower()}"] = pivot_mae[m]
    consolidado["mejor_modelo_rmse"] = pivot_rmse.idxmin(axis=1)
    consolidado["mejor_modelo_mae"] = pivot_mae.idxmin(axis=1)
    consolidado = consolidado.reset_index().sort_values("departamento").reset_index(drop=True)

    ruta_consolidado = os.path.join(comparacion_dir, "comparacion_departamentos.csv")
    consolidado.to_csv(ruta_consolidado, index=False)
    logger.info(f"[OK] {ruta_consolidado}")

    fig, ax = plt.subplots(figsize=(1.4 * len(orden_modelos) + 2, 0.45 * len(pivot_rmse) + 2))
    im = ax.imshow(pivot_rmse.values, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(pivot_rmse.columns)))
    ax.set_xticklabels(pivot_rmse.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot_rmse.index)))
    ax.set_yticklabels(pivot_rmse.index)

    for i in range(pivot_rmse.shape[0]):
        fila = pivot_rmse.values[i]
        for j in range(pivot_rmse.shape[1]):
            valor = fila[j]
            es_mejor = valor == np.nanmin(fila)
            ax.text(
                j, i, f"{valor:.4f}",
                ha="center", va="center", fontsize=7.5,
                fontweight="bold" if es_mejor else "normal",
                color="black",
            )

    fig.colorbar(im, ax=ax, label="RMSE")
    fig.tight_layout()

    ruta_heatmap = os.path.join(comparacion_dir, "heatmap_departamentos.png")
    fig.savefig(ruta_heatmap, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[OK] {ruta_heatmap}")

    return consolidado


def graficar_boxplot_validacion_distrital(
    rutas_distrito_dl: dict, comparacion_dir: str, top_n: int = 10,
) -> str:
    """
    Boxplot del RMSE distrital de validación (walk-forward 2020-2024) para
    los 3 modelos R11 (MLP/LSTM/CNN). Resalta en rojo los distritos que
    aparecen, simultáneamente, en el top_n de mayor RMSE de los tres modelos
    (intersección) -- los demás valores por encima del bigote superior de
    cada caja quedan en gris. Idéntico en diseño a O2/r4_r5/pipeline_comparacion.py.
    """
    rmse_por_modelo = {}
    top_n_por_modelo = {}

    for nombre, ruta in rutas_distrito_dl.items():
        df = pd.read_csv(ruta, dtype={"geocode": str})
        df["geocode"] = df["geocode"].str.zfill(6)
        serie_rmse = df.set_index("geocode")["rmse"]
        rmse_por_modelo[nombre] = serie_rmse
        top_n_por_modelo[nombre] = set(serie_rmse.sort_values(ascending=False).head(top_n).index)

    interseccion = set.intersection(*top_n_por_modelo.values())

    nombres = list(rutas_distrito_dl.keys())
    valores = [rmse_por_modelo[n].values for n in nombres]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.boxplot(valores, labels=nombres, showfliers=False, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#a6c8e0"))

    rng = np.random.default_rng(42)
    for j, nombre in enumerate(nombres, start=1):
        serie = rmse_por_modelo[nombre]
        q1, q3 = np.percentile(serie.values, [25, 75])
        bigote_superior = q3 + 1.5 * (q3 - q1)
        atipicos = serie[serie > bigote_superior]

        x_jitter = j + rng.uniform(-0.06, 0.06, size=len(atipicos))
        colores_punto = ["crimson" if geo in interseccion else "gray" for geo in atipicos.index]
        ax.scatter(x_jitter, atipicos.values, c=colores_punto, s=24, zorder=3,
                   edgecolor="white", linewidth=0.6)

    ax.set_ylabel("RMSE distrital de validación (2020-2024)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    leyenda = [
        Line2D([0], [0], marker="o", linestyle="", color="crimson",
               label=f"Coincide en los top-{top_n} de mayor error de los 3 modelos"),
        Line2D([0], [0], marker="o", linestyle="", color="gray",
               label="Atípico solo en este modelo"),
    ]
    ax.legend(
        handles=leyenda,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=8,
        frameon=True,
    )

    fig.tight_layout()

    ruta_fig = os.path.join(comparacion_dir, "boxplot_rmse_distrital_3candidatos.png")
    fig.savefig(ruta_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[OK] {ruta_fig}")

    return ruta_fig


# ─────────────────────────────────────────────────────────────────────────────
# Orquestador
# ─────────────────────────────────────────────────────────────────────────────

def pipeline_comparacion(
    panel_original: np.ndarray,
    df_distritos_info: pd.DataFrame,
    tamanio_entrenamiento: int = TAMANIO_ENTRENAMIENTO,
    anio_inicio: int = ANIO_INICIO,
    comparacion_dir: str = R11_COMPARACION_DIR,
    rutas_departamento: dict = None,
    rutas_distrito_dl: dict = None,
) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("COMPARACIÓN O2 (base) vs R11 (extendido)")
    logger.info("=" * 60)

    resultados_o2  = cargar_resultados_o2()
    resultados_r11 = cargar_resultados_r11()

    if not resultados_r11:
        logger.warning("[PENDIENTE] Ningún modelo R11 tiene resultados de Fase 2. Ejecuta Fase 2 primero.")
        return pd.DataFrame()

    ruta_csv = os.path.join(comparacion_dir, "comparacion_base_vs_extendido.csv")
    df_comp  = exportar_tabla_comparacion(resultados_o2, resultados_r11, ruta_csv)

    logger.info("Generando gráficos de predicciones por distrito (R11)...")
    graficar_predicciones_por_distrito(
        resultados_r11, panel_original, df_distritos_info,
        tamanio_entrenamiento, comparacion_dir,
        n=3, anio_inicio=anio_inicio,
    )

    if rutas_departamento:
        logger.info("Generando comparación por departamento (modelos R11)...")
        comparar_departamentos(rutas_departamento, df_distritos_info, comparacion_dir)
    else:
        logger.info("[SKIP] Comparación por departamento — rutas_departamento no provisto.")

    if rutas_distrito_dl:
        logger.info("Generando boxplot de RMSE distrital (modelos R11)...")
        graficar_boxplot_validacion_distrital(rutas_distrito_dl, comparacion_dir)
    else:
        logger.info("[SKIP] Boxplot distrital R11 — rutas_distrito_dl no provisto.")

    if df_comp.empty:
        return df_comp

    mejor = df_comp.iloc[0]
    logger.info(f"[GANADOR] {mejor['etiqueta']}  RMSE={mejor['rmse']:.4f}  MAE={mejor['mae']:.4f}")
    return df_comp
