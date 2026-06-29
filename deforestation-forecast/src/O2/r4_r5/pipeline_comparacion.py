import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def exportar_comparacion(resultados, ruta_csv):
    df = (
        pd.DataFrame([
            {"modelo": r["modelo"], "rmse": r["rmse"], "mae": r["mae"]}
            for r in resultados
        ])
        .sort_values("rmse")
        .reset_index(drop=True)
    )
    df.to_csv(ruta_csv, index=False)
    print(f"[OK] Comparación CSV: {ruta_csv}")
    print(df.to_string(index=False))
    return df


def graficar_predicciones(
    resultados, series, df_distritos_info, tamanio_entrenamiento,
    comparacion_dir, n=5, anio_inicio=1985,
):
    """
    Genera n gráficos de mejores + n de peores predicciones por distrito.

    Incluye todos los modelos con y_pred disponible (Persistencia, ARIMA, MLP, LSTM).
    Ranking: MAE promedio entre todos los modelos.
    Cada gráfico muestra la serie real completa (entrenamiento + test) con las
    predicciones de todos los modelos superpuestas en el período de test.
    """
    modelos_con_pred = [r for r in resultados if r.get("y_pred") is not None]
    if not modelos_con_pred:
        print("[WARN] Ningún modelo tiene y_pred disponible. No se generan gráficos.")
        return

    y_true_test = series[:, tamanio_entrenamiento:]   # (n_dist, horizonte)
    horizonte   = y_true_test.shape[1]

    # RMSE por modelo y por distrito
    rmse_matrix = np.stack([
        np.sqrt(np.mean((y_true_test - np.asarray(r["y_pred"])) ** 2, axis=1))
        for r in modelos_con_pred
    ], axis=0)                                         # (n_modelos, n_dist)

    # Mejores: max-RMSE más bajo → todos los modelos aciertan (consistentemente fácil)
    # Peores:  min-RMSE más alto → incluso el mejor modelo falla (consistentemente difícil)
    max_rmse = rmse_matrix.max(axis=0)                 # (n_dist,)
    min_rmse = rmse_matrix.min(axis=0)                 # (n_dist,)

    idx_best  = np.argsort(max_rmse)[:n]
    idx_worst = np.argsort(min_rmse)[-n:][::-1]

    colores = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    anio_inicio_plot = 2000
    offset  = anio_inicio_plot - anio_inicio          # años a omitir del inicio
    x_train = list(range(anio_inicio_plot, anio_inicio + tamanio_entrenamiento))
    x_test  = list(range(
        anio_inicio + tamanio_entrenamiento,
        anio_inicio + tamanio_entrenamiento + horizonte,
    ))

    for grupo, indices in [("mejores", idx_best), ("peores", idx_worst)]:
        for rank, i in enumerate(indices, 1):
            info = df_distritos_info.iloc[i]
            dep  = info["departamento"]
            dist = info["distrito"]
            geo  = info["geocode"]

            y_train_i = series[i, offset:tamanio_entrenamiento]
            y_test_i  = series[i, tamanio_entrenamiento:]

            # Panel izquierdo (2000–2019) | Panel derecho (2020–2024), 3:1
            fig = plt.figure(figsize=(10, 4))
            gs  = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.02)
            ax_tr = fig.add_subplot(gs[0])
            ax_te = fig.add_subplot(gs[1], sharey=ax_tr)

            # — Train
            ax_tr.plot(x_train, y_train_i, color="black", linewidth=1.5,
                       label="Real (entrenamiento)")
            ax_tr.set_xlabel("Año")
            ax_tr.set_ylabel("% Cobertura boscosa")
            ax_tr.grid(True, alpha=0.3)

            # — Test + predicciones
            ax_te.plot(x_test, y_test_i, color="black", linewidth=2,
                       linestyle="--", label="Real (test)")
            for j, r in enumerate(modelos_con_pred):
                y_pred_i      = np.asarray(r["y_pred"])[i]
                rmse_modelo_i = float(np.sqrt(np.mean((y_test_i - y_pred_i) ** 2)))
                nombre_corto  = r["modelo"].split("_")[0]
                ax_te.plot(x_test, y_pred_i,
                           color=colores[j % len(colores)],
                           linewidth=1.5, marker="o", markersize=5,
                           label=f"{nombre_corto} (RMSE={rmse_modelo_i:.4f})")
            ax_te.set_xlabel("Año")
            ax_te.grid(True, alpha=0.3)

            # — Ticks enteros en eje X
            ax_tr.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax_te.set_xticks(x_test)

            # — Separación limpia entre paneles (sin marcas diagonales)
            ax_tr.spines["right"].set_visible(False)
            ax_te.spines["left"].set_visible(False)
            ax_te.tick_params(axis="y", left=False, labelleft=False)

            # — Leyenda unificada en el panel de test
            h_tr, l_tr = ax_tr.get_legend_handles_labels()
            h_te, l_te = ax_te.get_legend_handles_labels()
            ax_te.legend(h_tr + h_te, l_tr + l_te, fontsize=8,
                         loc="upper left", bbox_to_anchor=(1.02, 1),
                         borderaxespad=0)

            metrica_titulo = (
                f"max RMSE entre modelos: {max_rmse[i]:.4f}"
                if grupo == "mejores"
                else f"min RMSE entre modelos: {min_rmse[i]:.4f}"
            )
            fig.suptitle(
                f"{dep} — {dist} (geocode: {geo})\n{metrica_titulo}",
                y=1.02,
            )
            fig.tight_layout()

            nombre_png = f"{grupo}_{rank:02d}_{geo}.png"
            fig.savefig(os.path.join(comparacion_dir, nombre_png),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] {nombre_png}")


def graficar_mejora_relativa(df_comp, comparacion_dir):
    """
    Gráfico de barras horizontales: mejora relativa de RMSE frente a Persistencia.
    ARIMA se muestra como referencia estadística clásica y MLP, LSTM y CNN1D
    como modelos de aprendizaje profundo.
    """

    # =========================
    # 1. Obtener RMSE base
    # =========================
    df_persist = df_comp[df_comp["modelo"].str.contains("Persistencia", case=False, na=False)]

    if df_persist.empty:
        raise ValueError("No se encontró un modelo de Persistencia en df_comp.")

    rmse_persistencia = df_persist["rmse"].iloc[0]

    # =========================
    # 2. Calcular mejora relativa
    # =========================
    df_mejora = df_comp[
        ~df_comp["modelo"].str.contains("Persistencia", case=False, na=False)
    ].copy()

    df_mejora["mejora_pct"] = (
        (rmse_persistencia - df_mejora["rmse"]) / rmse_persistencia
    ) * 100

    # Nombre corto del modelo
    df_mejora["modelo_corto"] = df_mejora["modelo"].apply(lambda x: x.split("_")[0])

    # Ordenar de menor a mayor para que el mejor quede arriba en barh
    df_mejora = df_mejora.sort_values("mejora_pct", ascending=True)

    # =========================
    # 3. Colores por familia
    # =========================
    color_arima = "#E69F00"   # naranja
    color_dl = "#4C78A8"      # azul
    color_negativo = "#B0B0B0"

    colores = []
    for _, row in df_mejora.iterrows():
        if row["mejora_pct"] < 0:
            colores.append(color_negativo)
        elif row["modelo_corto"] == "ARIMA":
            colores.append(color_arima)
        else:
            colores.append(color_dl)

    # =========================
    # 4. Crear gráfico
    # =========================
    fig, ax = plt.subplots(figsize=(9, 5.5))

    barras = ax.barh(
        df_mejora["modelo_corto"],
        df_mejora["mejora_pct"],
        color=colores,
        edgecolor="white",
        linewidth=1.0,
        height=0.58,
        zorder=3
    )

    # Línea base de Persistencia
    ax.axvline(
        0,
        color="black",
        linewidth=1.1,
        linestyle="--",
        zorder=2
    )

    # Etiquetas de porcentaje
    max_abs = max(abs(df_mejora["mejora_pct"].min()), abs(df_mejora["mejora_pct"].max()))
    desplazamiento = max_abs * 0.025 if max_abs > 0 else 0.1

    for barra, valor in zip(barras, df_mejora["mejora_pct"]):
        x = barra.get_width()
        y = barra.get_y() + barra.get_height() / 2

        if valor >= 0:
            ax.text(
                x + desplazamiento,
                y,
                f"{valor:.2f}%",
                va="center",
                ha="left",
                fontsize=10,
                fontweight="bold"
            )
        else:
            ax.text(
                x - desplazamiento,
                y,
                f"{valor:.2f}%",
                va="center",
                ha="right",
                fontsize=10,
                fontweight="bold"
            )

    # =========================
    # 5. Estética del gráfico
    # =========================
    ax.set_xlabel(
        "Mejora relativa del RMSE (%)",
        fontsize=11
    )

    ax.set_ylabel("Modelo", fontsize=11)

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.35,
        zorder=0
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(axis="y", length=0)

    # Límites dinámicos
    xmin = df_mejora["mejora_pct"].min()
    xmax = df_mejora["mejora_pct"].max()

    margen_derecho = max_abs * 0.10 if max_abs > 0 else 1

    if xmin >= 0:
        ax.set_xlim(0, xmax + margen_derecho)
    else:
        margen_izquierdo = max_abs * 0.10 if max_abs > 0 else 1
        ax.set_xlim(xmin - margen_izquierdo, xmax + margen_derecho)

    # Leyenda arriba, fuera del gráfico
    ax.legend(
        handles=[
            Patch(facecolor=color_arima, label="ARIMA: referencia estadística clásica"),
            Patch(facecolor=color_dl, label="MLP, LSTM y CNN1D: aprendizaje profundo"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=8.5,
        frameon=True,
        framealpha=0.9
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    # =========================
    # 6. Guardar imagen
    # =========================
    ruta = os.path.join(comparacion_dir, "mejora_relativa_persistencia.png")
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] {ruta}")


def comparar_departamentos(rutas_departamento, df_distritos_info, comparacion_dir):
    """
    Construye una única tabla de comparación departamental para todos los
    modelos en rutas_departamento: RMSE y MAE de cada modelo por departamento,
    número de distritos (para juzgar qué tan robusto es cada resultado), y el
    mejor modelo por RMSE y por MAE. También genera el heatmap de RMSE a
    partir de esa misma tabla, resaltando en negrita el mejor modelo de cada
    fila.

    rutas_departamento: dict {nombre_modelo: ruta_csv}, en el orden en que se
    quiere que aparezcan en las columnas.
    """
    dfs = []
    for nombre, ruta in rutas_departamento.items():
        if not os.path.exists(ruta):
            print(f"[SKIP] {nombre}: no existe {ruta}")
            continue
        df = pd.read_csv(ruta)[["departamento", "rmse", "mae"]].copy()
        df["modelo"] = nombre
        dfs.append(df)

    if not dfs:
        print("[WARN] comparar_departamentos: no hay archivos disponibles, se omite.")
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
    print(f"[OK] {ruta_consolidado}")

    # — Heatmap: filas=departamentos, columnas=modelos, color=RMSE, negrita=mejor de la fila
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
    print(f"[OK] {ruta_heatmap}")

    return consolidado


def graficar_boxplot_validacion_distrital(rutas_distrito_dl, comparacion_dir, top_n=10):
    """
    Boxplot del RMSE distrital de validación (walk-forward 2020-2024) para los
    candidatos de aprendizaje profundo (MLP, LSTM, CNN1D), para el anexo de R7.

    Se resaltan en rojo los distritos que aparecen, simultáneamente, en el
    top_n de mayor RMSE de los tres modelos (intersección) -- los demás
    valores por encima del bigote superior de cada caja quedan en gris.
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
    print(f"[OK] {ruta_fig}")

    return ruta_fig


def pipeline_comparacion(
    resultados, series, df_distritos_info, tamanio_entrenamiento,
    comparacion_dir, rutas_departamento=None, rutas_distrito_dl=None, anio_inicio=1985,
):
    print("\n" + "=" * 60)
    print(" COMPARACIÓN DE MODELOS ")
    print("=" * 60)

    ruta_csv = os.path.join(comparacion_dir, "comparacion_modelos.csv")
    df_comp  = exportar_comparacion(resultados, ruta_csv)

    print("\n[INFO] Generando gráfico de mejora relativa frente a Persistencia...")
    graficar_mejora_relativa(df_comp, comparacion_dir)

    print("\n[INFO] Generando gráficos (3 mejores + 3 peores distritos)...")
    graficar_predicciones(
        resultados, series, df_distritos_info, tamanio_entrenamiento,
        comparacion_dir, n=3, anio_inicio=anio_inicio,
    )

    if rutas_departamento:
        print("\n[INFO] Generando comparación por departamento...")
        comparar_departamentos(rutas_departamento, df_distritos_info, comparacion_dir)
    else:
        print("\n[SKIP] Comparación por departamento — rutas_departamento no provisto.")

    if rutas_distrito_dl:
        print("\n[INFO] Generando boxplot de RMSE distrital (candidatos DL, anexo R7)...")
        graficar_boxplot_validacion_distrital(rutas_distrito_dl, comparacion_dir)
    else:
        print("\n[SKIP] Boxplot distrital DL — rutas_distrito_dl no provisto.")

    mejor = df_comp.iloc[0]
    print(f"\n[GANADOR] {mejor['modelo']}  RMSE={mejor['rmse']:.4f}  MAE={mejor['mae']:.4f}")
    return df_comp
