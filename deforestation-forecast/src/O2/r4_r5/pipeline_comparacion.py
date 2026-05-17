import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def exportar_comparacion_departamentos(resultados, series, df_distritos_info,
                                       tamanio_entrenamiento, comparacion_dir):
    """
    Para cada departamento calcula RMSE y MAE de forma exacta:
    aplana todos los pares (distrito, paso) del departamento en un solo vector
    antes de calcular la métrica — sin promediar RMSEs individuales.

    Exporta:
      - comparacion_departamentos.csv  (métricas por departamento y modelo)
      - comparacion_departamentos_rmse.png  (barras agrupadas por modelo)
    """
    modelos_con_pred = [r for r in resultados if r.get("y_pred") is not None]
    if not modelos_con_pred:
        return

    y_true_test = series[:, tamanio_entrenamiento:]   # (n_dist, horizonte)

    departamentos = df_distritos_info["departamento"].values
    registros = []
    for dep in sorted(np.unique(departamentos)):
        mask       = departamentos == dep
        n_dist_dep = int(mask.sum())
        y_true_dep = y_true_test[mask].ravel()   # (n_dist_dep * horizonte,)
        fila = {"departamento": dep, "n_distritos": n_dist_dep}
        for r in modelos_con_pred:
            y_pred_dep = np.asarray(r["y_pred"])[mask].ravel()
            nombre     = r["modelo"].split("_")[0]
            fila[f"rmse_{nombre}"] = round(
                float(np.sqrt(np.mean((y_true_dep - y_pred_dep) ** 2))), 6
            )
            fila[f"mae_{nombre}"] = round(
                float(np.mean(np.abs(y_true_dep - y_pred_dep))), 6
            )
        registros.append(fila)

    df_dep = pd.DataFrame(registros)
    ruta_csv = os.path.join(comparacion_dir, "comparacion_departamentos.csv")
    df_dep.to_csv(ruta_csv, index=False)
    print(f"[OK] {ruta_csv}")
    print(df_dep.to_string(index=False))

    # — Gráfico de barras agrupadas (RMSE por departamento, coloreado por modelo)
    rmse_cols   = [c for c in df_dep.columns if c.startswith("rmse_")]
    nombres_mod = [c.replace("rmse_", "") for c in rmse_cols]
    x     = np.arange(len(df_dep))
    width = 0.8 / len(rmse_cols)
    colores = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    fig, ax = plt.subplots(figsize=(max(10, len(df_dep) * 1.2), 5))
    for k, (col, nom) in enumerate(zip(rmse_cols, nombres_mod)):
        offset = (k - len(rmse_cols) / 2 + 0.5) * width
        bars = ax.bar(x + offset, df_dep[col], width,
                      label=nom, color=colores[k % len(colores)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df_dep["departamento"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE por departamento — todos los modelos")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    ruta_png = os.path.join(comparacion_dir, "comparacion_departamentos_rmse.png")
    fig.savefig(ruta_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {ruta_png}")

    return df_dep


def pipeline_comparacion(
    resultados, series, df_distritos_info, tamanio_entrenamiento,
    comparacion_dir, anio_inicio=1985,
):
    print("\n" + "=" * 60)
    print(" COMPARACIÓN DE MODELOS ")
    print("=" * 60)

    ruta_csv = os.path.join(comparacion_dir, "comparacion_modelos.csv")
    df_comp  = exportar_comparacion(resultados, ruta_csv)

    print("\n[INFO] Generando gráficos (mejor + peor distrito)...")
    graficar_predicciones(
        resultados, series, df_distritos_info, tamanio_entrenamiento,
        comparacion_dir, n=1, anio_inicio=anio_inicio,
    )

    print("\n[INFO] Generando métricas y gráficos por departamento...")
    exportar_comparacion_departamentos(
        resultados, series, df_distritos_info, tamanio_entrenamiento,
        comparacion_dir,
    )

    mejor = df_comp.iloc[0]
    print(f"\n[GANADOR] {mejor['modelo']}  RMSE={mejor['rmse']:.4f}  MAE={mejor['mae']:.4f}")
    return df_comp
