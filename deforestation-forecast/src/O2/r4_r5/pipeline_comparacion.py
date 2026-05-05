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
    comparacion_dir, n=5, contexto_anios=5, anio_inicio=1985,
):
    """
    Genera n gráficos de mejores + n de peores predicciones por distrito.

    Ranking: MAE promedio entre todos los modelos con y_pred disponible.
    Cada gráfico muestra `contexto_anios` años de entrenamiento + el período de test,
    con las predicciones de cada modelo superpuestas.
    """
    modelos_con_pred = [r for r in resultados if r.get("y_pred") is not None]
    if not modelos_con_pred:
        print("[WARN] Ningún modelo tiene y_pred disponible. No se generan gráficos.")
        return

    y_true_test  = series[:, tamanio_entrenamiento:]   # (n_dist, horizonte)
    horizonte    = y_true_test.shape[1]
    n_distritos  = series.shape[0]

    # MAE por distrito para cada modelo → media entre modelos
    mae_matrix = np.stack([
        np.mean(np.abs(y_true_test - np.asarray(r["y_pred"])), axis=1)
        for r in modelos_con_pred
    ], axis=0)                                          # (n_modelos, n_dist)
    mean_mae = mae_matrix.mean(axis=0)                  # (n_dist,)

    idx_sorted = np.argsort(mean_mae)
    idx_best   = idx_sorted[:n]
    idx_worst  = idx_sorted[-n:][::-1]

    colores = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    x_context = list(range(
        anio_inicio + tamanio_entrenamiento - contexto_anios,
        anio_inicio + tamanio_entrenamiento,
    ))
    x_test = list(range(
        anio_inicio + tamanio_entrenamiento,
        anio_inicio + tamanio_entrenamiento + horizonte,
    ))

    for grupo, indices in [("mejores", idx_best), ("peores", idx_worst)]:
        for rank, i in enumerate(indices, 1):
            info = df_distritos_info.iloc[i]
            dep  = info["departamento"]
            dist = info["distrito"]
            geo  = info["geocode"]

            y_ctx  = series[i, tamanio_entrenamiento - contexto_anios:tamanio_entrenamiento]
            y_test = series[i, tamanio_entrenamiento:]

            fig, ax = plt.subplots(figsize=(8, 4))

            ax.plot(x_context, y_ctx,  color="black", linewidth=2,
                    label="Real (contexto)")
            ax.plot(x_test,    y_test, color="black", linewidth=2,
                    linestyle="--", label="Real (test)")
            ax.axvline(x=anio_inicio + tamanio_entrenamiento - 0.5,
                       color="gray", linestyle=":", linewidth=1, alpha=0.7)

            for j, r in enumerate(modelos_con_pred):
                y_pred_i     = np.asarray(r["y_pred"])[i]
                mae_modelo_i = float(np.mean(np.abs(y_test - y_pred_i)))
                nombre_corto = r["modelo"].split("_")[0]
                ax.plot(x_test, y_pred_i,
                        color=colores[j % len(colores)],
                        linewidth=1.5, marker="o", markersize=4,
                        label=f"{nombre_corto} (MAE={mae_modelo_i:.4f})")

            ax.set_title(
                f"{dep} — {dist} (geocode: {geo})\n"
                f"MAE promedio entre modelos: {mean_mae[i]:.4f}"
            )
            ax.set_xlabel("Año")
            ax.set_ylabel("% Cobertura boscosa")
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            nombre_png = f"{grupo}_{rank:02d}_{geo}.png"
            fig.savefig(os.path.join(comparacion_dir, nombre_png), dpi=150)
            plt.close(fig)
            print(f"[OK] {nombre_png}")


def pipeline_comparacion(
    resultados, series, df_distritos_info, tamanio_entrenamiento,
    comparacion_dir, anio_inicio=1985,
):
    print("\n" + "=" * 60)
    print(" COMPARACIÓN DE MODELOS ")
    print("=" * 60)

    ruta_csv = os.path.join(comparacion_dir, "comparacion_modelos.csv")
    df_comp  = exportar_comparacion(resultados, ruta_csv)

    print("\n[INFO] Generando gráficos (5 mejores + 5 peores predicciones)...")
    graficar_predicciones(
        resultados, series, df_distritos_info, tamanio_entrenamiento,
        comparacion_dir, n=5, anio_inicio=anio_inicio,
    )

    mejor = df_comp.iloc[0]
    print(f"\n[GANADOR] {mejor['modelo']}  RMSE={mejor['rmse']:.4f}  MAE={mejor['mae']:.4f}")
    return df_comp
