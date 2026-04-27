import pandas as pd
import numpy as np


def construir_dataset(ruta_series_entrenamiento):
    """
    Construye dataset de entrenamiento para pronóstico multistep.

    Retorna:
        X_train: (n_distritos, 35, 1)
        y_train: (n_distritos, 5)
    """

    print("[INFO] Cargando series temporales...")

    df = pd.read_csv(ruta_series_entrenamiento)

    # ------------------------------------------------------------------
    # 🔹 Seleccionar columnas relevantes
    # ------------------------------------------------------------------
    df = df[["geocode", "departamento", "distrito", "anio", "pct_bosque"]].copy()

    df_pivot = df.pivot_table(
        index="geocode",
        columns="anio",
        values="pct_bosque"
    )

    df_pivot = df_pivot.sort_index(axis=1)
    df_distritos_info = df[["geocode", "departamento", "distrito"]].drop_duplicates()
    df_distritos_info = df_distritos_info.set_index("geocode").loc[df_pivot.index].reset_index()

    print(f"[INFO] Shape después de pivot: {df_pivot.shape}")

    series = df_pivot.values

    n_distritos, n_anios = series.shape

    print(f"[INFO] Distritos: {n_distritos}")
    print(f"[INFO] Años por serie: {n_anios}")

    if n_anios < 40:
        raise ValueError("Se esperaban al menos 40 años de datos")

    X_train = series[:, :35]   # entrada
    y_train = series[:, 35:40] # salida (5 años futuros)

    X_train = X_train[..., np.newaxis]  # (n, 35, 1)

    print("\n[OK] Dataset de entrenamiento construido")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")

    return X_train, y_train, df_distritos_info