import pandas as pd
import numpy as np
import torch

def cargar_series(ruta_series):
    """
    Carga y pivotea las series temporales.

    Returns:
        series: np.array (n_distritos, n_anios)
        df_distritos_info: DataFrame
    """

    print("[INFO] Cargando series...")

    df = pd.read_csv(ruta_series)

    df = df[["geocode", "departamento", "distrito", "anio", "pct_bosque"]].copy()

    df_pivot = df.pivot_table(
        index="geocode",
        columns="anio",
        values="pct_bosque"
    )

    df_pivot = df_pivot.sort_index(axis=1)

    series = df_pivot.values

    df_distritos_info = (
        df[["geocode", "departamento", "distrito"]]
        .drop_duplicates()
        .set_index("geocode")
        .loc[df_pivot.index]
        .reset_index()
    )

    print(f"[INFO] Shape series: {series.shape}")

    return series, df_distritos_info


def construir_dataset_estadistico(series, train_size=35, horizon=5):
    """
    Construye dataset para modelos estadísticos.

    Returns:
        X_train_stat: (n_distritos, train_size, 1)
        y_train_stat: (n_distritos, horizon)
    """

    print("\n[INFO] Construyendo dataset estadístico...")

    X_train = series[:, :train_size]
    y_train = series[:, train_size:train_size + horizon]

    X_train = X_train[..., np.newaxis]

    print(f"X_train_stat: {X_train.shape}")
    print(f"y_train_stat: {y_train.shape}")

    return X_train, y_train


def crear_ventanas_split(series, window_size, tamanieo_entrenamiento):

    X_train, y_train = [], []
    X_test, y_test = [], []

    n_distritos, n_anios = series.shape

    for i in range(n_distritos):
        serie = series[i]

        for t in range(n_anios - window_size):

            X_window = serie[t:t + window_size]
            y_target = serie[t + window_size]

            # TRAIN
            if t + window_size < tamanieo_entrenamiento:
                X_train.append(X_window)
                y_train.append(y_target)

            # TEST
            else:
                X_test.append(X_window)
                y_test.append(y_target)

    X_train = np.array(X_train)[..., np.newaxis]
    y_train = np.array(y_train).reshape(-1, 1)

    X_test = np.array(X_test)[..., np.newaxis]
    y_test = np.array(y_test).reshape(-1, 1)

    return X_train, y_train, X_test, y_test


def construir_dataset_dl(series, window_sizes, tamanio_entrenamiento):

    print("\n[INFO] Construyendo datasets Deep Learning...")

    datasets = {}

    for w in window_sizes:

        print(f"\n[INFO] Ventana = {w}")

        X_train, y_train, X_test, y_test = crear_ventanas_split(
            series,
            w,
            tamanio_entrenamiento
        )

        print(f"X_train: {X_train.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_test: {y_test.shape}")

        # convertir a tensores
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        datasets[w] = {
            "train": (X_train, y_train),
            "test": (X_test, y_test)
        }

    return datasets