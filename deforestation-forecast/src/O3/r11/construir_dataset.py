"""
Construcción de ventanas deslizantes para el panel multivariable.

El panel de entrada tiene shape (n_distritos, n_anios, n_canales).
Canal 0 = pct_bosque (variable objetivo).

Protocolo walk-forward:
    X: ventana [t, t+w)  de los 7 canales  → shape (window_size, 7)
    y: pct_bosque en t+w                   → escalar

Split temporal:
    TRAIN: t + window_size < tamanio_entrenamiento
    TEST:  t + window_size >= tamanio_entrenamiento
"""

import logging

import numpy as np
import torch

from O3.config import TAMANIO_ENTRENAMIENTO

logger = logging.getLogger(__name__)


def crear_ventanas_split(
    panel: np.ndarray,
    window_size: int,
    tamanio_entrenamiento: int = TAMANIO_ENTRENAMIENTO,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    panel : (n_distritos, n_anios, n_canales) — ya escalado

    Returns
    -------
    X_train : (n_train, window_size, n_canales)
    y_train : (n_train, 1)
    X_test  : (n_test,  window_size, n_canales)
    y_test  : (n_test,  1)
    """
    n_distritos, n_anios, n_canales = panel.shape
    variables_entrenamiento, objetivo_entrenamiento = [], []
    variables_prueba, objetivo_prueba = [], []

    for i in range(n_distritos):
        panel_i = panel[i]  # (n_anios, n_canales)

        for t in range(n_anios - window_size):
            ventana = panel_i[t : t + window_size, :]   # (window_size, 7)
            objetivo = panel_i[t + window_size, 0]       # pct_bosque (canal 0)

            if t + window_size < tamanio_entrenamiento:
                variables_entrenamiento.append(ventana)
                objetivo_entrenamiento.append(objetivo)
            else:
                variables_prueba.append(ventana)
                objetivo_prueba.append(objetivo)

    X_train = np.array(variables_entrenamiento, dtype=np.float32)      # (n, w, 7)
    y_train = np.array(objetivo_entrenamiento, dtype=np.float32).reshape(-1, 1)
    X_test  = np.array(variables_prueba,       dtype=np.float32)
    y_test  = np.array(objetivo_prueba,         dtype=np.float32).reshape(-1, 1)

    return X_train, y_train, X_test, y_test


def construir_datasets(
    panel_escalado: np.ndarray,
    ventanas: list[int],
    tamanio_entrenamiento: int = TAMANIO_ENTRENAMIENTO,
) -> dict:
    """
    Construye datasets para todas las ventanas y los convierte a tensores.

    Returns
    -------
    dict  {window_size: {"train": (X_t, y_t), "test": (X_t, y_t)}}
    """
    logger.info("Construyendo datasets de ventanas deslizantes multivariables...")
    datasets = {}

    for w in ventanas:
        X_train, y_train, X_test, y_test = crear_ventanas_split(
            panel_escalado, w, tamanio_entrenamiento
        )

        if X_train.shape[0] == 0:
            logger.warning(f"Ventana w={w} omitida: sin muestras de entrenamiento.")
            continue

        logger.info(
            f"w={w} | X_train={X_train.shape} y_train={y_train.shape} "
            f"X_test={X_test.shape} y_test={y_test.shape}"
        )

        datasets[w] = {
            "train": (
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
            ),
            "test": (
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32),
            ),
        }

    if not datasets:
        raise RuntimeError("No se generaron datasets; revisa ventanas y tamaño de entrenamiento.")

    return datasets
