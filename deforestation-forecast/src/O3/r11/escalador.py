"""
Ajuste y serialización del StandardScaler para el panel multivariable de R11.

El scaler se ajusta exclusivamente sobre el período de entrenamiento (1985–2019)
de los 180 distritos. Se aplica a todo el panel (1985–2024) con los mismos
parámetros, sin re-ajuste.
"""

import json
import logging
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler

from O3.config import (
    COLUMNAS_PREDICTORAS,
    R11_ESCALADOR_DIR,
    R11_ESCALADOR_META,
    R11_ESCALADOR_PKL,
    TAMANIO_ENTRENAMIENTO,
)

import os

logger = logging.getLogger(__name__)


def ajustar_escalador(panel: np.ndarray) -> StandardScaler:
    """
    Ajusta StandardScaler sobre el sub-panel de entrenamiento.

    Parameters
    ----------
    panel : (n_distritos, n_anios, n_canales)

    Returns
    -------
    escalador : StandardScaler ajustado sobre las filas de entrenamiento
    """
    n_distritos, n_anios, n_canales = panel.shape

    panel_train = panel[:, :TAMANIO_ENTRENAMIENTO, :]  # (180, 35, 7)
    filas_train = panel_train.reshape(-1, n_canales)    # (6300, 7)

    escalador = StandardScaler()
    escalador.fit(filas_train)

    logger.info(
        f"Escalador ajustado sobre {len(filas_train)} filas "
        f"(primeros {TAMANIO_ENTRENAMIENTO} años × {n_distritos} distritos)"
    )
    for i, col in enumerate(COLUMNAS_PREDICTORAS):
        logger.info(f"  {col}: media={escalador.mean_[i]:.6f}  std={escalador.scale_[i]:.6f}")

    return escalador


def escalar_panel(panel: np.ndarray, escalador: StandardScaler) -> np.ndarray:
    """
    Aplica el escalador a todo el panel (entrenamiento + test).

    Parameters
    ----------
    panel : (n_distritos, n_anios, n_canales)

    Returns
    -------
    panel_escalado : misma shape, dtype float32
    """
    n_distritos, n_anios, n_canales = panel.shape
    filas = panel.reshape(-1, n_canales)
    filas_escaladas = escalador.transform(filas).astype(np.float32)
    panel_escalado = filas_escaladas.reshape(n_distritos, n_anios, n_canales)
    logger.info(f"Panel escalado: shape={panel_escalado.shape}")
    return panel_escalado


def guardar_escalador(escalador: StandardScaler) -> None:
    os.makedirs(R11_ESCALADOR_DIR, exist_ok=True)

    with open(R11_ESCALADOR_PKL, "wb") as f:
        pickle.dump(escalador, f)
    logger.info(f"[OK] Escalador guardado: {R11_ESCALADOR_PKL}")

    meta = {}
    for i, col in enumerate(COLUMNAS_PREDICTORAS):
        meta[col] = {
            "media":   float(escalador.mean_[i]),
            "std":     float(escalador.scale_[i]),
            "var":     float(escalador.var_[i]),
            "min_obs": float(escalador.data_min_[i]) if hasattr(escalador, "data_min_") else None,
            "max_obs": float(escalador.data_max_[i]) if hasattr(escalador, "data_max_") else None,
        }
    with open(R11_ESCALADOR_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    logger.info(f"[OK] Metadatos guardados: {R11_ESCALADOR_META}")


def cargar_escalador() -> StandardScaler:
    if not os.path.exists(R11_ESCALADOR_PKL):
        raise FileNotFoundError(f"Escalador no encontrado: {R11_ESCALADOR_PKL}")
    with open(R11_ESCALADOR_PKL, "rb") as f:
        escalador = pickle.load(f)
    logger.info(f"[OK] Escalador cargado: {R11_ESCALADOR_PKL}")
    return escalador


def ajustar_y_escalar(panel: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """
    Ajusta el escalador (sobre periodo de entrenamiento), escala el panel completo,
    serializa el scaler a disco y devuelve ambos objetos.

    Si el escalador ya existe en disco, lo carga y lo usa directamente.
    """
    import os
    if os.path.exists(R11_ESCALADOR_PKL):
        logger.info("[SKIP] Escalador ya existe — cargando desde disco.")
        escalador = cargar_escalador()
    else:
        escalador = ajustar_escalador(panel)
        guardar_escalador(escalador)

    panel_escalado = escalar_panel(panel, escalador)
    return panel_escalado, escalador
