"""
Transformaciones deterministas sobre el panel multivariable de R11,
aplicadas ANTES de ajustar_y_escalar().

A diferencia de StandardScaler, log1p no estima ningún parámetro a partir
de los datos (no tiene media, std, ni lambda) — es una formula fija, por lo
que se aplica al panel completo (1985-2024) en una sola pasada, sin separar
entrenamiento/test, sin riesgo de fuga de información.
"""

import logging

import numpy as np

from O3.config import COLUMNAS_PREDICTORAS, VARIABLES_LOG1P

logger = logging.getLogger(__name__)


def aplicar_transformaciones(panel: np.ndarray) -> np.ndarray:
    """
    Aplica log1p a los canales de VARIABLES_LOG1P (pct_anp,
    densidad_carreteras_km_km2, densidad_rios_km_km2). El resto de canales,
    incluido el canal 0 (pct_bosque, el objetivo), queda sin cambios.

    Debe llamarse antes de ajustar_y_escalar(): log1p no está definido para
    valores < -1, y StandardScaler centra los datos en 0.

    Parameters
    ----------
    panel : (n_distritos, n_anios, n_canales), valores crudos.

    Returns
    -------
    panel_transformado : misma shape, mismo dtype.
    """
    panel_transformado = panel.copy()

    for variable in VARIABLES_LOG1P:
        idx = COLUMNAS_PREDICTORAS.index(variable)
        canal = panel_transformado[:, :, idx]

        if (canal < 0).any():
            raise ValueError(
                f"log1p requiere valores >= 0, pero el canal '{variable}' tiene "
                f"valores negativos (min={canal.min()})."
            )

        panel_transformado[:, :, idx] = np.log1p(canal)
        logger.info(f"  log1p aplicado a canal {idx} ({variable})")

    return panel_transformado
