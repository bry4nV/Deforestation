"""
Carga el panel integrado de entrenamiento y lo pivota a matriz 3D.

Salida principal:
    panel:            np.ndarray (n_distritos, n_anios, n_canales)
    df_distritos_info: pd.DataFrame  (geocode, departamento, distrito)
"""

import logging

import numpy as np
import pandas as pd

from O3.config import COLUMNAS_PREDICTORAS, PANEL_ENTRENAMIENTO_CSV

logger = logging.getLogger(__name__)


def cargar_panel(
    anio_inicio: int = 1985,
    anio_fin: int = 2024,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Lee panel_integrado_entrenamiento.csv, valida columnas y construye la
    matriz (n_distritos, n_anios, n_canales) con canales en el orden de
    COLUMNAS_PREDICTORAS definido en config.py.

    Canal 0 siempre es pct_bosque (variable objetivo).

    Returns
    -------
    panel : np.ndarray de shape (n_distritos, n_anios, n_canales)
    df_distritos_info : DataFrame con columnas geocode, departamento, distrito
    """
    logger.info(f"Cargando panel: {PANEL_ENTRENAMIENTO_CSV}")
    marco = pd.read_csv(PANEL_ENTRENAMIENTO_CSV, dtype={"geocode": str})

    # ── Validar columnas requeridas ──────────────────────────────────────────
    columnas_requeridas = {"geocode", "departamento", "distrito", "anio"} | set(COLUMNAS_PREDICTORAS)
    columnas_faltantes = columnas_requeridas - set(marco.columns)
    if columnas_faltantes:
        raise RuntimeError(
            f"Panel faltante: columnas no encontradas → {sorted(columnas_faltantes)}\n"
            f"Columnas presentes: {sorted(marco.columns.tolist())}"
        )

    # ── Filtrar rango temporal ───────────────────────────────────────────────
    marco = marco[(marco["anio"] >= anio_inicio) & (marco["anio"] <= anio_fin)].copy()

    anios_disponibles = sorted(marco["anio"].unique())
    n_distritos = marco["geocode"].nunique()
    n_anios = len(anios_disponibles)
    n_canales = len(COLUMNAS_PREDICTORAS)

    logger.info(
        f"Panel: {n_distritos} distritos × {n_anios} años ({anios_disponibles[0]}–{anios_disponibles[-1]}) "
        f"× {n_canales} canales"
    )

    # ── Validar completitud ──────────────────────────────────────────────────
    n_filas_esperadas = n_distritos * n_anios
    if len(marco) != n_filas_esperadas:
        raise RuntimeError(
            f"Panel incompleto: se esperaban {n_filas_esperadas} filas "
            f"({n_distritos} distritos × {n_anios} años), se encontraron {len(marco)}."
        )

    # ── Validar sin NaN en columnas de interés ────────────────────────────────
    n_nan = marco[COLUMNAS_PREDICTORAS].isna().sum().sum()
    if n_nan > 0:
        cols_con_nan = marco[COLUMNAS_PREDICTORAS].isna().any()
        raise RuntimeError(
            f"Panel con NaN ({n_nan} valores nulos) en: "
            f"{cols_con_nan[cols_con_nan].index.tolist()}"
        )

    # ── Ordenar: primero por geocode, luego por año ───────────────────────────
    marco = marco.sort_values(["geocode", "anio"]).reset_index(drop=True)

    # ── Construir matriz 3D ───────────────────────────────────────────────────
    # Extraer geocodes en orden canónico (igual al sort anterior)
    geocodes_ordenados = marco["geocode"].unique()  # preserva orden de aparición tras sort

    panel = np.zeros((n_distritos, n_anios, n_canales), dtype=np.float32)

    for idx_d, gc in enumerate(geocodes_ordenados):
        sub = marco[marco["geocode"] == gc].sort_values("anio")
        panel[idx_d, :, :] = sub[COLUMNAS_PREDICTORAS].values

    # ── DataFrame de información por distrito ─────────────────────────────────
    df_distritos_info = (
        marco[["geocode", "departamento", "distrito"]]
        .drop_duplicates(subset=["geocode"])
        .set_index("geocode")
        .loc[geocodes_ordenados]
        .reset_index()
    )

    logger.info(f"Panel construido: shape={panel.shape}, dtype={panel.dtype}")
    return panel, df_distritos_info
