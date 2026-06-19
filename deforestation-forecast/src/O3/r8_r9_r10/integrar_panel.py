import logging
import os
from datetime import datetime

import pandas as pd

from O3.config import (
    AGROPECUARIA_CSV,
    ANIOS,
    ANP_CSV,
    CARRETERAS_CSV,
    DISTRITOS_ENTRENAMIENTO_CSV,
    DISTRITOS_GENERALIZACION_CSV,
    ELEVACION_CSV,
    GPKG_COL_DEPARTAMENTO,
    GPKG_COL_DISTRITO,
    GPKG_COL_GEOCODE,
    PANEL_CSV,
    PANEL_ENTRENAMIENTO_CSV,
    PANEL_GENERALIZACION_CSV,
    PANEL_LIGHT_CSV,
    PANEL_REPORTE_CSV,
    PENDIENTE_CSV,
    RIOS_CSV,
    RIOS_LAGOS_CSV,
    URBANO_CSV,
)
from O3.utils import guardar_csv, guardar_metadatos

logger = logging.getLogger(__name__)

# Columnas del panel light — las variables que entran al modelo
_COLS_LIGHT_VARS = [
    "pct_agropecuario",
    "pct_anp",
    "densidad_carreteras_km_km2",
    "densidad_rios_km_km2",
    "elev_media_m",
    "pendiente_media_deg",
]


def cargar_si_existe(ruta, nombre):
    """Carga un CSV si existe; si no, devuelve None con advertencia."""
    if os.path.exists(ruta):
        return pd.read_csv(ruta, dtype={"geocode": str})
    logger.warning(f"  [FALTA] {nombre}: {ruta}")
    return None


def integrar_panel(distritos_gdf):
    """Integra todas las variables locales en dos paneles por (geocode, anio).

    panel_integrado.csv  — todas las variables disponibles (exploración).
    panel_integrado_light.csv — solo las variables del modelo:
        pct_agropecuario, pct_anp, densidad_carreteras_km_km2, densidad_rios_km_km2,
        elev_media_m, pendiente_media_deg.

    panel_integrado_entrenamiento.csv / _generalizacion.csv → versión light
    dividida por los geocodes de O1.
    """
    if (
        os.path.exists(PANEL_CSV)
        and os.path.exists(PANEL_LIGHT_CSV)
        and os.path.exists(PANEL_REPORTE_CSV)
    ):
        logger.info("[SKIP] Panel integrado ya existe — cargando CSV existente")
        return pd.read_csv(PANEL_CSV, dtype={"geocode": str})

    logger.info("Integrando panel de variables locales...")

    # ── 1. Esqueleto base: 200 distritos × 40 años ─────────────────────────
    geocode_info = (
        distritos_gdf[[GPKG_COL_GEOCODE, GPKG_COL_DEPARTAMENTO, GPKG_COL_DISTRITO]]
        .drop_duplicates()
        .rename(columns={
            GPKG_COL_GEOCODE:      "geocode",
            GPKG_COL_DEPARTAMENTO: "departamento",
            GPKG_COL_DISTRITO:     "distrito",
        })
    )
    geocode_info["geocode"] = geocode_info["geocode"].astype(str)

    panel = pd.MultiIndex.from_product(
        [geocode_info["geocode"].tolist(), ANIOS],
        names=["geocode", "anio"],
    ).to_frame(index=False)
    panel = panel.merge(geocode_info, on="geocode", how="left")
    panel = panel[["geocode", "departamento", "distrito", "anio"]]
    logger.info(f"  Esqueleto base: {len(panel)} filas ({len(ANIOS)} años × {len(geocode_info)} distritos)")

    # ── 2. Variables temporales ─────────────────────────────────────────────
    join_cols = ["geocode", "anio"]

    agropecuaria = cargar_si_existe(AGROPECUARIA_CSV, "agropecuaria")
    if agropecuaria is not None:
        panel = panel.merge(
            agropecuaria[join_cols + ["pct_agropecuario"]],
            on=join_cols, how="left",
        )

    anp_df = cargar_si_existe(ANP_CSV, "anp")
    if anp_df is not None:
        cols_anp = [c for c in ["pct_anp", "tiene_anp"] if c in anp_df.columns]
        panel = panel.merge(
            anp_df[join_cols + cols_anp],
            on=join_cols, how="left",
        )

    # Variables de respaldo (exploratorias)
    rios_lagos = cargar_si_existe(RIOS_LAGOS_CSV, "rios_lagos (respaldo)")
    if rios_lagos is not None:
        panel = panel.merge(
            rios_lagos[join_cols + ["pct_rios_lagos"]],
            on=join_cols, how="left",
        )

    urbano = cargar_si_existe(URBANO_CSV, "urbano (respaldo)")
    if urbano is not None:
        panel = panel.merge(
            urbano[join_cols + ["pct_urbano"]],
            on=join_cols, how="left",
        )

    # ── 3. Variables estáticas ──────────────────────────────────────────────
    carreteras = cargar_si_existe(CARRETERAS_CSV, "carreteras")
    if carreteras is not None:
        cols = [c for c in [
            "geocode", "km_carreteras", "area_utm_km2", "densidad_carreteras_km_km2",
        ] if c in carreteras.columns]
        panel = panel.merge(carreteras[cols], on="geocode", how="left")

    rios = cargar_si_existe(RIOS_CSV, "rios")
    if rios is not None:
        cols = [c for c in [
            "geocode", "km_rios", "densidad_rios_km_km2",
        ] if c in rios.columns]
        panel = panel.merge(rios[cols], on="geocode", how="left")

    elevacion = cargar_si_existe(ELEVACION_CSV, "elevacion")
    if elevacion is not None:
        cols = [c for c in [
            "geocode", "elev_media_m", "elev_mediana_m", "elev_std_m",
            "elev_min_m", "elev_max_m", "elev_count_px",
        ] if c in elevacion.columns]
        panel = panel.merge(elevacion[cols], on="geocode", how="left")

    pendiente = cargar_si_existe(PENDIENTE_CSV, "pendiente")
    if pendiente is not None:
        cols = [c for c in [
            "geocode", "pendiente_media_deg", "pendiente_mediana_deg", "pendiente_std_deg",
            "pendiente_min_deg", "pendiente_max_deg", "pendiente_count_px",
        ] if c in pendiente.columns]
        panel = panel.merge(pendiente[cols], on="geocode", how="left")

    # ── 4. Guardar panel completo ───────────────────────────────────────────
    panel = panel.sort_values(["geocode", "anio"]).reset_index(drop=True)
    guardar_csv(panel, PANEL_CSV)
    logger.info(f"  Panel completo:  {len(panel)} filas, {len(panel.columns)} columnas")
    logger.info(f"     Columnas: {list(panel.columns)}")

    # ── 5. Panel light — solo variables del modelo ──────────────────────────
    _base = ["geocode", "departamento", "distrito", "anio"]
    cols_light = _base + [c for c in _COLS_LIGHT_VARS if c in panel.columns]
    cols_faltantes = [c for c in _COLS_LIGHT_VARS if c not in panel.columns]
    if cols_faltantes:
        logger.warning(f"  [WARN] Variables light no disponibles: {cols_faltantes}")

    panel_light = panel[cols_light].copy()
    guardar_csv(panel_light, PANEL_LIGHT_CSV)
    logger.info(f"  Panel light:     {len(panel_light)} filas, {len(panel_light.columns)} columnas")
    logger.info(f"     Variables: {cols_light[4:]}")

    # ── 6. Split entrenamiento / generalización ─────────────────────────────
    # Base: panels de O1 (pct_bosque, pix_bosque, etc.) + variables O3 light.
    # Se usa left-merge sobre (geocode, anio) para no perder ninguna fila de O1.
    df_train_o1 = pd.read_csv(DISTRITOS_ENTRENAMIENTO_CSV, dtype={"geocode": str})
    df_gen_o1   = pd.read_csv(DISTRITOS_GENERALIZACION_CSV, dtype={"geocode": str})

    cols_o3 = ["geocode", "anio"] + [c for c in _COLS_LIGHT_VARS if c in panel_light.columns]
    o3_para_merge = panel_light[cols_o3]

    panel_train = df_train_o1.merge(o3_para_merge, on=["geocode", "anio"], how="left")
    panel_gen   = df_gen_o1.merge(o3_para_merge,   on=["geocode", "anio"], how="left")

    guardar_csv(panel_train, PANEL_ENTRENAMIENTO_CSV)
    guardar_csv(panel_gen,   PANEL_GENERALIZACION_CSV)
    logger.info(
        f"  Entrenamiento:   {panel_train['geocode'].nunique()} distritos, "
        f"{len(panel_train)} filas  —  columnas: {list(panel_train.columns)}"
    )
    logger.info(
        f"  Generalización:  {panel_gen['geocode'].nunique()} distritos, "
        f"{len(panel_gen)} filas"
    )

    # ── 7. Reporte de completitud (panel completo) ──────────────────────────
    variables_presentes = [c for c in panel.columns if c not in ("geocode", "departamento", "distrito", "anio")]
    completitud = {v: round(panel[v].notna().mean() * 100, 2) for v in variables_presentes}

    reporte = pd.DataFrame([
        {
            "variable":        v,
            "completitud_pct": completitud[v],
            "n_faltantes":     int(panel[v].isna().sum()),
            "media":           round(panel[v].mean(skipna=True), 6)
                               if pd.api.types.is_numeric_dtype(panel[v]) else None,
            "en_panel_light":  v in _COLS_LIGHT_VARS,
        }
        for v in variables_presentes
    ])
    guardar_csv(reporte, PANEL_REPORTE_CSV)

    guardar_metadatos(
        {
            "n_filas_total":               len(panel),
            "n_columnas_completo":         len(panel.columns),
            "n_columnas_light":            len(panel_light.columns),
            "n_distritos":                 panel["geocode"].nunique(),
            "n_anios":                     len(ANIOS),
            "anio_inicio":                 ANIOS[0],
            "anio_fin":                    ANIOS[-1],
            "n_distritos_entrenamiento":   panel_train["geocode"].nunique(),
            "n_distritos_generalizacion":  panel_gen["geocode"].nunique(),
            "variables_completo":          str(list(panel.columns)),
            "variables_light":             str(cols_light[4:]),
            "fecha_procesamiento":         datetime.now().isoformat(),
        },
        os.path.join(os.path.dirname(PANEL_CSV), "panel_metadatos.csv"),
    )

    logger.info(f"[OK] Panel integrado guardado: {PANEL_CSV}")
    logger.info(f"[OK] Panel light guardado:     {PANEL_LIGHT_CSV}")
    logger.info(f"     Reporte de completitud:   {PANEL_REPORTE_CSV}")
    return panel
