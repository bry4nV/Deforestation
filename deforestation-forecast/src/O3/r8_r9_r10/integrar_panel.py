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
    PANEL_COMPLETO_CSV,
    PANEL_ENTRENAMIENTO_CSV,
    PANEL_GENERALIZACION_CSV,
    PANEL_MODELO_CSV,
    PANEL_REPORTE_ANUAL_CSV,
    PANEL_REPORTE_CSV,
    PENDIENTE_CSV,
    RIOS_CSV,
    RIOS_LAGOS_CSV,
    URBANO_CSV,
    VAR_AGROPECUARIA_DIR,
    VAR_ANP_DIR,
    VAR_CARRETERAS_DIR,
    VAR_ELEVACION_DIR,
    VAR_PENDIENTE_DIR,
    VAR_RIOS_DIR,
    VAR_RIOS_LAGOS_DIR,
    VAR_URBANO_DIR,
)
from O3.r8_r9_r10.metadata_fuentes import NOMBRE_ARCHIVO_POR_CARPETA
from O3.utils import guardar_csv, guardar_metadatos

logger = logging.getLogger(__name__)

# Columnas del panel modelo — las variables que entran al modelo
_COLS_MODELO_VARS = [
    "pct_agropecuario",
    "pct_anp",
    "densidad_carreteras_km_km2",
    "densidad_rios_km_km2",
    "elev_media_m",
    "pendiente_media_deg",
]

# Carpeta de origen y tipo (temporal/estática) de cada variable del panel —
# usado por el reporte de completitud para (a) reportar la base de cálculo
# correcta por tipo y (b) apuntar al `{variable}_metadatos_raw.csv` de
# metadata_fuentes.py (trazabilidad fuente → variable → completitud, ver
# §9.11 de O3_DOCUMENTATION.md). "temporal" = varía por año (merge en
# geocode+anio); "estatica" = un valor por distrito, replicado en sus 40
# filas anuales (merge en geocode solo).
_ORIGEN_VARIABLE = {
    "pct_agropecuario":           (VAR_AGROPECUARIA_DIR, "temporal"),
    "pct_anp":                    (VAR_ANP_DIR, "temporal"),
    "tiene_anp":                  (VAR_ANP_DIR, "temporal"),
    "pct_rios_lagos":             (VAR_RIOS_LAGOS_DIR, "temporal"),
    "pct_urbano":                 (VAR_URBANO_DIR, "temporal"),
    "km_carreteras":              (VAR_CARRETERAS_DIR, "estatica"),
    "area_utm_km2":               (VAR_CARRETERAS_DIR, "estatica"),
    "densidad_carreteras_km_km2": (VAR_CARRETERAS_DIR, "estatica"),
    "km_rios":                    (VAR_RIOS_DIR, "estatica"),
    "densidad_rios_km_km2":       (VAR_RIOS_DIR, "estatica"),
    "elev_media_m":               (VAR_ELEVACION_DIR, "estatica"),
    "elev_mediana_m":             (VAR_ELEVACION_DIR, "estatica"),
    "elev_std_m":                 (VAR_ELEVACION_DIR, "estatica"),
    "elev_min_m":                 (VAR_ELEVACION_DIR, "estatica"),
    "elev_max_m":                 (VAR_ELEVACION_DIR, "estatica"),
    "elev_count_px":              (VAR_ELEVACION_DIR, "estatica"),
    "pendiente_media_deg":        (VAR_PENDIENTE_DIR, "estatica"),
    "pendiente_mediana_deg":      (VAR_PENDIENTE_DIR, "estatica"),
    "pendiente_std_deg":          (VAR_PENDIENTE_DIR, "estatica"),
    "pendiente_min_deg":          (VAR_PENDIENTE_DIR, "estatica"),
    "pendiente_max_deg":          (VAR_PENDIENTE_DIR, "estatica"),
    "pendiente_count_px":         (VAR_PENDIENTE_DIR, "estatica"),
}


def cargar_si_existe(ruta, nombre):
    """Carga un CSV si existe; si no, devuelve None con advertencia."""
    if os.path.exists(ruta):
        return pd.read_csv(ruta, dtype={"geocode": str})
    logger.warning(f"  [FALTA] {nombre}: {ruta}")
    return None


def _generar_reporte_integracion(panel: pd.DataFrame) -> pd.DataFrame:
    """Completitud por variable, con base de cálculo explícita por tipo.

    Para variables "temporal" (merge en geocode+anio) la base es el panel
    completo (n_distritos × 40 años). Para "estatica" (merge en geocode) el
    valor se replica idéntico en las 40 filas anuales de cada distrito, así
    que la base conceptualmente correcta son los distritos, no las filas
    (aunque numéricamente da el mismo % porque la réplica es perfecta — un
    distrito nunca tiene 39 filas no nulas y 1 nula para una variable
    estática). `metadatos_raw_csv` apunta al `{variable}_metadatos_raw.csv`
    de `metadata_fuentes.py` (PASO 0) para trazabilidad fuente → variable →
    completitud sin fusionar ambos reportes (ver §9.11).
    """
    n_filas_total     = len(panel)
    n_distritos_total = panel["geocode"].nunique()
    variables_presentes = [
        c for c in panel.columns if c not in ("geocode", "departamento", "distrito", "anio")
    ]

    filas = []
    for v in variables_presentes:
        origen_dir, tipo = _ORIGEN_VARIABLE.get(v, (None, "desconocido"))
        no_nulos = panel[v].notna()
        n_distritos_con_dato = panel.loc[no_nulos, "geocode"].nunique()

        if tipo == "temporal":
            base_calculo = (
                f"panel completo: {n_distritos_total} distritos x {len(ANIOS)} "
                f"anios = {n_filas_total} filas"
            )
            n_base = n_filas_total
        elif tipo == "estatica":
            base_calculo = (
                f"{n_distritos_total} distritos (valor estatico, replicado en "
                f"las {len(ANIOS)} filas anuales de cada distrito)"
            )
            n_base = n_distritos_total
        else:
            logger.warning(f"  [WARN] Variable sin clasificar en _ORIGEN_VARIABLE: {v}")
            base_calculo = f"panel completo: {n_filas_total} filas (origen no clasificado)"
            n_base = n_filas_total

        ruta_raw = None
        if origen_dir is not None:
            nombre_archivo = NOMBRE_ARCHIVO_POR_CARPETA.get(origen_dir)
            if nombre_archivo:
                ruta_raw = os.path.join(origen_dir, nombre_archivo)

        filas.append({
            "variable":               v,
            "tipo_variable":          tipo,
            "base_calculo":           base_calculo,
            "n_base":                 n_base,
            "completitud_pct":        round(no_nulos.mean() * 100, 2),
            "n_faltantes":            int((~no_nulos).sum()),
            "n_distritos_con_dato":   int(n_distritos_con_dato),
            "n_distritos_total":      int(n_distritos_total),
            "pct_distritos_con_dato": round(n_distritos_con_dato / n_distritos_total * 100, 2)
                                      if n_distritos_total else None,
            "media":                  round(panel[v].mean(skipna=True), 6)
                                      if pd.api.types.is_numeric_dtype(panel[v]) else None,
            "en_panel_modelo":        v in _COLS_MODELO_VARS,
            "metadatos_raw_csv":      ruta_raw,
        })

    reporte = pd.DataFrame(filas)
    guardar_csv(reporte, PANEL_REPORTE_CSV)
    return reporte


def _generar_reporte_completitud_anual(panel: pd.DataFrame) -> pd.DataFrame:
    """Desagrega por año (1985-2024) la completitud de pct_agropecuario/pct_anp.

    Ambas variables nunca quedan NaN por diseño: `construir_agropecuario` y
    `construir_anp` devuelven 0.0 (no NaN) cuando no hay píxeles/área válida
    en un distrito-año (ver construir_agropecuaria.py, construir_anp.py). Por
    eso `completitud_pct` siempre sale 100% aquí y NO es la métrica que
    detectaría una caída real de cobertura (p. ej. nubosidad alta en el
    raster MapBiomas de un año). El único indicador real para eso es
    `pix_total` (píxeles válidos por distrito-año, ya excluyendo la clase "no
    observado" del denominador — ver AGROPECUARIA_CSV / §9.6), agregado por
    año, con una alerta si el promedio de ese año cae más de 2 desviaciones
    estándar por debajo de la media de la serie 1985-2024. `pct_anp` no tiene
    un indicador equivalente porque proviene de capas vectoriales SERNANP
    (acumulación de polígonos), no de píxeles satelitales — se documenta
    explícitamente en `fuente_cobertura` en vez de fabricar una métrica que
    no aplica.
    """
    filas = []
    for variable in ("pct_agropecuario", "pct_anp"):
        if variable not in panel.columns:
            continue
        for anio, grupo in panel.groupby("anio"):
            no_nulos = grupo[variable].notna()
            filas.append({
                "anio":             int(anio),
                "variable":         variable,
                "n_distritos":      len(grupo),
                "n_no_nulos":       int(no_nulos.sum()),
                "completitud_pct":  round(no_nulos.mean() * 100, 2),
                "fuente_cobertura": (
                    "raster MapBiomas - pix_total (pixeles validos)"
                    if variable == "pct_agropecuario"
                    else "vectorial SERNANP - sin concepto de pixeles validos"
                ),
            })
    reporte = pd.DataFrame(filas)

    if os.path.exists(AGROPECUARIA_CSV):
        agro = pd.read_csv(AGROPECUARIA_CSV, dtype={"geocode": str})
        pix_anual = (
            agro.groupby("anio")["pix_total"]
            .agg(pix_total_promedio="mean", pix_total_mediana="median",
                 pix_total_min="min", pix_total_max="max")
            .reset_index()
        )
        umbral = pix_anual["pix_total_promedio"].mean() - 2 * pix_anual["pix_total_promedio"].std()
        pix_anual["alerta_baja_cobertura"] = pix_anual["pix_total_promedio"] < umbral
        pix_anual["pix_total_promedio"] = pix_anual["pix_total_promedio"].round(1)
        pix_anual["pix_total_mediana"]  = pix_anual["pix_total_mediana"].round(1)

        # Split + concat en vez de asignar columnas nuevas por máscara: evita
        # el FutureWarning de pandas al mezclar bool/float/NaN en un .loc[] de
        # columnas que no existían antes del merge en las filas de pct_anp.
        es_agro = reporte["variable"] == "pct_agropecuario"
        reporte = pd.concat([
            reporte[es_agro].merge(pix_anual, on="anio", how="left"),
            reporte[~es_agro],
        ], ignore_index=True).sort_values(["anio", "variable"]).reset_index(drop=True)

    guardar_csv(reporte, PANEL_REPORTE_ANUAL_CSV)
    if "alerta_baja_cobertura" in reporte.columns:
        n_alertas = int((reporte["alerta_baja_cobertura"] == True).sum())  # noqa: E712 (NaN-safe; evita fillna+astype)
    else:
        n_alertas = 0
    logger.info(
        f"  Reporte de completitud anual: {PANEL_REPORTE_ANUAL_CSV} "
        f"({n_alertas} año(s) con alerta de baja cobertura en pct_agropecuario)"
    )
    return reporte


def integrar_panel(distritos_gdf):
    """Integra todas las variables locales en dos paneles por (geocode, anio).

    panel_integrado_completo.csv — todas las variables disponibles (exploración).
    panel_integrado_modelo.csv — solo las variables del modelo:
        pct_agropecuario, pct_anp, densidad_carreteras_km_km2, densidad_rios_km_km2,
        elev_media_m, pendiente_media_deg.

    panel_integrado_entrenamiento.csv / _generalizacion.csv → versión reducida
    a las variables del modelo, dividida por los geocodes de O1.

    Los reportes de completitud (PASO 7) se regeneran siempre, incluso si los
    cuatro paneles ya existen en caché — son baratos de recalcular sobre el
    panel ya guardado y deben reflejar el estado actual, no quedar congelados
    en la primera corrida (misma filosofía que metadata_fuentes.py, ver
    §9.10/§9.11 de O3_DOCUMENTATION.md).
    """
    ya_existe = (
        os.path.exists(PANEL_COMPLETO_CSV)
        and os.path.exists(PANEL_MODELO_CSV)
        and os.path.exists(PANEL_ENTRENAMIENTO_CSV)
        and os.path.exists(PANEL_GENERALIZACION_CSV)
    )

    if ya_existe:
        logger.info(
            "[SKIP] Panel integrado ya existe — cargando CSVs existentes "
            "(los reportes de completitud se regeneran siempre)"
        )
        panel        = pd.read_csv(PANEL_COMPLETO_CSV, dtype={"geocode": str})
        panel_modelo = pd.read_csv(PANEL_MODELO_CSV, dtype={"geocode": str})
        panel_train  = pd.read_csv(PANEL_ENTRENAMIENTO_CSV, dtype={"geocode": str})
        panel_gen    = pd.read_csv(PANEL_GENERALIZACION_CSV, dtype={"geocode": str})
    else:
        panel, panel_modelo, panel_train, panel_gen = _construir_paneles(distritos_gdf)

    # ── 7. Reportes de completitud — SIEMPRE se regeneran ───────────────────
    _generar_reporte_integracion(panel)
    _generar_reporte_completitud_anual(panel)

    guardar_metadatos(
        {
            "n_filas_total":               len(panel),
            "n_columnas_completo":         len(panel.columns),
            "n_columnas_modelo":           len(panel_modelo.columns),
            "n_distritos":                 panel["geocode"].nunique(),
            "n_anios":                     len(ANIOS),
            "anio_inicio":                 ANIOS[0],
            "anio_fin":                    ANIOS[-1],
            "n_distritos_entrenamiento":   panel_train["geocode"].nunique(),
            "n_distritos_generalizacion":  panel_gen["geocode"].nunique(),
            "variables_completo":          str(list(panel.columns)),
            "variables_modelo":            str([
                c for c in panel_modelo.columns
                if c not in ("geocode", "departamento", "distrito", "anio")
            ]),
            "fecha_procesamiento":         datetime.now().isoformat(),
        },
        os.path.join(os.path.dirname(PANEL_COMPLETO_CSV), "panel_metadatos.csv"),
    )

    logger.info(f"[OK] Panel completo guardado:  {PANEL_COMPLETO_CSV}")
    logger.info(f"[OK] Panel modelo guardado:    {PANEL_MODELO_CSV}")
    logger.info(f"     Reporte de completitud:        {PANEL_REPORTE_CSV}")
    logger.info(f"     Reporte de completitud anual:  {PANEL_REPORTE_ANUAL_CSV}")
    return panel


def _construir_paneles(distritos_gdf):
    """Construye desde cero los cuatro paneles (solo se llama si no están en caché)."""
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
    guardar_csv(panel, PANEL_COMPLETO_CSV)
    logger.info(f"  Panel completo:  {len(panel)} filas, {len(panel.columns)} columnas")
    logger.info(f"     Columnas: {list(panel.columns)}")

    # ── 5. Panel modelo — solo variables seleccionadas para el modelo ───────
    _base = ["geocode", "departamento", "distrito", "anio"]
    cols_modelo = _base + [c for c in _COLS_MODELO_VARS if c in panel.columns]
    cols_faltantes = [c for c in _COLS_MODELO_VARS if c not in panel.columns]
    if cols_faltantes:
        logger.warning(f"  [WARN] Variables del panel modelo no disponibles: {cols_faltantes}")

    panel_modelo = panel[cols_modelo].copy()
    guardar_csv(panel_modelo, PANEL_MODELO_CSV)
    logger.info(f"  Panel modelo:    {len(panel_modelo)} filas, {len(panel_modelo.columns)} columnas")
    logger.info(f"     Variables: {cols_modelo[4:]}")

    # ── 6. Split entrenamiento / generalización ─────────────────────────────
    # Base: panels de O1 (pct_bosque, pix_bosque, etc.) + variables O3 (panel modelo).
    # Se usa left-merge sobre (geocode, anio) para no perder ninguna fila de O1.
    df_train_o1 = pd.read_csv(DISTRITOS_ENTRENAMIENTO_CSV, dtype={"geocode": str})
    df_gen_o1   = pd.read_csv(DISTRITOS_GENERALIZACION_CSV, dtype={"geocode": str})

    cols_o3 = ["geocode", "anio"] + [c for c in _COLS_MODELO_VARS if c in panel_modelo.columns]
    o3_para_merge = panel_modelo[cols_o3]

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

    return panel, panel_modelo, panel_train, panel_gen
