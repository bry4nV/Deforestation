"""
EDA metodológico del panel integrado de O3.

No es un EDA descriptivo: es un filtro metodológico que decide, con reglas
explícitas y reproducibles, qué variables locales son aptas para modelar
pct_bosque como serie temporal de panel (distrito × año).

Entrada:  data/interim/O3/panel-integrado/panel_integrado_entrenamiento.csv
Salidas:  data/interim/O3/eda/*.csv  (tablas reproducibles)

Uso:
    python -m O3.eda_panel
"""

import logging
import os

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from O3.config import PANEL_ENTRENAMIENTO_CSV, O3_INTERIM_DIR

logger = logging.getLogger(__name__)

EDA_DIR = os.path.join(O3_INTERIM_DIR, "eda")

VARIABLE_OBJETIVO = "pct_bosque"

COLUMNAS_PREDICTORAS = [
    "pct_agropecuario",
    "pct_anp",
    "densidad_carreteras_km_km2",
    "densidad_rios_km_km2",
    "elev_media_m",
    "pendiente_media_deg",
]

# Rangos plausibles por variable (None = sin cota teórica en ese extremo).
# pct_* están almacenados como fracción [0, 1], no como porcentaje [0, 100].
RANGOS_PLAUSIBLES = {
    "pct_bosque":                 (0.0, 1.0),
    "pct_agropecuario":           (0.0, 1.0),
    "pct_anp":                    (0.0, 1.0),
    "densidad_carreteras_km_km2": (0.0, None),
    "densidad_rios_km_km2":       (0.0, None),
    "elev_media_m":               (0.0, 6962.0),   # techo de Perú (Huascarán)
    "pendiente_media_deg":        (0.0, 90.0),
}

# ============================================================
# UMBRALES DEL MOTOR DE REGLAS (sección 8) — ajustables aquí
# ============================================================

ALFA_SIGNIFICANCIA          = 0.05
UMBRAL_R_DEBIL               = 0.10
UMBRAL_R_MODERADO            = 0.30
UMBRAL_R_FUERTE              = 0.50
UMBRAL_REDUNDANCIA_ALTA      = 0.70   # |r| entre predictoras
UMBRAL_REDUNDANCIA_MUY_ALTA  = 0.90
UMBRAL_VIF_ALTO              = 5.0
UMBRAL_VIF_MUY_ALTO          = 10.0
UMBRAL_SESGO_FUERTE          = 1.0
UMBRAL_FRACCION_CEROS_ALTA   = 0.50   # como fracción [0, 1]
UMBRAL_PCT_DISTRITOS_CONSTANTES_ESTATICA = 0.99  # fracción de distritos con var. intra = 0
UMBRAL_PCT_VARIANZA_WITHIN_CUASI_ESTATICA = 0.05


# ============================================================
# CARGA
# ============================================================

def cargar_panel_entrenamiento(anio_hasta: int = 2019) -> pd.DataFrame:
    """
    Carga el panel integrado de entrenamiento y filtra al período indicado.

    Se filtra a anio_hasta=2019 (35 años) para que las decisiones de selección
    de variables no se contaminen con el horizonte de prueba (2020-2024).

    Devuelve el DataFrame filtrado con dtype geocode=str.
    """
    df = pd.read_csv(PANEL_ENTRENAMIENTO_CSV, dtype={"geocode": str})
    df_filtrado = df[df["anio"] <= anio_hasta].copy()

    n_filas = len(df_filtrado)
    n_distritos = df_filtrado["geocode"].nunique()
    anios = sorted(df_filtrado["anio"].unique())

    logger.info(
        f"Panel cargado: {n_filas} filas | {n_distritos} distritos | "
        f"años {anios[0]}–{anios[-1]} (horizonte de prueba excluido del EDA)"
    )
    return df_filtrado


# ============================================================
# HELPERS DE INTERPRETACIÓN (compartidos entre secciones)
# ============================================================

def _clasificar_magnitud_r(r: float) -> str:
    magnitud = abs(r)
    if magnitud >= UMBRAL_R_FUERTE:
        return "FUERTE"
    if magnitud >= UMBRAL_R_MODERADO:
        return "MODERADA"
    if magnitud >= UMBRAL_R_DEBIL:
        return "DEBIL"
    return "NULA"


def _interpretar_correlacion(r: float, p: float, alfa: float = ALFA_SIGNIFICANCIA) -> str:
    direccion = "+" if r > 0 else "-" if r < 0 else ""
    magnitud = _clasificar_magnitud_r(r)
    sig = "sig." if p < alfa else "no sig."
    return f"{direccion}{magnitud} ({sig})"


def _clasificar_sesgo(sk: float) -> str:
    direccion = " a la derecha" if sk > 0 else " a la izquierda" if sk < 0 else ""
    magnitud = abs(sk)
    if magnitud > 1.0:
        return f"MUY SESGADA{direccion}"
    if magnitud > 0.5:
        return f"MODERADAMENTE SESGADA{direccion}"
    return "APROX. SIMETRICA"


def _clasificar_curtosis(ku: float) -> str:
    if ku > 1.0:
        return "LEPTOCURTICA (colas pesadas, valores extremos frecuentes)"
    if ku < -1.0:
        return "PLATICURTICA (distribucion achatada)"
    return "MESOCURTICA (similar a normal)"


def _clasificar_vif(vif: float) -> str:
    if vif >= UMBRAL_VIF_MUY_ALTO:
        return "MUY ALTA (multicolinealidad severa)"
    if vif >= UMBRAL_VIF_ALTO:
        return "ALTA"
    return "BAJA"


# ============================================================
# SECCIÓN 1 — Validación de calidad de datos
# ============================================================

def diagnosticar_calidad_datos(
    df: pd.DataFrame,
    columnas: list = None,
) -> pd.DataFrame:
    """
    Por variable: nulos, valores fuera de rango plausible, si es constante
    a nivel global, proporción de ceros y distritos con la variable = 0
    en todos los años.
    """
    if columnas is None:
        columnas = [VARIABLE_OBJETIVO] + COLUMNAS_PREDICTORAS

    n_distritos = df["geocode"].nunique()
    registros = []

    for variable in columnas:
        serie = df[variable]
        n_nulos = int(serie.isna().sum())

        minimo_valido, maximo_valido = RANGOS_PLAUSIBLES.get(variable, (None, None))
        fuera_de_rango = pd.Series(False, index=serie.index)
        if minimo_valido is not None:
            fuera_de_rango |= serie < minimo_valido
        if maximo_valido is not None:
            fuera_de_rango |= serie > maximo_valido
        fuera_de_rango &= serie.notna()

        n_ceros = int((serie == 0).sum())
        maximo_por_distrito = df.groupby("geocode")[variable].max()
        distritos_siempre_cero = int((maximo_por_distrito == 0).sum())

        registros.append({
            "variable": variable,
            "n_filas": len(serie),
            "n_nulos": n_nulos,
            "pct_nulos": round(n_nulos / len(serie) * 100, 3),
            "rango_plausible": f"[{minimo_valido}, {maximo_valido}]",
            "n_fuera_de_rango": int(fuera_de_rango.sum()),
            "pct_fuera_de_rango": round(fuera_de_rango.mean() * 100, 3),
            "es_constante_global": bool(serie.nunique(dropna=True) <= 1),
            "pct_ceros": round(n_ceros / len(serie) * 100, 3),
            "distritos_siempre_cero": distritos_siempre_cero,
            "pct_distritos_siempre_cero": round(distritos_siempre_cero / n_distritos * 100, 3),
        })

    return pd.DataFrame(registros)


def detectar_outliers_iqr(
    df: pd.DataFrame,
    columnas: list = None,
    factor: float = 1.5,
) -> pd.DataFrame:
    """Outliers transversales por variable usando el criterio de Tukey (IQR)."""
    if columnas is None:
        columnas = [VARIABLE_OBJETIVO] + COLUMNAS_PREDICTORAS

    registros = []
    for variable in columnas:
        q1, q3 = df[variable].quantile([0.25, 0.75])
        iqr = q3 - q1
        lim_inf, lim_sup = q1 - factor * iqr, q3 + factor * iqr
        mascara = (df[variable] < lim_inf) | (df[variable] > lim_sup)

        registros.append({
            "variable": variable,
            "limite_inferior": round(lim_inf, 6),
            "limite_superior": round(lim_sup, 6),
            "n_outliers": int(mascara.sum()),
            "pct_outliers": round(mascara.mean() * 100, 3),
        })

    return pd.DataFrame(registros)


# ============================================================
# SECCIÓN 2 — Análisis descriptivo
# ============================================================

def calcular_estadisticas_descriptivas(df: pd.DataFrame) -> pd.DataFrame:
    """count, mean, std, min, p25, mediana, p75, max para pct_bosque y predictoras."""
    columnas = [VARIABLE_OBJETIVO] + COLUMNAS_PREDICTORAS
    tabla = df[columnas].describe().round(4).T
    tabla.index.name = "variable"
    return tabla


def calcular_asimetria(
    df: pd.DataFrame,
    columnas: list = None,
) -> pd.DataFrame:
    """Skewness y kurtosis por variable, con interpretación breve de forma."""
    if columnas is None:
        columnas = [VARIABLE_OBJETIVO] + COLUMNAS_PREDICTORAS

    registros = []
    for variable in columnas:
        sk = df[variable].skew()
        ku = df[variable].kurtosis()
        forma_sesgo = _clasificar_sesgo(sk)
        forma_curtosis = _clasificar_curtosis(ku)

        registros.append({
            "variable": variable,
            "skewness": round(sk, 3),
            "kurtosis": round(ku, 3),
            "forma_sesgo": forma_sesgo,
            "forma_curtosis": forma_curtosis,
            "interpretacion": f"{forma_sesgo} / {forma_curtosis}",
        })

    return pd.DataFrame(registros)


# ============================================================
# SECCIÓN 3 — Análisis local/territorial
# ============================================================

def calcular_correlacion_objetivo(
    df: pd.DataFrame,
    columnas: list = None,
    variable_objetivo: str = VARIABLE_OBJETIVO,
) -> pd.DataFrame:
    """Pearson y Spearman contemporáneos de cada predictor con pct_bosque."""
    if columnas is None:
        columnas = COLUMNAS_PREDICTORAS

    registros = []
    for variable in columnas:
        r_p, p_p = scipy_stats.pearsonr(df[variable_objetivo], df[variable])
        r_s, p_s = scipy_stats.spearmanr(df[variable_objetivo], df[variable])

        registros.append({
            "variable": variable,
            "r_pearson": round(r_p, 4),
            "p_pearson": float(f"{p_p:.4e}"),
            "r_spearman": round(r_s, 4),
            "p_spearman": float(f"{p_s:.4e}"),
            "interpretacion_pearson": _interpretar_correlacion(r_p, p_p),
            "interpretacion_spearman": _interpretar_correlacion(r_s, p_s),
        })

    tabla = pd.DataFrame(registros).sort_values(
        "r_pearson", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)
    return tabla


def calcular_matriz_correlacion_predictoras(
    df: pd.DataFrame,
    columnas: list = None,
) -> pd.DataFrame:
    """Matriz de correlación de Pearson entre las variables predictoras."""
    if columnas is None:
        columnas = COLUMNAS_PREDICTORAS
    return df[columnas].corr().round(4)


def identificar_pares_colineales(
    matriz_correlacion: pd.DataFrame,
    umbral: float = UMBRAL_REDUNDANCIA_ALTA,
) -> pd.DataFrame:
    """Pares de predictoras con |r| > umbral, clasificados por severidad."""
    columnas = list(matriz_correlacion.columns)
    registros = []

    for i, v1 in enumerate(columnas):
        for j, v2 in enumerate(columnas):
            if j > i:
                r = matriz_correlacion.loc[v1, v2]
                if abs(r) > umbral:
                    severidad = "MUY ALTA" if abs(r) >= UMBRAL_REDUNDANCIA_MUY_ALTA else "ALTA"
                    registros.append({"variable_1": v1, "variable_2": v2, "r": r, "severidad": severidad})

    if not registros:
        return pd.DataFrame(columns=["variable_1", "variable_2", "r", "severidad"])

    tabla = pd.DataFrame(registros).sort_values("r", key=lambda s: s.abs(), ascending=False)
    return tabla.reset_index(drop=True)


def calcular_vif(
    df: pd.DataFrame,
    columnas: list = None,
) -> pd.DataFrame:
    """
    Variance Inflation Factor por predictora. Complementa la matriz de
    correlación: detecta redundancia que surge de combinaciones de 3+
    variables y que un análisis de pares puede pasar por alto.
    """
    if columnas is None:
        columnas = COLUMNAS_PREDICTORAS

    X = add_constant(df[columnas].dropna())
    registros = [
        {"variable": col, "vif": round(variance_inflation_factor(X.values, i), 4)}
        for i, col in enumerate(columnas, start=1)
    ]
    tabla = pd.DataFrame(registros)
    tabla["interpretacion"] = tabla["vif"].apply(_clasificar_vif)
    return tabla.sort_values("vif", ascending=False).reset_index(drop=True)


# ============================================================
# SECCIÓN 4 — Análisis temporal de pct_bosque
# ============================================================

def calcular_evolucion_anual(
    df: pd.DataFrame,
    variable: str = VARIABLE_OBJETIVO,
) -> pd.DataFrame:
    """Media, mediana, dispersión y rango de pct_bosque por año."""
    tabla = (
        df.groupby("anio")[variable]
        .agg(n_distritos="count", media="mean", mediana="median", std="std", minimo="min", maximo="max")
        .round(4)
        .reset_index()
    )
    return tabla


def agregar_delta_pct_bosque(
    df: pd.DataFrame,
    variable: str = VARIABLE_OBJETIVO,
) -> pd.DataFrame:
    """Agrega delta_pct_bosque(t) = pct_bosque(t) - pct_bosque(t-1) por distrito."""
    df_ordenado = df.sort_values(["geocode", "anio"]).copy()
    df_ordenado["delta_pct_bosque"] = df_ordenado.groupby("geocode")[variable].diff()
    return df_ordenado


def resumir_delta_anual(df_con_delta: pd.DataFrame) -> pd.DataFrame:
    """Por año: magnitud del cambio y proporción de distritos en pérdida/ganancia/sin cambio."""
    registros = []
    for anio, grupo in df_con_delta.groupby("anio"):
        delta = grupo["delta_pct_bosque"].dropna()
        if len(delta) == 0:
            continue
        registros.append({
            "anio": anio,
            "n_distritos": len(delta),
            "media_delta": round(delta.mean(), 5),
            "mediana_delta": round(delta.median(), 5),
            "std_delta": round(delta.std(), 5),
            "pct_distritos_perdida": round((delta < 0).mean() * 100, 2),
            "pct_distritos_ganancia": round((delta > 0).mean() * 100, 2),
            "pct_distritos_sin_cambio": round((delta == 0).mean() * 100, 2),
        })
    return pd.DataFrame(registros)


def detectar_cambios_extremos(
    df_con_delta: pd.DataFrame,
    factor: float = 1.5,
) -> pd.DataFrame:
    """Distrito-año con delta_pct_bosque fuera del rango de Tukey (IQR), ordenados por magnitud."""
    delta_valido = df_con_delta["delta_pct_bosque"].dropna()
    q1, q3 = delta_valido.quantile([0.25, 0.75])
    iqr = q3 - q1
    lim_inf, lim_sup = q1 - factor * iqr, q3 + factor * iqr

    mascara = (df_con_delta["delta_pct_bosque"] < lim_inf) | (df_con_delta["delta_pct_bosque"] > lim_sup)
    tabla = df_con_delta.loc[mascara, ["geocode", "distrito", "anio", "delta_pct_bosque"]].copy()
    tabla["tipo_cambio"] = np.where(tabla["delta_pct_bosque"] < 0, "PERDIDA EXTREMA", "GANANCIA EXTREMA")
    tabla["limite_inferior"] = round(lim_inf, 5)
    tabla["limite_superior"] = round(lim_sup, 5)

    tabla = tabla.sort_values("delta_pct_bosque", key=lambda s: s.abs(), ascending=False)
    return tabla.reset_index(drop=True)


# ============================================================
# SECCIÓN 5 — Análisis de dependencia temporal
# ============================================================

def calcular_autocorrelacion_pct_bosque(
    df: pd.DataFrame,
    variable: str = VARIABLE_OBJETIVO,
    lags: range = range(1, 6),
) -> pd.DataFrame:
    """Correlación de Pearson entre pct_bosque(t) y pct_bosque(t-lag), agrupado por distrito."""
    df_ordenado = df.sort_values(["geocode", "anio"]).copy()
    registros = []

    for lag in lags:
        col_lag = f"_lag{lag}"
        df_ordenado[col_lag] = df_ordenado.groupby("geocode")[variable].shift(lag)
        df_valido = df_ordenado.dropna(subset=[variable, col_lag])

        r, p = scipy_stats.pearsonr(df_valido[variable], df_valido[col_lag])
        registros.append({
            "lag": lag,
            "r": round(r, 4),
            "p_valor": float(f"{p:.4e}"),
            "interpretacion": _interpretar_correlacion(r, p),
        })
        df_ordenado.drop(columns=[col_lag], inplace=True)

    return pd.DataFrame(registros)


# ============================================================
# SECCIÓN 7 — Clasificación empírica dinámica / estática
# ============================================================
# (Se calcula antes de la sección 6 en la ejecución porque el tratamiento
#  de cada variable en el punto 6 depende de esta clasificación.)

def clasificar_variables_dinamicas_estaticas(
    df: pd.DataFrame,
    columnas: list = None,
) -> pd.DataFrame:
    """
    Descompone la varianza de cada predictora en intra-distrito (within) y
    entre-distritos (between) -- descomposición clásica de panel balanceado --
    y clasifica la variable como ESTATICA, CUASI-ESTATICA o DINAMICA sin
    asumir la categoría de antemano.
    """
    if columnas is None:
        columnas = COLUMNAS_PREDICTORAS

    registros = []
    for variable in columnas:
        por_distrito = df.groupby("geocode")[variable]
        varianza_intra = por_distrito.var(ddof=0).mean()
        varianza_entre = por_distrito.mean().var(ddof=0)
        varianza_total = varianza_intra + varianza_entre
        pct_within = varianza_intra / varianza_total if varianza_total > 0 else 0.0
        pct_distritos_constantes = (por_distrito.std(ddof=0).fillna(0) < 1e-9).mean()

        if pct_distritos_constantes >= UMBRAL_PCT_DISTRITOS_CONSTANTES_ESTATICA:
            clasificacion = "ESTATICA"
            interpretacion = (
                "Constante en practicamente todos los distritos durante todo el periodo. "
                "No aporta senal temporal dentro de un distrito; tratar como atributo de "
                "contexto territorial (variable transversal / efecto fijo). No usar con "
                "rezagos ni leads."
            )
        elif pct_within < UMBRAL_PCT_VARIANZA_WITHIN_CUASI_ESTATICA:
            clasificacion = "CUASI-ESTATICA"
            interpretacion = (
                "Cambia en pocos distritos y pocas veces durante el periodo (ej. creacion "
                "de un area protegida). La mayor parte de su varianza es entre distritos, "
                "no dentro de cada distrito. Evaluar con el enfoque temporal (rezagos/leads) "
                "y con el transversal, e interpretar el primero con cautela por la escasa "
                "variacion intra-distrito disponible."
            )
        else:
            clasificacion = "DINAMICA"
            interpretacion = (
                "Varia ano a ano dentro de cada distrito. Apta para analisis temporal "
                "(rezagos, leads, delta) ademas del transversal."
            )

        registros.append({
            "variable": variable,
            "varianza_intra_within": round(varianza_intra, 8),
            "varianza_entre_between": round(varianza_entre, 8),
            "pct_varianza_within": round(pct_within * 100, 3),
            "pct_distritos_constantes": round(pct_distritos_constantes * 100, 3),
            "clasificacion": clasificacion,
            "interpretacion_metodologica": interpretacion,
        })

    return pd.DataFrame(registros)


# ============================================================
# SECCIÓN 6 — Relación predictiva con el futuro
# ============================================================

def calcular_relacion_futura_dinamicas(
    df: pd.DataFrame,
    variables: list,
    variable_objetivo: str = VARIABLE_OBJETIVO,
) -> pd.DataFrame:
    """
    Para variables dinámicas/cuasi-estáticas: correlación de var(t) con
    pct_bosque(t+1) y con delta_pct_bosque(t+1), agrupado por distrito.
    """
    if not variables:
        return pd.DataFrame()

    df_ordenado = df.sort_values(["geocode", "anio"]).copy()
    df_ordenado["_objetivo_lead"] = df_ordenado.groupby("geocode")[variable_objetivo].shift(-1)
    df_ordenado["_delta_lead"] = df_ordenado["_objetivo_lead"] - df_ordenado[variable_objetivo]

    registros = []
    for variable in variables:
        df_valido = df_ordenado.dropna(subset=[variable, "_objetivo_lead", "_delta_lead"])

        r_p_nivel, p_p_nivel = scipy_stats.pearsonr(df_valido[variable], df_valido["_objetivo_lead"])
        r_s_nivel, p_s_nivel = scipy_stats.spearmanr(df_valido[variable], df_valido["_objetivo_lead"])
        r_p_delta, p_p_delta = scipy_stats.pearsonr(df_valido[variable], df_valido["_delta_lead"])
        r_s_delta, p_s_delta = scipy_stats.spearmanr(df_valido[variable], df_valido["_delta_lead"])

        registros.append({
            "variable": variable,
            "r_pearson_vs_nivel_t1": round(r_p_nivel, 4),
            "p_pearson_vs_nivel_t1": float(f"{p_p_nivel:.4e}"),
            "r_spearman_vs_nivel_t1": round(r_s_nivel, 4),
            "p_spearman_vs_nivel_t1": float(f"{p_s_nivel:.4e}"),
            "r_pearson_vs_delta_t1": round(r_p_delta, 4),
            "p_pearson_vs_delta_t1": float(f"{p_p_delta:.4e}"),
            "r_spearman_vs_delta_t1": round(r_s_delta, 4),
            "p_spearman_vs_delta_t1": float(f"{p_s_delta:.4e}"),
            "interpretacion": _interpretar_correlacion(r_p_delta, p_p_delta) + " sobre delta_pct_bosque(t+1)",
        })

    return pd.DataFrame(registros)


def calcular_relacion_transversal_estaticas(
    df: pd.DataFrame,
    variables: list,
    variable_objetivo: str = VARIABLE_OBJETIVO,
) -> pd.DataFrame:
    """
    Para variables estáticas: una correlación lead/lag es matemáticamente
    idéntica a la contemporánea (el valor no cambia), así que no aporta
    información. En su lugar se evalúa de forma transversal -- un registro
    por distrito -- contra el nivel medio de pct_bosque y contra su tasa de
    cambio promedio en el periodo, evitando además la pseudo-replicación de
    correlacionar un valor constante contra cada año del panel.
    """
    if not variables:
        return pd.DataFrame()

    base = df.groupby("geocode").agg(
        objetivo_medio=(variable_objetivo, "mean"),
        objetivo_primero=(variable_objetivo, "first"),
        objetivo_ultimo=(variable_objetivo, "last"),
        n_anios=(variable_objetivo, "count"),
    )
    for variable in variables:
        base[variable] = df.groupby("geocode")[variable].first()
    base["tasa_cambio_anual_promedio"] = (
        (base["objetivo_ultimo"] - base["objetivo_primero"]) / (base["n_anios"] - 1)
    )

    registros = []
    for variable in variables:
        r_p_nivel, p_p_nivel = scipy_stats.pearsonr(base[variable], base["objetivo_medio"])
        r_s_nivel, p_s_nivel = scipy_stats.spearmanr(base[variable], base["objetivo_medio"])
        r_p_tasa, p_p_tasa = scipy_stats.pearsonr(base[variable], base["tasa_cambio_anual_promedio"])
        r_s_tasa, p_s_tasa = scipy_stats.spearmanr(base[variable], base["tasa_cambio_anual_promedio"])

        registros.append({
            "variable": variable,
            "n_distritos": len(base),
            "r_pearson_vs_pct_bosque_medio": round(r_p_nivel, 4),
            "p_pearson_vs_pct_bosque_medio": float(f"{p_p_nivel:.4e}"),
            "r_spearman_vs_pct_bosque_medio": round(r_s_nivel, 4),
            "p_spearman_vs_pct_bosque_medio": float(f"{p_s_nivel:.4e}"),
            "r_pearson_vs_tasa_cambio": round(r_p_tasa, 4),
            "p_pearson_vs_tasa_cambio": float(f"{p_p_tasa:.4e}"),
            "r_spearman_vs_tasa_cambio": round(r_s_tasa, 4),
            "p_spearman_vs_tasa_cambio": float(f"{p_s_tasa:.4e}"),
            "interpretacion": _interpretar_correlacion(r_p_nivel, p_p_nivel) + " (transversal, n=distritos)",
        })

    return pd.DataFrame(registros)


# ============================================================
# SECCIÓN 8 — Tabla final de decisión por variable
# ============================================================

def _decidir_variable(r: dict) -> tuple:
    """Motor de reglas determinista. Devuelve (decision, justificacion)."""

    if r["pct_nulos"] > 0 or r["pct_fuera_de_rango"] > 0:
        return "EXCLUIR", (
            f"Calidad de datos insuficiente: {r['pct_nulos']}% nulos, "
            f"{r['pct_fuera_de_rango']}% fuera de rango plausible."
        )

    if r["es_constante_global"]:
        return "EXCLUIR", (
            "Variable constante en todo el panel (varianza total = 0): no discrimina "
            "ni entre distritos ni en el tiempo."
        )

    redundancia_severa = (
        abs(r["redundancia_r"]) >= UMBRAL_REDUNDANCIA_MUY_ALTA or r["vif"] >= UMBRAL_VIF_MUY_ALTO
    )
    if redundancia_severa:
        return "TRANSFORMAR", (
            f"Redundancia muy alta con '{r['redundancia_con']}' (r={r['redundancia_r']}, "
            f"VIF={r['vif']}). Combinar ambas, conservar solo una, o aplicar reducción de "
            "dimensionalidad antes de modelar."
        )

    if r["clasificacion_temporal"] == "ESTATICA":
        tiene_senal = r["sig_contemporanea"] or r["sig_futuro"]
        redundancia_moderada = abs(r["redundancia_r"]) >= UMBRAL_REDUNDANCIA_ALTA
        nota_redundancia = (
            f" Además, tiene redundancia alta con '{r['redundancia_con']}' "
            f"(r={r['redundancia_r']}); preferir solo una de las dos en modelos lineales."
            if redundancia_moderada else ""
        )
        if tiene_senal:
            return "MANTENER CON CAUTELA", (
                f"Atributo territorial fijo (varianza intra-distrito = 0). Relación "
                f"transversal r={r['r_contemporanea']} "
                f"({'sig.' if r['sig_contemporanea'] else 'no sig.'}). Útil como covariable "
                "de contexto / efecto fijo entre distritos, no como insumo de dinámica "
                f"temporal (no usar con rezagos/leads).{nota_redundancia}"
            )
        return "EXCLUIR", (
            "Variable estática sin relación transversal significativa con pct_bosque: no "
            "aporta como covariable de contexto ni como insumo temporal."
        )

    relacion_util = r["sig_contemporanea"] or r["sig_futuro"]
    magnitud_relevante = r["r_contemporanea"] >= UMBRAL_R_DEBIL or r["r_futuro"] >= UMBRAL_R_DEBIL
    if not (relacion_util and magnitud_relevante):
        return "EXCLUIR", (
            f"Relación débil/no significativa con pct_bosque (r={r['r_contemporanea']}) y con "
            f"su evolución futura (r={r['r_futuro']}). No justifica incluirla."
        )

    necesita_transformacion = (
        abs(r["skewness"]) > UMBRAL_SESGO_FUERTE or r["pct_ceros"] >= UMBRAL_FRACCION_CEROS_ALTA * 100
    )
    if necesita_transformacion:
        return "TRANSFORMAR", (
            f"Relación útil con pct_bosque (r_contemporanea={r['r_contemporanea']}, "
            f"r_futuro={r['r_futuro']}) pero distribución problemática (skew={r['skewness']}, "
            f"{r['pct_ceros']}% ceros). Aplicar log1p, winsorizing o un indicador binario "
            "de presencia/ausencia antes de usarla."
        )

    redundancia_moderada = (
        abs(r["redundancia_r"]) >= UMBRAL_REDUNDANCIA_ALTA or r["vif"] >= UMBRAL_VIF_ALTO
    )
    if redundancia_moderada:
        return "MANTENER CON CAUTELA", (
            f"Relación útil con pct_bosque pero redundancia moderada-alta con "
            f"'{r['redundancia_con']}' (r={r['redundancia_r']}, VIF={r['vif']}). Vigilar "
            "estabilidad de coeficientes en modelos lineales."
        )

    return "MANTENER", (
        f"Calidad de datos OK, variación suficiente, relación "
        f"{'contemporánea' if r['sig_contemporanea'] else 'futura'} significativa "
        f"(r_contemporanea={r['r_contemporanea']}, r_futuro={r['r_futuro']}), sin redundancia "
        "relevante."
    )


def construir_tabla_decision_final(
    tabla_calidad: pd.DataFrame,
    tabla_outliers: pd.DataFrame,
    tabla_asimetria: pd.DataFrame,
    tabla_correlacion_objetivo: pd.DataFrame,
    tabla_vif: pd.DataFrame,
    pares_colineales: pd.DataFrame,
    tabla_clasificacion: pd.DataFrame,
    tabla_futura_dinamicas: pd.DataFrame,
    tabla_transversal_estaticas: pd.DataFrame,
) -> pd.DataFrame:
    """Combina todas las secciones anteriores en una fila de decisión por variable."""
    calidad_idx = tabla_calidad.set_index("variable")
    outliers_idx = tabla_outliers.set_index("variable")
    asimetria_idx = tabla_asimetria.set_index("variable")
    corr_idx = tabla_correlacion_objetivo.set_index("variable")
    vif_idx = tabla_vif.set_index("variable")
    clasif_idx = tabla_clasificacion.set_index("variable")
    futura_idx = tabla_futura_dinamicas.set_index("variable") if len(tabla_futura_dinamicas) else None
    transversal_idx = tabla_transversal_estaticas.set_index("variable") if len(tabla_transversal_estaticas) else None

    filas = []
    for variable in COLUMNAS_PREDICTORAS:
        calidad = calidad_idx.loc[variable]
        outliers = outliers_idx.loc[variable]
        asimetria = asimetria_idx.loc[variable]
        corr_obj = corr_idx.loc[variable]
        vif = vif_idx.loc[variable, "vif"]
        clasif = clasif_idx.loc[variable]

        pares_var = pares_colineales[
            (pares_colineales["variable_1"] == variable) | (pares_colineales["variable_2"] == variable)
        ]
        if len(pares_var) > 0:
            fila_max = pares_var.loc[pares_var["r"].abs().idxmax()]
            redundancia_con = fila_max["variable_2"] if fila_max["variable_1"] == variable else fila_max["variable_1"]
            redundancia_r = fila_max["r"]
        else:
            redundancia_con, redundancia_r = None, 0.0

        es_dinamica_o_cuasi = clasif["clasificacion"] in ("DINAMICA", "CUASI-ESTATICA")
        if es_dinamica_o_cuasi and futura_idx is not None and variable in futura_idx.index:
            fut = futura_idx.loc[variable]
            r_futuro = max(abs(fut["r_pearson_vs_nivel_t1"]), abs(fut["r_pearson_vs_delta_t1"]))
            p_futuro = min(fut["p_pearson_vs_nivel_t1"], fut["p_pearson_vs_delta_t1"])
        else:
            fut = transversal_idx.loc[variable]
            r_futuro = max(abs(fut["r_pearson_vs_pct_bosque_medio"]), abs(fut["r_pearson_vs_tasa_cambio"]))
            p_futuro = min(fut["p_pearson_vs_pct_bosque_medio"], fut["p_pearson_vs_tasa_cambio"])

        r_contemporanea = max(abs(corr_obj["r_pearson"]), abs(corr_obj["r_spearman"]))
        p_contemporanea = min(corr_obj["p_pearson"], corr_obj["p_spearman"])

        resumen = {
            "variable": variable,
            "pct_nulos": calidad["pct_nulos"],
            "pct_fuera_de_rango": calidad["pct_fuera_de_rango"],
            "pct_outliers_iqr": outliers["pct_outliers"],
            "es_constante_global": calidad["es_constante_global"],
            "clasificacion_temporal": clasif["clasificacion"],
            "pct_varianza_within": clasif["pct_varianza_within"],
            "r_contemporanea": round(float(r_contemporanea), 4),
            "sig_contemporanea": bool(p_contemporanea < ALFA_SIGNIFICANCIA),
            "r_futuro": round(float(r_futuro), 4),
            "sig_futuro": bool(p_futuro < ALFA_SIGNIFICANCIA),
            "vif": vif,
            "redundancia_r": round(float(redundancia_r), 4),
            "redundancia_con": redundancia_con,
            "skewness": asimetria["skewness"],
            "pct_ceros": calidad["pct_ceros"],
        }
        decision, justificacion = _decidir_variable(resumen)
        resumen["decision_final"] = decision
        resumen["justificacion"] = justificacion
        filas.append(resumen)

    return pd.DataFrame(filas)


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    os.makedirs(EDA_DIR, exist_ok=True)

    df_train = cargar_panel_entrenamiento(anio_hasta=2019)

    # --- Sección 1: Validación de calidad de datos ---
    logger.info("=" * 60)
    logger.info("SECCIÓN 1 — Validación de calidad de datos")
    tabla_calidad = diagnosticar_calidad_datos(df_train)
    logger.info("\n" + tabla_calidad.to_string(index=False))
    tabla_calidad.to_csv(os.path.join(EDA_DIR, "01_calidad_datos.csv"), index=False)

    tabla_outliers = detectar_outliers_iqr(df_train)
    logger.info("\n  Outliers transversales (IQR):\n" + tabla_outliers.to_string(index=False))
    tabla_outliers.to_csv(os.path.join(EDA_DIR, "01b_outliers_iqr.csv"), index=False)
    logger.info("[OK] Sección 1 guardada")

    # --- Sección 2: Análisis descriptivo ---
    logger.info("=" * 60)
    logger.info("SECCIÓN 2 — Análisis descriptivo")
    tabla_desc = calcular_estadisticas_descriptivas(df_train)
    logger.info("\n" + tabla_desc.to_string())
    tabla_desc.to_csv(os.path.join(EDA_DIR, "02_estadisticas_descriptivas.csv"))

    tabla_asimetria = calcular_asimetria(df_train)
    logger.info("\n" + tabla_asimetria.to_string(index=False))
    tabla_asimetria.to_csv(os.path.join(EDA_DIR, "02b_asimetria_kurtosis.csv"), index=False)
    logger.info("[OK] Sección 2 guardada")

    # --- Sección 3: Análisis local/territorial ---
    logger.info("=" * 60)
    logger.info("SECCIÓN 3 — Relación con pct_bosque y redundancia entre predictoras")
    tabla_correlacion_objetivo = calcular_correlacion_objetivo(df_train)
    logger.info("\n" + tabla_correlacion_objetivo.to_string(index=False))
    tabla_correlacion_objetivo.to_csv(
        os.path.join(EDA_DIR, "03_correlacion_objetivo.csv"), index=False
    )

    matriz_corr = calcular_matriz_correlacion_predictoras(df_train)
    logger.info("\n  Matriz de correlación entre predictoras:\n" + matriz_corr.to_string())
    matriz_corr.to_csv(os.path.join(EDA_DIR, "03b_matriz_correlacion_predictoras.csv"))

    pares_colineales = identificar_pares_colineales(matriz_corr)
    logger.info(f"\n  Pares con |r| > {UMBRAL_REDUNDANCIA_ALTA}:\n" + pares_colineales.to_string(index=False))
    pares_colineales.to_csv(os.path.join(EDA_DIR, "03c_pares_colineales.csv"), index=False)

    tabla_vif = calcular_vif(df_train)
    logger.info("\n  VIF:\n" + tabla_vif.to_string(index=False))
    tabla_vif.to_csv(os.path.join(EDA_DIR, "03d_vif_predictoras.csv"), index=False)
    logger.info("[OK] Sección 3 guardada")

    # --- Sección 4: Análisis temporal de pct_bosque ---
    logger.info("=" * 60)
    logger.info("SECCIÓN 4 — Evolución temporal de pct_bosque")
    tabla_evolucion = calcular_evolucion_anual(df_train)
    logger.info("\n" + tabla_evolucion.to_string(index=False))
    tabla_evolucion.to_csv(os.path.join(EDA_DIR, "04_evolucion_anual_pct_bosque.csv"), index=False)

    df_delta = agregar_delta_pct_bosque(df_train)
    tabla_delta_anual = resumir_delta_anual(df_delta)
    logger.info("\n  delta_pct_bosque por año:\n" + tabla_delta_anual.to_string(index=False))
    tabla_delta_anual.to_csv(os.path.join(EDA_DIR, "04b_delta_pct_bosque_anual.csv"), index=False)

    tabla_cambios_extremos = detectar_cambios_extremos(df_delta)
    logger.info(f"\n  Distrito-años con cambio extremo: {len(tabla_cambios_extremos)}")
    tabla_cambios_extremos.to_csv(os.path.join(EDA_DIR, "04c_distritos_cambios_extremos.csv"), index=False)
    logger.info("[OK] Sección 4 guardada")

    # --- Sección 5: Dependencia temporal ---
    logger.info("=" * 60)
    logger.info("SECCIÓN 5 — Dependencia temporal de pct_bosque (rezagos 1-5)")
    tabla_autocorr = calcular_autocorrelacion_pct_bosque(df_train)
    logger.info("\n" + tabla_autocorr.to_string(index=False))
    tabla_autocorr.to_csv(os.path.join(EDA_DIR, "05_autocorrelacion_pct_bosque.csv"), index=False)
    if tabla_autocorr.loc[0, "r"] >= UMBRAL_R_FUERTE:
        logger.info(
            "  Conclusión: autocorrelación fuerte en lag=1 -> existe estructura temporal "
            "suficiente para justificar modelos de series temporales."
        )
    logger.info("[OK] Sección 5 guardada")

    # --- Sección 7 (calculada aquí porque la sección 6 depende de ella) ---
    tabla_clasificacion = clasificar_variables_dinamicas_estaticas(df_train)
    variables_dinamicas = tabla_clasificacion.loc[
        tabla_clasificacion["clasificacion"].isin(["DINAMICA", "CUASI-ESTATICA"]), "variable"
    ].tolist()
    variables_estaticas = tabla_clasificacion.loc[
        tabla_clasificacion["clasificacion"] == "ESTATICA", "variable"
    ].tolist()

    # --- Sección 6: Relación predictiva con el futuro ---
    logger.info("=" * 60)
    logger.info("SECCIÓN 6 — Relación predictiva con el futuro")
    logger.info(f"  Dinámicas/cuasi-estáticas (evaluadas con lead temporal): {variables_dinamicas}")
    logger.info(f"  Estáticas (evaluadas de forma transversal): {variables_estaticas}")

    tabla_futura_dinamicas = calcular_relacion_futura_dinamicas(df_train, variables_dinamicas)
    if len(tabla_futura_dinamicas):
        logger.info("\n" + tabla_futura_dinamicas.to_string(index=False))
        tabla_futura_dinamicas.to_csv(os.path.join(EDA_DIR, "06_relacion_futura_dinamicas.csv"), index=False)

    tabla_transversal_estaticas = calcular_relacion_transversal_estaticas(df_train, variables_estaticas)
    if len(tabla_transversal_estaticas):
        logger.info("\n" + tabla_transversal_estaticas.to_string(index=False))
        tabla_transversal_estaticas.to_csv(
            os.path.join(EDA_DIR, "06b_relacion_transversal_estaticas.csv"), index=False
        )
    logger.info("[OK] Sección 6 guardada")

    # --- Sección 7: guardar clasificación ya calculada ---
    logger.info("=" * 60)
    logger.info("SECCIÓN 7 — Clasificación empírica dinámica / estática")
    logger.info("\n" + tabla_clasificacion.to_string(index=False))
    tabla_clasificacion.to_csv(os.path.join(EDA_DIR, "07_clasificacion_dinamica_estatica.csv"), index=False)
    logger.info("[OK] Sección 7 guardada")

    # --- Sección 8: Tabla final de decisión ---
    logger.info("=" * 60)
    logger.info("SECCIÓN 8 — Tabla final de decisión por variable")
    tabla_decision = construir_tabla_decision_final(
        tabla_calidad=tabla_calidad,
        tabla_outliers=tabla_outliers,
        tabla_asimetria=tabla_asimetria,
        tabla_correlacion_objetivo=tabla_correlacion_objetivo,
        tabla_vif=tabla_vif,
        pares_colineales=pares_colineales,
        tabla_clasificacion=tabla_clasificacion,
        tabla_futura_dinamicas=tabla_futura_dinamicas,
        tabla_transversal_estaticas=tabla_transversal_estaticas,
    )
    logger.info("\n" + tabla_decision[["variable", "decision_final", "justificacion"]].to_string(index=False))
    tabla_decision.to_csv(os.path.join(EDA_DIR, "08_tabla_decision_final.csv"), index=False)
    logger.info("[OK] Sección 8 guardada")

    logger.info("=" * 60)
    logger.info(f"EDA completado. Tablas guardadas en: {EDA_DIR}")
