import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# -----------------------------
# Selección de distritos
# -----------------------------
def seleccionar_distritos_representativos(ruta_estadisticas):
    df = pd.read_csv(ruta_estadisticas)
    df["rango"] = df["pct_bosque_max"] - df["pct_bosque_min"]

    distrito_mayor = df.loc[df["rango"].idxmax(), "geocode"]
    distrito_menor = df.loc[df["rango"].idxmin(), "geocode"]

    print(f"[INFO] Distrito mayor cambio: {distrito_mayor}")
    print(f"[INFO] Distrito menor cambio: {distrito_menor}")

    return distrito_mayor, distrito_menor


# -----------------------------
# Construcción de series
# -----------------------------
def construir_series(ruta_series, distrito_mayor, distrito_menor):
    df = pd.read_csv(ruta_series)

    df_mayor = df[df["geocode"] == distrito_mayor].sort_values("anio")
    df_menor = df[df["geocode"] == distrito_menor].sort_values("anio")
    df_mediana = df.groupby("anio")["pct_bosque"].median().reset_index()

    return (
        (df_mayor["anio"].values,    df_mayor["pct_bosque"].values),
        (df_menor["anio"].values,    df_menor["pct_bosque"].values),
        (df_mediana["anio"].values,  df_mediana["pct_bosque"].values),
    )


# -----------------------------
# Diferenciación
# -----------------------------
def diferenciar_serie(serie, d=1):
    serie_diff = pd.Series(serie)
    for _ in range(d):
        serie_diff = serie_diff.diff()
    return serie_diff.dropna().values


# -----------------------------
# Guardar series
# -----------------------------
def guardar_serie(anios, serie, titulo, ruta):
    plt.figure()
    plt.plot(anios, serie)
    plt.title(titulo)
    plt.xlabel("Año")
    plt.ylabel("% cobertura boscosa")
    plt.grid()
    plt.savefig(ruta)
    plt.close()


# -----------------------------
# Guardar ACF/PACF
# -----------------------------
def guardar_acf_pacf(serie, nombre, carpeta):
    plt.figure()
    plot_acf(serie, lags=10)
    plt.title(f"ACF - {nombre}")
    plt.savefig(os.path.join(carpeta, f"acf_{nombre}.png"))
    plt.close()

    plt.figure()
    plot_pacf(serie, lags=10)
    plt.title(f"PACF - {nombre}")
    plt.savefig(os.path.join(carpeta, f"pacf_{nombre}.png"))
    plt.close()


# -----------------------------
# Función principal
# -----------------------------
def generar_analisis_arima(ruta_estadisticas, ruta_series, ruta_analisis_arima):
    print("\n" + "=" * 60)
    print(" ANÁLISIS EXPLORATORIO ARIMA ")
    print("=" * 60)

    os.makedirs(ruta_analisis_arima, exist_ok=True)

    d_mayor, d_menor = seleccionar_distritos_representativos(ruta_estadisticas)

    (anios_mayor,   serie_mayor), \
    (anios_menor,   serie_menor), \
    (anios_mediana, serie_mediana) = construir_series(ruta_series, d_mayor, d_menor)

    # -----------------------------
    # Guardar series originales
    # -----------------------------
    guardar_serie(anios_mayor,   serie_mayor,   "Serie - Mayor cambio", os.path.join(ruta_analisis_arima, "serie_mayor.png"))
    guardar_serie(anios_menor,   serie_menor,   "Serie - Menor cambio", os.path.join(ruta_analisis_arima, "serie_menor.png"))
    guardar_serie(anios_mediana, serie_mediana, "Serie - Mediana",      os.path.join(ruta_analisis_arima, "serie_mediana.png"))

    # -----------------------------
    # Diferenciar (solo mayor y mediana)
    # -----------------------------
    serie_mayor_diff = diferenciar_serie(serie_mayor, d=1)
    serie_mediana_diff = diferenciar_serie(serie_mediana, d=1)

    anios_mayor_diff = anios_mayor[1:]
    anios_mediana_diff = anios_mediana[1:]

    # Guardar series diferenciadas
    guardar_serie(anios_mayor_diff, serie_mayor_diff,
                  "Serie Diferenciada - Mayor",
                  os.path.join(ruta_analisis_arima, "serie_mayor_diff.png"))

    guardar_serie(anios_mediana_diff, serie_mediana_diff,
                  "Serie Diferenciada - Mediana",
                  os.path.join(ruta_analisis_arima, "serie_mediana_diff.png"))

    # -----------------------------
    # ACF/PACF
    # -----------------------------

    # Originales (diagnóstico)
    guardar_acf_pacf(serie_mayor,   "mayor_raw",   ruta_analisis_arima)
    guardar_acf_pacf(serie_mediana, "mediana_raw", ruta_analisis_arima)

    # Diferenciadas (las correctas)
    guardar_acf_pacf(serie_mayor_diff,   "mayor_diff",   ruta_analisis_arima)
    guardar_acf_pacf(serie_mediana_diff, "mediana_diff", ruta_analisis_arima)

    # Menor (sin diferenciar)
    guardar_acf_pacf(serie_menor, "menor", ruta_analisis_arima)

    print(f"[OK] Gráficos guardados en: {ruta_analisis_arima}")