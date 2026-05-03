import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def seleccionar_distritos_representativos(ruta_estadisticas):
    df = pd.read_csv(ruta_estadisticas)

    # Crear métrica de variabilidad
    df["rango"] = df["pct_bosque_max"] - df["pct_bosque_min"]

    distrito_mayor = df.loc[df["rango"].idxmax(), "geocode"]
    distrito_menor = df.loc[df["rango"].idxmin(), "geocode"]

    print(f"[INFO] Distrito mayor cambio: {distrito_mayor}")
    print(f"[INFO] Distrito menor cambio: {distrito_menor}")

    return distrito_mayor, distrito_menor


def construir_series(ruta_series, distrito_mayor, distrito_menor):
    df = pd.read_csv(ruta_series)

    df_mayor = df[df["geocode"] == distrito_mayor].sort_values("anio")
    df_menor = df[df["geocode"] == distrito_menor].sort_values("anio")

    serie_mayor = df_mayor["pct_bosque"].values
    anios_mayor = df_mayor["anio"].values

    serie_menor = df_menor["pct_bosque"].values
    anios_menor = df_menor["anio"].values

    df_mediana = df.groupby("anio")["pct_bosque"].median().reset_index()

    serie_mediana = df_mediana["pct_bosque"].values
    anios_mediana = df_mediana["anio"].values

    return (
        (anios_mayor, serie_mayor),
        (anios_menor, serie_menor),
        (anios_mediana, serie_mediana)
    )


def guardar_grafico_serie(anios, serie, titulo, ruta):
    plt.figure()
    plt.plot(anios, serie)
    plt.title(titulo)
    plt.xlabel("Año")   # 🔥 importante
    plt.ylabel("pct_bosque")
    plt.grid()
    plt.savefig(ruta)
    plt.close()


def guardar_acf_pacf(serie, nombre, carpeta_salida):
    # ACF
    plt.figure()
    plot_acf(serie, lags=10)
    plt.title(f"ACF - {nombre}")
    plt.savefig(os.path.join(carpeta_salida, f"acf_{nombre}.png"))
    plt.close()

    # PACF
    plt.figure()
    plot_pacf(serie, lags=10)
    plt.title(f"PACF - {nombre}")
    plt.savefig(os.path.join(carpeta_salida, f"pacf_{nombre}.png"))
    plt.close()


def generar_analisis_arima(
    ruta_estadisticas,
    ruta_series,
    carpeta_salida
):
    print("\n" + "="*60)
    print(" ANÁLISIS EXPLORATORIO ARIMA ")
    print("="*60)

    os.makedirs(carpeta_salida, exist_ok=True)

    # 1. Seleccionar distritos
    d_mayor, d_menor = seleccionar_distritos_representativos(ruta_estadisticas)

    # 2. Construir series
    (serie_mayor, serie_menor, serie_mediana) = construir_series(
        ruta_series,
        d_mayor,
        d_menor
    )

    # 3. Guardar gráficos
    guardar_grafico_serie(
        serie_mayor[0], serie_mayor[1],
        "Serie - Mayor cambio",
        os.path.join(carpeta_salida, "serie_mayor.png")
    )

    guardar_grafico_serie(
        serie_menor[0], serie_menor[1],
        "Serie - Menor cambio",
        os.path.join(carpeta_salida, "serie_menor.png")
    )

    guardar_grafico_serie(
        serie_mediana[0], serie_mediana[1],
        "Serie - Mediana",
        os.path.join(carpeta_salida, "serie_mediana.png")
    )

    # 4. Guardar ACF y PACF
    guardar_acf_pacf(serie_mayor[1], "mayor", carpeta_salida)
    guardar_acf_pacf(serie_menor[1], "menor", carpeta_salida)
    guardar_acf_pacf(serie_mediana[1], "mediana", carpeta_salida)

    print(f"[OK] Gráficos guardados en: {carpeta_salida}")