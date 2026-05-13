import os

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RUTA_CAMBIOS = os.path.join(
    BASE_DIR, "data", "interim", "O1",
    "distritos-alto-cambio", "distritos_alto_cambio.csv"
)
RUTA_TRAIN = os.path.join(
    BASE_DIR, "data", "interim", "O1",
    "series-temporales", "entrenamiento", "distritos_entrenamiento.csv"
)
RUTA_GEN = os.path.join(
    BASE_DIR, "data", "interim", "O1",
    "series-temporales", "generalizacion-espacial", "distritos_generalizacion_espacial.csv"
)

# ── Carga datos ──────────────────────────────────────────────────────────────
df_cambios = pd.read_csv(RUTA_CAMBIOS)
df_series  = pd.concat([pd.read_csv(RUTA_TRAIN), pd.read_csv(RUTA_GEN)], ignore_index=True)

df_cambios["GEOCODE"] = df_cambios["GEOCODE"].astype(str).str.zfill(6)
df_series["geocode"]  = df_series["geocode"].astype(str).str.zfill(6)

# ── Identificar extremos por % Cambio ────────────────────────────────────────
fila_max = df_cambios.loc[df_cambios["% Cambio"].idxmax()]
fila_min = df_cambios.loc[df_cambios["% Cambio"].idxmin()]

mayor = {"geocode": fila_max["GEOCODE"], "distrito": fila_max["Distrito"],
         "depto": fila_max["Departamento"], "pct_cambio": fila_max["% Cambio"]}
menor = {"geocode": fila_min["GEOCODE"], "distrito": fila_min["Distrito"],
         "depto": fila_min["Departamento"], "pct_cambio": fila_min["% Cambio"]}

print(f"Mayor cambio : {mayor['distrito']} ({mayor['depto']})  — {mayor['pct_cambio']:.2f} %")
print(f"Menor cambio : {menor['distrito']} ({menor['depto']})  — {menor['pct_cambio']:.2f} %")

# ── Extraer series desde entrenamiento o generalización ──────────────────────
def serie_distrito(geocode):
    df = df_series[df_series["geocode"] == geocode].sort_values("anio")
    if df.empty:
        raise ValueError(f"Geocode {geocode} no encontrado en ningún CSV de series.")
    return df["anio"].values, df["pct_bosque"].values

x_mayor, y_mayor = serie_distrito(mayor["geocode"])
x_menor, y_menor = serie_distrito(menor["geocode"])

# ── Gráfica ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    x_mayor, y_mayor,
    color="#d62728", linewidth=2, marker="o", markersize=4,
    label=f"{mayor['distrito']} ({mayor['depto']})  —  cambio histórico: {mayor['pct_cambio']:.1f} %",
)
ax.plot(
    x_menor, y_menor,
    color="#1f77b4", linewidth=2, marker="o", markersize=4,
    label=f"{menor['distrito']} ({menor['depto']})  —  cambio histórico: {menor['pct_cambio']:.1f} %",
)

ax.set_xlabel("Año")
ax.set_ylabel("% Cobertura boscosa")
ax.set_title("Distrito con mayor y menor % de cambio histórico de cobertura boscosa (1985–2024)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()

ruta_salida = os.path.join(BASE_DIR, "series_extremas.png")
fig.savefig(ruta_salida, dpi=150)
plt.show()
print(f"\n[OK] Gráfica guardada: {ruta_salida}")
