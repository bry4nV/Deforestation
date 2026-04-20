import geopandas as gpd
import pandas as pd
import numpy as np

def seleccionar_distritos_entrenamiento(
    ruta_mapa_cambios_distrito,
    ruta_mapa_cambios_deforestacion_distrito,
    ruta_mapa_cambios_reforestacion_distrito,
    particion_entrenamiento=0.7,
    frac_top=0.7,
    random_state=42
):

    gdf_cambio = gpd.read_file(ruta_mapa_cambios_distrito)
    gdf_def = gpd.read_file(ruta_mapa_cambios_deforestacion_distrito)
    gdf_ref = gpd.read_file(ruta_mapa_cambios_reforestacion_distrito)

    # =====================================================
    # 2. Preparar columnas clave
    # =====================================================
    gdf = gdf_cambio[[
        "GEOCODE", "LEVEL_2", "LEVEL_4", "geometry", "porcentaje_cambio"
    ]].rename(columns={
        "porcentaje_cambio": "pct_cambio"
    })

    gdf = gdf.merge(
        gdf_def[["GEOCODE", "porcentaje_cambio"]].rename(
            columns={"porcentaje_cambio": "pct_deforestacion"}
        ),
        on="GEOCODE",
        how="inner"
    )

    gdf = gdf.merge(
        gdf_ref[["GEOCODE", "porcentaje_cambio"]].rename(
            columns={"porcentaje_cambio": "pct_reforestacion"}
        ),
        on="GEOCODE",
        how="inner"
    )

    gdf["score"] = (
        0.5 * gdf["pct_cambio"] +
        0.4 * gdf["pct_deforestacion"] +
        0.1 * gdf["pct_reforestacion"]
    )

    seleccionados = []
    n_total = int(len(gdf) * particion_entrenamiento)

    for _, grupo in gdf.groupby("LEVEL_2"):

        proporcion = len(grupo) / len(gdf)
        n_depto = max(1, int(proporcion * n_total))

        grupo = grupo.sort_values("score", ascending=False)

        n_top = max(1, round(n_depto * frac_top))
        n_top = min(n_top, len(grupo))
        n_rand = max(0, n_depto - n_top)

        top = grupo.head(n_top)

        if n_rand > 0:
            resto = grupo.iloc[n_top:]

            if len(resto) > 0:
                pesos = resto["score"] - resto["score"].min()
                pesos = pesos / pesos.sum() if pesos.sum() > 0 else None

                aleatorio = resto.sample(
                    n=min(n_rand, len(resto)),
                    weights=pesos,
                    random_state=random_state
                )

                seleccion = pd.concat([top, aleatorio])
            else:
                seleccion = top
        else:
            seleccion = top

        seleccionados.append(seleccion)

    gdf_sel = pd.concat(seleccionados).drop_duplicates()

    if len(gdf_sel) > n_total:
        exceso = len(gdf_sel) - n_total
        gdf_sel = gdf_sel.sort_values("score", ascending=True).iloc[exceso:]
    elif len(gdf_sel) < n_total:
        faltantes = n_total - len(gdf_sel)
        restantes = gdf[~gdf["GEOCODE"].isin(gdf_sel["GEOCODE"])]
        adicionales = restantes.sample(
            n=faltantes,
            weights=restantes["score"],
            random_state=random_state
        )
        gdf_sel = pd.concat([gdf_sel, adicionales])

    gdf["dataset"] = "generalizacion"
    gdf.loc[gdf_sel.index, "dataset"] = "entrenamiento"

    return gdf, gdf_sel