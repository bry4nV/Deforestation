"""Validaciones exploratorias — módulo O3.

  run_test(nombre, funcion, col, distritos_gdf)
      Valida el output de construir_*: cobertura, nulos y estadísticos.

  validar_longitud_rios()
      GeoPandas UTM vs LONG_KM (ANA).

  validar_longitud_carreteras()
      GeoPandas UTM vs LONGITUD (MTC) — tres capas concatenadas.

Ejecución:
    python -m O3.r8_r9_r10.test_construir
"""
import os
import subprocess
import sys
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize as _rasterize
from rasterio.transform import array_bounds as _array_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, reproject as warp_reproject
from rasterio.windows import Window as _Window, from_bounds as _wfb
from scipy.ndimage import convolve as _convolve
from shapely.geometry import box as _box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from O3.config import (
    CARRETERAS_DEPARTAMENTAL_SHP,
    CARRETERAS_NACIONAL_SHP,
    CARRETERAS_VECINAL_SHP,
    CRS_PROYECTADO,
    DISTRITOS_ALTO_CAMBIO_GPKG,
    ELEVACION_CSV,
    ELEVACION_MOSAIC,
    GPKG_COL_DEPARTAMENTO,
    GPKG_COL_DISTRITO,
    GPKG_COL_GEOCODE,
    RIOS_SHP,
    SLOPE_NODATA,
    SRTM_NODATA,
)
from O3.r8_r9_r10.construir_carreteras import construir_carreteras
# from O3.r8_r9_r10.construir_rios import construir_rios

SEP            = "=" * 60
UMBRAL_OUTLIER = 90   # Δ% a partir del cual un segmento se considera sospechoso


# ─────────────────────────────────────────────────────────────────────────────
# run_test — validación genérica de cualquier función construir_*
# ─────────────────────────────────────────────────────────────────────────────

def run_test(nombre, funcion, col_variable, distritos_gdf):
    """Valida cobertura de distritos, ausencia de nulos y resumen estadístico."""
    n_esperado = len(distritos_gdf)

    print(f"\n{SEP}\n  TEST: {nombre}\n{SEP}")

    print(f"\n[0] CRS de distritos_gdf:")
    print(f"    {distritos_gdf.crs}")

    print(f"\n[1] Llamando {nombre}(distritos_gdf)...")
    df = funcion(distritos_gdf)
    print(f"    Retorno: {type(df).__name__}")

    print(f"\n[2] Primeras 5 filas:")
    print(df.head().to_string(index=False))

    n_real = len(df)
    ok_n   = "✓" if n_real == n_esperado else "✗"
    print(f"\n[3] Distritos — {ok_n} esperados: {n_esperado}  obtenidos: {n_real}")
    if n_real != n_esperado:
        faltantes = set(distritos_gdf[GPKG_COL_GEOCODE].unique()) - set(df["geocode"].unique())
        print(f"    Faltantes: {sorted(faltantes)[:10]} ...")

    n_nulos = df[col_variable].isna().sum()
    print(f"\n[4] Nulos en '{col_variable}' — {'✓' if n_nulos == 0 else '✗'} {n_nulos}")
    if n_nulos > 0:
        print(df[df[col_variable].isna()][["geocode", "distrito", col_variable]])

    s     = df[col_variable].dropna()
    n_cer = (s == 0).sum()
    print(f"\n[5] '{col_variable}':  min={s.min():.4f}  max={s.max():.4f}  "
          f"media={s.mean():.4f}  med={s.median():.4f}  "
          f"std={s.std():.4f}  ceros={n_cer}/{len(s)}")

    paso = (n_real == n_esperado) and (n_nulos == 0)
    print(f"\n{'[PASS]' if paso else '[FAIL]'} {nombre}\n{SEP}\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Validación de métodos de longitud (GeoPandas UTM vs columna de referencia)
# ─────────────────────────────────────────────────────────────────────────────

def _inspeccionar_outliers(sospechosos, umbral_pct, cols_id=None):
    """Detalle de segmentos con Δ% > umbral_pct: tipo de geometría y columnas identificadoras."""
    if cols_id is None:
        # Auto-detección: columnas descriptivas por palabras clave
        cols_id = [
            c for c in sospechosos.columns
            if any(kw in c.upper() for kw in (
                "NOMB", "NAME", "RIO", "DESC", "ORDEN", "CODIGO_ECA",
                "CODRU", "DEPART", "JERARQ", "TIPO",
            ))
            and c != "geometry"
        ]

    print(f"\n[4] {len(sospechosos)} segmentos con Δ% > {umbral_pct}%")
    print(f"    Columnas identificadoras: {cols_id or ['(ninguna)']}")

    for idx, row in sospechosos.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            geom_type, partes = "NULL/vacía", []
        else:
            geom_type = geom.geom_type
            partes = (list(geom.geoms) if geom_type == "MultiLineString"
                      else [geom]       if geom_type == "LineString"
                      else [])
        longitudes_km = [round(p.length / 1000, 4) for p in partes]

        print(f"\n  ── índice {idx} {'─'*35}")
        print(f"    ref km:    {row['longitud_ref']:.4f}")
        print(f"    geom km:   {row['geopandas_km']:.4f}")
        print(f"    Δ%:        {row['diff_pct']:.2f}%")
        print(f"    tipo geom: {geom_type}  |  partes: {len(partes)}")
        if len(partes) > 1:
            print(f"    km partes: {longitudes_km} → Σ={sum(longitudes_km):.4f}")
        for col in cols_id:
            if col in row.index and pd.notna(row[col]):
                print(f"    {col}: {row[col]}")

    print(f"\n    geom_types en outliers: {sospechosos.geometry.geom_type.value_counts().to_dict()}")


def _validar_longitud(gdf, col_ref, display_name, cols_id=None):
    """Compara longitud geométrica (GeoPandas UTM) vs columna de referencia.

    Parámetros
    ----------
    gdf          : GeoDataFrame con col_ref (en km) y geometry
    col_ref      : nombre de la columna de longitud de referencia
    display_name : etiqueta para el encabezado del reporte
    cols_id      : columnas identificadoras para el reporte de outliers;
                   None activa auto-detección por palabras clave
    """
    print(f"\n{SEP}\n  VALIDACIÓN: {display_name}\n{SEP}")

    if col_ref not in gdf.columns:
        print(f"  [ERROR] Columna {col_ref!r} no encontrada.")
        print(f"  Columnas disponibles: {list(gdf.columns)}")
        return None

    gdf_utm = gdf.to_crs(CRS_PROYECTADO).copy()
    gdf_utm["geopandas_km"] = gdf_utm.geometry.length / 1000
    gdf_utm["longitud_ref"] = gdf[col_ref].values

    validos = gdf_utm[gdf_utm["longitud_ref"] > 0].copy()
    n_excl  = len(gdf_utm) - len(validos)
    if n_excl:
        print(f"  ({n_excl} segmentos con {col_ref}=0 excluidos de la comparación)")

    validos["diff_abs_km"] = (validos["geopandas_km"] - validos["longitud_ref"]).abs()
    validos["diff_pct"]    = validos["diff_abs_km"] / validos["longitud_ref"] * 100

    muestra = (
        validos
        .sample(min(10, len(validos)), random_state=42)
        .sort_values("longitud_ref")
        [["longitud_ref", "geopandas_km", "diff_abs_km", "diff_pct"]]
        .rename(columns={
            "longitud_ref": f"{col_ref} (ref)",
            "geopandas_km": "GeoPandas km",
            "diff_abs_km":  "Δ abs km",
            "diff_pct":     "Δ %",
        })
    )
    print("\n[1] Muestra de 10 segmentos:")
    print(muestra.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    pct = validos["diff_pct"]
    print(f"\n[2] Δ% (n={len(pct)} segmentos):")
    print(f"    min={pct.min():.4f}  max={pct.max():.4f}  media={pct.mean():.4f}  "
          f"med={pct.median():.4f}  std={pct.std():.4f}  p95={pct.quantile(0.95):.4f}")
    print(f"    >1%:  {(pct > 1).sum():>5}  ({(pct > 1).mean()*100:.1f}%)")
    print(f"    >5%:  {(pct > 5).sum():>5}  ({(pct > 5).mean()*100:.1f}%)")
    print(f"    >10%: {(pct > 10).sum():>5}  ({(pct > 10).mean()*100:.1f}%)")

    sospechosos = validos[validos["diff_pct"] > UMBRAL_OUTLIER].sort_values("diff_pct", ascending=False)
    _inspeccionar_outliers(sospechosos, UMBRAL_OUTLIER, cols_id=cols_id)

    media = pct.mean()
    print(f"\n[3] Orientación (media={media:.2f}%):")
    if media < 2:
        print("    → cálculo UTM confiable; diferencia probablemente por curvatura de vértices.")
    elif media < 5:
        print("    → diferencia moderada; revisar proyección de origen o generalización.")
    else:
        print(f"    → diferencia alta; investigar fuente de {col_ref} antes de decidir.")

    print(SEP + "\n")
    return validos


def validar_longitud_rios():
    """Compara GeoPandas UTM vs LONG_KM (ANA) para orientar el método en construir_rios."""
    gdf = gpd.read_file(RIOS_SHP)
    print(f"  Segmentos cargados: {len(gdf)}  |  CRS: {gdf.crs}")
    return _validar_longitud(gdf, col_ref="LONG_KM", display_name="longitud ríos UTM vs LONG_KM (ANA)")


def validar_longitud_carreteras():
    """Compara GeoPandas UTM vs LONGITUD (MTC) para las tres capas de red vial."""
    # Columnas presentes en las tres capas; 'tipo' se agrega al concat
    COLS_COMUNES = ["LONGITUD", "CODRUTA", "DEPARTAMEN", "JERARQ_L", "geometry"]

    capas = []
    for shp, tipo in [
        (CARRETERAS_NACIONAL_SHP,      "nacional"),
        (CARRETERAS_DEPARTAMENTAL_SHP, "departamental"),
        (CARRETERAS_VECINAL_SHP,       "vecinal"),
    ]:
        gdf  = gpd.read_file(shp)
        cols = [c for c in COLS_COMUNES if c in gdf.columns]
        sub  = gdf[cols].copy()
        sub["tipo"] = tipo
        capas.append(sub)
        print(f"  {tipo}: {len(sub)} segmentos")

    carreteras = gpd.GeoDataFrame(
        pd.concat(capas, ignore_index=True),
        geometry="geometry",
        crs=capas[0].crs,
    )
    print(f"  Total: {len(carreteras)} segmentos  |  CRS: {carreteras.crs}")

    return _validar_longitud(
        carreteras,
        col_ref="LONGITUD",
        display_name="longitud carreteras UTM vs LONGITUD (MTC)",
        cols_id=["CODRUTA", "DEPARTAMEN", "JERARQ_L", "tipo"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validación: LONGITUD == FIN - INICIO (progresivas MTC) vs medición geométrica
# ─────────────────────────────────────────────────────────────────────────────

def validar_longitud_es_fin_menos_inicio():
    """Comprueba si LONGITUD ≈ FIN - INICIO en las tres capas viales del MTC.

    INICIO y FIN son progresivas (km acumulados desde el origen de la ruta).
    Si LONGITUD = FIN - INICIO, es una diferencia de progresivas, no una
    medición geométrica: no refleja la longitud real del tramo en el espacio
    (curvas, tolerancias de digitalización). Para intersección espacial con
    distritos se debe usar geometry.length en UTM.
    """
    TOLERANCIA_KM = 0.001   # 1 m — margen de redondeo aceptable
    COLS_REQUERIDAS = {"INICIO", "FIN", "LONGITUD"}

    for shp, tipo in [
        (CARRETERAS_NACIONAL_SHP,      "nacional"),
        (CARRETERAS_DEPARTAMENTAL_SHP, "departamental"),
        (CARRETERAS_VECINAL_SHP,       "vecinal"),
    ]:
        print(f"\n{SEP}\n  FIN-INICIO vs LONGITUD: {tipo}\n{SEP}")
        gdf = gpd.read_file(shp)
        print(f"  Segmentos: {len(gdf)}  |  CRS: {gdf.crs}")

        faltantes = COLS_REQUERIDAS - set(gdf.columns)
        if faltantes:
            print(f"  [SKIP] Columnas no encontradas: {sorted(faltantes)}")
            print(f"  Columnas disponibles: {sorted(gdf.columns.tolist())}")
            continue

        sub = gdf[["INICIO", "FIN", "LONGITUD"]].copy()
        n_nulos = sub.isna().any(axis=1).sum()
        if n_nulos:
            print(f"  ({n_nulos} filas con nulos excluidas)")
        sub = sub.dropna()

        sub = sub.copy()
        sub["fin_menos_inicio"] = sub["FIN"] - sub["INICIO"]
        sub["diff_abs"]         = (sub["LONGITUD"] - sub["fin_menos_inicio"]).abs()

        pct_exacto = (sub["diff_abs"] <= TOLERANCIA_KM).mean() * 100
        d = sub["diff_abs"]

        # Muestra representativa ordenada por progresiva
        muestra = (
            sub.sample(min(10, len(sub)), random_state=42)
            .sort_values("INICIO")
            [["INICIO", "FIN", "fin_menos_inicio", "LONGITUD", "diff_abs"]]
        )
        print("\n[1] Muestra de 10 segmentos (LONGITUD vs FIN-INICIO):")
        print(muestra.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

        print(f"\n[2] |LONGITUD - (FIN-INICIO)| (n={len(d)} segmentos):")
        print(f"    min={d.min():.6f}  max={d.max():.6f}  media={d.mean():.6f}  "
              f"p50={d.median():.6f}  p95={d.quantile(0.95):.6f}")
        print(f"    Dentro de ±{TOLERANCIA_KM} km: {pct_exacto:.1f}%")

        # Outliers: diferencia > 10 % de la longitud de referencia
        ratio = sub["diff_abs"] / sub["LONGITUD"].replace(0, pd.NA)
        n_outliers = (ratio > 0.10).sum()
        print(f"    Δ > 10% de LONGITUD:  {int(n_outliers)} segmentos ({n_outliers/len(sub)*100:.1f}%)")

        print(f"\n[3] Conclusión — {tipo}:")
        if pct_exacto >= 99.0:
            print(
                f"    LONGITUD = FIN - INICIO (diferencia residual por redondeo).\n"
                f"    Es una diferencia de progresivas (km de ruta), no una medición\n"
                f"    geométrica. Para análisis espacial usar geometry.length en UTM."
            )
        elif pct_exacto >= 90.0:
            print(
                f"    La mayoría coincide ({pct_exacto:.1f}%), pero hay {int((sub['diff_abs'] > TOLERANCIA_KM).sum())} "
                f"excepciones — revisar si son errores de digitalización o tramos especiales."
            )
        else:
            print(
                f"    Solo {pct_exacto:.1f}% coincide — LONGITUD puede tener otra fuente.\n"
                f"    Investigar antes de usar como referencia."
            )
        print(SEP)

def validar_geometrias_descartadas_overlay(distritos_gdf):
    """Cuantifica cuántos km de carretera se pierden por geometrías no-lineales
    en el overlay final carreteras × distritos (keep_geom_type=False).

    construir_carreteras.py descarta geometrías que el overlay devuelve como
    Point/MultiPoint (tangencias en el límite del distrito). Este test mide
    si esa pérdida es despreciable (como asume el comentario "esperado") o
    si representa una pérdida real de longitud no trivial.
    """
    from O3.r8_r9_r10.construir_carreteras import construir_carreteras
    import geopandas as gpd

    print(f"\n{SEP}\n  GEOMETRÍAS DESCARTADAS: overlay carreteras × distritos\n{SEP}")

    capas = [
        gpd.read_file(CARRETERAS_NACIONAL_SHP),
        gpd.read_file(CARRETERAS_DEPARTAMENTAL_SHP),
        gpd.read_file(CARRETERAS_VECINAL_SHP),
    ]
    carreteras_gdf = gpd.GeoDataFrame(
        pd.concat(capas, ignore_index=True), geometry="geometry", crs=capas[0].crs
    )

    distritos_utm  = distritos_gdf.to_crs(CRS_PROYECTADO)
    carreteras_utm = carreteras_gdf.to_crs(CRS_PROYECTADO)
    carreteras_utm["geometry"] = carreteras_utm.geometry.make_valid()
    distritos_utm["geometry"]  = distritos_utm.geometry.make_valid()

    wkb_series = carreteras_utm.geometry.apply(lambda g: g.wkb)
    carreteras_utm = carreteras_utm[~wkb_series.duplicated()].copy()

    from shapely.ops import unary_union as _uu
    red_unificada = _uu(carreteras_utm.geometry.tolist())
    carreteras_utm = (
        gpd.GeoDataFrame(geometry=[red_unificada], crs=CRS_PROYECTADO)
        .explode(index_parts=False)
        .reset_index(drop=True)
    )

    print("  Ejecutando overlay con keep_geom_type=False (puede tardar varios minutos)...")
    interseccion_raw = gpd.overlay(
        carreteras_utm[["geometry"]],
        distritos_utm[[GPKG_COL_GEOCODE, "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )

    es_lineal = interseccion_raw.geometry.geom_type.isin(["LineString", "MultiLineString"])
    descartadas = interseccion_raw[~es_lineal]
    n_descartadas = len(descartadas)

    print(f"\n[1] Geometrías totales del overlay: {len(interseccion_raw)}")
    print(f"    Descartadas (no lineales): {n_descartadas}")
    print(f"    Tipos descartados: {descartadas.geometry.geom_type.value_counts().to_dict() if n_descartadas else '(ninguno)'}")

    # Las descartadas son puntos/geometrías vacías -> longitud real perdida es 0
    # por definición geométrica (un Point no tiene length). Lo que SÍ importa es
    # si algún segmento LineString largo se degradó en Point: para eso comparamos
    # la longitud total de la red ANTES del overlay vs. la longitud capturada DESPUÉS.
    km_red_total = carreteras_utm.geometry.length.sum() / 1000
    km_capturado = interseccion_raw[es_lineal].geometry.length.sum() / 1000

    print(f"\n[2] Longitud total de la red unificada (antes del overlay): {km_red_total:,.2f} km")
    print(f"    Longitud capturada en intersecciones lineales (después): {km_capturado:,.2f} km")
    print(f"    Diferencia: {km_red_total - km_capturado:,.4f} km "
          f"({(km_red_total - km_capturado)/km_red_total*100:.4f}%)")

    print(f"\n[3] Conclusión:")
    diff_pct = (km_red_total - km_capturado) / km_red_total * 100 if km_red_total > 0 else 0
    if diff_pct < 0.01:
        print("    Pérdida despreciable — confirma que las geometrías descartadas son")
        print("    tangencias puntuales sin longitud asociada, como asume el comentario.")
    else:
        print(f"    *** Pérdida de {diff_pct:.4f}% — investigar si hay segmentos largos")
        print("    degenerados en Point por geometrías inválidas o errores de topología ***")
    print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# Validación: doble conteo por duplicados o solapamiento entre capas viales
# ─────────────────────────────────────────────────────────────────────────────

def validar_doble_conteo_carreteras():
    """Detecta duplicados y solapamiento entre capas nacional, departamental y vecinal.

    [1] Duplicados WKB exactos — dentro de cada capa y entre capas.
    [2] Longitud de intersección por par de capas — solapamiento geométrico real.
    [3] Ratio unión/suma — cuantifica el doble conteo efectivo en el dataset completo.

    Interpretación de [3]:
      ratio == 1.0  → sin solapamiento; suma == longitud real de red única.
      ratio <  1.0  → hay segmentos superpuestos; construir_carreteras sobreestima km.
      ratio >  1.0  → imposible para LineStrings (indica error en la validación).
    """
    print(f"\n{SEP}\n  DOBLE CONTEO: capas viales MTC\n{SEP}")

    capas = {}
    for shp, tipo in [
        (CARRETERAS_NACIONAL_SHP,      "nacional"),
        (CARRETERAS_DEPARTAMENTAL_SHP, "departamental"),
        (CARRETERAS_VECINAL_SHP,       "vecinal"),
    ]:
        gdf = gpd.read_file(shp).to_crs(CRS_PROYECTADO)
        gdf["geometry"] = gdf.geometry.make_valid()
        capas[tipo] = gdf[["geometry"]].copy()
        print(f"  {tipo}: {len(gdf)} segmentos cargados")

    # ── [1] Duplicados WKB exactos ────────────────────────────────────────────
    print(f"\n[1] Duplicados WKB exactos")
    for tipo, gdf in capas.items():
        wkb = gdf.geometry.apply(lambda g: g.wkb)
        n   = wkb.duplicated().sum()
        print(f"    {tipo}: {int(n)} duplicados internos ({n/len(gdf)*100:.2f}%)")

    # Duplicados cruzados: WKB presentes en más de una capa
    todas = pd.concat(
        [g.assign(tipo=t) for t, g in capas.items()],
        ignore_index=True,
    )
    todas["wkb"] = todas.geometry.apply(lambda g: g.wkb)
    dupes_cruzados = todas[todas.duplicated("wkb", keep=False)]
    n_cruzados     = dupes_cruzados["wkb"].nunique()
    print(f"    Duplicados entre capas distintas: {n_cruzados} geometrías únicas aparecen en ≥2 capas")

    # WKB por capa — reutilizados en [2] para descomponer el origen del solapamiento
    wkb_por_capa = {
        tipo: gdf.geometry.apply(lambda g: g.wkb)
        for tipo, gdf in capas.items()
    }

    # ── [2] Longitud de intersección por par de capas — con origen ────────────
    print(f"\n[2] Solapamiento geométrico por par de capas")
    print(f"    Descomposición: cuánto viene de duplicados exactos vs. solapamiento distinto")
    pares = [
        ("nacional",      "departamental"),
        ("nacional",      "vecinal"),
        ("departamental", "vecinal"),
    ]
    km_solapado_total = 0.0
    for a, b in pares:
        try:
            inter_total = gpd.overlay(
                capas[a], capas[b],
                how="intersection",
                keep_geom_type=True,
            )
            km_total = inter_total.geometry.length.sum() / 1000
        except Exception as exc:
            print(f"    {a} × {b}: error en overlay ({exc})")
            continue
        km_solapado_total += km_total

        # Segmentos con WKB común entre las dos capas del par
        wkb_comunes = set(wkb_por_capa[a]) & set(wkb_por_capa[b])

        if wkb_comunes:
            dup_a = capas[a][wkb_por_capa[a].isin(wkb_comunes)]
            dup_b = capas[b][wkb_por_capa[b].isin(wkb_comunes)]
            try:
                inter_dup = gpd.overlay(
                    dup_a, dup_b,
                    how="intersection",
                    keep_geom_type=True,
                )
                km_dup = inter_dup.geometry.length.sum() / 1000
            except Exception:
                km_dup = float("nan")
            km_otro = km_total - km_dup
            print(f"    {a} × {b}: {km_total:.2f} km totales")
            print(f"      → duplicados exactos ({len(wkb_comunes)} geom. comunes): {km_dup:.2f} km")
            print(f"      → solapamiento entre segmentos distintos:               {km_otro:.2f} km")
        else:
            print(f"    {a} × {b}: {km_total:.2f} km  (sin duplicados exactos entre estas capas)")

    # ── [3] Ratio longitud unión / suma individual ────────────────────────────
    print(f"\n[3] Ratio longitud union / suma individual")
    print(f"    (puede tardar varios minutos — dissolve de toda la red)")
    concat_gdf = gpd.GeoDataFrame(
        pd.concat(capas.values(), ignore_index=True),
        geometry="geometry",
        crs=list(capas.values())[0].crs,
    )
    km_suma  = concat_gdf.geometry.length.sum() / 1000
    from shapely.ops import unary_union as _uu
    km_union = _uu(concat_gdf.geometry.tolist()).length / 1000
    ratio    = km_union / km_suma if km_suma > 0 else float("nan")

    print(f"    Suma de longitudes individuales: {km_suma:,.1f} km")
    print(f"    Longitud de la unión (red única): {km_union:,.1f} km")
    print(f"    Ratio union/suma:                 {ratio:.4f}")
    print(f"    Solapamiento efectivo:            {(1 - ratio)*100:.2f}%")

    print(f"\n[4] Conclusión:")
    if ratio >= 0.999:
        print(
            "    Sin solapamiento significativo.\n"
            "    construir_carreteras no introduce doble conteo — el cálculo es correcto."
        )
    elif ratio >= 0.98:
        print(
            f"    Solapamiento menor al 2% ({(1-ratio)*100:.2f}%).\n"
            "    El impacto en densidad_carreteras_km_km2 es despreciable."
        )
    else:
        print(
            f"    Solapamiento significativo: {(1-ratio)*100:.2f}% del total.\n"
            "    construir_carreteras sobreestima km_carreteras.\n"
            "    Considerar dissolve antes del overlay con distritos."
        )
    print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# Validación: duplicados internos en la capa de ríos (ANA — capa única)
# ─────────────────────────────────────────────────────────────────────────────

def validar_duplicados_rios():
    """Verifica si la capa ANA tiene segmentos con geometría WKB idéntica.

    A diferencia de carreteras, ríos proviene de una sola fuente — no hay
    concatenación de capas. El doble conteo solo ocurriría si el shapefile
    ANA registró el mismo segmento dos veces.
    """
    print(f"\n{SEP}\n  DUPLICADOS INTERNOS: capa de ríos (ANA)\n{SEP}")

    gdf = gpd.read_file(RIOS_SHP).to_crs(CRS_PROYECTADO)
    gdf["geometry"] = gdf.geometry.make_valid()
    print(f"  Segmentos cargados: {len(gdf)}")

    wkb   = gdf.geometry.apply(lambda g: g.wkb)
    n_dup = int(wkb.duplicated().sum())
    n_uniq = wkb.nunique()

    print(f"\n[1] Duplicados WKB exactos: {n_dup} ({n_dup/len(gdf)*100:.3f}%)")
    print(f"    Geometrías únicas:        {n_uniq} de {len(gdf)} segmentos")

    km_suma  = gdf.geometry.length.sum() / 1000
    gdf_dedup = gdf[~wkb.duplicated()]
    km_dedup = gdf_dedup.geometry.length.sum() / 1000

    print(f"\n[2] Longitud total con duplicados:    {km_suma:,.1f} km")
    print(f"    Longitud sin duplicados exactos:   {km_dedup:,.1f} km")
    print(f"    Diferencia:                        {km_suma - km_dedup:.1f} km ({(km_suma-km_dedup)/km_suma*100:.4f}%)")

    print(f"\n[3] Conclusión:")
    if n_dup == 0:
        print("    Sin duplicados — construir_rios no tiene doble conteo por esta causa.")
    else:
        print(
            f"    {n_dup} segmentos duplicados detectados.\n"
            f"    Añadir drop_duplicates(WKB) en construir_rios.py si el impacto es relevante."
        )
    print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# Validación Punto 4: NaN en dem_utm post-reproyección y riesgo contaminación Horn
# ─────────────────────────────────────────────────────────────────────────────

def validar_void_propagacion_dem_utm(distritos_gdf):
    """Verifica si la reproyección WGS84→UTM introduce NaN en dem_utm y si
    algún distrito cae dentro de la zona de borde NaN (riesgo contaminación Horn).

    Usa WarpedVRT para leer solo tiras de borde y muestra central sin cargar
    el raster completo (~5 GB) en memoria.
    """
    print(f"\n{SEP}\n  VOID PROPAGATION: NaN en dem_utm post-reproyección WGS84→UTM\n{SEP}")

    if not os.path.exists(ELEVACION_MOSAIC):
        print(f"  [SKIP] {ELEVACION_MOSAIC} no encontrado")
        return

    _ND   = -9999.0   # nodata concreto para WarpedVRT (NaN no es un nodata declarable)
    MARG  = 5         # ancho de las tiras de borde a inspeccionar (px)
    SZ_IN = 100       # lado de la muestra interior (px)

    with rasterio.open(ELEVACION_MOSAIC) as src:
        with WarpedVRT(
            src,
            crs=CRS_PROYECTADO,
            resampling=Resampling.bilinear,
            nodata=_ND,
            src_nodata=SRTM_NODATA,
        ) as vrt:
            W, H = vrt.width, vrt.height
            tf    = vrt.transform
            px_m  = abs(tf.a)
            py_m  = abs(tf.e)

            print(f"\n[1] Grid UTM: {W}×{H} px  —  píxel {px_m:.2f}×{py_m:.2f} m")
            print(f"    Extensión: {W*px_m/1000:.1f}×{H*py_m/1000:.1f} km")

            def _lee_nan(col, row, w, h):
                data = vrt.read(1, window=_Window(col, row, min(w, W-col), min(h, H-row)))
                return (data == _ND).sum(), data.size

            n_top,    t_top    = _lee_nan(0,       0,       W,    MARG)
            n_bottom, t_bottom = _lee_nan(0,       H-MARG,  W,    MARG)
            n_left,   t_left   = _lee_nan(0,       0,       MARG, H)
            n_right,  t_right  = _lee_nan(W-MARG,  0,       MARG, H)

            cx, cy = max(0, W//2 - SZ_IN//2), max(0, H//2 - SZ_IN//2)
            n_int, t_int = _lee_nan(cx, cy, SZ_IN, SZ_IN)

    print(f"\n[2] NaN en tiras de borde ({MARG} px):")
    print(f"    Top:    {int(n_top):>8,}  /  {t_top:>8,}  ({n_top/t_top*100:.2f}%)")
    print(f"    Bottom: {int(n_bottom):>8,}  /  {t_bottom:>8,}  ({n_bottom/t_bottom*100:.2f}%)")
    print(f"    Left:   {int(n_left):>8,}  /  {t_left:>8,}  ({n_left/t_left*100:.2f}%)")
    print(f"    Right:  {int(n_right):>8,}  /  {t_right:>8,}  ({n_right/t_right*100:.2f}%)")

    print(f"\n[3] Muestra interior ({SZ_IN}×{SZ_IN} px en el centro del grid):")
    print(f"    NaN: {int(n_int)}  /  {t_int}  ({n_int/t_int*100:.2f}%)")
    if n_int > 0:
        print(f"    *** NaN en interior — revisar si hay voids SRTM o artefactos de reproyección ***")
    else:
        print(f"    Sin NaN — interior completamente cubierto (esperado para SRTMGL1 void-filled)")

    total_borde = int(n_top + n_bottom + n_left + n_right)
    if total_borde == 0 and n_int == 0:
        print(f"\n  Conclusión: sin NaN en dem_utm. Void propagation NO aplica para este dataset.")
        print(SEP)
        return

    # ── Solapamiento zona borde NaN vs distritos ─────────────────────────────
    print(f"\n[4] Solapamiento zona borde ({MARG} px = {MARG*max(px_m,py_m):.0f} m) vs distritos:")

    xmin      = tf.c
    ymax      = tf.f
    xmax      = xmin + W * px_m
    ymin      = ymax - H * py_m
    margen_m  = MARG * max(px_m, py_m)

    borde_zona = _box(xmin, ymin, xmax, ymax).difference(
        _box(xmin + margen_m, ymin + margen_m, xmax - margen_m, ymax - margen_m)
    )

    distritos_utm = distritos_gdf.to_crs(CRS_PROYECTADO)
    afectados     = distritos_utm[distritos_utm.geometry.intersects(borde_zona)]

    if afectados.empty:
        print(f"    Sin solapamiento — todos los distritos están dentro del grid interior.")
        print(f"    Conclusión: los NaN de borde NO afectan pendiente_media_deg.")
    else:
        print(f"    *** {len(afectados)} distritos tocan la zona de borde NaN:")
        for _, r in afectados.iterrows():
            print(f"      {r[GPKG_COL_GEOCODE]}: {r[GPKG_COL_DISTRITO]} ({r[GPKG_COL_DEPARTAMENTO]})")

    print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# Validación Punto 2: Horn (nuestro) vs np.gradient vs gdaldem slope
# ─────────────────────────────────────────────────────────────────────────────

def validar_horn_vs_gdaldem_slope(distritos_gdf):
    """Compara pendiente media distrital entre tres métodos sobre un subconjunto del DEM:
      - Horn (1981) — nuestra implementación (scipy.ndimage.convolve)
      - np.gradient — diferencias finitas centrales simples
      - gdaldem slope — referencia externa (si está disponible en PATH)

    Selecciona automáticamente el distrito de mayor y menor elevación media
    para cubrir terreno andino y amazónico.
    """
    print(f"\n{SEP}\n  VALIDACIÓN: Horn (nuestro) vs np.gradient vs gdaldem slope\n{SEP}")

    if not os.path.exists(ELEVACION_MOSAIC):
        print(f"  [SKIP] Mosaico DEM no encontrado: {ELEVACION_MOSAIC}")
        return
    if not os.path.exists(ELEVACION_CSV):
        print(f"  [SKIP] {ELEVACION_CSV} no encontrado — ejecuta construir_elevacion primero")
        return

    elev_df = pd.read_csv(ELEVACION_CSV).dropna(subset=["elev_media_m"])
    if len(elev_df) < 2:
        print(f"  [SKIP] Insuficientes distritos con elevación ({len(elev_df)})")
        return

    seleccion = pd.concat([
        elev_df.nlargest(1,  "elev_media_m").assign(_etiqueta="andino_max"),
        elev_df.nsmallest(1, "elev_media_m").assign(_etiqueta="amazonico_min"),
    ], ignore_index=True)

    print(f"\n  Distritos seleccionados:")
    for _, r in seleccion.iterrows():
        print(f"    [{r['_etiqueta']}] {r['distrito']} ({r['departamento']}) "
              f"— elev_media={r['elev_media_m']:.0f} m  geocode={r['geocode']}")

    # Verificar gdaldem en PATH
    gdaldem_ok = False
    try:
        res = subprocess.run(["gdaldem", "--version"], capture_output=True, timeout=5)
        gdaldem_ok = (res.returncode == 0)
        if gdaldem_ok:
            print(f"\n  gdaldem disponible: {res.stdout.decode().strip()[:60]}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    if not gdaldem_ok:
        print(f"\n  gdaldem no encontrado en PATH — comparación solo Horn vs np.gradient")

    BUF = 0.05   # buffer en grados (~5 km) alrededor del distrito para contexto de vecindad
    resultados = []

    for _, fila in seleccion.iterrows():
        gc = str(fila["geocode"]).zfill(6)
        etiq = fila["_etiqueta"]

        geom_row = distritos_gdf[distritos_gdf[GPKG_COL_GEOCODE].astype(str) == gc]
        if geom_row.empty:
            print(f"\n  [WARN] {gc} no encontrado en distritos_gdf — saltando")
            continue

        print(f"\n  ── {etiq}: {fila['distrito']} ({fila['departamento']})  {'─'*30}")

        # Leer ventana WGS84 con buffer
        b  = geom_row.total_bounds   # minx miny maxx maxy
        wb = (b[0]-BUF, b[1]-BUF, b[2]+BUF, b[3]+BUF)

        with rasterio.open(ELEVACION_MOSAIC) as src:
            win    = _wfb(*wb, src.transform)
            raw    = src.read(1, window=win, boundless=True, fill_value=SRTM_NODATA)
            sub_tf = src.window_transform(win)
            sub_crs = src.crs

        sub = raw.astype(np.float64)
        sub[sub == SRTM_NODATA] = np.nan

        # Reproyectar subset a UTM
        sub_bounds = _array_bounds(sub.shape[0], sub.shape[1], sub_tf)
        tf_u, w_u, h_u = calculate_default_transform(
            sub_crs, CRS_PROYECTADO, sub.shape[1], sub.shape[0], *sub_bounds
        )
        dem_u = np.full((h_u, w_u), np.nan, dtype=np.float64)
        warp_reproject(
            source=sub, destination=dem_u,
            src_transform=sub_tf, src_crs=sub_crs,
            dst_transform=tf_u, dst_crs=CRS_PROYECTADO,
            src_nodata=np.nan, dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        px = abs(tf_u.a)
        py = abs(tf_u.e)
        print(f"    DEM subset UTM: {w_u}×{h_u} px  —  {px:.2f}×{py:.2f} m/px")

        # ── Método 1: Horn (scipy.ndimage.convolve) ───────────────────────────
        kx = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / (8. * px)
        ky = np.array([[ 1., 2., 1.], [ 0., 0., 0.], [-1.,-2.,-1.]]) / (8. * py)
        slope_horn = np.where(
            np.isnan(dem_u), np.nan,
            np.degrees(np.arctan(np.sqrt(
                _convolve(dem_u, kx, mode="nearest")**2 +
                _convolve(dem_u, ky, mode="nearest")**2
            )))
        )

        # ── Método 2: np.gradient (diferencias centrales simples) ────────────
        dz_dy_g, dz_dx_g = np.gradient(dem_u, py, px)
        slope_npg = np.where(
            np.isnan(dem_u), np.nan,
            np.degrees(np.arctan(np.sqrt(dz_dx_g**2 + dz_dy_g**2)))
        )

        # Máscara del distrito en UTM
        geom_utm = geom_row.to_crs(CRS_PROYECTADO).geometry.iloc[0]
        mask = _rasterize(
            [geom_utm], out_shape=(h_u, w_u), transform=tf_u,
            fill=0, default_value=1, dtype=np.uint8, all_touched=False,
        ).astype(bool)

        valid = mask & ~np.isnan(dem_u)
        n_px  = int(valid.sum())
        if n_px == 0:
            print(f"    [WARN] Sin píxeles válidos en el distrito — saltando")
            continue

        m_horn = float(np.nanmean(slope_horn[valid]))
        m_npg  = float(np.nanmean(slope_npg[valid]))
        d_abs  = abs(m_horn - m_npg)
        d_pct  = d_abs / m_horn * 100 if m_horn > 0 else 0.0

        print(f"    Píxeles en distrito: {n_px:,}")
        print(f"    Horn (nuestro):      {m_horn:.4f}°")
        print(f"    np.gradient:         {m_npg:.4f}°")
        print(f"    Δ Horn–np.gradient:  {d_abs:.4f}°  ({d_pct:.2f}%)")

        # ── Método 3: gdaldem slope ────────────────────────────────────────────
        m_gdal = None
        if gdaldem_ok:
            _ND_G = float(SLOPE_NODATA)
            meta_tmp = {
                "driver": "GTiff", "dtype": "float32", "count": 1, "nodata": _ND_G,
                "crs": CRS_PROYECTADO, "transform": tf_u, "width": w_u, "height": h_u,
            }
            dem_for_gdal = np.where(np.isnan(dem_u), _ND_G, dem_u).astype(np.float32)
            tmp_dem   = tempfile.NamedTemporaryFile(suffix="_dem_utm.tif",    delete=False).name
            tmp_slope = tempfile.NamedTemporaryFile(suffix="_slope_gdal.tif", delete=False).name
            try:
                with rasterio.open(tmp_dem, "w", **meta_tmp) as dst:
                    dst.write(dem_for_gdal, 1)
                subprocess.run(
                    ["gdaldem", "slope", "-compute_edges", tmp_dem, tmp_slope, "-of", "GTiff"],
                    capture_output=True, check=True, timeout=120,
                )
                with rasterio.open(tmp_slope) as slp:
                    s_gdal = slp.read(1).astype(np.float64)
                    nd_val = slp.nodata
                if nd_val is not None:
                    s_gdal[s_gdal == nd_val] = np.nan

                m_gdal = float(np.nanmean(s_gdal[valid]))
                d_hg   = abs(m_horn - m_gdal)
                ok_str = "✓ validado (<0.05°)" if d_hg < 0.05 else "*** DISCREPANCIA — revisar kernels/escala ***"
                print(f"    gdaldem slope:       {m_gdal:.4f}°")
                print(f"    Δ Horn–gdaldem:      {d_hg:.4f}°  ({d_hg/m_horn*100:.3f}%)  → {ok_str}")
            except Exception as exc:
                print(f"    [gdaldem error]: {exc}")
            finally:
                for _f in [tmp_dem, tmp_slope]:
                    if os.path.exists(_f):
                        os.remove(_f)

        resultados.append({
            "etiqueta": etiq, "geocode": gc, "distrito": fila["distrito"],
            "n_px": n_px,
            "horn_deg": round(m_horn, 4),
            "npg_deg":  round(m_npg,  4),
            "gdal_deg": round(m_gdal, 4) if m_gdal is not None else None,
            "Δ_abs_deg": round(d_abs, 4),
            "Δ_pct":     round(d_pct, 2),
        })

    if resultados:
        print(f"\n  Resumen comparativo:")
        df_r  = pd.DataFrame(resultados)
        cols  = ["etiqueta", "distrito", "n_px", "horn_deg", "npg_deg", "Δ_abs_deg", "Δ_pct"]
        if any(r["gdal_deg"] is not None for r in resultados):
            cols.append("gdal_deg")
        print(df_r[cols].to_string(index=False))

    print(SEP)

def validar_solapamiento_interno_rios():
    """Detecta si segmentos DISTINTOS dentro de Rios.shp se solapan parcialmente
    entre sí (no duplicados exactos, sino tramos compartidos). Si existe,
    construir_rios.py contaría ese tramo dos veces en km_rios.
    """
    print(f"\n{SEP}\n  SOLAPAMIENTO INTERNO: capa de ríos (ANA)\n{SEP}")

    gdf = gpd.read_file(RIOS_SHP).to_crs(CRS_PROYECTADO)
    gdf["geometry"] = gdf.geometry.make_valid()

    km_suma = gdf.geometry.length.sum() / 1000

    from shapely.ops import unary_union as _uu
    print("  Calculando unary_union de todos los segmentos (puede tardar)...")
    km_union = _uu(gdf.geometry.tolist()).length / 1000

    ratio = km_union / km_suma if km_suma > 0 else float("nan")

    print(f"\n[1] Suma de longitudes individuales: {km_suma:,.2f} km")
    print(f"    Longitud de la unión (red única):  {km_union:,.2f} km")
    print(f"    Ratio unión/suma:                  {ratio:.4f}")
    print(f"    Solapamiento efectivo:             {(1-ratio)*100:.4f}%")

    print(f"\n[2] Conclusión:")
    if ratio >= 0.999:
        print("    Sin solapamiento interno significativo — cada segmento es")
        print("    geométricamente independiente. km_rios no tiene doble conteo.")
    elif ratio >= 0.98:
        print(f"    Solapamiento menor al 2% ({(1-ratio)*100:.2f}%) — impacto despreciable.")
    else:
        print(f"    *** Solapamiento de {(1-ratio)*100:.2f}% — algunos segmentos comparten")
        print("    tramos. construir_rios.py sobreestima km_rios. Investigar.")
    print(SEP)

def validar_geometrias_descartadas_rios(distritos_gdf):
    """Cuantifica geometrías no-lineales en el overlay rios × distritos
    (keep_geom_type=False). Misma lógica que para carreteras, pero sin
    unary_union previo — rios.shp es fuente única sin solapamiento entre capas.
    """
    print(f"\n{SEP}\n  GEOMETRÍAS DESCARTADAS: overlay ríos × distritos\n{SEP}")

    rios_gdf = gpd.read_file(RIOS_SHP)
    distritos_utm = distritos_gdf.to_crs(CRS_PROYECTADO)
    rios_utm = rios_gdf.to_crs(CRS_PROYECTADO)
    rios_utm["geometry"] = rios_utm.geometry.make_valid()
    distritos_utm["geometry"] = distritos_utm.geometry.make_valid()

    interseccion_raw = gpd.overlay(
        rios_utm[["geometry"]],
        distritos_utm[[GPKG_COL_GEOCODE, "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )

    es_lineal = interseccion_raw.geometry.geom_type.isin(["LineString", "MultiLineString"])
    n_descartadas = int((~es_lineal).sum())

    print(f"[1] Geometrías totales del overlay: {len(interseccion_raw)}")
    print(f"    Descartadas (no lineales): {n_descartadas}")
    if n_descartadas:
        print(f"    Tipos: {interseccion_raw[~es_lineal].geometry.geom_type.value_counts().to_dict()}")

    print(f"\n[2] Conclusión:")
    if n_descartadas == 0:
        print("    Sin pérdida — el overlay no degenera ningún segmento a Point.")
    else:
        pct = n_descartadas / len(interseccion_raw) * 100
        print(f"    {n_descartadas} geometrías descartadas ({pct:.4f}% del total) — revisar si")
        print(f"    representan segmentos largos degenerados o solo tangencias triviales.")
    print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE PRINCIPAL — activar/desactivar secciones según lo que se quiera probar
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── 1. Validación de módulo construir_* ───────────────────────────────
    print("Cargando distritos...")
    distritos_gdf = gpd.read_file(DISTRITOS_ALTO_CAMBIO_GPKG)
    print(f"  {len(distritos_gdf)} distritos cargados")

    run_test("construir_carreteras", construir_carreteras, "km_carreteras", distritos_gdf)

    # from O3.r8_r9_r10.construir_rios import construir_rios
    # run_test("construir_rios", construir_rios, "km_rios", distritos_gdf)

    # ── 2. Validación de método de longitud ───────────────────────────────
    # validar_longitud_rios()
    # validar_longitud_carreteras()
    # validar_longitud_es_fin_menos_inicio()

    # ── 3. Validación de doble conteo entre capas viales ─────────────────
    # validar_doble_conteo_carreteras()

    # ── 4. Geometrías descartadas en overlay final (keep_geom_type=False) ──
    # validar_geometrias_descartadas_overlay(distritos_gdf)

    # ── 5. Duplicados internos en capa de ríos ────────────────────────────
    # validar_duplicados_rios()
    validar_geometrias_descartadas_rios(distritos_gdf)
    validar_solapamiento_interno_rios()

    # ── 6. Void propagation: NaN en dem_utm post-reproyección ─────────────
    # (Punto 6 del análisis metodológico de pendiente)
    # validar_void_propagacion_dem_utm(distritos_gdf)

    # ── 7. Horn vs np.gradient vs gdaldem slope (Punto 2) ─────────────────
    # Requiere construir_elevacion() ejecutada (necesita elevacion_por_distrito.csv)
    # validar_horn_vs_gdaldem_slope(distritos_gdf)
