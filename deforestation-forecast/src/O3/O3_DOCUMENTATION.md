# O3 — Módulo de Variables Locales

**Propósito:** Construir un panel de variables contextuales por distrito amazónico (presión antrópica, infraestructura, protección legal, topografía) para enriquecer los paneles de cobertura forestal de O1 antes del entrenamiento de modelos.  
**Universo:** 200 distritos amazónicos seleccionados por O1 (top-N por % de cambio forestal).  
**Período temporal:** 1985–2024 (40 años, alineado con O1).  
**Punto de entrada:** `python -m O3.r8_r9_r10.main`

---

## Tabla de contenidos

1. [Propósito del módulo](#1-propósito-del-módulo)
2. [Estructura del módulo](#2-estructura-del-módulo)
3. [Flujo de procesamiento](#3-flujo-de-procesamiento)
4. [Módulos y scripts](#4-módulos-y-scripts)
5. [Variables producidas](#5-variables-producidas)
6. [Datos de entrada](#6-datos-de-entrada)
7. [Inventario de salidas](#7-inventario-de-salidas)
8. [Referencia de configuración](#8-referencia-de-configuración)
9. [Decisiones de diseño](#9-decisiones-de-diseño)
10. [Convenciones internas](#10-convenciones-internas)
11. [Dependencias entre módulos](#11-dependencias-entre-módulos)
12. [Mejoras técnicas aplicadas](#12-mejoras-técnicas-aplicadas)

---

## 1. Propósito del módulo

O1 produce un panel de cobertura boscosa por distrito y año (`pct_bosque`). O2 entrena modelos predictivos sobre ese panel. O3 extiende el panel con **variables locales** que representan los factores contextuales asociados a la dinámica de deforestación: presión agropecuaria, infraestructura vial, red hidrográfica, áreas de protección legal y condiciones topográficas.

El módulo resuelve tres problemas metodológicos distintos:

1. **Heterogeneidad de fuentes.** Las variables provienen de cuatro tipos de fuente incompatibles (rasters MapBiomas, shapefiles vectoriales, DEM SRTM) que requieren cadenas de procesamiento diferentes.
2. **Mezcla temporal/estática.** Algunas variables cambian año a año (cobertura agropecuaria, ANP); otras son constantes en el período estudiado (red vial, ríos, topografía). El pipeline distingue explícitamente ambos tipos.
3. **Integración en un panel único.** Todas las variables deben converger en un formato largo consistente `(geocode, anio)` para el merge con O1.

**Producto final:** cuatro CSVs en `data/interim/O3/panel-integrado/`:

| Archivo | Descripción |
|---------|-------------|
| `panel_integrado.csv` | Panel completo: todas las variables disponibles |
| `panel_integrado_light.csv` | Panel reducido: las 6 variables del modelo |
| `panel_integrado_entrenamiento.csv` | O1 train + variables O3 light |
| `panel_integrado_generalizacion.csv` | O1 gen + variables O3 light |

---

## 2. Estructura del módulo

### Árbol de código fuente

```
src/O3/
├── __init__.py
├── config.py                        — rutas, constantes, logging, creación de directorios
├── utils.py                         — guardar_csv(), guardar_metadatos(), log_config(), validar_fuentes()
├── O3_DOCUMENTATION.md              — este documento
│
└── r8_r9_r10/                       — pipeline completo R8-R10
    ├── __init__.py
    ├── main.py                      — orquestador (4 pasos secuenciales)
    │
    ├── construir_agropecuaria.py    — cobertura agropecuaria por año (MapBiomas)
    ├── construir_rios_lagos.py      — cobertura ríos/lagos por año (MapBiomas, respaldo)
    ├── construir_urbano.py          — cobertura urbana por año (MapBiomas, respaldo)
    ├── construir_carreteras.py      — densidad vial estática (MTC 2018)
    ├── construir_rios.py            — densidad hidrográfica estática (ANA)
    ├── construir_anp.py             — cobertura ANP acumulada por año (SERNANP)
    ├── construir_elevacion.py       — elevación media por distrito (SRTM 30m)
    ├── construir_pendiente.py       — pendiente media por distrito (Horn 1981 sobre SRTM)
    ├── integrar_panel.py            — integra todos los CSV en los cuatro paneles finales
    └── test_construir.py            — validaciones y diagnósticos opcionales
```

### Árbol de datos

```
data/
├── raw/
│   └── variables-locales/
│       ├── redes-viales/
│       │   ├── nacional/red_vial_nacional_dic18.shp
│       │   ├── departamental/red_vial_departamental_dic18.shp
│       │   └── vecinal/red_vial_vecinal_dic18.shp
│       ├── rios/Rios.shp
│       ├── anp/
│       │   ├── ANP Nacional Definitivas/ANPNacionalDefinitivas.shp
│       │   ├── Zonas Reservadas/ZonasReservadas.shp
│       │   ├── Áreas de Conservación Regional/AreasdeConservacionRegional.shp
│       │   └── Áreas de Conservación Privada/AreasdeConservacionPrivada.shp
│       └── elevacion/
│           └── S##W###.SRTMGL1.hgt/S##W###.hgt   (tiles SRTM, glob recursivo)
│
└── interim/
    └── O3/
        ├── variables/
        │   ├── agropecuaria/
        │   │   ├── agropecuaria_por_distrito.csv
        │   │   └── agropecuaria_metadatos.csv
        │   ├── carreteras/
        │   │   ├── carreteras_por_distrito.csv
        │   │   └── carreteras_metadatos.csv
        │   ├── rios/
        │   │   ├── rios_por_distrito.csv
        │   │   └── rios_metadatos.csv
        │   ├── anp/
        │   │   ├── anp_por_distrito.csv
        │   │   ├── anp_metadatos.csv
        │   │   └── anp_resumen_anual.csv
        │   ├── elevacion/
        │   │   ├── elevacion_por_distrito.csv
        │   │   ├── elevacion_metadatos.csv
        │   │   └── dem_mosaico.tif
        │   └── pendiente/
        │       ├── pendiente_por_distrito.csv
        │       ├── pendiente_metadatos.csv
        │       └── pendiente.tif              (BigGeoTIFF float32 ~5.5 GB)
        ├── variables-respaldo/
        │   ├── rios_lagos/
        │   │   ├── rios_lagos_por_distrito.csv
        │   │   └── rios_lagos_metadatos.csv
        │   └── urbano/
        │       ├── urbano_por_distrito.csv
        │       └── urbano_metadatos.csv
        └── panel-integrado/
            ├── panel_integrado.csv
            ├── panel_integrado_light.csv
            ├── panel_integrado_entrenamiento.csv
            ├── panel_integrado_generalizacion.csv
            ├── reporte_integracion.csv
            └── panel_metadatos.csv
```

---

## 3. Flujo de procesamiento

```
ENTRADAS
  data/interim/O1/distritos-alto-cambio/distritos_alto_cambio.gpkg  (200 distritos)
  data/raw/variables-locales/  (redes viales, ríos, ANP, SRTM)
  data/interim/O1/mapas-amazonia/peru_amazonia_YYYY.tif  (40 rasters MapBiomas)
        │
        ▼
┌────────────────────────────────────────────────────────────────────┐
│ PASO 0 / main.py                                                   │
│  Carga distritos_alto_cambio.gpkg                                  │
│  validar_fuentes() — verifica existencia de todas las fuentes      │
└─────────────────────────────────┬──────────────────────────────────┘
                                  │
        ▼
┌────────────────────────────────────────────────────────────────────┐
│ PASO 1 — Variables MapBiomas (temporales, 1985-2024)               │
│                                                                    │
│  construir_agropecuaria()  →  pct_agropecuario (200×40)           │
│  construir_rios_lagos()    →  pct_rios_lagos (respaldo)            │
│  construir_urbano()        →  pct_urbano (respaldo)                │
└─────────────────────────────────┬──────────────────────────────────┘
                                  │
        ▼
┌────────────────────────────────────────────────────────────────────┐
│ PASO 2 — Variables vectoriales                                     │
│                                                                    │
│  construir_carreteras()  →  densidad_carreteras_km_km2 (estática)  │
│  construir_rios()        →  densidad_rios_km_km2 (estática)        │
│  construir_anp()         →  pct_anp, tiene_anp (temporal, 200×40)  │
└─────────────────────────────────┬──────────────────────────────────┘
                                  │
        ▼
┌────────────────────────────────────────────────────────────────────┐
│ PASO 3 — Variables SRTM (estáticas)                                │
│                                                                    │
│  construir_elevacion()                                             │
│     • crear_mosaico_dem() si no existe dem_mosaico.tif            │
│     • zonal_stats → elev_media_m, elev_mediana_m, ... (200 filas) │
│                                                                    │
│  construir_pendiente()                                             │
│     • calcular_slope_raster() si no existe pendiente.tif          │
│       (DEM → UTM → Horn 1981 → BigGeoTIFF float32)                │
│     • zonal_stats → pendiente_media_deg, ... (200 filas)          │
└─────────────────────────────────┬──────────────────────────────────┘
                                  │
        ▼
┌────────────────────────────────────────────────────────────────────┐
│ PASO 4 — Panel integrado / integrar_panel()                        │
│                                                                    │
│  1. Esqueleto 200×40 filas                                         │
│  2. Merge left de todas las variables (temporales + estáticas)     │
│  3. panel_integrado.csv  (todas las columnas)                      │
│  4. panel_integrado_light.csv  (6 variables del modelo)            │
│  5. left-merge O1 train + O3 light → panel_integrado_entrenamiento │
│  6. left-merge O1 gen   + O3 light → panel_integrado_generalizacion│
│  7. reporte_integracion.csv (completitud % por variable)           │
└────────────────────────────────────────────────────────────────────┘
                                  │
        ▼
SALIDA para O4 (modelo multivariable)
```

---

## 4. Módulos y scripts

### `config.py`
Centraliza todas las rutas de fuentes RAW, rutas de salida, constantes de clases MapBiomas, CRS y el rango temporal. Crea todos los directorios de salida de forma idempotente al importarse.

### `utils.py`

| Función | Descripción |
|---------|-------------|
| `guardar_csv(df, ruta)` | Elimina columna `geometry` si existe; guarda UTF-8 sin índice |
| `guardar_metadatos(meta_dict, ruta)` | Guarda diccionario como CSV de una fila |
| `log_config()` | Registra via `logging.info` todos los parámetros activos al inicio |
| `validar_fuentes()` | Verifica existencia de todas las fuentes RAW antes de ejecutar; lanza `RuntimeError` con lista completa de faltantes |

### `main.py`
Orquestador con estructura `def main()`. Ejecuta los 4 pasos en orden secuencial; cada paso es idempotente. Llama a `log_config()` y `validar_fuentes()` antes de iniciar el procesamiento.

### `construir_agropecuaria.py`
Calcula `pct_agropecuario` por (distrito, año). Usa `rasterstats.zonal_stats()` con `categorical=True` sobre los rasters `peru_amazonia_YYYY.tif` de O1. El denominador excluye la clase 27 (no observado) para no penalizar años con alta nubosidad.

### `construir_rios_lagos.py` / `construir_urbano.py`
Misma lógica que `construir_agropecuaria` para las clases {33} (ríos/lagos) y {24} (urbano). Se clasifican como **variables de respaldo** porque no forman parte del panel light del modelo pero se incluyen en el panel completo para análisis exploratorio.

### `construir_carreteras.py`
Calcula `km_carreteras` y `densidad_carreteras_km_km2` por distrito. Concatena las tres capas MTC (nacional, departamental, vecinal), aplica deduplicación WKB exacta, unifica la red con `unary_union` para eliminar solapamientos entre capas, y realiza un overlay de intersección con los distritos en UTM. Variable **estática** (sin columna `anio`).

### `construir_rios.py`
Calcula `km_rios` y `densidad_rios_km_km2` por distrito. Intersecta la red hidrográfica ANA con los distritos en UTM. Variable **estática**.

### `construir_anp.py`
Calcula `pct_anp` (fracción del área distrital cubierta por ANP) acumulada año a año. Unifica las cuatro capas SERNANP (ANP Nacionales, Zonas Reservadas, ACR, ACP) con fecha de establecimiento estandarizada. Estrategia incremental: pre-calcula la intersección ANP × distritos una sola vez; en cada año solo actualiza los distritos con ANPs nuevas (`felec.year == anio`); los demás reutilizan la unión del año anterior. Variable **temporal**.

### `construir_elevacion.py`
Calcula estadísticas de elevación por distrito usando el DEM SRTM 30m. Crea un mosaico GeoTIFF de los tiles disponibles recortado a la extensión de los distritos (una sola vez), luego aplica `zonal_stats`. Variable **estática**.

### `construir_pendiente.py`
Calcula estadísticas de pendiente en grados por distrito. Dos etapas internas:
1. `calcular_slope_raster()` — reproyecta el mosaico DEM a UTM (píxeles cuadrados en metros), aplica el algoritmo de Horn (1981) con `scipy.ndimage.convolve`, escribe el resultado como BigGeoTIFF float32.
2. `construir_pendiente()` — aplica `zonal_stats` sobre el raster de pendiente con los distritos reproyectados al mismo CRS.

Incluye detección de raster corrupto: si `pendiente.tif` existe pero `rasterio.open()` falla, lo elimina y lo recalcula.

### `integrar_panel.py`
Ensambla los ocho CSV individuales en los cuatro paneles finales. Construye primero un esqueleto 200 × 40 filas, luego hace left-merge sucesivo de cada variable sobre las claves `(geocode, anio)` para variables temporales o `geocode` para variables estáticas. Garantiza que `geocode` sea `str` en todos los merges.

### `test_construir.py`
Funciones de diagnóstico opcionales (no se ejecutan en el pipeline normal):

| Función | Propósito |
|---------|-----------|
| `validar_void_propagacion_dem_utm()` | Detecta si los NaN de borde UTM del DEM reproyectado solapan con algún distrito |
| `validar_horn_vs_gdaldem_slope()` | Compara Horn vs `np.gradient` en distritos de máxima/mínima elevación; llama a `gdaldem` si está disponible |

---

## 5. Variables producidas

### Variables temporales (200 distritos × 40 años = 8 000 filas)

| Variable | Fuente | Descripción |
|----------|--------|-------------|
| `pct_agropecuario` | MapBiomas C3 | Fracción del área distrital cubierta por clases agropecuarias {9,15,21,35,40} |
| `pct_anp` | SERNANP (4 capas) | Fracción del área distrital dentro de ANP establecidas hasta ese año |
| `tiene_anp` | Derivada | 1 si `pct_anp > 0`, 0 en caso contrario |
| `pct_rios_lagos` | MapBiomas C3, clase {33} | Fracción de cobertura de ríos/lagos (respaldo) |
| `pct_urbano` | MapBiomas C3, clase {24} | Fracción de cobertura urbana (respaldo) |

### Variables estáticas (200 distritos, 1 fila por distrito)

| Variable | Fuente | Descripción |
|----------|--------|-------------|
| `km_carreteras` | MTC dic-2018 | Longitud total de red vial dentro del distrito |
| `area_utm_km2` | Geometría distrito | Área en km² (UTM Zona 18S) |
| `densidad_carreteras_km_km2` | Derivada | km de carreteras por km² de distrito |
| `km_rios` | ANA | Longitud total de ejes fluviales dentro del distrito |
| `densidad_rios_km_km2` | Derivada | km de ríos por km² de distrito |
| `elev_media_m` | SRTM SRTMGL1 30m | Elevación media (m) |
| `elev_mediana_m` | SRTM | Elevación mediana (m) |
| `elev_std_m` | SRTM | Desviación estándar de elevación (m) |
| `elev_min_m` / `elev_max_m` | SRTM | Rango de elevación (m) |
| `elev_count_px` | SRTM | Píxeles válidos contados |
| `pendiente_media_deg` | Derivada SRTM | Pendiente media (grados), algoritmo Horn (1981) |
| `pendiente_mediana_deg` | Derivada SRTM | Pendiente mediana (grados) |
| `pendiente_std_deg` | Derivada SRTM | Desviación estándar de pendiente (grados) |
| `pendiente_min_deg` / `pendiente_max_deg` | Derivada SRTM | Rango de pendiente (grados) |
| `pendiente_count_px` | Derivada SRTM | Píxeles válidos contados |

### Variables del modelo (panel light)

Las 6 variables seleccionadas como predictores del modelo O4:

```
pct_agropecuario, pct_anp,
densidad_carreteras_km_km2, densidad_rios_km_km2,
elev_media_m, pendiente_media_deg
```

---

## 6. Datos de entrada

### Fuentes RAW (data/raw/variables-locales/)

| Recurso | Tipo | Descripción |
|---------|------|-------------|
| `redes-viales/nacional/*.shp` | Shapefile lineal | Red vial nacional MTC, diciembre 2018 |
| `redes-viales/departamental/*.shp` | Shapefile lineal | Red vial departamental MTC, diciembre 2018 |
| `redes-viales/vecinal/*.shp` | Shapefile lineal | Red vial vecinal MTC, diciembre 2018 |
| `rios/Rios.shp` | Shapefile lineal | Red hidrográfica nacional ANA |
| `anp/ANP Nacional Definitivas/*.shp` | Shapefile poligonal | ANP nacionales definitivas SERNANP |
| `anp/Zonas Reservadas/*.shp` | Shapefile poligonal | Zonas Reservadas SERNANP |
| `anp/Áreas de Conservación Regional/*.shp` | Shapefile poligonal | ACR SERNANP |
| `anp/Áreas de Conservación Privada/*.shp` | Shapefile poligonal | ACP SERNANP |
| `elevacion/**/*.hgt` | Binario SRTM | Tiles SRTM SRTMGL1 1 arc-second (~30 m), formato HGT |

### Fuentes de O1 consumidas por O3

| Recurso | Descripción |
|---------|-------------|
| `data/interim/O1/distritos-alto-cambio/distritos_alto_cambio.gpkg` | Geometrías de los 200 distritos |
| `data/interim/O1/mapas-amazonia/peru_amazonia_YYYY.tif` | 40 rasters MapBiomas C3 (clases crudas) |
| `data/interim/O1/series-temporales/entrenamiento/distritos_entrenamiento.csv` | Panel de cobertura forestal O1 (train) |
| `data/interim/O1/series-temporales/generalizacion-espacial/distritos_generalizacion_espacial.csv` | Panel de cobertura forestal O1 (gen) |

---

## 7. Inventario de salidas

```
data/interim/O3/
│
├── variables/
│   ├── agropecuaria/
│   │   ├── agropecuaria_por_distrito.csv     (8 000 filas: geocode, departamento, distrito, anio, pct_agropecuario)
│   │   └── agropecuaria_metadatos.csv
│   │
│   ├── carreteras/
│   │   ├── carreteras_por_distrito.csv        (200 filas: geocode, ..., km_carreteras, area_utm_km2, densidad_carreteras_km_km2)
│   │   └── carreteras_metadatos.csv
│   │
│   ├── rios/
│   │   ├── rios_por_distrito.csv              (200 filas: geocode, ..., km_rios, area_utm_km2, densidad_rios_km_km2)
│   │   └── rios_metadatos.csv
│   │
│   ├── anp/
│   │   ├── anp_por_distrito.csv               (8 000 filas: geocode, ..., anio, pct_anp, tiene_anp)
│   │   ├── anp_metadatos.csv
│   │   └── anp_resumen_anual.csv              (40 filas: expansión temporal de la cobertura ANP)
│   │
│   ├── elevacion/
│   │   ├── dem_mosaico.tif                    (mosaico SRTM, int16, CRS_GEOG, LZW)
│   │   ├── elevacion_por_distrito.csv         (200 filas: elev_media_m, elev_mediana_m, ...)
│   │   └── elevacion_metadatos.csv
│   │
│   └── pendiente/
│       ├── pendiente.tif                      (BigGeoTIFF float32 ~5.5 GB, EPSG:32718, Horn 1981)
│       ├── pendiente_por_distrito.csv         (200 filas: pendiente_media_deg, pendiente_mediana_deg, ...)
│       └── pendiente_metadatos.csv
│
├── variables-respaldo/
│   ├── rios_lagos/
│   │   ├── rios_lagos_por_distrito.csv        (8 000 filas: geocode, ..., pct_rios_lagos)
│   │   └── rios_lagos_metadatos.csv
│   └── urbano/
│       ├── urbano_por_distrito.csv            (8 000 filas: geocode, ..., pct_urbano)
│       └── urbano_metadatos.csv
│
└── panel-integrado/
    ├── panel_integrado.csv                    (8 000 filas, ~17 columnas — todas las variables)
    ├── panel_integrado_light.csv              (8 000 filas, 10 columnas — 6 vars del modelo + id)
    ├── panel_integrado_entrenamiento.csv      (7 200 filas — O1 train + O3 light)
    ├── panel_integrado_generalizacion.csv     (800 filas  — O1 gen + O3 light)
    ├── reporte_integracion.csv                (completitud % por variable)
    └── panel_metadatos.csv
```

### Esquema de los paneles finales

**panel_integrado_entrenamiento.csv / _generalizacion.csv:**

| Columna | Origen | Descripción |
|---------|--------|-------------|
| `geocode` | O1 | Código geográfico del distrito |
| `departamento` | O1 | Departamento |
| `distrito` | O1 | Nombre del distrito |
| `anio` | O1 | Año (1985–2024) |
| `pix_total` | O1 | Total de píxeles válidos |
| `pix_bosque` | O1 | Píxeles de bosque |
| `pix_no_bosque` | O1 | Píxeles de no-bosque |
| `pct_bosque` | O1 | Fracción de cobertura boscosa **[variable objetivo O2]** |
| `pct_no_bosque` | O1 | Fracción de no-bosque |
| `pct_agropecuario` | O3 | Fracción agropecuaria |
| `pct_anp` | O3 | Fracción ANP acumulada |
| `densidad_carreteras_km_km2` | O3 | Densidad vial |
| `densidad_rios_km_km2` | O3 | Densidad hidrográfica |
| `elev_media_m` | O3 | Elevación media |
| `pendiente_media_deg` | O3 | Pendiente media |

---

## 8. Referencia de configuración

### Sistemas de referencia

| Constante | Valor | Uso |
|-----------|-------|-----|
| `CRS_PROYECTADO` | `EPSG:32718` | Cálculos de área y longitud; reproyección DEM para pendiente |
| `CRS_GEOG` | `EPSG:4326` | Almacenamiento de geometrías y rasters (mosaico DEM) |

### Clases MapBiomas C3

| Constante | Valor | Variable |
|-----------|-------|----------|
| `CLASES_AGROPECUARIA` | `{9, 15, 21, 35, 40}` | `pct_agropecuario` |
| `CLASES_RIOS_LAGOS` | `{33}` | `pct_rios_lagos` (respaldo) |
| `CLASES_URBANO` | `{24}` | `pct_urbano` (respaldo) |
| `CLASE_NO_OBSERVADO` | `27` | Excluido del denominador en todos los cálculos de fracción |
| `CLASE_FONDO` | `0` | Excluido via `nodata=0` en `zonal_stats` |

### Constantes SRTM

| Constante | Valor | Descripción |
|-----------|-------|-------------|
| `SRTM_NODATA` | `-32768` | Valor nodata estándar SRTMGL1 (int16) |
| `SLOPE_NODATA` | `-9999.0` | Valor nodata del raster de pendiente derivado |

### Rango temporal

| Constante | Valor | Descripción |
|-----------|-------|-------------|
| `ANIOS` | `range(1985, 2025)` | 40 años, alineado con O1 |

---

## 9. Decisiones de diseño

### 9.1 Algoritmo de pendiente: Horn (1981)

**Decisión: Horn (1981) en lugar de `np.gradient`**

El algoritmo de Horn (1981) calcula las derivadas parciales `dz/dx` y `dz/dy` con una ventana 3×3 ponderada idéntica a la que usan `gdaldem slope`, QGIS y ArcGIS. `np.gradient` aplica diferencias centradas de primer orden que equivalen a una ventana 2×2 sin ponderación.

La diferencia es relevante en terreno plano (Amazonía baja): `np.gradient` sobrestima la pendiente en ~7.84% porque amplifica el ruido vertical del SRTM al operar sobre un vecindario más pequeño. En zonas montañosas la diferencia baja a ~0.27%. Horn es más robusto porque promedia sobre un vecindario mayor antes de dividir.

**Implementación:**

```python
from scipy.ndimage import convolve as _convolve

kern_dx = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=np.float64) / (8.0 * px_m)
kern_dy = np.array([[ 1., 2., 1.], [ 0., 0., 0.], [-1.,-2.,-1.]], dtype=np.float64) / (8.0 * py_m)
slope   = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
```

---

### 9.2 Reproyección UTM para el DEM antes de calcular la pendiente

**Decisión: Reproyectar el DEM a EPSG:32718 antes de aplicar Horn**

El DEM SRTM en EPSG:4326 tiene píxeles en grados. Para que los kernels de Horn sean correctos, cada píxel debe tener una dimensión física conocida y constante. Reproyectar a UTM produce píxeles cuadrados en metros, eliminando la corrección aproximada `111 000 × cos(lat)` que sería necesaria de otro modo. El error de esa aproximación varía con la latitud a lo largo de la Amazonía peruana.

---

### 9.3 BigGeoTIFF para el raster de pendiente

**Decisión: `"bigtiff": "YES"` en los metadatos de escritura del raster de pendiente**

El raster de pendiente como float32 sobre la extensión amazónica peruana (~5.55 GB) supera el límite de 4 GB del formato TIFF estándar. Sin BigGeoTIFF, la escritura falla con `TIFFAppendToStrip: Maximum TIFF file size exceeded` y deja un archivo parcial en disco. La detección de raster corrupto (try/except sobre `rasterio.open`) permite eliminarlo y recalcular automáticamente en la siguiente ejecución.

---

### 9.4 Acumulación incremental de ANP

**Decisión: Union geométrica acumulada por distrito, año a año**

Las ANP no desaparecen una vez establecidas: la fracción del año `t` incluye todas las ANP con `felec.year ≤ t`. La implementación pre-calcula la intersección ANP × distritos una sola vez y mantiene un diccionario `{geocode → union_geometry}` que se actualiza solo para los distritos que tienen ANPs nuevas en cada año. Esto reduce el costo de O(N_anios × N_distritos × N_ANPs) a O(N_intersecciones × N_activaciones), que es mucho menor porque la mayoría de ANPs se establecieron después de 1985.

`unary_union` por distrito resuelve el solapamiento entre categorías ANP (p. ej., ACP dentro de ANP Nacional) sin doble conteo.

---

### 9.5 Red vial: deduplicación y unión topológica

**Decisión: WKB-dedup + `unary_union` antes de la intersección de carreteras**

Las tres capas MTC (nacional, departamental, vecinal) contienen segmentos que se solapan entre capas (la misma carretera aparece tanto en la capa nacional como en la departamental). Sin deduplicación, el `km_carreteras` por distrito se inflaría contando la misma longitud varias veces. La secuencia de limpieza es:
1. Eliminar segmentos duplicados por WKB exacto
2. `unary_union` sobre toda la red para fusionar tramos concurrentes

Este mismo patrón no se aplica a ríos porque la red hidrográfica ANA es una capa única sin solapamientos entre capas.

---

### 9.6 Denominador en las fracciones MapBiomas

**Decisión: Excluir la clase 27 (no observado) del denominador**

En años con alta cobertura de nubes, la clase 27 puede cubrir una fracción significativa del área distrital. Si se incluye en el denominador, `pct_agropecuario` se subestimaría artificialmente. La exclusión garantiza que la fracción se calcule sobre píxeles con información real, haciendo las series comparables entre años de distinta nubosidad.

---

### 9.7 Arquitectura del panel: completo vs. light vs. train/gen

**Decisión: Tres niveles de panel con responsabilidades distintas**

| Panel | Filas | Propósito |
|-------|-------|-----------|
| `panel_integrado.csv` | 8 000 | Exploración: todas las variables disponibles para análisis descriptivo |
| `panel_integrado_light.csv` | 8 000 | Modelo: solo las 6 variables seleccionadas como predictores |
| `panel_integrado_entrenamiento.csv` | 7 200 | Entrenamiento: O1 base (pct_bosque + pixtotales) + O3 light, left-merge |
| `panel_integrado_generalizacion.csv` | 800 | Generalización: ídem sobre los 20 distritos de prueba |

Los paneles train/gen se construyen como **left-merge sobre O1** (no como filtro de panel_light). Esto preserva todas las columnas de O1 — incluyendo `pct_bosque`, `pix_bosque`, `pix_no_bosque` — que son la base del problema y no se regeneran en O3.

---

### 9.8 Normalización de `geocode` a tipo `str`

**Decisión: Forzar `dtype={"geocode": str}` en todos los `pd.read_csv`**

Los GPKG devuelven `geocode` como objeto (str). `pd.read_csv` infiere automáticamente `int64` para columnas numéricas. Mezclar ambos tipos en un merge produce `ValueError: merging on object and int64 columns`. La normalización a `str` es la solución más robusta porque `geocode` es un identificador, no un número.

---

### 9.9 Idempotencia y validación de fuentes

**Decisión: Verificar existencia de fuentes al inicio; skip por CSV + META**

`validar_fuentes()` comprueba la existencia de todas las rutas RAW antes de ejecutar cualquier paso. Un error de ruta que se manifestara a mitad del pipeline (p. ej., dentro de `construir_anp` después de 20 minutos de cómputo) sería costoso; detectarlo al inicio es más rápido.

Cada módulo de construcción verifica la existencia de tanto el CSV como el archivo de metadatos antes de hacer skip. Verificar solo uno puede dejar el sistema en estado inconsistente si la ejecución se interrumpió a mitad.

---

## 10. Convenciones internas

### Nomenclatura de archivos

| Tipo | Patrón |
|------|--------|
| CSV de variable | `{variable}_por_distrito.csv` |
| Metadatos de variable | `{variable}_metadatos.csv` |
| Mosaico DEM | `dem_mosaico.tif` |
| Raster de pendiente | `pendiente.tif` |
| Panel final | `panel_integrado{_sufijo}.csv` |
| Reporte de completitud | `reporte_integracion.csv` |

### Estándares de datos

| Aspecto | Estándar |
|---------|----------|
| CRS almacenamiento vectorial | EPSG:4326 |
| CRS cálculo de áreas/longitudes | EPSG:32718 (UTM 18S) |
| dtype raster DEM | int16 (SRTM original) |
| dtype raster pendiente | float32 (BigGeoTIFF) |
| Compresión rasters | LZW |
| Encoding CSV | UTF-8 |
| Formato tabular | CSV en formato largo, una fila por (distrito, año) para variables temporales; una fila por distrito para variables estáticas |
| Tipo `geocode` | `str` en todos los CSV y merges |
| Columnas tabular | snake_case en español (`pct_agropecuario`, `densidad_rios_km_km2`) |

### Principios de diseño

| Principio | Manifestación en O3 |
|-----------|---------------------|
| Idempotencia | Cada módulo verifica CSV + META antes de procesar |
| Validación anticipada | `validar_fuentes()` comprueba todas las rutas RAW al inicio |
| Trazabilidad | Metadatos CSV paralelos a cada output de variable |
| Separación tipo/uso | Variables temporales vs. estáticas documentadas explícitamente |
| Configuración centralizada | Cero hardcoding; todos los parámetros en `config.py` |
| Normalización de tipos | `geocode` forzado a `str` en toda la cadena |

---

## 11. Dependencias entre módulos

### Grafo de importaciones

```
config.py
    ├── utils.py
    ├── main.py
    ├── construir_agropecuaria.py   (ANIOS, MAPBIOMAS_AMAZONIA_PATRON, CLASES_AGROPECUARIA, ...)
    ├── construir_rios_lagos.py     (ANIOS, MAPBIOMAS_AMAZONIA_PATRON, CLASES_RIOS_LAGOS, ...)
    ├── construir_urbano.py         (ANIOS, MAPBIOMAS_AMAZONIA_PATRON, CLASES_URBANO, ...)
    ├── construir_carreteras.py     (CRS_PROYECTADO, CARRETERAS_*_SHP, ...)
    ├── construir_rios.py           (CRS_PROYECTADO, RIOS_SHP, ...)
    ├── construir_anp.py            (CRS_PROYECTADO, ANP_*_SHP, ANP_COL_FECHA_*, ...)
    ├── construir_elevacion.py      (CRS_GEOG, SRTM_TILES, SRTM_NODATA, ELEVACION_MOSAIC, ...)
    ├── construir_pendiente.py      (CRS_PROYECTADO, ELEVACION_MOSAIC, SLOPE_NODATA, ...)
    └── integrar_panel.py           (PANEL_CSV, PANEL_LIGHT_CSV, PANEL_ENTRENAMIENTO_CSV, ...)
```

### Orden de ejecución requerido

```
O1 (R1–R3)  → distritos_alto_cambio.gpkg + peru_amazonia_YYYY.tif
                    ↓
main.py [PASO 0]    → carga distritos_gdf
    ↓
construir_agropecuaria / construir_rios_lagos / construir_urbano
    ↓
construir_carreteras / construir_rios / construir_anp
    ↓
construir_elevacion  →  dem_mosaico.tif
    ↓
construir_pendiente  →  dem_mosaico.tif (insumo) → pendiente.tif
    ↓
integrar_panel  →  todos los CSV individuales → paneles finales
```

### Dependencias de O3 hacia O1

O3 consume cuatro productos de O1:

1. `distritos_alto_cambio.gpkg` — geometrías base de los 200 distritos
2. `peru_amazonia_YYYY.tif` (×40) — rasters anuales para variables MapBiomas
3. `distritos_entrenamiento.csv` — base del panel de entrenamiento
4. `distritos_generalizacion_espacial.csv` — base del panel de generalización

Si se regeneran los productos de O1 (p. ej., por cambio en `N_DISTRITOS_ALTO_CAMBIO`), los CSV de O3 deben borrarse manualmente y re-ejecutar el pipeline.

---

## 12. Mejoras técnicas aplicadas

| Problema | Solución | Archivos |
|----------|----------|----------|
| `np.gradient` sobrestima pendiente en terreno plano (~7.84% en Amazonía baja) | Reemplazado por algoritmo Horn (1981) con `scipy.ndimage.convolve` | `construir_pendiente.py` |
| Raster de pendiente float32 supera límite 4 GB de TIFF estándar | `"bigtiff": "YES"` en `meta_utm.update()` | `construir_pendiente.py` |
| Archivo parcial corrupto tras fallo BIGTIFF persiste y causa skip silencioso | Detección via try/except `rasterio.open`; elimina y recalcula si falla | `construir_pendiente.py` |
| `ValueError` al mergear `geocode` (str GPKG vs. int64 CSV) | `dtype={"geocode": str}` en todos `pd.read_csv` + `.astype(str)` en `geocode_info` | `integrar_panel.py` |
| Paneles train/gen perdían columnas de O1 (`pct_bosque`, `pix_bosque`, etc.) | Cambio de filtro de panel_light a left-merge sobre O1 CSV como base | `integrar_panel.py` |
| Solapamiento entre capas MTC inflaba `km_carreteras` | WKB-dedup + `unary_union` antes de la intersección | `construir_carreteras.py` |
| Doble conteo de área ANP cuando una ACP cae dentro de un ANP Nacional | `unary_union` incremental por distrito produce la unión real | `construir_anp.py` |
| Cálculo ANP O(N×M) por año: muy lento con 40 años y muchos fragmentos | Pre-intersección única + actualización incremental solo para nuevas ANP por año | `construir_anp.py` |
