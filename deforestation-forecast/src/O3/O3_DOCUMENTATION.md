# O3 — Variables Locales y Modelos Multivariables

**Propósito:** (1) Construir un panel de variables contextuales por distrito amazónico (presión antrópica, infraestructura, protección legal, topografía) para enriquecer los paneles de cobertura forestal de O1 — submódulo `r8_r9_r10/`; (2) entrenar y evaluar modelos de pronóstico multivariables (MLP, LSTM, CNN 1D) sobre ese panel y compararlos contra los modelos univariables de O2 — submódulo `r11/`.  
**Universo:** 200 distritos amazónicos seleccionados por O1 para `r8_r9_r10/`; 180 distritos de entrenamiento de O1/O2 para `r11/` (el panel multivariable hereda el universo de entrenamiento de O2).  
**Período temporal:** 1985–2024 (40 años, alineado con O1/O2).  
**Puntos de entrada:** `python -m O3.r8_r9_r10.main` (construcción de variables) y `python -m O3.r11.main` (modelado multivariable), ejecutados en ese orden — R11 depende de los paneles que produce r8_r9_r10.

---

## Tabla de contenidos

### Parte I — Construcción de variables locales (`r8_r9_r10/`)

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

### Parte II — Modelos multivariables (`r11/`)

13. [Propósito del módulo R11](#13-propósito-del-módulo-r11)
14. [Estructura del módulo R11](#14-estructura-del-módulo-r11)
15. [Flujo de ejecución R11](#15-flujo-de-ejecución-r11)
16. [Diseño del problema](#16-diseño-del-problema)
17. [Construcción del panel y los datasets](#17-construcción-del-panel-y-los-datasets)
18. [Arquitectura de pipeline en dos fases](#18-arquitectura-de-pipeline-en-dos-fases)
19. [Protocolo de evaluación walk-forward](#19-protocolo-de-evaluación-walk-forward)
20. [Modelo MLP](#20-modelo-mlp)
21. [Modelo LSTM](#21-modelo-lstm)
22. [Modelo CNN 1D](#22-modelo-cnn-1d)
23. [Utilidades compartidas — utils.py](#23-utilidades-compartidas--utilspy)
24. [Verificación de configuraciones finales — seleccion_fase1.py](#24-verificación-de-configuraciones-finales--seleccion_fase1py)
25. [Comparación: O2 (base) vs. R11 (extendido)](#25-comparación-o2-base-vs-r11-extendido)
26. [Pronóstico 2025](#26-pronóstico-2025)
27. [Referencia de configuración R11](#27-referencia-de-configuración-r11)
28. [Inventario de salidas R11](#28-inventario-de-salidas-r11)
29. [Decisiones de diseño específicas de R11](#29-decisiones-de-diseño-específicas-de-r11)

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
| `panel_integrado_completo.csv` | Panel completo: todas las variables disponibles |
| `panel_integrado_modelo.csv` | Panel reducido: las 6 variables del modelo |
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
    ├── main.py                      — orquestador (5 pasos secuenciales)
    │
    ├── metadata_fuentes.py          — PASO 0: metadata estructural de las fuentes RAW (CRS, geometría/resolución, dimensiones)
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
        ├── distritos-alto-cambio/
        │   └── distritos_alto_cambio_metadatos_raw.csv   (PASO 0 — única fuente sin variable propia, ver §9.10)
        ├── variables/
        │   ├── agropecuaria/
        │   │   ├── agropecuaria_por_distrito.csv
        │   │   ├── agropecuaria_metadatos.csv
        │   │   └── agropecuaria_metadatos_raw.csv     (PASO 0 — los 40 años de MapBiomas, exhaustivo)
        │   ├── carreteras/
        │   │   ├── carreteras_por_distrito.csv
        │   │   ├── carreteras_metadatos.csv
        │   │   └── carreteras_metadatos_raw.csv       (PASO 0 — 3 shapefiles MTC)
        │   ├── rios/
        │   │   ├── rios_por_distrito.csv
        │   │   ├── rios_metadatos.csv
        │   │   └── rios_metadatos_raw.csv             (PASO 0 — Rios.shp)
        │   ├── anp/
        │   │   ├── anp_por_distrito.csv
        │   │   ├── anp_metadatos.csv
        │   │   ├── anp_resumen_anual.csv
        │   │   └── anp_metadatos_raw.csv              (PASO 0 — 4 shapefiles SERNANP)
        │   ├── elevacion/
        │   │   ├── elevacion_por_distrito.csv
        │   │   ├── elevacion_metadatos.csv
        │   │   ├── elevacion_metadatos_raw.csv        (PASO 0 — 163 tiles SRTM)
        │   │   └── dem_mosaico.tif
        │   └── pendiente/
        │       ├── pendiente_por_distrito.csv
        │       ├── pendiente_metadatos.csv
        │       ├── pendiente_metadatos_raw.csv        (PASO 0 — mismos 163 tiles SRTM)
        │       └── pendiente.tif              (BigGeoTIFF float32 ~5.5 GB)
        ├── variables-respaldo/
        │   ├── rios_lagos/
        │   │   ├── rios_lagos_por_distrito.csv
        │   │   ├── rios_lagos_metadatos.csv
        │   │   └── rios_lagos_metadatos_raw.csv       (PASO 0 — los 40 años de MapBiomas, exhaustivo)
        │   └── urbano/
        │       ├── urbano_por_distrito.csv
        │       ├── urbano_metadatos.csv
        │       └── urbano_metadatos_raw.csv           (PASO 0 — los 40 años de MapBiomas, exhaustivo)
        └── panel-integrado/
            ├── panel_integrado_completo.csv
            ├── panel_integrado_modelo.csv
            ├── panel_integrado_entrenamiento.csv
            ├── panel_integrado_generalizacion.csv
            ├── reporte_integracion.csv             (completitud por variable — base de cálculo, cobertura distrital, puntero a metadata_raw, ver §9.11)
            ├── reporte_completitud_anual.csv        (pct_agropecuario/pct_anp por año 1985-2024, ver §9.11)
            ├── reporte_ejecucion.csv                (metadata de la corrida completa de main.py — evidencia IOV R9, ver §9.11)
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
│  validar_fuentes() — verifica existencia de todas las fuentes      │
│  generar_metadata_fuentes() — abre cada fuente y extrae su         │
│     metadata estructural (CRS, geometría/resolución, dimensiones)  │
│     → {variable}_metadatos_raw.csv en cada carpeta de variable     │
│     → distritos-alto-cambio/..._metadatos_raw.csv (sin var. propia)│
│  Carga distritos_alto_cambio.gpkg                                  │
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
┌───────────────────────────────────────────────────────────────────────┐
│  PASO 4 — Panel integrado / integrar_panel()                          │
│                                                                       │
│   1. Esqueleto 200×40 filas                                           │
│   2. Merge left de todas las variables (temporales + estáticas)       │
│   3. panel_integrado_completo.csv  (todas las columnas)               │
│   4. panel_integrado_modelo.csv  (6 variables del modelo)             │
│   5. left-merge O1 train + O3 modelo → panel_integrado_entrenamiento  │
│   6. left-merge O1 gen   + O3 modelo → panel_integrado_generalizacion │
│   7. reporte_integracion.csv (completitud por variable: base de       │
│      cálculo según tipo, cobertura distrital, puntero a metadata_raw  │
│   8. reporte_completitud_anual.csv (pct_agropecuario/pct_anp × año)   │
│      — (7) y (8) se regeneran SIEMPRE, incluso si (3)-(6) ya existen  │
│      en caché (ver §9.11)                                             │
└────────────────────────────────────┬────────────────────────────────────┘
                                  │
        ▼
┌────────────────────────────────────────────────────────────────────┐
│ CIERRE main.py — reporte_ejecucion.csv (bloque finally)            │
│  fecha_inicio/fin, duración, último paso completado, estado, error │
│  se escribe incluso si el pipeline falla — evidencia del IOV de R9 │
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
| `validar_fuentes()` | Verifica existencia de todas las fuentes RAW antes de ejecutar; lanza `RuntimeError` con lista completa de faltantes. Solo existencia — no abre ningún archivo |

### `main.py`
Orquestador con estructura `def main()`. Ejecuta 5 pasos en orden secuencial (PASO 0 a PASO 4); los pasos de construcción de variables (1-3) son idempotentes, PASO 0 y PASO 4 no. Llama a `log_config()` y `validar_fuentes()` antes de iniciar el procesamiento. Todo el cuerpo (PASO 0 a PASO 4) corre dentro de un `try/except/finally`: el `finally` escribe `reporte_ejecucion.csv` (fecha de inicio/fin, duración, último paso completado, estado OK/ERROR) tanto si el pipeline termina bien como si falla a mitad de camino — es la evidencia que sustenta el IOV de R9 ("el pipeline se ejecuta de forma reproducible y sin errores"). Ver §9.11. Al ejecutarse como script (`__main__`) también llama a `iniciar_log_archivo("r8_r9_r10")`, igual que `r11/main.py` y `O4/r12_r13_r14/main.py`, para dejar un `.log` con timestamp en `data/logs/` además de la consola.

### `metadata_fuentes.py` (PASO 0)
Complementa a `validar_fuentes()` con una auditoría estructural: abre cada fuente RAW (9 vectoriales + los 40 años de MapBiomas + los 163 tiles SRTM, todos exhaustivos) y registra sus atributos (CRS, tipo de geometría/resolución, dimensiones, nodata, etc.) en `{variable}_metadatos_raw.csv`, junto al `_metadatos.csv` ya procesado de cada variable (`distritos_alto_cambio_metadatos_raw.csv`, sin variable propia, es la única excepción). Ver §9.10 para el detalle de diseño (por qué un archivo por variable y no un consolidado, por qué CSV, por qué no idempotente, y por qué `distritos_alto_cambio` no vive dentro de `variables/`).

### `construir_agropecuaria.py`
Calcula `pct_agropecuario` por (distrito, año). Usa `rasterstats.zonal_stats()` con `categorical=True` sobre los rasters `peru_amazonia_YYYY.tif` de O1. El denominador excluye la clase 27 (no observado) para no penalizar años con alta nubosidad.

### `construir_rios_lagos.py` / `construir_urbano.py`
Misma lógica que `construir_agropecuaria` para las clases {33} (ríos/lagos) y {24} (urbano). Se clasifican como **variables de respaldo** porque no forman parte del panel modelo pero se incluyen en el panel completo para análisis exploratorio.

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
Ensambla los ocho CSV individuales en los cuatro paneles finales. `integrar_panel()` solo orquesta: si los cuatro paneles ya existen en caché los carga; si no, delega la construcción a `_construir_paneles()` (esqueleto 200×40 filas, left-merge sucesivo de cada variable sobre `(geocode, anio)` para variables temporales o `geocode` para variables estáticas, garantizando que `geocode` sea `str` en todos los merges). Después de eso, **siempre** — esté el panel cacheado o recién construido — llama a `_generar_reporte_integracion()` y `_generar_reporte_completitud_anual()`, que escriben `reporte_integracion.csv` y `reporte_completitud_anual.csv` respectivamente. Ver §9.11 para el detalle de ambos reportes y por qué no están gateados por la misma caché que los paneles.

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

### Variables del modelo (panel modelo)

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
├── distritos-alto-cambio/
│   └── distritos_alto_cambio_metadatos_raw.csv   (PASO 0 — única fuente sin variable propia, ver §9.10)
│
├── variables/
│   ├── agropecuaria/
│   │   ├── agropecuaria_por_distrito.csv     (8 000 filas: geocode, departamento, distrito, anio, pct_agropecuario)
│   │   ├── agropecuaria_metadatos.csv
│   │   └── agropecuaria_metadatos_raw.csv     (40 filas: un raster MapBiomas por año, exhaustivo)
│   │
│   ├── carreteras/
│   │   ├── carreteras_por_distrito.csv        (200 filas: geocode, ..., km_carreteras, area_utm_km2, densidad_carreteras_km_km2)
│   │   ├── carreteras_metadatos.csv
│   │   └── carreteras_metadatos_raw.csv       (3 filas: shapefiles MTC nacional/departamental/vecinal)
│   │
│   ├── rios/
│   │   ├── rios_por_distrito.csv              (200 filas: geocode, ..., km_rios, area_utm_km2, densidad_rios_km_km2)
│   │   ├── rios_metadatos.csv
│   │   └── rios_metadatos_raw.csv             (1 fila: Rios.shp)
│   │
│   ├── anp/
│   │   ├── anp_por_distrito.csv               (8 000 filas: geocode, ..., anio, pct_anp, tiene_anp)
│   │   ├── anp_metadatos.csv
│   │   ├── anp_resumen_anual.csv              (40 filas: expansión temporal de la cobertura ANP)
│   │   └── anp_metadatos_raw.csv              (4 filas: shapefiles Nacional/ZR/ACR/ACP)
│   │
│   ├── elevacion/
│   │   ├── dem_mosaico.tif                    (mosaico SRTM, int16, CRS_GEOG, LZW)
│   │   ├── elevacion_por_distrito.csv         (200 filas: elev_media_m, elev_mediana_m, ...)
│   │   ├── elevacion_metadatos.csv
│   │   └── elevacion_metadatos_raw.csv        (163 filas: un tile SRTM por fila)
│   │
│   └── pendiente/
│       ├── pendiente.tif                      (BigGeoTIFF float32 ~5.5 GB, EPSG:32718, Horn 1981)
│       ├── pendiente_por_distrito.csv         (200 filas: pendiente_media_deg, pendiente_mediana_deg, ...)
│       ├── pendiente_metadatos.csv
│       └── pendiente_metadatos_raw.csv        (163 filas: mismos tiles SRTM que elevacion/)
│
├── variables-respaldo/
│   ├── rios_lagos/
│   │   ├── rios_lagos_por_distrito.csv        (8 000 filas: geocode, ..., pct_rios_lagos)
│   │   ├── rios_lagos_metadatos.csv
│   │   └── rios_lagos_metadatos_raw.csv       (40 filas: un raster MapBiomas por año, exhaustivo)
│   └── urbano/
│       ├── urbano_por_distrito.csv            (8 000 filas: geocode, ..., pct_urbano)
│       ├── urbano_metadatos.csv
│       └── urbano_metadatos_raw.csv           (40 filas: un raster MapBiomas por año, exhaustivo)
│
└── panel-integrado/
    ├── panel_integrado_completo.csv                    (8 000 filas, 26 columnas — todas las variables)
    ├── panel_integrado_modelo.csv              (8 000 filas, 10 columnas — 6 vars del modelo + id)
    ├── panel_integrado_entrenamiento.csv      (7 200 filas — O1 train + O3 light)
    ├── panel_integrado_generalizacion.csv     (800 filas  — O1 gen + O3 light)
    ├── reporte_integracion.csv                (22 filas: 1 por variable — completitud, base de cálculo
    │                                            por tipo, cobertura distrital, puntero a metadata_raw)
    ├── reporte_completitud_anual.csv          (80 filas: pct_agropecuario + pct_anp × 40 años 1985-2024)
    ├── reporte_ejecucion.csv                  (1 fila: metadata de la corrida completa de main.py)
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

### Esquema de los reportes de completitud y ejecución (R9/R10)

**reporte_integracion.csv** (1 fila por variable del panel completo, 22 filas):

| Columna | Descripción |
|---------|-------------|
| `variable` | Nombre de la columna en `panel_integrado_completo.csv` |
| `tipo_variable` | `temporal` (varía por año, merge en `geocode+anio`) o `estatica` (1 valor por distrito, merge en `geocode`) |
| `base_calculo` | Texto explícito: para temporales, "200 distritos × 40 años = 8 000 filas"; para estáticas, "200 distritos (valor replicado en las 40 filas anuales)" |
| `n_base` | Denominador numérico usado en `completitud_pct` (8 000 o 200 según `tipo_variable`) |
| `completitud_pct` | % de `n_base` con valor no nulo |
| `n_faltantes` | Conteo de filas nulas (sobre el panel completo, 8 000 filas, para ambos tipos) |
| `n_distritos_con_dato` | Distritos únicos (de 200) con al menos un valor no nulo |
| `n_distritos_total` | 200, constante — incluida para que el porcentaje siguiente sea autocontenido |
| `pct_distritos_con_dato` | `n_distritos_con_dato / 200 × 100` |
| `media` | Media de la variable (solo numéricas) |
| `en_panel_modelo` | `True` si la variable es una de las 6 del panel modelo |
| `metadatos_raw_csv` | Ruta al `{variable}_metadatos_raw.csv` de `metadata_fuentes.py` (PASO 0) — trazabilidad fuente → variable, sin fusionar ambos reportes (ver §9.11) |

**reporte_completitud_anual.csv** (1 fila por variable temporal del panel modelo × año, 80 filas: `pct_agropecuario` + `pct_anp` × 40 años):

| Columna | Descripción |
|---------|-------------|
| `anio` | 1985–2024 |
| `variable` | `pct_agropecuario` o `pct_anp` |
| `n_distritos` | 200, constante (filas de ese año en el panel) |
| `n_no_nulos` | Filas no nulas ese año (siempre 200 — ver nota abajo) |
| `completitud_pct` | Siempre 100.0 para ambas variables (ver nota) |
| `fuente_cobertura` | Texto: de dónde vendría una caída de cobertura real para esa variable |
| `pix_total_promedio/mediana/min/max` | Solo en filas `pct_agropecuario`: estadísticos de píxeles válidos (`pix_total` de `agropecuaria_por_distrito.csv`) agregados entre los 200 distritos ese año; `NaN` en filas `pct_anp` |
| `alerta_baja_cobertura` | `True` si `pix_total_promedio` de ese año cae > 2 desviaciones estándar por debajo de la media de la serie 1985-2024; `NaN` en filas `pct_anp` |

> **Nota importante:** `completitud_pct` sale 100% en este reporte para ambas variables porque ninguna de las dos queda NaN por diseño (`construir_agropecuaria.py`/`construir_anp.py` devuelven `0.0`, no `NaN`, cuando no hay píxeles/área válida — ver §9.6). La columna que sí puede revelar una caída real de cobertura es `pix_total_promedio` (solo para `pct_agropecuario`, que viene de píxeles satelitales). **Hallazgo verificado en la corrida del 2026-06-29:** `pix_total` es idéntico (std = 0) para los 200 distritos en los 40 años — ningún año tiene menos píxeles válidos que otro, es decir, la clase "no observado" (27) nunca aparece dentro de estos 200 distritos en ningún año de la serie MapBiomas C3 ya recortada por O1. `alerta_baja_cobertura` sale `False` en las 40 filas. `pct_anp` no tiene un indicador de píxeles porque proviene de capas vectoriales SERNANP (acumulación de polígonos), no de un raster satelital — se documenta así explícitamente en `fuente_cobertura` en vez de fabricar una métrica que no aplica.

**reporte_ejecucion.csv** (1 fila, sobrescrita en cada corrida de `main.py`):

| Columna | Descripción |
|---------|-------------|
| `fecha_inicio` / `fecha_fin` | Timestamps ISO de la corrida completa (Pasos 0-4) |
| `duracion_segundos` | `fecha_fin - fecha_inicio` |
| `pasos_esperados` | Texto fijo: `"PASO 0 a PASO 4"` |
| `ultimo_paso_completado` | Último paso que terminó sin excepción (`"PASO 0"`...`"PASO 4"`, o `None` si falló antes de PASO 0) |
| `estado` | `OK` o `ERROR` |
| `error` | `None` si `estado=OK`; `"{TipoExcepcion}: {mensaje}"` si `estado=ERROR` |

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

### Metadata de fuentes (PASO 0)

| Constante | Valor | Descripción |
|-----------|-------|-------------|
| `DISTRITOS_METADATA_RAW_CSV` | `data/interim/O3/distritos-alto-cambio/distritos_alto_cambio_metadatos_raw.csv` | Única fuente sin variable propia, en su propia carpeta (igual convención que O1); el resto de la metadata cruda se escribe en `VAR_*_DIR`, ver §9.10 |

### Reportes de completitud y ejecución (PASO 4 / cierre de `main.py`)

| Constante | Valor | Descripción |
|-----------|-------|-------------|
| `PANEL_REPORTE_CSV` | `data/interim/O3/panel-integrado/reporte_integracion.csv` | Completitud por variable, ver §9.11 |
| `PANEL_REPORTE_ANUAL_CSV` | `data/interim/O3/panel-integrado/reporte_completitud_anual.csv` | Completitud anual de `pct_agropecuario`/`pct_anp`, ver §9.11 |
| `PANEL_REPORTE_EJECUCION_CSV` | `data/interim/O3/panel-integrado/reporte_ejecucion.csv` | Metadata de la corrida completa de `main.py`, ver §9.11 |

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
| `panel_integrado_completo.csv` | 8 000 | Exploración: todas las variables disponibles para análisis descriptivo |
| `panel_integrado_modelo.csv` | 8 000 | Modelo: solo las 6 variables seleccionadas como predictores |
| `panel_integrado_entrenamiento.csv` | 7 200 | Entrenamiento: O1 base (pct_bosque + pixtotales) + O3 light, left-merge |
| `panel_integrado_generalizacion.csv` | 800 | Generalización: ídem sobre los 20 distritos de prueba |

Los paneles train/gen se construyen como **left-merge sobre O1** (no como filtro de panel_modelo). Esto preserva todas las columnas de O1 — incluyendo `pct_bosque`, `pix_bosque`, `pix_no_bosque` — que son la base del problema y no se regeneran en O3.

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

### 9.10 Metadata estructural de fuentes RAW (`metadata_fuentes.py`)

**Decisión: etapa nueva y separada, complementaria a `validar_fuentes()` — no la reemplaza**

`validar_fuentes()` (§9.9) responde una pregunta binaria: *¿existe el archivo?* (`os.path.exists`), sin abrirlo. `generar_metadata_fuentes()` responde una pregunta distinta: *¿qué características tiene exactamente el archivo que sí existe?* (CRS, tipo de geometría/resolución, dimensiones, nodata...). Son complementarias por diseño: `validar_fuentes()` debe correr primero porque es la guarda rápida que falla temprano si algo no existe; `generar_metadata_fuentes()` solo tiene sentido ejecutarla sobre fuentes que ya se confirmó que existen. Ambas corren dentro de PASO 0, antes de construir cualquier variable — la metadata de una fuente no debería describirse *después* de haber calculado variables con ella.

**Atributos registrados según tipo de fuente:**

| Tipo | Atributos |
|------|-----------|
| Vectorial (carreteras, ríos, ANP, distritos) | CRS, tipo(s) de geometría, número de features, bounding box (`minx, miny, maxx, maxy`) |
| Raster (MapBiomas, SRTM) | CRS, resolución espacial (`res_x, res_y`), dimensiones (`width, height`), tipo de dato, número de bandas, valor `nodata` |

**Decisión: un archivo por variable, no un consolidado único — columnas recortadas por tipo**

La primera versión de este módulo escribía un único CSV consolidado (`metadata_fuentes.csv`) con las 174 fuentes mezcladas, donde las columnas que no aplicaban a un tipo quedaban vacías en esa fila (p. ej. `n_features` vacío en una fila raster). Se reemplazó por un archivo de metadata cruda **por variable** (`{variable}_metadatos_raw.csv`), escrito junto al `{variable}_metadatos.csv` ya existente de esa variable, por dos razones:

1. **Ubicación.** Quien revisa `variables/anp/` para entender la variable ANP encuentra ahí mismo tanto la metadata procesada como la cruda, sin tener que saltar a un archivo separado a nivel de todo O3.
2. **Cada archivo resultante es homogéneo en tipo** (una variable nunca mezcla fuentes vectoriales y raster), así que ya no hace falta el esquema con columnas vacías que un consolidado heterogéneo sí necesitaba: `{variable}_metadatos_raw.csv` solo tiene las columnas que aplican a su tipo (`COLUMNAS_VECTORIAL` o `COLUMNAS_RASTER`).

Una fuente compartida por varias variables se duplica en cada carpeta que la consume: MapBiomas (los 40 años) en `agropecuaria/`, `rios_lagos/` y `urbano/`; los 163 tiles SRTM en `elevacion/` y `pendiente/`.

**Decisión: `distritos_alto_cambio.gpkg` tiene su propia carpeta -- ni dentro de `variables/`, ni un archivo suelto**

Es la única fuente sin carpeta de variable propia: ningún `construir_*.py` es su "dueño" -- los 8 lo reciben por igual como parámetro `distritos_gdf`, a diferencia de las demás fuentes, que cada una alimenta exactamente una (o un grupo fijo de) variable(s). Se descartaron tres alternativas:

- **Crear `variables/distritos/`:** se descartó porque `variables/` está reservado para variables *derivadas* (cada subcarpeta = un `pct_*`/`densidad_*`/`elev_*` calculado); `distritos_alto_cambio` no es una variable, es la geometría base sobre la que se calculan todas -- meterla ahí mezclaría dos conceptos distintos.
- **No auditarla en absoluto** (es un output de O1, no una fuente verdaderamente externa, a diferencia de los shapefiles de MTC/ANA/SERNANP): se descartó porque, igual que MapBiomas (también un output de O1, y sí se audita), su CRS afecta a *todos* los cálculos de overlay/zonal_stats de O3 -- es, si acaso, el insumo cuyo error sería más caro pasar por alto, no menos.
- **Dejarla como archivo suelto en `O3_INTERIM_DIR`** (primera versión): se descartó porque rompía la convención de que todo archivo de O3 vive dentro de alguna carpeta.

La solución final le da su propia carpeta de primer nivel, `distritos-alto-cambio/` -- mismo nombre que usa O1 para el `.gpkg` original (`O1/distritos-alto-cambio/distritos_alto_cambio.gpkg`) -- en vez de anidarla bajo `variables/` o dejarla suelta.

**Decisión: formato CSV, no JSON — justificación**

Se evaluaron ambos formatos explícitamente en vez de asumir uno por defecto:

- **A favor de JSON:** los atributos son heterogéneos entre vectorial y raster (geometría vs. resolución/bandas), y JSON anida naturalmente esa heterogeneidad sin columnas vacías.
- **A favor de CSV (elegido):**
  1. **Consistencia con toda la metadata ya existente en el proyecto.** O1 ya usa CSV para la metadata de sus 40 rásters MapBiomas (`metadatos_mapas_amazonia.csv`), y O3 ya usa CSV para la metadata de cada variable (`*_metadatos.csv`) y para `reporte_integracion.csv`. Introducir JSON aquí rompería esa convención sin necesidad.
  2. **Tras dividir por variable, cada archivo es homogéneo en tipo** (§ arriba) -- la heterogeneidad que originalmente motivaba considerar JSON ya no existe en estos archivos; un CSV plano con columnas fijas por tipo representa la información sin ninguna columna vacía.
  3. **Todo el resto del pipeline ya consume CSV con `pandas`.** Pasar a JSON obligaría a un camino de lectura distinto (`json.load` en vez de `pd.read_csv`) que no existe en ningún otro punto de O1/O2/O3, solo para esta tabla.
  4. **Trazabilidad en `git diff`.** Una fila por archivo en CSV permite ver de un vistazo, en el historial de git, exactamente qué cambió entre dos corridas (p. ej., un tile SRTM con `nodata` distinto); un JSON anidado dificulta ese diff línea por línea.

**Decisión: MapBiomas y SRTM, ambos exhaustivos -- sin muestreo**

La primera versión muestreaba MapBiomas (solo 1985 y 2024 de los 40 años), con el mismo criterio que ya usa `validar_fuentes()` (la serie comparte grilla espacial y solo cambia el año) y apoyándose en que `O1/mapas-amazonia/metadatos_mapas_amazonia.csv` ya documenta los 40 años exhaustivamente. Se cambió a exhaustivo (40 filas) porque ese argumento dejó de sostenerse al mover la auditoría a `agropecuaria_metadatos_raw.csv`: ese archivo vive *dentro* de la carpeta de la variable que efectivamente abre los 40 rásters uno por uno para calcular `pct_agropecuario` por año -- una muestra de 2 ahí parece una auditoría completa de "qué se usó para construir esta variable" sin serlo realmente. Que `O1` ya tenga su propio CSV exhaustivo no evita esa confusión, porque vive en otra carpeta, no junto a `agropecuaria/`. SRTM ya era exhaustivo desde el diseño original (163 filas) por la misma razón estructural: cada tile cubre una extensión geográfica distinta, no son repeticiones de la misma grilla.

**Decisión: no idempotente, a diferencia de los `construir_*`**

Cada `construir_*` evita recalcular si su CSV+metadatos ya existen, porque su cómputo es costoso (minutos en `construir_anp`, horas en `construir_pendiente` por la reproyección/Horn sobre el DEM completo). Leer la cabecera de 212 archivos (9 vectoriales + 40 MapBiomas + 163 SRTM) con `rasterio`/`geopandas` toma segundos (~4-5 s medido en este equipo), así que no hay beneficio de cachear -- y sí hay un costo: si las fuentes RAW cambiaran entre corridas (p. ej. una capa MTC actualizada), una versión cacheada del reporte ocultaría ese cambio. Por eso `generar_metadata_fuentes()` siempre sobrescribe cada `{variable}_metadatos_raw.csv`, a propósito, en cada ejecución de `main.py` -- incluso aunque el `construir_*` correspondiente haga skip por idempotencia.

---

### 9.11 Reportes de completitud reforzados para sustentar R10/R9

**Contexto:** `reporte_integracion.csv` original solo reportaba `completitud_pct` agregado (100% para todas las variables) sin distinguir la base de cálculo por tipo de variable, sin desagregación temporal, y sin evidencia de que la corrida que lo produjo terminó sin errores. Insuficiente para sustentar R10 (base integrada de variables locales) ante un comité que puede pedir el desglose exacto. Se reforzó en cuatro frentes, cada uno con su propia decisión de diseño.

**1) Base de cálculo explícita por tipo de variable**

`reporte_integracion.csv` ahora distingue `tipo_variable` (`temporal` vs. `estatica`) y reporta `base_calculo`/`n_base` en consecuencia: para variables temporales (`pct_agropecuario`, `pct_anp`, `pct_rios_lagos`, `pct_urbano`, `tiene_anp`) la base es el panel completo (200 distritos × 40 años = 8 000 filas, porque el merge de `_construir_paneles()` es sobre `geocode+anio`); para variables estáticas (carreteras, ríos, elevación, pendiente) la base son los 200 distritos, porque el merge es solo sobre `geocode` y el mismo valor se replica idéntico en las 40 filas anuales de ese distrito. Esta distinción se deriva de un diccionario explícito (`_ORIGEN_VARIABLE` en `integrar_panel.py`), no se infiere estadísticamente del panel, porque la clasificación ya está determinada por qué merge construyó cada columna -- inferirla sería redundante y más frágil.

Nota matemática: para una variable estática perfectamente replicada, `completitud_pct` calculado sobre las 8 000 filas y sobre los 200 distritos da el mismo número (un distrito nunca tiene 39 filas no nulas y 1 nula). La distinción no cambia el valor reportado, pero hace explícito *por qué* es válido leer ese 100% como "200/200 distritos" y no como "8 000/8 000 observaciones independientes" -- la segunda lectura sería estadísticamente incorrecta (pseudo-replicación) si alguien la usara para, por ejemplo, un intervalo de confianza.

**2) Desagregación anual de `pct_agropecuario`/`pct_anp` (`reporte_completitud_anual.csv`)**

Ambas variables nunca quedan `NaN` por diseño: `construir_agropecuaria.py` y `construir_anp.py` devuelven `0.0`, no `NaN`, cuando un distrito-año no tiene píxeles/área válida (ver §9.6). Esto significa que `completitud_pct` agregado es ciego a una caída real de cobertura (p. ej. nubosidad alta en el raster MapBiomas de un año específico) -- el dato "existe" (es 0.0 o un valor bajo), pero podría estar mal estimado si el denominador de píxeles válidos de ese año fue anormalmente chico.

Se agregó un reporte separado, por año, con el único indicador que sí mide eso para `pct_agropecuario`: `pix_total` (píxeles válidos por distrito-año, ya excluyendo la clase 27 "no observado" del denominador -- ver §9.6 y `AGROPECUARIA_CSV`), agregado (media/mediana/min/max) entre los 200 distritos de cada año, con una alerta si la media de ese año cae más de 2 desviaciones estándar por debajo de la media de toda la serie 1985-2024. `pct_anp` no tiene un indicador equivalente porque proviene de capas vectoriales SERNANP acumuladas por fecha de establecimiento, no de píxeles satelitales -- en vez de fabricar una métrica que no aplica, la columna `fuente_cobertura` lo documenta explícitamente fila por fila.

**Hallazgo verificado en la corrida del 2026-06-29:** `pix_total` resultó **idéntico** (desviación estándar = 0) para los 200 distritos en los 40 años de la serie. Es decir, dentro de estos 200 distritos amazónicos, la clase "no observado" nunca aparece en ningún año del producto MapBiomas C3 ya recortado por O1 -- no hay evidencia de nubosidad ni de huecos de cobertura que distorsionen el denominador de `pct_agropecuario` en ningún año. `alerta_baja_cobertura` sale `False` en las 40 filas. Este resultado responde directamente la pregunta abierta de R10 sobre variabilidad de cobertura por nubosidad: no la hay, en este producto y este universo de distritos.

**3) Cobertura distrital explícita (no solo % agregado)**

`reporte_integracion.csv` agrega `n_distritos_con_dato` / `n_distritos_total` / `pct_distritos_con_dato` para **todas** las variables (no solo las 6 del panel modelo, aunque son las que más importan para R10 -- filtrar por `en_panel_modelo=True` da exactamente esas 6). Es una columna casi gratis de calcular (`panel.loc[no_nulos, "geocode"].nunique()`) sobre el panel ya en memoria, y responde directamente "¿cuántos de los 200 distritos tienen al menos un dato real?" sin que el lector tenga que inferirlo del `completitud_pct` agregado.

**4) Trazabilidad fuente → variable → completitud, sin fusionar reportes**

Se evaluó explícitamente cruzar `reporte_integracion.csv` con la metadata de fuentes RAW de `metadata_fuentes.py` (§9.10) para mostrar la cadena completa fuente → variable construida → completitud final. Se descartó la fusión real (un solo CSV con columnas de ambos) por la misma razón que ya motivó no consolidar la metadata de fuentes en §9.10: cada reporte tiene una granularidad distinta (`metadata_fuentes` es 1 fila por archivo RAW -- p. ej. 40 filas para los 40 años de MapBiomas que alimentan `pct_agropecuario`; `reporte_integracion` es 1 fila por variable ya agregada), y fusionarlos forzaría a elegir entre perder esa granularidad o repetir la fila de variable 40 veces.

En su lugar, `reporte_integracion.csv` agrega una columna `metadatos_raw_csv` con la ruta exacta al `{variable}_metadatos_raw.csv` correspondiente (reutilizando el diccionario `NOMBRE_ARCHIVO_POR_CARPETA` ya definido en `metadata_fuentes.py`, sin duplicarlo). Es un puntero, no una fusión: quien revisa la completitud de `pct_agropecuario` encuentra ahí mismo la ruta al archivo que documenta exactamente qué 40 rásters MapBiomas se auditaron para construirla, sin que ningún archivo en disco mezcle ambas granularidades.

**5) Metadata de ejecución del pipeline completo (`reporte_ejecucion.csv`)**

El IOV de R9 exige evidencia de que "el pipeline se ejecuta de forma reproducible y sin errores". Antes de este cambio, esa evidencia solo existía como texto en la consola/log de una corrida puntual -- no quedaba un archivo verificable. `main.py` ahora envuelve los Pasos 0-4 en un `try/except/finally`: registra `fecha_inicio`, y en el `finally` (que corre tanto si el pipeline tuvo éxito como si lanzó una excepción) escribe `fecha_fin`, `duracion_segundos`, `ultimo_paso_completado` y `estado` (`OK`/`ERROR`) en `reporte_ejecucion.csv`. La excepción se relanza después del `finally` (no se silencia) -- el propósito es dejar constancia de *dónde* falló, no ocultar el fallo. Verificado con una corrida real completa (`python -m O3.r8_r9_r10.main`): `estado=OK`, `ultimo_paso_completado=PASO 4`, `duracion_segundos=4.8` (con los 8 `construir_*` ya cacheados; el costo real con cómputo desde cero es el de cada `construir_*` individual, no el de este wrapper).

**Decisión: los reportes (7-8) ya no dependen de la misma caché que los paneles (3-6)**

`integrar_panel()` originalmente hacía `return` temprano si `panel_integrado_completo.csv`/`panel_integrado_modelo.csv`/`reporte_integracion.csv` ya existían, sin tocar nada. Con los nuevos reportes, ese diseño dejaría `reporte_completitud_anual.csv` sin generar nunca en un repositorio donde el panel ya se calculó antes de este cambio (como este). Se separó la función en `_construir_paneles()` (la construcción cara: 6 merges sobre el panel completo, gateada por la existencia de los 4 paneles) y dos llamadas a reporte (`_generar_reporte_integracion`, `_generar_reporte_completitud_anual`) que **siempre** se ejecutan al final de `integrar_panel()`, estén los paneles cacheados o recién construidos -- misma filosofía que `metadata_fuentes.py` (§9.10): leer un panel de 8 000 filas ya en memoria y agruparlo por año cuesta milisegundos, así que cachearlo no ahorra nada y sí arriesga dejar un reporte desactualizado respecto al panel real en disco.

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
| Reporte de completitud anual | `reporte_completitud_anual.csv` |
| Reporte de ejecución del pipeline | `reporte_ejecucion.csv` |

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
    ├── metadata_fuentes.py         (ANIOS, DISTRITOS_METADATA_RAW_CSV, VAR_*_DIR, *_SHP, SRTM_TILES, MAPBIOMAS_AMAZONIA_PATRON, ...)
    ├── construir_agropecuaria.py   (ANIOS, MAPBIOMAS_AMAZONIA_PATRON, CLASES_AGROPECUARIA, ...)
    ├── construir_rios_lagos.py     (ANIOS, MAPBIOMAS_AMAZONIA_PATRON, CLASES_RIOS_LAGOS, ...)
    ├── construir_urbano.py         (ANIOS, MAPBIOMAS_AMAZONIA_PATRON, CLASES_URBANO, ...)
    ├── construir_carreteras.py     (CRS_PROYECTADO, CARRETERAS_*_SHP, ...)
    ├── construir_rios.py           (CRS_PROYECTADO, RIOS_SHP, ...)
    ├── construir_anp.py            (CRS_PROYECTADO, ANP_*_SHP, ANP_COL_FECHA_*, ...)
    ├── construir_elevacion.py      (CRS_GEOG, SRTM_TILES, SRTM_NODATA, ELEVACION_MOSAIC, ...)
    ├── construir_pendiente.py      (CRS_PROYECTADO, ELEVACION_MOSAIC, SLOPE_NODATA, ...)
    └── integrar_panel.py           (PANEL_COMPLETO_CSV, PANEL_MODELO_CSV, PANEL_ENTRENAMIENTO_CSV,
                                      PANEL_REPORTE_ANUAL_CSV, VAR_*_DIR, ...)

integrar_panel.py
    └── metadata_fuentes.py         (NOMBRE_ARCHIVO_POR_CARPETA — solo para construir la columna
                                      metadatos_raw_csv del reporte, ver §9.11; no hay dependencia
                                      de orden de ejecución, ambos importan de config.py de forma
                                      independiente)
```

### Orden de ejecución requerido

```
O1 (R1–R3)  → distritos_alto_cambio.gpkg + peru_amazonia_YYYY.tif
                    ↓
main.py [PASO 0]    → validar_fuentes() → generar_metadata_fuentes() → carga distritos_gdf
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
| Paneles train/gen perdían columnas de O1 (`pct_bosque`, `pix_bosque`, etc.) | Cambio de filtro de panel_modelo a left-merge sobre O1 CSV como base | `integrar_panel.py` |
| Solapamiento entre capas MTC inflaba `km_carreteras` | WKB-dedup + `unary_union` antes de la intersección | `construir_carreteras.py` |
| Doble conteo de área ANP cuando una ACP cae dentro de un ANP Nacional | `unary_union` incremental por distrito produce la unión real | `construir_anp.py` |
| Cálculo ANP O(N×M) por año: muy lento con 40 años y muchos fragmentos | Pre-intersección única + actualización incremental solo para nuevas ANP por año | `construir_anp.py` |
| `validar_fuentes()` solo confirma existencia, sin registrar CRS/resolución/dimensiones de las fuentes RAW antes de construir variables | Nuevo PASO 0: `metadata_fuentes.py` audita exhaustivamente las 9 fuentes vectoriales + los 40 años de MapBiomas + los 163 tiles SRTM, en un `{variable}_metadatos_raw.csv` por carpeta, ver §9.10 | `metadata_fuentes.py`, `main.py` |
| `reporte_integracion.csv` solo reportaba completitud % agregada (siempre 100%), sin base de cálculo por tipo, sin desagregación temporal, sin cobertura distrital explícita ni evidencia de ejecución sin errores -- insuficiente para sustentar R9/R10 | `tipo_variable`/`base_calculo`/`n_base`/`n_distritos_con_dato`/`metadatos_raw_csv` en `reporte_integracion.csv`; nuevo `reporte_completitud_anual.csv` (pix_total por año, alerta 2σ); nuevo `reporte_ejecucion.csv` (try/except/finally en `main.py`), ver §9.11 | `integrar_panel.py`, `main.py` |
| Los reportes de completitud quedarían sin regenerar si los 4 paneles ya estaban cacheados de una corrida anterior al cambio anterior | `integrar_panel()` separa la construcción cara (`_construir_paneles()`, gateada por caché) de los reportes, que siempre se recalculan al final | `integrar_panel.py` |

---

# Parte II — Modelos multivariables (R11)

## 13. Propósito del módulo R11

`src/O3/r11/` extiende el modelo univariable de O2 (solo `pct_bosque`) con las 6 variables locales que construye `r8_r9_r10/` (presión agropecuaria, ANP, infraestructura vial e hidrográfica, topografía), para ver si el contexto territorial reduce el error de pronóstico a 5 años. Entrena y evalúa MLP, LSTM y CNN 1D sobre un panel de 7 canales (`pct_bosque` + 6 variables locales) y compara el resultado contra los 5 modelos univariables de O2, leyendo los CSV de O2 ya existentes sin re-ejecutarlos.

R11 reutiliza, con las adaptaciones que exige el caso multivariable, exactamente la misma **arquitectura de pipeline en dos fases** y el mismo **protocolo de evaluación walk-forward con oracle** que O2 (ver `O2_DOCUMENTATION.md` §6–§7) — es la razón por la que ambos módulos comparten tanto código y tantas decisiones de diseño.

**Punto de entrada:** `python -m O3.r11.main`  
**Dependencia:** `panel_integrado_entrenamiento.csv` debe existir previamente (salida de `r8_r9_r10/integrar_panel.py`).

---

## 14. Estructura del módulo R11

```
src/O3/r11/
├── __init__.py
├── cargar_panel.py            ← Lee el panel integrado y lo pivota a tensor (distritos, años, canales)
├── transformaciones.py        ← log1p sobre variables sesgadas, antes de escalar
├── escalador.py                ← StandardScaler ajustado solo sobre 1985–2019, serializado a disco
├── construir_dataset.py       ← Ventanas deslizantes multivariables por distrito
├── utils.py                    ← Funciones compartidas por los 3 pipelines DL (autónomo respecto a O2)
├── final_configs.py           ← Configuraciones finales elegidas por el investigador
├── seleccion_fase1.py         ← Verifica la config final contra el grid search de Fase 1
│
├── pipeline_mlp.py            ← Grid search + entrenamiento final MLP multivariable
├── pipeline_lstm.py           ← Grid search + entrenamiento final LSTM multivariable
├── pipeline_cnn.py            ← Grid search + entrenamiento final CNN 1D multivariable
├── pipeline_comparacion.py    ← Comparación O2 vs. R11, por departamento y boxplot distrital
├── pronostico_2025.py         ← Pronóstico 2025 (sin reentrenamiento) para MLP/LSTM/CNN
└── main.py                    ← Orquestador principal
```

### 14.1 Responsabilidad por archivo

| Archivo | Rol |
|---------|-----|
| `cargar_panel.py` | `cargar_panel()`: lee `panel_integrado_entrenamiento.csv`, valida completitud/NaN, pivota a `(180, 40, 7)` |
| `transformaciones.py` | `aplicar_transformaciones()`: `log1p` sobre `VARIABLES_LOG1P`, antes de escalar |
| `escalador.py` | `ajustar_y_escalar()`, `cargar_escalador()`: `StandardScaler` ajustado solo sobre 1985–2019 |
| `construir_dataset.py` | `construir_datasets()`: ventanas deslizantes multivariables por ventana DL |
| `utils.py` | `fijar_semilla`, `calcular_metricas`, `diagnosticar_ajuste`, `obtener_activacion`, `inversa_pct_bosque`, `construir_df_predicciones`, `graficar_curva` |
| `final_configs.py` | Diccionarios con la configuración ganadora de cada modelo (claves en español, independientes de O2) |
| `seleccion_fase1.py` | Verifica que la config de `final_configs.py` exista en el grid search y exporta su RMSE/MAE/posición exactos |
| `pipeline_mlp.py` | Grid search sobre ventanas DL; Fase 2 con walk-forward geográfico e inversión de escala |
| `pipeline_lstm.py` | Igual que MLP pero arquitectura LSTM |
| `pipeline_cnn.py` | Igual que MLP pero arquitectura CNN 1D |
| `pipeline_comparacion.py` | Tabla O2-vs-R11, gráficos de mejores/peores distritos, comparación por departamento, boxplot distrital |
| `pronostico_2025.py` | Pronóstico 2025 con los 3 modelos finales; helpers de deforestación en km² y gráficos de anexo (uso manual) |
| `main.py` | Orquesta: panel → escalado → datasets → Fase 1/2 de los 3 modelos → verificación → comparación → pronóstico 2025 |

---

## 15. Flujo de ejecución R11

```
data/interim/O3/panel-integrado/panel_integrado_entrenamiento.csv
  │
  ├─ cargar_panel()
  │      └── panel: ndarray (180, 40, 7)   df_distritos_info: DataFrame
  │
  ├─ aplicar_transformaciones()   ← log1p sobre pct_anp, densidad_carreteras, densidad_rios
  ├─ ajustar_y_escalar()          ← StandardScaler sobre 1985–2019, aplicado a 1985–2024
  │
  ├─ construir_datasets()
  │      └── {w: {"train": (X, y), "test": (X, y)}}   w ∈ {3,4,5,6,7}
  │               │
  │               ├── pipeline_mlp   ──► modelos/mlp/
  │               ├── pipeline_lstm  ──► modelos/lstm/
  │               └── pipeline_cnn   ──► modelos/cnn/
  │
  ├─ [revisión manual → final_configs.py]
  │
  ├─ generar_seleccion_final() ──► comparacion/seleccion_configuraciones_finales.csv
  │
  ├─ pipeline_comparacion ──► comparacion/comparacion_base_vs_extendido.csv
  │                          comparacion/mejores_01–03_*.png / peores_01–03_*.png
  │                          comparacion/comparacion_departamentos.csv + heatmap_departamentos.png
  │                          comparacion/boxplot_rmse_distrital_3candidatos.png
  │
  └─ generar_pronostico_2025 ──► comparacion/pronostico_2025.csv
```

**Punto de entrada:** `python -m O3.r11.main`  
**Dependencia:** debe ejecutarse después de `O3.r8_r9_r10.main` (provee el panel integrado) y de `O2.r4_r5.main` (provee los 5 CSV `_final_global.csv` que se leen en la comparación, sin re-ejecutarlos).

---

## 16. Diseño del problema

A diferencia de O2 (predicción de series temporales **univariadas**: solo `pct_bosque`), R11 plantea el pronóstico como **multivariable**: cada distrito aporta una ventana de 7 canales (`pct_bosque` + 6 variables locales) y el modelo predice únicamente `pct_bosque` (canal 0) en el siguiente paso. Se evalúan las mismas tres arquitecturas DL que en O2 (MLP, LSTM, CNN 1D); R11 no reimplementa Persistencia ni ARIMA porque no tiene sentido extenderlos con variables exógenas de la misma forma y porque O2 ya los reporta como referencia.

| Modelo | Tipo | Entrada |
|--------|------|---------|
| MLP | Red neuronal feed-forward | Ventana `(window_size, 7)` aplanada a `window_size × 7` |
| LSTM | Red neuronal recurrente | Secuencia `(window_size, 7)` |
| CNN 1D | Red convolucional | Ventana permutada a `(7, window_size)` — 7 canales de entrada |

El criterio de selección del mejor modelo en cada familia es el mismo que en O2: **RMSE global walk-forward** sobre 2020–2024, en la escala original de `pct_bosque` (no en la escala estandarizada usada internamente para entrenar).

---

## 17. Construcción del panel y los datasets

### 17.1 Carga (`cargar_panel`)

```
panel_integrado_entrenamiento.csv (largo) → pivot por (geocode, anio)
panel: ndarray (180, 40, 7)            ← canal 0 = pct_bosque, canales 1–6 = COLUMNAS_PREDICTORAS[1:]
df_distritos_info: DataFrame           ← geocode, departamento, distrito (alineado con panel)
```

`COLUMNAS_PREDICTORAS` (en `O3/config.py`) fija el orden canónico de canales:

```python
COLUMNAS_PREDICTORAS = [
    "pct_bosque", "pct_agropecuario", "pct_anp",
    "densidad_carreteras_km_km2", "densidad_rios_km_km2",
    "elev_media_m", "pendiente_media_deg",
]
```

`cargar_panel()` valida columnas requeridas, completitud (`n_distritos × n_anios` filas exactas) y ausencia de NaN antes de construir la matriz — un panel incompleto o con huecos lanza `RuntimeError` en vez de propagar silenciosamente un tensor mal formado.

### 17.2 Transformaciones (`aplicar_transformaciones`)

`log1p` se aplica a los 3 canales con distribución sesgada/cero-inflada detectados en el EDA de O3 (`pct_anp`, `densidad_carreteras_km_km2`, `densidad_rios_km_km2`), definidos en `VARIABLES_LOG1P` de `config.py`. El canal 0 (`pct_bosque`, el objetivo) nunca se transforma. `log1p` se aplica **antes** de escalar y sobre el panel completo (1985–2024) en una sola pasada, sin riesgo de fuga de información porque no estima ningún parámetro desde la muestra (a diferencia de `StandardScaler`).

### 17.3 Escalado (`escalador.py`)

`StandardScaler` se ajusta **solo** sobre 1985–2019 (35 años × 180 distritos = 6300 filas) y se aplica a todo el panel 1985–2024 sin re-ajuste — la misma disciplina de "ajustar solo con datos de entrenamiento" que ya usa O2 para sus modelos, extendida aquí a la normalización de entrada. El escalador se serializa en `data/interim/O3/modelos/escalador/`; si ya existe en disco, se reutiliza (`ajustar_y_escalar()` es idempotente). `inversa_pct_bosque()` en `utils.py` invierte el escalado del canal 0 directamente con `escalador.mean_[0]` / `escalador.scale_[0]`, sin reconstruir una fila completa de 7 canales.

### 17.4 Ventanas deslizantes (`construir_dataset.py`)

Mismo split estrictamente temporal que O2 (`t + window_size < TAMANIO_ENTRENAMIENTO` → train, si no → test), pero cada ventana tiene shape `(window_size, 7)` en vez de `(window_size, 1)`:

```python
datasets[w] = {
    "train": (X_train_t, y_train_t),   # X: (n, w, 7)   y: (n, 1) — pct_bosque escalado
    "test":  (X_test_t,  y_test_t),
}
```

`DL_VENTANAS = [3, 4, 5, 6, 7]` — idéntico a `DL_WINDOW_VALUES` de O2.

---

## 18. Arquitectura de pipeline en dos fases

Idéntica en diseño a O2 (ver `O2_DOCUMENTATION.md` §6): Fase 1 (grid search exploratorio, evaluación directa sobre `(X_test, y_true)`, guarda solo métricas agregadas) → revisión humana → `final_configs.py` → Fase 2 (entrenamiento final, walk-forward geográfico, todos los artefactos). Mismas salidas comunes (`_resultados.csv`, `_top5_configuraciones.csv`, `_mejores_por_ventana.csv` en Fase 1; `_final_model.pth`, `_final_curva.png`, `_final_config.json`, `_final_global.csv`, `_final_distrito.csv`, `_final_departamento.csv`, `_final_predicciones.csv`, `_final_ypred.npy` en Fase 2) y misma guarda de idempotencia simétrica (`if exists(npy) and exists(gbl)`).

**Diferencia con O2:** las claves de hiperparámetros de R11 están en español (`capas_ocultas`, `activacion`, `epocas`, `lote`, `canales_conv`, `tamanio_denso`) mientras que O2 usa inglés (`hidden_sizes`, `activation`, `epochs`, `batch_size`, `conv_channels`, `dense_size`) — son convenciones independientes fijadas en cada módulo por separado, sin relación funcional entre sí; no se ha unificado la nomenclatura entre O2 y R11 porque implicaría invalidar los CSV de grid search ya generados.

---

## 19. Protocolo de evaluación walk-forward

Mismo protocolo one-step-ahead con oracle que O2 (ver `O2_DOCUMENTATION.md` §7.1), adaptado a que el modelo opera en espacio escalado pero las métricas se reportan en escala original:

```
history = panel_escalado[:, :35, :]        ← 35 años de los 7 canales, ya transformados/escalados
Para t ∈ {2020, ..., 2024}:
    ŷ_t_escalado = modelo(history[-ventana:, :])
    ŷ_t = inversa_pct_bosque(ŷ_t_escalado, escalador)   ← solo el canal 0 se invierte
    history avanza con el panel_escalado REAL (oracle), nunca con la predicción propia
```

Igual que en O2, esto no es forecasting multistep genuino: cada paso tiene acceso al valor real (de los 7 canales) del año anterior, así que subestima el error de un despliegue real a 2–5 años donde no se conocerían los valores futuros de las variables exógenas tampoco.

---

## 20. Modelo MLP

### 20.1 Representación de la entrada

```
(n, window_size, 7) → reshape → (n, window_size × 7)
```

### 20.2 Arquitectura

```
Input(window_size × 7) → [Linear → Activación → Dropout] × n_capas → Linear(1)
```

### 20.3 Grid search (`O3/config.py`)

```python
MLP_CAPAS_OCULTAS_VALORES = [[32, 16], [64, 32], [128, 64, 32]]
MLP_ACTIVACION_VALORES    = ["relu", "leaky_relu"]
MLP_DROPOUT_VALORES       = [0.0, 0.1]
MLP_EPOCAS_VALORES        = [50]
MLP_LR_VALORES            = [0.001, 0.0005]
MLP_LOTE_VALORES          = [8, 16]
```

### 20.4 Configuración final elegida

```python
FINAL_CONFIG_MLP = {"window_size": 6, "capas_ocultas": [64, 32],
                    "activacion": "leaky_relu", "dropout": 0.0,
                    "epocas": 50, "lr": 0.001, "lote": 16}
```

**Estado:** Fase 1 y Fase 2 de MLP aún no se han ejecutado en R11 (`data/interim/O3/modelos/mlp/` vacío) — esta config está escrita pero pendiente de correr.

---

## 21. Modelo LSTM

### 21.1 Representación de la entrada

Secuencia ordenada `(n, window_size, 7)` directa — el estado interno de la LSTM acumula información de los 7 canales en cada paso temporal.

### 21.2 Grid search

```python
LSTM_UNIDADES_OCULTAS_VALORES = [16, 32, 64]
LSTM_NUM_CAPAS_VALORES        = [1, 2]
LSTM_DROPOUT_VALORES          = [0.0, 0.1]
LSTM_EPOCAS_VALORES           = [50]
LSTM_LR_VALORES               = [0.001, 0.0005]
LSTM_LOTE_VALORES             = [8, 16]
```

### 21.3 Configuración final elegida

```python
FINAL_CONFIG_LSTM = {"window_size": 6, "unidades_ocultas": 16, "num_capas": 2,
                     "dropout": 0.0, "epocas": 50, "lr": 0.001, "lote": 8}
```

**Estado:** Fase 1 y Fase 2 de LSTM aún no se han ejecutado en R11 (`data/interim/O3/modelos/lstm/` vacío).

---

## 22. Modelo CNN 1D

### 22.1 Representación de la entrada

```
(n, window_size, 7) → permute(0, 2, 1) → (n, 7, window_size)
```

`canales_entrada = 7` (una por variable), a diferencia de O2 donde `input_channels = 1`.

### 22.2 Grid search

```python
CNN_CANALES_CONV_VALORES  = [[16], [32], [16, 32]]
CNN_KERNEL_VALORES        = [2, 3]
CNN_DROPOUT_VALORES       = [0.0, 0.1]
CNN_ACTIVACION_VALORES    = ["relu", "leaky_relu"]
CNN_TAMANIO_DENSO_VALORES = [16, 32]
CNN_EPOCAS_VALORES        = [50]
CNN_LR_VALORES            = [0.001, 0.0005]
CNN_LOTE_VALORES          = [8, 16]
```

### 22.3 Configuración final elegida

```python
FINAL_CONFIG_CNN = {"window_size": 6, "canales_conv": [32], "kernel_size": 2,
                    "activacion": "relu", "dropout": 0.0, "tamanio_denso": 16,
                    "epocas": 50, "lr": 0.001, "lote": 8}
```

**Estado: único modelo con Fase 2 corrida en R11** (RMSE=0.011143, MAE=0.007945) — ya mejor que el ARIMA de O2 (RMSE=0.011521, el ganador de O2 hasta ahora).

---

## 23. Utilidades compartidas — utils.py

`r11/utils.py` es deliberadamente **autónomo respecto a O2** (no importa de `O2.r4_r5.utils`): usa `logging` en vez de `print`, define `DEVICE` una sola vez para todo el módulo, y agrega `inversa_pct_bosque` (sin equivalente en O2, porque O2 nunca escala sus datos).

| Función | Firma | Descripción |
|---------|-------|-------------|
| `fijar_semilla` | `(seed=SEMILLA)` | Idéntica a O2 |
| `calcular_metricas` | `(y_true, y_pred) → (rmse, mae)` | Idéntica a O2 |
| `diagnosticar_ajuste` | `(rmse_tr, mae_tr, rmse_te, mae_te) → dict` | Idéntica a O2 |
| `obtener_activacion` | `(nombre) → nn.Module` | Idéntica a O2 |
| `inversa_pct_bosque` | `(valores_escalados, escalador) → ndarray` | Invierte el escalado del canal 0 (`pct_bosque`) sin reconstruir las 7 columnas |
| `construir_df_predicciones` | `(modelo_nombre, y_true, y_pred, df_info, anios_test) → DataFrame` | Idéntica a O2 (ya en escala original) |
| `graficar_curva` | `(train_losses, nombre, ruta_png)` | Idéntica a O2 |

---

## 24. Verificación de configuraciones finales — seleccion_fase1.py

Mismo propósito y mismo algoritmo que `O2/r4_r5/seleccion_fase1.py` (ver `O2_DOCUMENTATION.md` §14): no decide la configuración final, solo verifica que la elegida en `final_configs.py` exista en el `_resultados.csv` de Fase 1 de MLP/LSTM/CNN, recupera su RMSE/MAE oficial y su posición en el ranking, y exporta todo a `comparacion/seleccion_configuraciones_finales.csv`.

**Diferencia con O2:** `generar_seleccion_final()` en R11 recibe los directorios como parámetros opcionales (`mlp_dir`, `lstm_dir`, `cnn_dir`, con default a las constantes de `config.py`) en vez de un diccionario `dirs`, y no incluye ARIMA (R11 no tiene modelo ARIMA propio).

---

## 25. Comparación: O2 (base) vs. R11 (extendido)

### 25.1 Tabla global (`exportar_tabla_comparacion`)

Lee los `_final_global.csv` de los 5 modelos de O2 (Persistencia, ARIMA, MLP, LSTM, CNN1D — sin re-ejecutarlos) y de los modelos R11 que ya tengan Fase 2 completa, y exporta una tabla única ordenada por RMSE:

```
comparacion_base_vs_extendido.csv: etiqueta | modelo | rmse | mae | conjunto   (hasta 8 filas: 5 "base_O2" + 3 "extendido_R11")
```

### 25.2 Gráficos de predicciones por distrito (`graficar_predicciones_por_distrito`)

Igual criterio de consenso que O2 (ver `O2_DOCUMENTATION.md` §15.3): 3 mejores (mínimo de los máximos RMSE entre modelos R11) y 3 peores (máximo de los mínimos RMSE) distritos, panel doble 2000–2019 / 2020–2024, usando solo los modelos R11 con `y_pred` disponible.

### 25.3 Comparación por departamento (`comparar_departamentos`)

Idéntica en diseño a `O2/r4_r5/pipeline_comparacion.py::comparar_departamentos` (ver `O2_DOCUMENTATION.md` §15.5), aplicada a los 3 modelos propios de R11 (MLP/LSTM/CNN) en vez de a los 5 de O2: tabla consolidada de RMSE/MAE por departamento + heatmap con el mejor modelo de cada fila en negrita.

Salidas: `comparacion/comparacion_departamentos.csv`, `comparacion/heatmap_departamentos.png`.

### 25.4 Boxplot distrital (`graficar_boxplot_validacion_distrital`)

Idéntica en diseño a `O2_DOCUMENTATION.md` §15.6, aplicada a los 3 modelos de R11. Salida: `comparacion/boxplot_rmse_distrital_3candidatos.png`.

**Condición de ejecución:** las 3 secciones anteriores (25.1–25.4) solo corren si los 3 modelos R11 (MLP, LSTM, CNN) tienen Fase 2 completa; mientras falte alguno, `main.py` imprime `[PENDIENTE]` y omite todo el bloque de comparación (y, en consecuencia, también el pronóstico 2025 de §26).

---

## 26. Pronóstico 2025

`pronostico_2025.py` es el análogo multivariable de `O2/r4_r5/pronostico_r7.py` (ver `O2_DOCUMENTATION.md` §17) — mismo diseño, adaptado a que las predicciones viven en espacio escalado y deben invertirse.

### 26.1 `generar_pronostico_2025` (orquestado por `main.py`)

Carga el checkpoint `.pth` de cada modelo final (`CARGADORES` reconstruye la arquitectura exacta desde la config guardada en el propio checkpoint), toma como entrada los últimos `window_size` años del panel **escalado** (cuyo último año es 2024) y predice 2025 en espacio escalado; `inversa_pct_bosque` lo convierte a la escala original antes de guardarlo. El ancla explícita es siempre `pct_bosque_real_2024` tomado del panel **original** (sin escalar), nunca una predicción propia — misma disciplina que en O2.

Salida: `comparacion/pronostico_2025.csv` (`geocode`, `departamento`, `distrito`, `pct_bosque_real_2024`, `mlp_pred_2025`, `lstm_pred_2025`, `cnn_pred_2025`).

### 26.2 Deforestación en km² y gráficos de anexo (uso manual)

`calcular_deforestacion_km2`, `graficar_deforestacion_departamento_km2` y `graficar_correlacion_rmse_divergencia_2025` están definidas en el mismo archivo pero **no se invocan desde `main.py`**, igual que sus equivalentes de O2 — se ejecutan manualmente sobre un `deforestacion_2025.csv` con las fracciones de deforestación ya calculadas (insumo externo al pipeline). `calcular_deforestacion_km2` usa `pix_total` de 2024 del propio panel de O3 (`PANEL_ENTRENAMIENTO_CSV`), no de O1 directamente, porque el panel de O3 ya lo incluye.

---

## 27. Referencia de configuración R11

### 27.1 Constantes globales (`O3/config.py`)

| Constante | Valor | Descripción |
|-----------|-------|-------------|
| `COLUMNAS_PREDICTORAS` | 7 columnas | Orden canónico de canales del panel; canal 0 = `pct_bosque` |
| `VARIABLES_LOG1P` | 3 columnas | Canales transformados con `log1p` antes de escalar |
| `ANIO_INICIO` | `1985` | Igual que O2 |
| `TAMANIO_ENTRENAMIENTO` | `35` | Igual que O2 |
| `HORIZONTE` | `5` | Igual que O2 |
| `SEMILLA` | `42` | Igual que O2 |
| `DL_VENTANAS` | `[3,4,5,6,7]` | Igual que `DL_WINDOW_VALUES` de O2 |
| `NOMBRES_DEPARTAMENTO_DISPLAY` | dict | Duplicado de `O2.config` para no romper la autonomía de R11 respecto a O2 |

### 27.2 Grid search por modelo

| Modelo | Espacio | Total configs |
|--------|---------|----------------|
| MLP | 3×2×2×1×2×2 por ventana | 240 configs |
| LSTM | 3×2×2×1×2×2 por ventana | 240 configs |
| CNN | 3×2×2×2×2×1×2×2 por ventana (filtrado kernel) | ≤960 configs |

### 27.3 Rutas de salida (`O3/config.py`)

| Variable | Ruta |
|----------|------|
| `R11_MODELOS_DIR` | `data/interim/O3/modelos/` |
| `R11_ESCALADOR_DIR` | `…/modelos/escalador/` |
| `R11_MLP_DIR` | `…/modelos/mlp/` |
| `R11_LSTM_DIR` | `…/modelos/lstm/` |
| `R11_CNN_DIR` | `…/modelos/cnn/` |
| `R11_COMPARACION_DIR` | `…/modelos/comparacion/` |
| `O2_PERSISTENCIA_GLOBAL_CSV` … `O2_CNN_GLOBAL_CSV` | Rutas de los 5 `_final_global.csv` de O2, leídas sin re-ejecutar O2 |

---

## 28. Inventario de salidas R11

```
data/interim/O3/modelos/
│
├── escalador/
│   ├── escalador_standard.pkl
│   └── escalador_metadatos.json
│
├── mlp/                                  ← misma estructura que arima/ de O2 (sin analisis_arima/)
│   ├── mlp_resultados.csv                ← Fase 1
│   ├── mlp_top5_configuraciones.csv
│   ├── mlp_mejores_por_ventana.csv
│   ├── mlp_final_model.pth               ← Fase 2
│   ├── mlp_final_curva.png
│   ├── mlp_final_config.json
│   ├── mlp_final_global.csv
│   ├── mlp_final_distrito.csv
│   ├── mlp_final_departamento.csv
│   ├── mlp_final_predicciones.csv
│   └── mlp_final_ypred.npy
│
├── lstm/                                 ← misma estructura que mlp/ (pendiente de Fase 1+2)
├── cnn/                                  ← misma estructura que mlp/ (único con Fase 2 corrida)
│
└── comparacion/
    ├── seleccion_configuraciones_finales.csv   ← seleccion_fase1
    ├── comparacion_base_vs_extendido.csv        ← O2 (5 modelos) vs. R11 (hasta 3 modelos)
    ├── mejores_01_<geocode>.png … mejores_03_<geocode>.png
    ├── peores_01_<geocode>.png  … peores_03_<geocode>.png
    ├── comparacion_departamentos.csv
    ├── heatmap_departamentos.png
    ├── boxplot_rmse_distrital_3candidatos.png
    └── pronostico_2025.csv
```

> `deforestacion_2025.csv` y las figuras `deforestacion_2025_departamento_km2.png` /
> `dispersion_rmse_divergencia_2025.png` (helpers de `pronostico_2025.py`, ver §26.2)
> se generan manualmente, no por `main.py` — no forman parte de este inventario automático.

**Estado de ejecución (a la fecha de esta documentación):** solo CNN tiene Fase 1 y Fase 2 completas; MLP y LSTM tienen su configuración final ya escrita en `final_configs.py` pero ninguna corrida todavía, por lo que `comparacion/` está vacío hasta que se completen las 3 fases 2.

---

## 29. Decisiones de diseño específicas de R11

**Panel multivariable derivado del de O2, no independiente.** R11 no construye su propio universo de distritos ni su propio split temporal: hereda ambos de O1/O2 a través de `panel_integrado_entrenamiento.csv`, para que la comparación O2-vs-R11 sea sobre exactamente los mismos 180 distritos y los mismos años de test.

**`log1p` antes de escalar, nunca después.** `StandardScaler` centra en 0 y admite valores negativos; aplicar `log1p` después del escalado produciría errores para los valores ya centrados por debajo de -1. Por eso `aplicar_transformaciones()` se ejecuta siempre antes de `ajustar_y_escalar()`.

**`StandardScaler` ajustado una sola vez sobre 1985–2019.** Igual razón que en cualquier split train/test: ajustar el escalador sobre el panel completo (incluyendo 2020–2024) filtraría información del período de prueba a la normalización del período de entrenamiento.

**R11 es autónomo respecto a O2 en su código (`utils.py`, `config.py`), pero no en sus datos.** El módulo deliberadamente no importa funciones de `O2.r4_r5.utils` ni de `O2.config` (de ahí constantes duplicadas como `NOMBRES_DEPARTAMENTO_DISPLAY` o `SEMILLA`), para que un cambio interno en O2 no pueda romper R11 silenciosamente. Sí depende de los **resultados** de O2 (los 5 `_final_global.csv`) para la comparación final, pero los lee como datos, nunca re-ejecutando su código.

**Nomenclatura de hiperparámetros en español, distinta de O2 (en inglés).** Ver §18 — es una divergencia de convención heredada de cuando se escribió cada módulo, no una inconsistencia funcional; cambiarla invalidaría los CSV de grid search ya generados en R11.

**Mismo protocolo walk-forward con oracle que O2, con un paso adicional de inversión de escala.** El walk-forward de R11 avanza con los valores reales del panel **escalado** (los 7 canales), nunca con la predicción propia, y solo invierte la escala del canal 0 para reportar RMSE/MAE en unidades de `pct_bosque` — ver §19. Esto preserva la comparabilidad metodológica con O2 (ver `O2_DOCUMENTATION.md` §7.1) pese a que R11 entrena en un espacio numérico distinto.
