# O4 — Generalización Espacial

**Propósito:** evaluar si el modelo final extendido de O3/R11 mantiene un desempeño estable en distritos no utilizados durante el entrenamiento ni durante la selección del modelo.

**Resultado de tesis asociado:** Objetivo específico 4, compuesto por R12, R13 y R14:

- **R12:** Conjunto de datos de nuevas zonas para evaluar generalización espacial.
- **R13:** Conjunto de pronósticos de pérdida de bosque en las nuevas zonas de estudio.
- **R14:** Informe de análisis explicativo de las causas de la generalización espacial.

**Punto de entrada:** `python -m O4.r12_r13_r14.main`

---

## Tabla de contenidos

1. [Propósito del módulo](#1-propósito-del-módulo)
2. [Estructura del módulo](#2-estructura-del-módulo)
3. [Flujo de ejecución](#3-flujo-de-ejecución)
4. [Diseño metodológico](#4-diseño-metodológico)
5. [R12 — Dataset de generalización](#5-r12--dataset-de-generalización)
6. [R13 — Pronósticos en zonas nuevas](#6-r13--pronósticos-en-zonas-nuevas)
7. [R14 — Informe de generalización](#7-r14--informe-de-generalización)
8. [Inventario de salidas](#8-inventario-de-salidas)
9. [Relación con O1, O2 y O3](#9-relación-con-o1-o2-y-o3)
10. [Decisiones de diseño](#10-decisiones-de-diseño)

---

## 1. Propósito del módulo

O4 toma el modelo final extendido entrenado en O3/R11 y lo aplica sobre el conjunto de generalización espacial reservado desde O1. La finalidad no es reentrenar ni ajustar hiperparámetros, sino verificar el comportamiento del modelo en distritos no vistos.

El módulo responde tres preguntas:

1. ¿El conjunto de generalización está separado del entrenamiento y completo?
2. ¿Qué pronósticos produce el modelo final en esos distritos?
3. ¿Qué factores territoriales ayudan a explicar las diferencias de error entre zonas?

---

## 2. Estructura del módulo

```text
src/O4/
├── __init__.py
├── config.py                         — rutas de entrada/salida y constantes O4
├── O4_DOCUMENTATION.md               — este documento
│
└── r12_r13_r14/
    ├── __init__.py
    ├── main.py                       — orquestador O4
    ├── construir_dataset_generalizacion.py
    ├── modelos.py                    — carga del CNN1D final de O3/R11
    ├── pipeline_pronosticos.py       — evaluación 2020-2024 y pronóstico 2025
    └── analisis_generalizacion.py    — tablas, factores y figuras R14
```

---

## 3. Flujo de ejecución

```text
Entradas de O1/O3/R11
  │
  ├─ O1: distritos_generalizacion_espacial.gpkg
  ├─ O3: panel_integrado_generalizacion.csv
  ├─ O3/R11: cnn_final_model.pth
  ├─ O3/R11: escalador_standard.pkl
  └─ O3/R11: cnn_final_global.csv
        │
        ▼
R12 — construir_dataset_generalizacion()
  Verifica separación train/generalización y completitud del panel.
        │
        ▼
R13 — pipeline_pronosticos()
  Evalúa el CNN final en 2020-2024 y genera el pronóstico 2025.
        │
        ▼
R14 — analizar_generalizacion()
  Compara desempeño, calcula métricas distritales y analiza factores territoriales.
```

---

## 4. Diseño metodológico

La generalización espacial se evalúa sobre **20 distritos reservados**, distintos de los 180 distritos usados para el desarrollo experimental.

El modelo usado es el **CNN1D extendido de O3/R11**, elegido como modelo final del proyecto. O4 reutiliza:

- arquitectura y pesos entrenados;
- configuración guardada en el checkpoint;
- transformaciones del panel multivariable;
- escalador ajustado sobre entrenamiento;
- protocolo de evaluación geográfica usado en O3/R11.

La evaluación 2020-2024 usa el mismo criterio walk-forward aplicado previamente. Para cada año de prueba, el modelo pronostica un paso adelante usando la información real disponible hasta el año anterior. Esto permite comparar el RMSE/MAE de generalización contra el RMSE/MAE obtenido durante el desarrollo experimental.

---

## 5. R12 — Dataset de generalización

**Archivo:** `construir_dataset_generalizacion.py`

R12 no reconstruye los datos desde cero. Verifica que las salidas de O1 y O3 cumplan las condiciones necesarias para evaluar generalización espacial:

- existencia del archivo geoespacial de distritos reservados;
- existencia del panel integrado de generalización;
- ausencia de solapamiento entre distritos de entrenamiento y generalización;
- completitud de variables de entrada para el 100% de las zonas.

**Salida principal:**

```text
data/interim/O4/r12_dataset_generalizacion/reporte_r12.csv
```

**Resultado actual:**

| Indicador | Valor |
|---|---:|
| Distritos de generalización | 20 |
| Años por distrito | 40 |
| Filas del panel | 800 |
| Geocodes solapados con entrenamiento | 0 |
| Completitud de variables de entrada | 100.0% |
| Geometrías GPKG | 20 |

---

## 6. R13 — Pronósticos en zonas nuevas

**Archivo:** `pipeline_pronosticos.py`

R13 carga el CNN1D final entrenado en O3/R11 y genera dos grupos de salidas:

1. **Evaluación 2020-2024:** predicciones y métricas sobre los 20 distritos reservados.
2. **Pronóstico 2025:** estimación de cobertura boscosa y pérdida neta esperada.

**Modelo usado:**

```text
CNN_FINAL_w6_c32_k2_actrelu_d0.0_dense16_e50_lr0.001_b8
```

**Métricas de generalización:**

| Métrica | Valor |
|---|---:|
| RMSE | 0.011290 |
| MAE | 0.007862 |

**Pronóstico 2025 agregado:**

La pérdida neta estimada para las 20 zonas de generalización es:

```text
148.78 km²
```

Distribución departamental:

| Departamento | Distritos | Pérdida neta 2025 (km²) |
|---|---:|---:|
| Ucayali | 1 | 49.674 |
| Loreto | 1 | 46.047 |
| San Martin | 9 | 25.590 |
| Huanuco | 2 | 21.088 |
| Cajamarca | 2 | 3.879 |
| Amazonas | 4 | 1.338 |
| Cusco | 1 | 1.160 |

---

## 7. R14 — Informe de generalización

**Archivo:** `analisis_generalizacion.py`

R14 convierte las predicciones de R13 en tablas y figuras interpretables para el informe de tesis. Evalúa:

- comparación de desempeño entre desarrollo experimental y generalización espacial;
- métricas distritales de RMSE, MAE y sesgo;
- casos extremos de menor y mayor error;
- relación entre error distrital y variables locales;
- perfil territorial promedio por grupo de error.

### Comparación global

| Conjunto evaluado | Distritos | Predicciones | RMSE | MAE |
|---|---:|---:|---:|---:|
| Desarrollo experimental | 180 | 900 | 0.010912 | 0.007740 |
| Generalización espacial | 20 | 100 | 0.011290 | 0.007862 |

El incremento relativo es bajo: aproximadamente **3.46% en RMSE** y **1.58% en MAE**, lo que indica que el modelo mantiene un desempeño cercano al observado durante el desarrollo experimental.

### Casos extremos

Distritos de menor error:

- Mariscal Castilla, Amazonas — RMSE 0.002760
- Chachapoyas, Amazonas — RMSE 0.003559
- Olleros, Amazonas — RMSE 0.005198

Distritos de mayor error:

- Namballe, Cajamarca — RMSE 0.020564
- Caspisapa, San Martin — RMSE 0.021004
- Shanao, San Martin — RMSE 0.023815

### Factores territoriales

La evidencia más consistente se observa en:

1. `elev_media_m`
2. `pct_anp`
3. `pct_agropecuario`

La correlación más marcada corresponde a `elev_media_m`, con Spearman `-0.4962` y `p = 0.0261`. Esto sugiere que la heterogeneidad biofísica ayuda a explicar parte de las diferencias de error entre distritos.

---

## 8. Inventario de salidas

### R12

```text
data/interim/O4/r12_dataset_generalizacion/
└── reporte_r12.csv
```

### R13

```text
data/interim/O4/r13_pronosticos/
├── cnn_generalizacion_global.csv
├── cnn_generalizacion_distrito.csv
├── cnn_generalizacion_departamento.csv
├── cnn_generalizacion_predicciones.csv
├── cnn_generalizacion_ypred.npy
├── cnn_generalizacion_deforestacion_2025.csv
└── grafico_pronostico_2025_departamento.png
```

### R14

```text
data/interim/O4/r14_informe/
├── informe_generalizacion.csv
├── informe_generalizacion.md
├── tabla_distrital_completa.csv
├── factores_generalizacion.csv
├── factores_por_grupo.csv
├── casos_extremos.csv
├── casos_extremos_detalle.csv
├── recomendacion_factores.csv
├── grafico_en_muestra_vs_generalizacion.png
├── grafico_boxplot_departamento.png
├── grafico_factores_territoriales.png
├── grafico_mejor_peor_zona.png
└── final/
    ├── tabla_r14_comparacion_global.csv/.md
    ├── tabla_r14_metricas_distritales.csv/.md
    ├── tabla_r14_correlaciones_factores.csv/.md
    ├── tabla_r14_perfil_grupos_error.csv/.md
    ├── tabla_r14_casos_extremos.csv/.md
    ├── figura_r14_rmse_factores_territoriales.png
    └── figura_r14_observado_predicho_extremos.png
```

---

## 9. Relación con O1, O2 y O3

| Módulo | Dependencia usada por O4 |
|---|---|
| O1 | Distritos reservados para generalización espacial |
| O2 | Criterio metodológico de evaluación temporal y comparación de desempeño |
| O3/R10 | Panel integrado de generalización con variables locales |
| O3/R11 | CNN1D extendido, escalador, transformaciones y métricas del modelo final |

O4 no modifica ni reentrena los modelos previos. Su función es aplicar el modelo final en un conjunto espacialmente separado y producir evidencia sobre su transferibilidad.

---

## 10. Decisiones de diseño

1. **Sin reentrenamiento:** la evaluación busca medir transferencia espacial, no ajustar el modelo a nuevos distritos.
2. **Mismo protocolo de evaluación:** se reutiliza el esquema de O3/R11 para que RMSE y MAE sean comparables.
3. **Separación estricta por distrito:** los 20 distritos de generalización no aparecen en entrenamiento.
4. **Métricas globales y distritales:** el análisis combina desempeño agregado con variabilidad territorial.
5. **Explicabilidad territorial:** R14 usa las mismas variables locales de O3 para interpretar dónde el modelo generaliza mejor o peor.

