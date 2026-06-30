# O2 — Módulo de Pronóstico de Deforestación

**Variable objetivo:** Porcentaje de cobertura boscosa (`pct_bosque`) por distrito y año.  
**Período histórico:** 1985–2024 (40 años).  
**Entrenamiento:** 1985–2019 (35 años).  
**Horizonte de pronóstico:** 2020–2024 (5 años).  
**Universo:** 180 distritos amazónicos del Perú con mayor cambio histórico de cobertura boscosa.

---

## Tabla de contenidos

1. [Propósito del módulo](#1-propósito-del-módulo)
2. [Estructura del módulo](#2-estructura-del-módulo)
3. [Flujo de ejecución](#3-flujo-de-ejecución)
4. [Diseño del problema](#4-diseño-del-problema)
5. [Construcción del dataset](#5-construcción-del-dataset)
6. [Arquitectura de pipeline en dos fases](#6-arquitectura-de-pipeline-en-dos-fases)
7. [Protocolo de evaluación compartido](#7-protocolo-de-evaluación-compartido)
8. [Modelo Persistencia — baseline](#8-modelo-persistencia--baseline)
9. [Modelo ARIMA](#9-modelo-arima)
10. [Modelo MLP](#10-modelo-mlp)
11. [Modelo LSTM](#11-modelo-lstm)
12. [Modelo CNN 1D](#12-modelo-cnn-1d)
13. [Utilidades compartidas — utils.py](#13-utilidades-compartidas--utilspy)
14. [Verificación de configuraciones finales — seleccion_fase1.py](#14-verificación-de-configuraciones-finales--seleccion_fase1py)
15. [Comparación final](#15-comparación-final)
16. [Verificación nivel vs. deforestación derivada](#16-verificación-nivel-vs-deforestación-derivada)
17. [R7 — Pronóstico 2025](#17-r7--pronóstico-2025)
18. [Referencia de configuración](#18-referencia-de-configuración)
19. [Inventario de salidas](#19-inventario-de-salidas)
20. [Decisiones de diseño](#20-decisiones-de-diseño)
21. [Mejoras aplicadas](#21-mejoras-aplicadas)

---

## 1. Propósito del módulo

O2 toma como entrada el panel de series temporales de cobertura boscosa producido por O1 (180 distritos × 40 años) y entrena, evalúa y compara cinco modelos de pronóstico: un baseline de persistencia, ARIMA, y tres arquitecturas de aprendizaje profundo (MLP, LSTM, CNN 1D).

El módulo implementa un **protocolo de evaluación homogéneo** para garantizar que los resultados de todos los modelos sean comparables entre sí, y una **arquitectura de pipeline en dos fases** que intercala la revisión humana del investigador entre el grid search y el entrenamiento final.

---

## 2. Estructura del módulo

```
src/O2/
├── config.py                     ← Constantes centralizadas: rutas, grids de hiperparámetros, nombres de display
├── O2_DOCUMENTATION.md           ← Documentación única del módulo
│
└── r4_r5/
    ├── main.py                   ← Orquestador principal
    ├── construir_dataset.py      ← Carga series y construye datasets estadístico / DL
    ├── utils.py                  ← Funciones compartidas por todos los pipelines DL
    ├── final_configs.py          ← Configuraciones finales elegidas por el investigador
    │
    ├── analisis_arima.py         ← Diagnóstico ACF/PACF previo al grid search
    ├── seleccion_fase1.py        ← Verifica la config final contra el grid search de Fase 1
    │
    ├── pipeline_persistencia.py  ← Baseline walk-forward (ŷ = último valor)
    ├── pipeline_arima.py         ← Grid search + evaluación walk-forward ARIMA
    ├── pipeline_mlp.py           ← Grid search + entrenamiento final MLP
    ├── pipeline_lstm.py          ← Grid search + entrenamiento final LSTM
    ├── pipeline_cnn.py           ← Grid search + entrenamiento final CNN 1D
    ├── pipeline_comparacion.py   ← Ranking global, comparación por departamento y gráficos de mejores/peores distritos
    ├── verificacion_deforestacion.py  ← Confirma que el error de nivel == error de deforestación derivada
    └── pronostico_r7.py          ← R7: pronóstico 2025 (sin reentrenamiento) para MLP/LSTM/CNN1D
```

### 2.1 Responsabilidad por archivo

| Archivo | Rol |
|---------|-----|
| `config.py` | Rutas de salida, listas de hiperparámetros para grid search, `SEMILLA`, `NOMBRES_DEPARTAMENTO_DISPLAY` |
| `r4_r5/main.py` | Orquesta todos los pasos: carga → modelos estadísticos → DL → comparación → verificación → pronóstico R7 |
| `r4_r5/construir_dataset.py` | `cargar_series`, `construir_dataset_estadistico`, `construir_dataset_dl` |
| `r4_r5/utils.py` | `fijar_semilla`, `calcular_metricas`, `diagnosticar_ajuste`, `obtener_activacion`, `construir_df_predicciones`, `graficar_curva` |
| `r4_r5/final_configs.py` | Diccionarios con la configuración ganadora de cada modelo (editado manualmente) |
| `r4_r5/analisis_arima.py` | ACF/PACF sobre tres series representativas: mayor cambio, menor cambio, mediana |
| `r4_r5/seleccion_fase1.py` | Verifica que la config de `final_configs.py` exista en el grid search y exporta su RMSE/MAE/posición exactos |
| `r4_r5/pipeline_persistencia.py` | Walk-forward ŷ = history[-1]; genera todos los artefactos finales directamente |
| `r4_r5/pipeline_arima.py` | Grid search con walk-forward completo por configuración; Fase 2 re-genera artefactos |
| `r4_r5/pipeline_mlp.py` | Grid search sobre ventanas DL (evaluación directa); Fase 2 con walk-forward geográfico |
| `r4_r5/pipeline_lstm.py` | Igual que MLP pero arquitectura LSTM (sin `obtener_activacion`) |
| `r4_r5/pipeline_cnn.py` | Igual que MLP pero arquitectura CNN 1D |
| `r4_r5/pipeline_comparacion.py` | Ranking CSV, comparación por departamento (heatmap), boxplot distrital de candidatos DL, y gráficos de panel (histórico + pronóstico) para 3 mejores y 3 peores distritos |
| `r4_r5/verificacion_deforestacion.py` | Verifica fila por fila que RMSE/MAE de nivel y de deforestación derivada coinciden, para los 5 modelos |
| `r4_r5/pronostico_r7.py` | Pronóstico 2025 con MLP/LSTM/CNN1D ya entrenados; helpers de deforestación en km² y gráficos de anexo (uso manual, no orquestados por `main.py`) |

---

## 3. Flujo de ejecución

```
O1/series-temporales/entrenamiento/distritos_entrenamiento.csv
  │
  ├─ cargar_series()
  │      └── series: ndarray (180, 40)   df_distritos_info: DataFrame
  │
  ├─ construir_dataset_estadistico()
  │      └── X_stat (180, 35, 1)   y_stat (180, 5)
  │               │
  │               ├── pipeline_persistencia  ──► persistencia/
  │               └── pipeline_arima         ──► arima/
  │
  ├─ construir_dataset_dl()
  │      └── {w: {"train": (X, y), "test": (X, y)}}   w ∈ {3,4,5,6,7}
  │               │
  │               ├── pipeline_mlp   ──► mlp/
  │               ├── pipeline_lstm  ──► lstm/
  │               └── pipeline_cnn   ──► cnn/
  │
  ├─ [revisión manual → final_configs.py]
  │
  ├─ analisis_arima            ──► arima/analisis_arima/
  ├─ generar_seleccion_final() ──► comparacion/seleccion_configuraciones_finales.csv
  │
  ├─ pipeline_comparacion ──► comparacion/comparacion_modelos.csv
  │                          comparacion/mejora_relativa_persistencia.png
  │                          comparacion/mejores_01–03_*.png / peores_01–03_*.png
  │                          comparacion/comparacion_departamentos.csv + heatmap_departamentos.png
  │                          comparacion/boxplot_rmse_distrital_3candidatos.png
  │
  ├─ verificar_identidad_deforestacion ──► comparacion/verificacion_deforestacion.csv
  │
  └─ pipeline_r7 ──► comparacion/deforestacion_2025.csv
                     comparacion/deforestacion_2025_departamento_km2.png
                     comparacion/deforestacion_2025_dispersion_rmse_divergencia.png
                     comparacion/deforestacion_2025_dispersion_stats.csv
```

**Punto de entrada:** `python -m O2.r4_r5.main`  
**Dependencia:** el CSV de entrenamiento debe existir previamente (salida de O1).

---

## 4. Diseño del problema

El pronóstico de deforestación se plantea como **predicción de series temporales univariadas**: dado el histórico de `pct_bosque` de un distrito, predecir los 5 años siguientes. Cada distrito se trata como una serie independiente; los modelos no explotan correlaciones espaciales entre distritos vecinos.

Se evalúan cinco modelos con complejidad creciente:

| Modelo | Tipo | Supuesto principal |
|--------|------|--------------------|
| Persistencia | Baseline naive | El futuro igual al último valor observado |
| ARIMA | Estadístico clásico | Autocorrelación lineal (estacionaria tras d=1) |
| MLP | Red neuronal feed-forward | Patrones no lineales en una ventana plana |
| LSTM | Red neuronal recurrente | Dependencias secuenciales con memoria explícita |
| CNN 1D | Red convolucional | Patrones locales repetibles en la secuencia temporal |

El criterio de selección del mejor modelo en cada familia es el **RMSE global walk-forward** sobre el período 2020–2024.

---

## 5. Construcción del dataset

### 5.1 Carga (`cargar_series`)

El panel CSV en formato largo se pivota a matriz:

```
df (largo) → pivot_table(index=geocode, columns=anio, values=pct_bosque)
series: ndarray (180, 40)      ← 180 distritos × 40 años (1985–2024)
df_distritos_info: DataFrame   ← geocode, departamento, distrito (alineado con series)
```

El orden de filas de `df_distritos_info` está garantizado igual al de `series` mediante `.loc[df_pivot.index]`, lo que permite indexar con `iloc[i]` en la evaluación.

### 5.2 Dataset estadístico (`construir_dataset_estadistico`)

```
X_train_stat: (n_distritos, 35, 1)   ← histórico 1985–2019; dim extra para uniformidad
y_train_stat: (n_distritos, 5)        ← valores reales 2020–2024 (ground truth walk-forward)
```

Tanto Persistencia como ARIMA usan este dataset.

### 5.3 Dataset de aprendizaje profundo (`construir_dataset_dl`)

Para cada ventana `w ∈ DL_WINDOW_VALUES`:

```python
datasets[w] = {
    "train": (X_train_t, y_train_t),   # tensores float32
    "test":  (X_test_t,  y_test_t),
}
```

Forma de los tensores:

```
X: (n_muestras, window_size, 1)
y: (n_muestras, 1)
```

La dimensión extra de tamaño 1 (`unsqueeze`) es el número de variables. Cada arquitectura lo transforma con su función `preparar_X_*` (ver §10–§13).

### 5.4 Ventanas deslizantes y split temporal

```python
if t + window_size < TAMANIO_ENTRENAMIENTO:   # año objetivo en 1985–2019
    → train
else:                                          # año objetivo en 2020–2024
    → test
```

El split es **estrictamente temporal**: ninguna observación posterior a 2019 entra en el ajuste. La separación por ventanas (en lugar de por distritos) multiplica el número de muestras de entrenamiento, ya que cada distrito contribuye con múltiples ventanas.

### 5.5 Tamaños de ventana explorados

```python
DL_WINDOW_VALUES = [3, 4, 5, 6, 7]
```

Ventanas < 3 no capturan tendencias; ventanas > 7 reducen las muestras de entrenamiento disponibles dado el horizonte de 5 años. El tamaño óptimo se determina por grid search y forma parte de la especificación del modelo final.

---

## 6. Arquitectura de pipeline en dos fases

Todos los modelos con hiperparámetros (ARIMA, MLP, LSTM, CNN) siguen una arquitectura de dos fases que intercala la revisión humana entre la búsqueda y la evaluación final.

### 6.1 Fase 1 — Grid search exploratorio

El pipeline ejecuta un grid search sobre el espacio de hiperparámetros y guarda solo métricas agregadas en CSV. No guarda modelos entrenados ni predicciones individuales.

**Evaluación en Fase 1 para modelos DL (MLP, LSTM, CNN):** Se evalúa directamente sobre los pares `(X_test, y_true)` del dataset de ventanas. Esto no es el walk-forward geográfico: se mide el error sobre las ventanas cuyo objetivo cae en 2020–2024, sin actualización oracle entre pasos. Esta simplificación hace tractable el grid search para cientos de configuraciones.

**Evaluación en Fase 1 para ARIMA:** por su naturaleza de ajuste continuo, ya realiza el walk-forward completo en Fase 1 (ver §9.3).

Salidas comunes a todos los modelos con grid search:

```
_resultados.csv            ← todas las combinaciones evaluadas
_top5_configuraciones.csv  ← las 5 mejores por rmse
_mejores_por_ventana.csv   ← la mejor configuración por tamaño de ventana
```

ARIMA genera adicionalmente `_boxplot_ventanas.png` con la distribución de RMSE por distrito.

### 6.2 Revisión intermedia

El investigador revisa los CSV de Fase 1 y configura la elección final en `final_configs.py`:

```python
FINAL_CONFIG_ARIMA = {...}   # configuración elegida — ver §9.5
FINAL_CONFIG_MLP   = {...}   # configuración elegida — ver §10.4
FINAL_CONFIG_LSTM  = {...}   # configuración elegida — ver §11.5
FINAL_CONFIG_CNN   = {...}   # configuración elegida — ver §12.5
```

Mientras un valor sea `None`, `main.py` imprime `[PENDIENTE]` para ese modelo y la comparación final queda bloqueada.

### 6.3 Fase 2 — Entrenamiento y evaluación final

Con la configuración elegida, el pipeline ejecuta el entrenamiento completo y guarda todos los artefactos. La guarda de idempotencia verifica la existencia de **ambos** archivos críticos antes de hacer skip:

```python
if os.path.exists(ruta_modelo_npy) and os.path.exists(ruta_modelo_gbl):
    # cargar resultados sin re-entrenar
```

Salidas de Fase 2 comunes a todos los modelos:

```
_final_config.json         ← hiperparámetros y métricas globales
_final_global.csv          ← RMSE y MAE globales walk-forward (fila única)
_final_distrito.csv        ← métricas por distrito, ordenadas por MAE desc
_final_departamento.csv    ← métricas por departamento, ordenadas por MAE desc
_final_predicciones.csv    ← predicciones en formato largo (ver §6.4)
_final_ypred.npy           ← array (n_distritos, horizonte) de predicciones
```

MLP, LSTM y CNN guardan adicionalmente `_final_model.pth` y `_final_curva.png`. ARIMA no tiene objeto de modelo persistible.

### 6.4 Formato de predicciones largas

Todos los modelos generan `_final_predicciones.csv` con esquema idéntico:

| Columna | Descripción |
|---------|-------------|
| `modelo` | Nombre del modelo |
| `geocode` | Código geográfico del distrito |
| `departamento` | Departamento al que pertenece |
| `distrito` | Nombre del distrito |
| `horizonte` | Paso de predicción (1–5) |
| `anio` | Año calendario (2020–2024) |
| `y_true` | Valor real observado |
| `y_pred` | Predicción del modelo |
| `error` | `y_pred − y_true` |
| `abs_error` | `|y_pred − y_true|` |
| `squared_error` | `(y_pred − y_true)²` |

Este formato permite `pd.concat` directo sobre todos los modelos para cualquier análisis comparativo.

### 6.5 Persistencia — pipeline de fase única

El modelo de Persistencia no tiene hiperparámetros, por lo que no sigue la arquitectura de dos fases. Sus salidas usan sufijos sin `_final_`:

```
persistencia_resultados.csv
persistencia_resultados_departamento.csv
persistencia_resultados_global.csv
persistencia_resultados_config.json
persistencia_resultados_predicciones.csv
persistencia_resultados_ypred.npy
```

---

## 7. Protocolo de evaluación compartido

### 7.1 Walk-forward one-step-ahead con oracle

Todos los modelos se evalúan con el mismo protocolo:

```
history = [obs_1985, ..., obs_2019]   ← 35 años reales
Para t ∈ {2020, 2021, 2022, 2023, 2024}:
    ŷ_t = modelo(history[-ventana:])
    history.append(obs_t)              ← valor REAL, no la predicción anterior
```

El uso del valor real en cada paso (oracle) evita la acumulación de error propio del multi-step forecasting recursivo, y es la estimación más honesta de la capacidad predictiva paso a paso.

### 7.2 Métricas

Se reportan RMSE y MAE en la escala original de `pct_bosque` (fracción, no porcentaje):

| Nivel | Descripción |
|-------|-------------|
| **Global** | Todos los residuos: 180 distritos × 5 años = 900 valores |
| **Departamento** | Residuos pooled de todos los distritos del departamento |
| **Distrito** | Residuos de los 5 pasos del horizonte de ese distrito |

Las métricas son **pooled** (un cálculo sobre el conjunto de residuos), no el promedio de métricas individuales. Esto equivale a dar el mismo peso a cada predicción año-distrito.

### 7.3 Diagnóstico de sobreajuste (modelos DL)

Para los modelos DL se reporta adicionalmente en el grid search:

```
gap_rmse   = rmse_test − rmse_train
ratio_rmse = rmse_test / rmse_train
```

`gap > 0` indica sobreajuste. El **criterio primario de selección es `rmse_test`**; el gap actúa como desempate entre configuraciones con RMSE similar.

---

## 8. Modelo Persistencia — baseline

Predice `ŷ_t = history[-1]` (último valor observado). Su propósito es establecer el **piso de rendimiento**: cualquier modelo con RMSE superior al de Persistencia no aporta valor predictivo.

La evaluación usa el mismo protocolo walk-forward que los modelos complejos, garantizando comparabilidad metodológica.

---

## 9. Modelo ARIMA

### 9.1 Análisis exploratorio ACF/PACF (`analisis_arima.py`)

Antes del grid search se genera diagnóstico visual para tres series representativas:

| Serie | Criterio |
|-------|----------|
| Alta variabilidad | Distrito con mayor rango `pct_bosque_max − pct_bosque_min` |
| Baja variabilidad | Distrito con menor rango |
| Mediana nacional | Serie mediana entre todos los distritos por año |

Para cada serie se generan: serie original, serie diferenciada en orden 1, ACF y PACF de ambas. El análisis muestra que las autocorrelaciones significativas desaparecen tras diferenciación d=1, justificando `ARIMA_D_VALUES = [1]`.

Salidas en `arima/analisis_arima/`: `serie_mayor.png`, `serie_menor.png`, `serie_mediana.png`, `serie_mayor_diff.png`, `serie_mediana_diff.png`, y correlogramas `acf/pacf_*.png`.

### 9.2 Ventana rodante

En lugar de ajustar sobre los 35 años completos, se ajusta sobre los últimos `w` años de la historia disponible en cada paso del walk-forward. Esto responde a posibles rupturas estructurales en las series de larga duración.

```python
ARIMA_P_VALUES      = [0, 1, 2]
ARIMA_D_VALUES      = [1]
ARIMA_Q_VALUES      = [0, 1, 2]
ARIMA_WINDOW_VALUES = [3, 4, 5, 6, 7, 10, 15, 20, 25, 30, 35, None]
```

`window=None` equivale al histórico completo disponible en cada paso.

### 9.3 Naturaleza de la evaluación

ARIMA no separa "ajuste" de "inferencia": cada paso del walk-forward ajusta un modelo nuevo sobre la ventana disponible. Por eso:
- Fase 1 ya contiene las métricas definitivas para cada configuración
- Fase 2 re-ejecuta el mismo proceso para generar los artefactos adicionales (`_final_predicciones.csv`, etc.)
- Los resultados de Fase 2 son numéricamente idénticos a los de Fase 1 para esa configuración

### 9.4 Fallback ante divergencia

Si el ajuste no converge, la predicción cae al valor de persistencia. Esto se registra implícitamente en los residuos.

### 9.5 Configuración final elegida

```python
FINAL_CONFIG_ARIMA = {"window": 30, "p": 1, "d": 1, "q": 0}
```

---

## 10. Modelo MLP

### 10.1 Representación de la entrada (`preparar_X_mlp`)

El MLP recibe la ventana como un **vector plano** de longitud `window_size`. La posición temporal no se preserva explícitamente — el modelo aprende pesos independientes para cada posición.

```
preparar_X_mlp: (n, window_size, 1) → aplana → (n, window_size)
```

### 10.2 Arquitectura

```
Input(window_size) → [Linear → Activación → Dropout] × n_capas → Linear(1)
```

### 10.3 Grid search

```python
MLP_HIDDEN_SIZES_VALUES = [[32, 16], [64, 32], [128, 64, 32]]
MLP_ACTIVATION_VALUES   = ["relu", "leaky_relu"]
MLP_DROPOUT_VALUES      = [0.0, 0.1]
MLP_EPOCHS_VALUES       = [50]
MLP_LR_VALUES           = [0.001, 0.0005]
MLP_BATCH_SIZE_VALUES   = [8, 16]
```

Combinaciones por ventana: 3 × 2 × 2 × 1 × 2 × 2 = **48**; con 5 ventanas: **240 configuraciones**.

### 10.4 Configuración final elegida

```python
FINAL_CONFIG_MLP = {"window_size": 3, "hidden_sizes": [128, 64, 32],
                    "activation": "leaky_relu", "dropout": 0.0,
                    "epochs": 50, "lr": 0.001, "batch_size": 16}
```

---

## 11. Modelo LSTM

### 11.1 Representación de la entrada (`preparar_X_lstm`)

La LSTM recibe la ventana como una **secuencia ordenada** `(batch, window_size, 1)` — el estado interno acumula información de pasos anteriores.

```
preparar_X_lstm: ya tiene forma (n, window_size, 1) → devuelve float32
```

### 11.2 Activaciones internas — no son hiperparámetros

Las celdas LSTM tienen activaciones fijas por definición: sigmoid en las puertas; tanh en el candidato de celda y en la salida del estado oculto. Modificarlas produciría una arquitectura diferente (GRU, MGU). Por ello `activation` no aparece en el grid search de LSTM, a diferencia de MLP y CNN.

### 11.3 Dropout en arquitecturas multicapa

```python
lstm_dropout = dropout if num_layers > 1 else 0.0
```

`nn.LSTM(dropout=...)` aplica regularización entre capas, sin efecto con `num_layers=1`. El dropout sobre el estado oculto final se aplica siempre con `nn.Dropout` separado.

### 11.4 Grid search

```python
LSTM_HIDDEN_SIZE_VALUES = [16, 32, 64]
LSTM_NUM_LAYERS_VALUES  = [1, 2]
LSTM_DROPOUT_VALUES     = [0.0, 0.1]
LSTM_EPOCHS_VALUES      = [50]
LSTM_LR_VALUES          = [0.001, 0.0005]
LSTM_BATCH_SIZE_VALUES  = [8, 16]
```

Combinaciones por ventana: 3 × 2 × 2 × 1 × 2 × 2 = **48**; con 5 ventanas: **240 configuraciones**.

### 11.5 Configuración final elegida

```python
FINAL_CONFIG_LSTM = {"window_size": 6, "hidden_size": 64, "num_layers": 1,
                     "dropout": 0.0, "epochs": 50, "lr": 0.001, "batch_size": 16}
```

---

## 12. Modelo CNN 1D

### 12.1 Representación de la entrada (`preparar_X_cnn`)

```
preparar_X_cnn: (n, window_size, 1) → permute(0, 2, 1) → (n, 1, window_size)
```

El permute convierte de `(batch, tiempo, canales)` a `(batch, canales, tiempo)`, requerido por `nn.Conv1d`.

### 12.2 Arquitectura

```
Input(1, window_size) → [Conv1d → Activación → Dropout] × n_capas
                      → Flatten → Linear(dense_size) → Activación → Dropout → Linear(1)
```

`padding="same"` preserva la dimensión temporal entre capas convolucionales. `dense_size` comprime la representación `(conv_channels[-1] × window_size)` antes de la predicción.

### 12.3 Restricción kernel ≤ window\_size

El pipeline omite automáticamente las combinaciones con `kernel_size > window_size` antes de intentar entrenar.

### 12.4 Grid search

```python
CNN_CONV_CHANNELS_VALUES = [[16], [32], [16, 32]]
CNN_KERNEL_SIZE_VALUES   = [2, 3]
CNN_DROPOUT_VALUES       = [0.0, 0.1]
CNN_ACTIVATION_VALUES    = ["relu", "leaky_relu"]
CNN_DENSE_SIZE_VALUES    = [16, 32]
CNN_EPOCHS_VALUES        = [50]
CNN_LR_VALUES            = [0.001, 0.0005]
CNN_BATCH_SIZE_VALUES    = [8, 16]
```

Combinaciones antes de filtrar: 3 × 2 × 2 × 2 × 2 × 1 × 2 × 2 = **192**; con 5 ventanas: hasta **960 configuraciones** (reducidas por el filtro de kernel).

### 12.5 Configuración final elegida

```python
FINAL_CONFIG_CNN = {"window_size": 5, "conv_channels": [16, 32], "kernel_size": 3,
                    "activation": "relu", "dropout": 0.0, "dense_size": 32,
                    "epochs": 50, "lr": 0.001, "batch_size": 8}
```

---

## 13. Utilidades compartidas — utils.py

`r4_r5/utils.py` centraliza las funciones que eran idénticas en los tres pipelines DL. Todos importan de este módulo; las pipelines no definen estas funciones localmente.

| Función | Firma | Descripción |
|---------|-------|-------------|
| `fijar_semilla` | `(seed=SEMILLA)` | Fija `random`, `numpy`, `torch`, `cuda`; activa `deterministic=True` |
| `calcular_metricas` | `(y_true, y_pred) → (rmse, mae)` | RMSE y MAE con reshape a 1D; usada también en ARIMA y Persistencia |
| `diagnosticar_ajuste` | `(rmse_tr, mae_tr, rmse_te, mae_te) → dict` | Gap y ratio train/test para diagnóstico de sobreajuste |
| `obtener_activacion` | `(nombre) → nn.Module` | Mapea string a módulo PyTorch: relu, leaky_relu, tanh, elu, sigmoid |
| `construir_df_predicciones` | `(modelo_nombre, y_true, y_pred, df_info, anios_test) → DataFrame` | Construye el CSV de predicciones en formato largo |
| `graficar_curva` | `(train_losses, nombre, ruta_png)` | Guarda curva de pérdida MSE vs. época |

---

## 14. Verificación de configuraciones finales — seleccion_fase1.py

`generar_seleccion_final()` se llama desde `main.py` tras completar todos los grid searches. No decide la configuración final de ningún modelo — esa decisión sigue siendo manual e interpretativa (RMSE como criterio principal, MAE y complejidad de arquitectura como respaldo cuando las diferencias son marginales) y se registra a mano en `final_configs.py`.

Lo que hace este módulo es exclusivamente de verificación/trazabilidad:

1. Toma la configuración ya elegida en `final_configs.py` para cada modelo.
2. La busca por coincidencia exacta de hiperparámetros en el `_resultados.csv` de Fase 1 de ese modelo.
3. Recupera su RMSE/MAE oficial (sin volver a transcribirlo a mano) y su posición en el ranking por RMSE dentro de esa Fase 1.
4. Exporta todo en `comparacion/seleccion_configuraciones_finales.csv` — la fuente directa para la tabla de configuraciones finales del Anexo E.

Si la configuración elegida no es la posición 1 del ranking, se imprime un aviso (`[INFO]`) para recordar que esa desviación del mínimo absoluto debe quedar justificada en el texto del anexo, no para bloquear la ejecución. Si la configuración de `final_configs.py` no aparece en el CSV de Fase 1 (p. ej. por un valor mal copiado), se imprime un `[WARN]` explícito — este es el caso que reemplazó al antiguo gráfico de barras por ventana, que no detectaba este tipo de desincronización.

---

## 15. Comparación final

### 15.1 Ranking global (`exportar_comparacion`)

Los cinco modelos se ordenan por RMSE walk-forward global sobre 2020–2024:

```
comparacion_modelos.csv: modelo | rmse | mae   (5 filas, ordenadas por rmse asc)
```

### 15.2 Mejora relativa frente a Persistencia (`graficar_mejora_relativa`)

Gráfico de barras horizontales con la mejora porcentual de RMSE de cada modelo (ARIMA, MLP, LSTM, CNN1D) respecto al baseline de Persistencia, coloreando ARIMA distinto de los tres modelos DL. Salida: `comparacion/mejora_relativa_persistencia.png`.

### 15.3 Criterio de selección de distritos para visualización

Se identifican los 3 mejores y 3 peores distritos usando un criterio de **consenso entre modelos**:

- **Mejores:** distritos donde `max(RMSE_modelo)` sobre todos los modelos es mínimo — todos los modelos aciertan. Representan series con patrón predecible.
- **Peores:** distritos donde `min(RMSE_modelo)` sobre todos los modelos es máximo — incluso el mejor modelo falla. Representan dinámicas que escapan a todos los enfoques evaluados.

Este criterio evita sesgar la selección hacia los casos donde un modelo en particular sobresale.

### 15.4 Visualización por distrito

Para cada uno de los 6 distritos (3 mejores + 3 peores) se genera un gráfico de panel doble:

```
Panel izquierdo (2:1) │ Panel derecho (1:1)
2000–2019 (contexto)  │ 2020–2024 (pronóstico)
serie histórica       │ y_true + ŷ de cada modelo superpuestos
```

El panel comienza en 2000 (no 1985) para mostrar el contexto reciente más relevante sin comprimir el período de pronóstico.

### 15.5 Comparación por departamento (`comparar_departamentos`)

Tabla consolidada con RMSE y MAE de los 5 modelos por departamento, número de distritos por departamento (para juzgar la robustez de cada resultado) y el mejor modelo por RMSE y por MAE. Genera además un heatmap de RMSE (filas=departamentos, columnas=modelos) que resalta en negrita el mejor modelo de cada fila.

Salidas: `comparacion/comparacion_departamentos.csv`, `comparacion/heatmap_departamentos.png`.

### 15.6 Boxplot distrital de candidatos DL (`graficar_boxplot_validacion_distrital`)

Boxplot del RMSE distrital de validación (walk-forward 2020–2024) restringido a los tres candidatos de aprendizaje profundo (MLP, LSTM, CNN1D), pensado para el anexo de R7. Resalta en rojo los distritos que caen simultáneamente en el top-10 de mayor RMSE de los tres modelos (intersección); los demás atípicos por encima del bigote superior de cada caja quedan en gris.

Salida: `comparacion/boxplot_rmse_distrital_3candidatos.png`.

---

## 16. Verificación nivel vs. deforestación derivada

`verificacion_deforestacion.py` confirma empíricamente que el error de pronóstico sobre el **nivel** de `pct_bosque` (el que reportan `comparacion_modelos.csv` y las tablas de tesis) es idéntico al error sobre la **deforestación derivada** (la variación interanual), para los 5 modelos.

Esta identidad se cumple porque cada predicción usa como ancla el valor **real** del año anterior, nunca una predicción encadenada del propio modelo:

```
Δ_pred(t) = pct_real(t-1) - pct_pred(t)
Δ_real(t) = pct_real(t-1) - pct_real(t)
Δ_pred(t) - Δ_real(t) = pct_real(t) - pct_pred(t) = -error_nivel(t)
```

Esa cancelación del ancla es lo que hace que RMSE/MAE coincidan entre nivel y deforestación, y depende de que los 5 pipelines (`pipeline_persistencia.py`, `pipeline_arima.py`, `pipeline_mlp.py`, `pipeline_lstm.py`, `pipeline_cnn.py`) usen siempre `history.append(valor_real)`, nunca `history.append(predicción)`.

`verificar_identidad_deforestacion()` no reentrena ni recalcula nada: lee las predicciones ya guardadas de Fase 2 (`*_final_predicciones.csv`) y confirma la identidad fila por fila con `np.allclose`. Si la identidad no se cumple para algún modelo, lanza `AssertionError` — señal de que algún walk-forward dejó de alimentarse con el valor real en algún paso.

Salida: `comparacion/verificacion_deforestacion.csv` (rmse/mae de nivel y de deforestación por modelo, y la bandera `identidad_confirmada`).

---

## 17. R7 — Pronóstico 2025

`pronostico_r7.py` genera el pronóstico 2025 con los tres candidatos de aprendizaje profundo (MLP, LSTM, CNN1D) ya entrenados y validados en R6/R7, sin reentrenar.

### 17.1 `pipeline_r7` — punto de entrada único (orquestado por `main.py`)

Punto de entrada único que integra en un solo paso: predicciones 2025, deforestación en fracción (`pct_bosque_real_2024 - {modelo}_pred_2025`) y en km² (usando `pix_total` de 2024 del panel de O1 y `PIXEL_AREA_KM2`), y la generación de todos los gráficos del bloque. Los cargadores de modelo (`_CARGADORES`) y las funciones de gráficos (`_graficar_*`) son privados y solo se invocan desde `pipeline_r7`.

Salida única: `comparacion/deforestacion_2025.csv` con columnas `geocode`, `departamento`, `distrito`, `pct_bosque_real_2024`, `{mlp,lstm,cnn1d}_pred_2025`, `deforestacion_2025_{mlp,lstm,cnn1d}`, `area_km2_2024`, `deforestacion_2025_{mlp,lstm,cnn1d}_km2`.

### 17.2 Gráficos generados por `pipeline_r7`

- `deforestacion_2025_departamento_km2.png`: barras horizontales agrupadas por departamento y modelo, ordenadas por promedio de los 3 modelos, con resaltado del departamento de interés (`Cajamarca` por defecto).
- `deforestacion_2025_dispersion_rmse_divergencia.png`: dispersión por distrito entre RMSE promedio de validación (2020–2024) y divergencia entre modelos en el pronóstico 2025 (desviación estándar de las 3 fracciones); anotación con Pearson r y p-valor sobre la figura.
- `deforestacion_2025_dispersion_stats.csv`: tres filas — `global` (Pearson/Spearman/pendiente para los 180 distritos), `{departamento_resaltado}` (medias de RMSE y divergencia), `excl_{departamento_resaltado}` (mismas estadísticas excluyendo ese departamento).

---

## 18. Referencia de configuración

### 18.1 Constantes globales (`config.py`)

| Constante | Valor | Descripción |
|-----------|-------|-------------|
| `ANIO_INICIO` | `1985` | Primer año del panel de series |
| `TAMANIO_ENTRENAMIENTO` | `35` | Años usados para ajuste (1985–2019) |
| `HORIZONTE` | `5` | Años de pronóstico (2020–2024) |
| `SEMILLA` | `42` | Semilla global de reproducibilidad |
| `DL_WINDOW_VALUES` | `[3,4,5,6,7]` | Tamaños de ventana explorados en DL |
| `NOMBRES_DEPARTAMENTO_DISPLAY` | dict | Nombres de departamento con tildes, solo para texto/figuras (no se aplica a los datos en disco) |

### 18.2 Grid search por modelo

| Modelo | Espacio | Total configs |
|--------|---------|---------------|
| ARIMA | p∈{0,1,2}, d=1, q∈{0,1,2}, w×12 | 108 configs |
| MLP | 3×2×2×1×2×2 por ventana | 240 configs |
| LSTM | 3×2×2×1×2×2 por ventana | 240 configs |
| CNN | 3×2×2×2×2×1×2×2 por ventana (filtrado kernel) | ≤960 configs |

### 18.3 Rutas de salida (`config.py`)

| Variable | Ruta |
|----------|------|
| `O2_INTERIM_DIR` | `data/interim/O2/` |
| `MODELOS_DIR` | `data/interim/O2/modelos/` |
| `PERSISTENCIA_DIR` | `…/modelos/persistencia/` |
| `ARIMA_DIR` | `…/modelos/arima/` |
| `ANALISIS_ARIMA_DIR` | `…/modelos/arima/analisis_arima/` |
| `MLP_DIR` | `…/modelos/mlp/` |
| `LSTM_DIR` | `…/modelos/lstm/` |
| `CNN_DIR` | `…/modelos/cnn/` |
| `COMPARACION_DIR` | `…/modelos/comparacion/` |

---

## 19. Inventario de salidas

```
data/interim/O2/modelos/
│
├── persistencia/
│   ├── persistencia_resultados.csv
│   ├── persistencia_resultados_departamento.csv
│   ├── persistencia_resultados_global.csv
│   ├── persistencia_resultados_config.json
│   ├── persistencia_resultados_predicciones.csv
│   └── persistencia_resultados_ypred.npy
│
├── arima/
│   ├── analisis_arima/
│   │   ├── serie_mayor.png / serie_menor.png / serie_mediana.png
│   │   ├── serie_mayor_diff.png / serie_mediana_diff.png
│   │   └── acf_*.png / pacf_*.png   (mayor_raw, mediana_raw, mayor_diff, mediana_diff, menor)
│   ├── arima_resultados.csv              ← Fase 1
│   ├── arima_top5_configuraciones.csv
│   ├── arima_mejores_por_ventana.csv
│   ├── arima_boxplot_ventanas.png
│   ├── arima_final_config.json           ← Fase 2
│   ├── arima_final_global.csv
│   ├── arima_final_distrito.csv
│   ├── arima_final_departamento.csv
│   ├── arima_final_predicciones.csv
│   └── arima_final_ypred.npy
│
├── mlp/                                  ← misma estructura que arima/ (sin analisis_arima/)
│   ├── mlp_resultados.csv
│   ├── mlp_top5_configuraciones.csv
│   ├── mlp_mejores_por_ventana.csv
│   ├── mlp_final_model.pth               ← pesos del modelo
│   ├── mlp_final_curva.png
│   ├── mlp_final_config.json
│   ├── mlp_final_global.csv
│   ├── mlp_final_distrito.csv
│   ├── mlp_final_departamento.csv
│   ├── mlp_final_predicciones.csv
│   └── mlp_final_ypred.npy
│
├── lstm/                                 ← misma estructura que mlp/
├── cnn/                                  ← misma estructura que mlp/
│
└── comparacion/
    ├── comparacion_modelos.csv
    ├── seleccion_configuraciones_finales.csv   ← seleccion_fase1
    ├── mejora_relativa_persistencia.png
    ├── mejores_01_<geocode>.png … mejores_03_<geocode>.png
    ├── peores_01_<geocode>.png  … peores_03_<geocode>.png
    ├── comparacion_departamentos.csv
    ├── heatmap_departamentos.png
    ├── boxplot_rmse_distrital_3candidatos.png
    ├── verificacion_deforestacion.csv            ← verificacion_deforestacion
    ├── deforestacion_2025.csv                    ← pipeline_r7 (R7)
    ├── deforestacion_2025_departamento_km2.png   ← pipeline_r7 (R7)
    ├── deforestacion_2025_dispersion_rmse_divergencia.png  ← pipeline_r7 (R7)
    └── deforestacion_2025_dispersion_stats.csv   ← pipeline_r7 (R7)
```

---

## 20. Decisiones de diseño

**Sin normalización de datos.** `pct_bosque ∈ [0, 1]` es adecuado para todas las arquitecturas sin normalización adicional. Los gradientes de MSELoss son naturalmente pequeños y Adam converge establemente en este rango.

**MSELoss + Adam en todos los modelos DL.** La función de pérdida es consistente con la métrica de evaluación (RMSE). Adam se eligió por su robustez ante diferentes escalas de gradiente y convergencia más rápida que SGD en datasets pequeños.

**Normalización de la pérdida por muestra, no por batch.** `epoch_loss / len(dataloader.dataset)` hace que el valor de loss sea independiente del `batch_size`, facilitando la comparación entre configuraciones.

**Sin early stopping.** Todos los modelos consumen el mismo presupuesto de épocas para la misma configuración, eliminando una fuente de asimetría en la comparación.

**Walk-forward con oracle, no recursivo.** Todos los modelos comparten exactamente el mismo protocolo (ver §7.1).

**Fase 1 DL vs. ARIMA diferente.** DL usa evaluación directa sobre ventanas (tractable para cientos de configs); ARIMA ya hace walk-forward completo en Fase 1 porque no tiene separación ajuste/inferencia.

**Revisión humana intermedia.** La arquitectura de dos fases obliga a que el investigador revise los resultados de Fase 1 antes de gastar cómputo en el entrenamiento final. `final_configs.py` es el contrato explícito entre Fase 1 y Fase 2.

**Idempotencia simétrica en Fase 2.** El skip verifica que existan tanto `_final_ypred.npy` como `_final_global.csv`. Verificar solo uno de los dos puede dejar al pipeline en un estado inconsistente si la ejecución se interrumpió a mitad.

**Formato largo homogéneo para predicciones.** El esquema idéntico de `_final_predicciones.csv` en todos los modelos es la columna vertebral de la sección de comparación.

---

## 21. Mejoras aplicadas

| Problema | Solución | Archivos |
|----------|----------|----------|
| 6 funciones duplicadas en los pipelines DL | Creado `utils.py`; pipelines importan en lugar de definir | `utils.py`, `pipeline_mlp/lstm/cnn.py` |
| `calcular_metricas` duplicada también en ARIMA y Persistencia | Ambos importan desde `utils.py` | `pipeline_arima.py`, `pipeline_persistencia.py` |
| Imports huérfanos (`random`, `matplotlib`, `sklearn`) tras extracción | Eliminados de los pipelines DL | `pipeline_mlp/lstm/cnn.py` |
| Idempotencia asimétrica en Fase 2: verificaba `_npy` pero leía `_gbl` sin verificar | `if exists(npy) and exists(gbl)` en los 5 modelos | `main.py` |
| `pd.read_csv(...).iloc[0]` sin validación en skip de Fase 2 | Añadido `if df_gbl.empty: raise RuntimeError(...)` | `main.py` |
| Documentación incompleta: faltaban utils, analisis_fase1, árbol, config ref | Creado `O2_DOCUMENTATION.md`; eliminado `documentacion_tecnica.md` | este archivo |
| `analisis_fase1.py` graficaba "mejor ventana" sin relación con la selección final real, y no detectaba si `final_configs.py` se desincronizaba del grid search (pasó con ARIMA: ventana 30 en el config vs. ventana 32 real) | Reemplazado por `seleccion_fase1.py`: verifica la config final contra el CSV de Fase 1 y exporta sus métricas exactas para el Anexo E | `seleccion_fase1.py`, `main.py` (idéntico en O3/r11) |
| `pipeline_tcn.py` se implementó pero nunca llegó a tener resultados en disco ni configuración final | Eliminado del repositorio; doc actualizada para reflejar 5 modelos (no 6) | `O2_DOCUMENTATION.md` |
| `comparar_departamentos` solo exportaba la matriz de RMSE y el "mejor modelo" por fila, sin MAE ni conteo de distritos en la misma tabla | Tabla consolidada única con RMSE, MAE, n_distritos y mejor modelo por RMSE/MAE | `pipeline_comparacion.py` (idéntico en O3/r11) |
| Sin vista de la dispersión de RMSE distrital entre los 3 candidatos DL para el anexo de R7 | Agregado `graficar_boxplot_validacion_distrital` | `pipeline_comparacion.py` (idéntico en O3/r11) |
| Visualización de mejores/peores distritos fijada en 5+5 sin relación con el nivel de detalle final del anexo | Reducida a 3+3 (`graficar_predicciones(..., n=3)`) | `pipeline_comparacion.py` (idéntico en O3/r11) |
