# Documentación Técnica — Pronóstico de Deforestación en Distritos Amazónicos del Perú

**Variable objetivo:** Porcentaje de cobertura boscosa (`pct_bosque`) por distrito y año.  
**Período de entrenamiento:** 1985–2019 (35 años).  
**Horizonte de pronóstico:** 2020–2024 (5 años).  
**Universo:** 180 distritos amazónicos del Perú con mayor cambio histórico de cobertura boscosa.

---

## Tabla de contenidos

1. [Diseño del problema](#1-diseño-del-problema)
2. [Preprocesamiento geoespacial (O1)](#2-preprocesamiento-geoespacial-o1)
3. [Protocolo de evaluación compartido](#3-protocolo-de-evaluación-compartido)
4. [Arquitectura de pipeline en dos fases](#4-arquitectura-de-pipeline-en-dos-fases)
5. [Construcción del dataset](#5-construcción-del-dataset)
6. [Modelo Persistencia — baseline](#6-modelo-persistencia--baseline)
7. [Modelo ARIMA](#7-modelo-arima)
8. [Modelo MLP](#8-modelo-mlp)
9. [Modelo LSTM](#9-modelo-lstm)
10. [Modelo CNN](#10-modelo-cnn)
11. [Criterios de comparabilidad entre modelos](#11-criterios-de-comparabilidad-entre-modelos)
12. [Comparación final](#12-comparación-final)
13. [Inventario de salidas](#13-inventario-de-salidas)

---

## 1. Diseño del problema

El pronóstico de deforestación se plantea como un problema de **predicción de series temporales univariadas**: dado el histórico de porcentaje de cobertura boscosa de un distrito, predecir los 5 años siguientes. Cada distrito se trata como una serie independiente; los modelos no explotan correlaciones espaciales entre distritos vecinos.

Se evalúan cinco familias de modelos con complejidad creciente:

| Modelo | Tipo | Supuesto principal |
|--------|------|--------------------|
| Persistencia | Estadístico naive | El futuro igual al presente |
| ARIMA | Estadístico clásico | Autocorrelación lineal estacionaria |
| MLP | Redes neuronales | Patrones no lineales en una ventana plana |
| LSTM | Redes neuronales recurrentes | Dependencias secuenciales explícitas |
| CNN 1D | Redes neuronales convolucionales | Patrones locales repetibles en la secuencia |

El criterio de selección del mejor modelo en cada familia es el **RMSE global walk-forward** sobre el período 2020–2024.

---

## 2. Preprocesamiento geoespacial (O1)

### 2.1 Delimitación del dominio de estudio

La Amazonía peruana se delimita a partir del shapefile oficial de biomas nacionales. Un distrito se considera "amazónico" si más del **50 % de su área** cae dentro del polígono de la Amazonía. Este umbral evita incluir distritos mayoritariamente andinos con pequeñas franjas de selva de altura, sin excluir distritos de ecotono con presencia forestal relevante.

Todos los cálculos de área se realizan en proyección **UTM zona 18S (EPSG:32718)**, la proyección métrica adecuada para el territorio peruano, y los productos se almacenan en **WGS84 (EPSG:4326)** para interoperabilidad.

### 2.2 Reclasificación bosque / no-bosque

MapBiomas Colección 3 Perú distingue más de 20 clases de uso del suelo. Para el pronóstico se adopta una **reclasificación binaria**: las clases 3, 4, 5 y 6 (formaciones forestales) se agrupan como bosque (1); el resto como no-bosque (0). Los píxeles con código 27 ("no observado") se marcan como NODATA (255) y se excluyen de todos los cálculos.

Esta simplificación está justificada porque el interés de la tesis es la pérdida neta de cobertura forestal, no la dinámica interna entre tipos de bosque.

### 2.3 Selección de los 200 distritos con mayor cambio

La detección de cambio opera píxel a píxel: un píxel "cambia" si su clase bosque/no-bosque varía en al menos un año consecutivo del período 1985–2024. El porcentaje de cambio de cada distrito es la fracción de sus píxeles válidos que experimentaron cambio. Los **200 distritos** con mayor porcentaje se seleccionan como dominio de estudio, garantizando que el universo analizado concentre la deforestación histórica relevante.

### 2.4 Split entrenamiento / generalización espacial

Los 200 distritos se dividen aleatoriamente (semilla 42) en **180 para entrenamiento** (90 %) y **20 para generalización espacial** (10 %). Los modelos se ajustan únicamente sobre los 180 distritos de entrenamiento. Los 20 de generalización están reservados para una evaluación de transferencia geográfica futura y no intervienen en ninguna decisión de hiperparámetros.

### 2.5 Variable objetivo

Para cada distrito y año se calcula:

```
pct_bosque = píxeles_bosque / píxeles_válidos
```

El resultado es un valor en **[0, 1]**, interpretable directamente como fracción de territorio con cobertura forestal. Esta escala es adecuada para redes neuronales sin necesidad de normalización adicional: los gradientes de la pérdida MSE son naturalmente pequeños y los optimizadores basados en Adam se comportan establemente en este rango.

---

## 3. Protocolo de evaluación compartido

### 3.1 Walk-forward one-step-ahead

Todos los modelos se evalúan con el mismo protocolo de **evaluación walk-forward**: el modelo predice un año a la vez, y tras cada predicción incorpora el valor **real** observado para construir la ventana del siguiente paso. El proceso se repite durante los 5 años del horizonte.

```
history = [obs_1985, ..., obs_2019]   ← 35 años reales
Para t ∈ {2020, 2021, 2022, 2023, 2024}:
    ŷ_t = modelo(history[-ventana:])
    history.append(obs_t)              ← valor real, no la predicción
```

Esta elección es deliberada: al usar valores reales en lugar de predicciones previas se evita el acumulado de error propio del multi-step forecasting recursivo, y se obtiene la estimación más honesta de la capacidad predictiva paso a paso de cada modelo.

### 3.2 Métricas

Se reportan RMSE y MAE en la escala original de `pct_bosque` (fracción de bosque, no porcentaje ×100). Las métricas se calculan a tres niveles de agregación:

| Nivel | Descripción |
|-------|-------------|
| **Global** | Todos los residuos de los 180 distritos × 5 años (900 valores) |
| **Departamento** | Residuos pooled de todos los distritos del departamento |
| **Distrito** | Residuos de los 5 años del horizonte de ese distrito |

Las métricas departamentales y global son **pooled** (un único cálculo sobre el conjunto de residuos), no el promedio de las métricas individuales. Esto es equivalente a darle el mismo peso a cada predicción año-distrito, independientemente de cuántos distritos tenga el departamento.

### 3.3 Diagnóstico de sobreajuste (modelos DL)

Para los modelos de aprendizaje profundo se reporta adicionalmente el gap de generalización:

```
gap_RMSE   = RMSE_test − RMSE_train
ratio_RMSE = RMSE_test / RMSE_train
```

Valores `gap > 0` indican sobreajuste; valores cercanos a 1 en el ratio indican buen ajuste generalizable. Esta información se guarda en los CSV de resultados del grid search y sirve como criterio de desempate en la revisión manual, pero el **criterio primario de selección es siempre `rmse_test`**.

---

## 4. Arquitectura de pipeline en dos fases

Todos los modelos con hiperparámetros (ARIMA, MLP, LSTM, CNN) siguen una arquitectura de **dos fases explícitas** que intercala la revisión humana entre la búsqueda y la evaluación final.

### 4.1 Fase 1 — Búsqueda exploratoria

El pipeline ejecuta un **grid search** sobre el espacio de hiperparámetros y guarda únicamente métricas agregadas en CSV. No guarda modelos entrenados ni predicciones individuales. Su única función es producir los archivos que el investigador revisará para elegir la configuración final.

**Evaluación en Fase 1 para modelos DL (MLP, LSTM, CNN):** El grid search evalúa cada configuración con una evaluación directa sobre el conjunto de ventanas deslizantes de test (las ventanas cuyo objetivo cae en 2020–2024). Esto **no** es el walk-forward geográfico: se evalúa el modelo sobre los pares `(X_test_window, y_true)` del dataset ya construido, no distrito a distrito con actualización oracle. El walk-forward geográfico completo solo se ejecuta en Fase 2, una vez elegida la configuración final. Esta distinción permite que el grid search sea computacionalmente tractable para cientos de configuraciones.

**Evaluación en Fase 1 para ARIMA:** A diferencia de los modelos DL, ARIMA ya ejecuta el walk-forward geográfico completo en Fase 1, ya que no tiene separación entre ajuste y evaluación (ver §7.3).

Salidas de Fase 1 (comunes a todos los modelos con grid search):

```
_resultados.csv            ← todas las combinaciones; DL ordenadas por [rmse_test, mae_test, gap_rmse]
_top5_configuraciones.csv  ← las 5 mejores
_mejores_por_ventana.csv   ← la mejor configuración por tamaño de ventana
```

ARIMA genera adicionalmente `_boxplot_ventanas.png` con la distribución de RMSE por distrito para cada mejor configuración por ventana.

### 4.2 Revisión intermedia

El investigador revisa los CSV de Fase 1 y configura la elección final en `final_configs.py`:

```python
FINAL_CONFIG_ARIMA = {"window": "full", "p": 1, "d": 1, "q": 0}
FINAL_CONFIG_MLP   = {"window_size": 5, "hidden_sizes": [64, 32], ...}
FINAL_CONFIG_LSTM  = {"window_size": 5, "hidden_size": 32, ...}
FINAL_CONFIG_CNN   = {"window_size": 5, "conv_channels": [16, 32], ...}
```

Mientras un valor sea `None`, el pipeline imprime `[PENDIENTE]` para ese modelo y la comparación final queda bloqueada.

### 4.3 Fase 2 — Entrenamiento y evaluación final

Con la configuración elegida, el pipeline ejecuta el entrenamiento completo (o re-evaluación para ARIMA) y guarda todos los artefactos finales. El criterio de skip es la existencia del archivo `_final_ypred.npy`; si ya existe, se cargan los resultados directamente sin re-ejecutar.

Salidas de Fase 2 (comunes a todos los modelos):

```
_final_config.json         ← configuración y métricas del modelo elegido
_final_global.csv          ← RMSE y MAE globales walk-forward
_final_distrito.csv        ← métricas por distrito, ordenadas por MAE desc
_final_departamento.csv    ← métricas por departamento, ordenadas por MAE desc
_final_predicciones.csv    ← predicciones en formato largo (ver §4.4)
_final_ypred.npy           ← array (n_distritos, horizonte) de predicciones
```

MLP, LSTM y CNN guardan adicionalmente `_final_model.pth` (pesos del modelo) y `_final_curva.png` (curva de pérdida de entrenamiento). ARIMA no tiene equivalente a estos archivos porque no existe un objeto "modelo entrenado" separado de la inferencia: cada paso del walk-forward ajusta un nuevo modelo ARIMA.

### 4.4 Formato de predicciones largas (`_final_predicciones.csv`)

Todos los modelos generan un archivo de predicciones en **formato largo** con esquema idéntico:

| Columna | Descripción |
|---------|-------------|
| `modelo` | Nombre completo del modelo |
| `geocode` | Código geográfico del distrito |
| `departamento` | Departamento al que pertenece |
| `distrito` | Nombre del distrito |
| `horizonte` | Paso de predicción (1 a 5) |
| `anio` | Año calendario del paso (2020 a 2024) |
| `y_true` | Valor real observado |
| `y_pred` | Predicción del modelo |
| `error` | `y_pred − y_true` |
| `abs_error` | `|y_pred − y_true|` |
| `squared_error` | `(y_pred − y_true)²` |

Este formato permite unir los archivos de todos los modelos con un simple `pd.concat` y construir cualquier análisis comparativo posterior (por horizonte, por departamento, por distrito, etc.).

### 4.5 Persistencia — pipeline de fase única

El modelo de Persistencia no tiene hiperparámetros que explorar, por lo que no sigue la arquitectura de dos fases. Se ejecuta una sola vez y genera directamente todos los artefactos finales con prefijos sin `_final_`:

```
_resultados.csv        ← métricas por distrito
_departamento.csv      ← métricas por departamento
_global.csv            ← RMSE y MAE globales
_config.json           ← identificador y métricas
_predicciones.csv      ← formato largo idéntico al resto
_ypred.npy             ← array de predicciones
```

---

## 5. Construcción del dataset

### 5.1 Dataset estadístico (Persistencia y ARIMA)

```
X_train_stat: (n_distritos, 35, 1)   ← histórico completo 1985–2019
y_train_stat: (n_distritos, 5)        ← valores reales 2020–2024
```

La última dimensión de `X_train_stat` (tamaño 1) es la variable; se accede con `X_train[i, :, 0]` para obtener la serie 1D del distrito `i`. `y_train_stat` contiene los valores reales del período de pronóstico y actúa como ground truth en el walk-forward con oracle.

### 5.2 Dataset de aprendizaje profundo (MLP, LSTM, CNN)

Los tres modelos de aprendizaje profundo reciben los datos en un formato canónico tridimensional:

```
X: (n_muestras, window_size, 1)
y: (n_muestras, 1)
```

La última dimensión (tamaño 1) representa el número de variables — en este caso uno, ya que la serie es univariada. Cada modelo transforma internamente este tensor según su arquitectura: el MLP lo aplana, la LSTM lo usa directamente como secuencia, y la CNN lo permuta a `(n_muestras, 1, window_size)`.

### 5.3 Ventanas deslizantes y split temporal

Para cada distrito se generan ventanas deslizantes de tamaño `window_size`. Una ventana `[t, t+w)` va al conjunto de **entrenamiento** si su año objetivo `t+w` cae dentro del período 1985–2019; va al **test** si cae en 2020–2024.

Esta separación es **estrictamente temporal**: ninguna observación posterior a 2019 interviene en el ajuste del modelo. La separación por ventanas, en lugar de por distritos, aumenta el número de muestras de entrenamiento disponibles ya que los 180 distritos contribuyen cada uno con múltiples ventanas.

### 5.4 Tamaños de ventana explorados

```
DL_WINDOW_VALUES = [3, 4, 5, 6, 7]
```

Se exploran ventanas de 3 a 7 años. Ventanas muy cortas (1–2) no capturan tendencias; ventanas largas (≥8) reducen el número de muestras de entrenamiento disponibles y podrían sobreajustarse a la trayectoria inicial de la serie. El tamaño de ventana óptimo se determina por grid search y forma parte de la especificación del mejor modelo reportado.

---

## 6. Modelo Persistencia — baseline

El modelo de persistencia predice que el valor de cobertura boscosa del próximo año será igual al del año actual. Su propósito es establecer el **piso de rendimiento**: cualquier modelo con RMSE superior al de Persistencia no aporta valor predictivo.

La evaluación sigue el mismo protocolo walk-forward que los modelos complejos, con la diferencia de que "el modelo" es simplemente `ŷ_t = history[-1]`. Este diseño garantiza que la comparación sea metodológicamente válida.

---

## 7. Modelo ARIMA

### 7.1 Análisis exploratorio ACF/PACF

Antes del grid search se ejecuta un pipeline de diagnóstico (`analisis_arima.py`) que genera visualizaciones para tres series representativas seleccionadas automáticamente:

| Serie representativa | Criterio de selección |
|----------------------|-----------------------|
| Alta variabilidad | Distrito con mayor rango de `pct_bosque` (máx − mín histórico) |
| Baja variabilidad | Distrito con menor rango |
| Mediana nacional | Distrito en la mediana del rango |

Para cada una de las tres series se generan cuatro gráficos: la serie original, la serie diferenciada en orden 1, y los correlogramas ACF y PACF de ambas. El resultado observado es que las autocorrelaciones significativas de la serie original desaparecen tras una diferenciación de orden 1; esto justifica fijar `d=1` como constante en el grid search. Las salidas se almacenan en el subdirectorio `analisis_arima/`.

### 7.2 Estrategia de ventana rodante

En lugar de ajustar ARIMA sobre los 35 años completos, se ajusta sobre una **ventana de los últimos `w` años**. Esto responde a dos consideraciones: (1) las series de deforestación pueden tener rupturas estructurales que invalidan el supuesto de estacionariedad en el largo plazo; (2) el ajuste sobre ventanas cortas es computacionalmente tractable para los 180 distritos × 5 pasos × múltiples configuraciones.

```
ARIMA_P_VALUES      = [0, 1, 2]
ARIMA_D_VALUES      = [1]
ARIMA_Q_VALUES      = [0, 1, 2]
ARIMA_WINDOW_VALUES = [3, 4, 5, 6, 7, 10, 15, 20, 25, 30, 35, None]
```

`window=None` equivale a usar el histórico completo disponible en cada paso.

### 7.3 Naturaleza de la evaluación en ARIMA

A diferencia de los modelos DL, ARIMA no tiene una separación entre "fase de entrenamiento" y "fase de inferencia": en cada paso del walk-forward se ajusta un nuevo modelo ARIMA sobre la ventana disponible y se obtiene la predicción de inmediato. Esto implica que:

- La Fase 1 (grid search) ya contiene las métricas definitivas para cada configuración.
- La Fase 2 re-ejecuta exactamente el mismo proceso para la configuración elegida, con el fin de generar los artefactos adicionales (`_final_predicciones.csv`, `_final_distrito.csv`, etc.) que el grid search descartó por eficiencia.
- Los resultados de Fase 2 son numéricamente idénticos a los que produjo el grid search para esa configuración.

### 7.4 Fallback ante divergencia

Si el ajuste ARIMA no converge para un distrito en un paso dado, la predicción cae de vuelta al valor de persistencia. Este fallback se registra implícitamente en los residuos y puede inflar ligeramente el RMSE del modelo ARIMA en distritos con series irregulares.

---

## 8. Modelo MLP

### 8.1 Representación de la entrada

El MLP recibe la ventana temporal como un **vector plano** de longitud `window_size`. Esta representación no preserva el orden temporal explícitamente — el modelo trata la posición `t-3` y la posición `t-1` como dos features independientes. La capacidad de capturar dependencias temporales depende enteramente de que el optimizador descubra los pesos correctos para cada posición.

### 8.2 Arquitectura

Capas lineales apiladas con activación no lineal y dropout entre capas:

```
Input(window_size) → [Linear → Activación → Dropout] × n_capas → Linear(1)
```

La activación es un hiperparámetro del grid search porque sin ella las capas colapsarían a una transformación lineal equivalente a una sola capa.

```
MLP_HIDDEN_SIZES_VALUES = [[32, 16], [64, 32], [128, 64, 32]]
MLP_ACTIVATION_VALUES   = ["relu", "leaky_relu"]
MLP_DROPOUT_VALUES      = [0.0, 0.1]
MLP_EPOCHS_VALUES       = [50]
MLP_LR_VALUES           = [0.001, 0.0005]
MLP_BATCH_SIZE_VALUES   = [8, 16]
```

Combinaciones totales (por ventana): 3 × 2 × 2 × 1 × 2 × 2 = **48**; con 5 ventanas: **240 configuraciones**.

### 8.3 Función de pérdida y optimizador

Se usa **MSELoss** como función de pérdida, consistente con la métrica de evaluación (RMSE). El optimizador **Adam** se eligió por su robustez ante diferentes escalas de gradiente y su convergencia más rápida que SGD puro en datasets pequeños. La pérdida de entrenamiento se normaliza por el número total de muestras del dataset (no por el número de batches), lo que hace que el valor reportado sea independiente del `batch_size`.

---

## 9. Modelo LSTM

### 9.1 Representación de la entrada

A diferencia del MLP, la LSTM recibe la ventana como una **secuencia ordenada** `(batch, window_size, 1)`. Las celdas recurrentes procesan un paso temporal a la vez, manteniendo un estado interno que acumula información de los pasos anteriores. Esto le permite capturar tendencias y patrones de cambio interanual sin que el diseñador tenga que especificar explícitamente qué posiciones de la ventana son relevantes.

### 9.2 Activaciones internas — no son hiperparámetros

Las celdas LSTM tienen activaciones fijas por definición matemática: sigmoid en las puertas de olvido, entrada y salida; tanh en el candidato de celda y en la salida del estado oculto. Modificarlas produciría una arquitectura diferente (GRU, MGU, etc.), no una variante de LSTM. Por ello el grid search de LSTM no incluye `activation` como parámetro, a diferencia del MLP y CNN.

### 9.3 Dropout en arquitecturas recurrentes multicapa

El parámetro `dropout` de `nn.LSTM` aplica regularización **entre capas** (no sobre la salida de la última capa). Con `num_layers=1` este mecanismo no tiene efecto matemático, y PyTorch emite una advertencia. La implementación resuelve esto con:

```
lstm_dropout = dropout  si num_layers > 1
lstm_dropout = 0.0      si num_layers = 1
```

El dropout sobre el estado oculto final (antes de la capa lineal de salida) se aplica siempre mediante un módulo `nn.Dropout` separado, independientemente del número de capas.

```
LSTM_HIDDEN_SIZE_VALUES = [16, 32, 64]
LSTM_NUM_LAYERS_VALUES  = [1, 2]
LSTM_DROPOUT_VALUES     = [0.0, 0.1]
LSTM_EPOCHS_VALUES      = [50]
LSTM_LR_VALUES          = [0.001, 0.0005]
LSTM_BATCH_SIZE_VALUES  = [8, 16]
```

Combinaciones totales: 3 × 2 × 2 × 1 × 2 × 2 = **48**; con 5 ventanas: **240 configuraciones**.

Se incluye `dropout=0.1` para evaluar el efecto de la regularización; el grid search determinará si aporta beneficio real dado el tamaño reducido de las series (≤35 puntos por distrito).

---

## 10. Modelo CNN

### 10.1 Mecanismo de extracción de características

Una convolución 1D desliza un kernel de tamaño `kernel_size` a lo largo de la dimensión temporal, detectando patrones locales (tendencias cortas, cambios bruscos, rupturas). Con `padding="same"` la longitud temporal de salida es idéntica a la de entrada, lo que permite apilar varias capas convolucionales sin reducir la resolución temporal.

La salida de la última capa convolucional se aplana y se procesa por una capa densa antes de la predicción final:

```
Input(1, window_size) → [Conv1d → Activación → Dropout] × n_capas
                      → Flatten → Linear(dense_size) → Activación → Dropout → Linear(1)
```

### 10.2 Restricción kernel ≤ window_size

Un kernel más grande que la ventana no tiene sentido físico y PyTorch produciría un error. El pipeline omite automáticamente las combinaciones inválidas durante el grid search antes de intentar entrenar, y el constructor de la clase valida el mismo criterio como red de seguridad.

### 10.3 `dense_size` como hiperparámetro adicional

La CNN incluye una capa densa intermedia entre el aplanado y la salida. Esta capa comprime la representación `(conv_channels[-1] × window_size)` antes de la predicción. Su tamaño es un hiperparámetro específico de la CNN que no tiene análogo directo en MLP o LSTM, pero cumple la misma función de regularización que `hidden_sizes` en el MLP.

```
CNN_CONV_CHANNELS_VALUES = [[16], [32], [16, 32]]
CNN_KERNEL_SIZE_VALUES   = [2, 3]
CNN_DROPOUT_VALUES       = [0.0, 0.1]
CNN_ACTIVATION_VALUES    = ["relu", "leaky_relu"]
CNN_DENSE_SIZE_VALUES    = [16, 32]
CNN_EPOCHS_VALUES        = [50]
CNN_LR_VALUES            = [0.001, 0.0005]
CNN_BATCH_SIZE_VALUES    = [8, 16]
```

Combinaciones antes de filtrar kernels inválidos: 3 × 2 × 2 × 2 × 2 × 1 × 2 × 2 = **192**; con 5 ventanas: hasta **960 configuraciones** (reducidas por el filtro de kernel).

### 10.4 Dropout en capas convolucionales

El dropout se aplica tras cada capa convolucional, a diferencia del MLP (una aplicación por capa oculta) y la LSTM (una aplicación antes del FC). Con dos capas conv y dropout=0.1, la CNN experimenta tres operaciones de dropout en total. Esto es comportamiento estándar para CNNs y el grid search seleccionará el valor de dropout que mejor equilibre regularización y capacidad.

---

## 11. Criterios de comparabilidad entre modelos

### 11.1 Condiciones compartidas por los modelos DL

Los tres modelos de aprendizaje profundo (MLP, LSTM, CNN) comparten las siguientes condiciones para garantizar comparabilidad:

| Criterio | Decisión |
|----------|----------|
| **Datos de entrada** | Mismo `dataset_dl` — idénticas muestras de train y test por ventana |
| **Normalización de datos** | Ninguna — `pct_bosque ∈ [0,1]` es adecuado para todas las arquitecturas |
| **Función de pérdida** | MSELoss en todos |
| **Optimizador** | Adam en todos |
| **Normalización de la loss** | `epoch_loss / len(dataloader.dataset)` en todos (por muestra, no por batch) |
| **Early stopping** | Ninguno — todos entrenan el número de épocas configurado |
| **Evaluación** | Walk-forward geográfico idéntico sobre los 180 distritos |
| **Métrica de selección** | `rmse_test` como criterio primario; `mae_test` y `gap_rmse` como desempate |
| **Semilla** | `SEMILLA = 42` fijada al inicio de cada entrenamiento |
| **Soporte GPU** | Todos usan `DEVICE = cuda` si disponible, `cpu` en caso contrario |

La ausencia de early stopping es deliberada: garantiza que todos los modelos consuman el mismo presupuesto de entrenamiento (en épocas) para la misma configuración, eliminando una fuente de asimetría.

### 11.2 Comparabilidad del walk-forward

Los cinco modelos (incluyendo Persistencia y ARIMA) usan **el mismo protocolo walk-forward con oracle**: la ventana de entrada se actualiza en cada paso con el valor real observado, no con la predicción anterior. Esto garantiza que las métricas de todos los modelos son comparables entre sí, ya que ninguno acumula error propio de predicciones previas durante la evaluación.

### 11.3 Esquema de salidas homogéneo

Todos los modelos generan un conjunto de salidas con el mismo esquema:

| Archivo | Descripción |
|---------|-------------|
| `_final_global.csv` | RMSE y MAE globales — fila única comparable entre modelos |
| `_final_distrito.csv` | Métricas por distrito — mismas columnas en todos |
| `_final_departamento.csv` | Métricas por departamento — mismas columnas en todos |
| `_final_predicciones.csv` | Formato largo — esquema idéntico, concatenable directamente |
| `_final_ypred.npy` | Array `(n_distritos, horizonte)` — forma idéntica en todos |
| `_final_config.json` | Configuración del modelo elegido |

---

## 12. Comparación final

### 12.1 Ranking global

Los cinco modelos se ordenan por RMSE walk-forward global sobre el período 2020–2024. Esta métrica resume el error medio por predicción año-distrito sobre los 900 pares (180 distritos × 5 años).

### 12.2 Criterio de selección de distritos para visualización

Se identifican los 5 distritos mejor predichos y los 5 peor predichos usando un criterio de **consenso entre modelos**:

- **Mejores:** distritos donde el `max(RMSE entre modelos)` es mínimo — todos los modelos aciertan consistentemente. Representan casos de deforestación con patrón predecible.
- **Peores:** distritos donde el `min(RMSE entre modelos)` es máximo — incluso el mejor modelo falla. Representan dinámicas de cobertura que escapan a cualquiera de los enfoques evaluados.

Este criterio evita sesgar la selección hacia los casos donde un modelo en particular sobresale, y ofrece una lectura del comportamiento colectivo del conjunto de modelos.

### 12.3 Visualización

Para cada uno de los 10 distritos se genera un gráfico de panel doble: el panel izquierdo muestra el histórico de cobertura desde el año 2000 hasta 2019 (contexto), y el panel derecho muestra el período 2020–2024 con los valores reales y las predicciones de cada modelo superpuestas. El RMSE individual de cada modelo se incluye en la leyenda.

---

## 13. Inventario de salidas

### O1

```
data/interim/O1/
├── mapas-amazonia/             ← 40 rasters recortados a la Amazonía
├── mapas-reclasificados/       ← 40 rasters binarios bosque/no-bosque
├── mapas-cambios/              ← Mapa de cambio histórico 1985–2024
├── metricas-distritos/         ← Métricas de cambio por distrito
├── distritos-alto-cambio/      ← Top 200 distritos seleccionados
└── series-temporales/
    ├── entrenamiento/          ← 180 distritos × 40 años
    └── generalizacion-espacial/← 20 distritos reservados
```

### O2

```
data/interim/O2/modelos/
│
├── persistencia/
│   ├── persistencia_resultados.csv          ← métricas por distrito
│   ├── persistencia_resultados_departamento.csv
│   ├── persistencia_resultados_global.csv
│   ├── persistencia_resultados_config.json
│   ├── persistencia_resultados_predicciones.csv  ← formato largo
│   └── persistencia_resultados_ypred.npy
│
├── arima/
│   ├── analisis_arima/                      ← gráficos ACF/PACF (3 distritos repr.)
│   │   ├── serie_alta_variabilidad.png
│   │   ├── serie_baja_variabilidad.png
│   │   └── serie_mediana.png
│   ├── arima_resultados.csv                 ← Fase 1: todas las combinaciones
│   ├── arima_top5_configuraciones.csv
│   ├── arima_mejores_por_ventana.csv
│   ├── arima_boxplot_ventanas.png
│   ├── arima_final_config.json              ← Fase 2
│   ├── arima_final_global.csv
│   ├── arima_final_distrito.csv
│   ├── arima_final_departamento.csv
│   ├── arima_final_predicciones.csv         ← formato largo
│   └── arima_final_ypred.npy
│
├── mlp/
│   ├── mlp_resultados.csv                   ← Fase 1
│   ├── mlp_top5_configuraciones.csv
│   ├── mlp_mejores_por_ventana.csv
│   ├── mlp_final_model.pth                  ← Fase 2
│   ├── mlp_final_curva.png
│   ├── mlp_final_config.json
│   ├── mlp_final_global.csv
│   ├── mlp_final_distrito.csv
│   ├── mlp_final_departamento.csv
│   ├── mlp_final_predicciones.csv           ← formato largo
│   └── mlp_final_ypred.npy
│
├── lstm/                                    ← misma estructura que mlp/
├── cnn/                                     ← misma estructura que mlp/
│
└── comparacion/
    ├── comparacion_modelos.csv              ← ranking de los 5 modelos por RMSE
    ├── mejores_01–05_<geocode>.png          ← 5 distritos mejor predichos
    └── peores_01–05_<geocode>.png           ← 5 distritos peor predichos
```

La columna de referencia para el join entre archivos de distintos modelos es `_final_predicciones.csv`; su esquema es idéntico en todos los modelos y permite construir análisis comparativos con un simple `pd.concat`.
