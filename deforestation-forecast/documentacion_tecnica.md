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
4. [Construcción del dataset para modelos DL](#4-construcción-del-dataset-para-modelos-dl)
5. [Modelo Persistencia — baseline](#5-modelo-persistencia--baseline)
6. [Modelo ARIMA](#6-modelo-arima)
7. [Modelo MLP](#7-modelo-mlp)
8. [Modelo LSTM](#8-modelo-lstm)
9. [Modelo CNN](#9-modelo-cnn)
10. [Criterios de comparabilidad entre modelos DL](#10-criterios-de-comparabilidad-entre-modelos-dl)
11. [Comparación final](#11-comparación-final)
12. [Inventario de salidas](#12-inventario-de-salidas)

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

### 3.3 Diagnóstico de sobreajuste

Para los modelos de aprendizaje profundo se reporta adicionalmente el gap de generalización:

```
gap_RMSE  = RMSE_test − RMSE_train
ratio_RMSE = RMSE_test / RMSE_train
```

Valores `gap > 0` indican sobreajuste; valores cercanos a 1 en el ratio indican buen ajuste generalizable. Esta información se guarda en los CSV de resultados del grid search pero no interviene en la selección del mejor modelo (que se rige únicamente por `rmse_test`).

---

## 4. Construcción del dataset para modelos DL

### 4.1 Formato canónico

Los tres modelos de aprendizaje profundo (MLP, LSTM, CNN) reciben los datos en un formato canónico tridimensional:

```
X: (n_muestras, window_size, 1)
y: (n_muestras, 1)
```

La última dimensión (tamaño 1) representa el número de variables — en este caso uno, ya que la serie es univariada. Cada modelo transforma internamente este tensor según su arquitectura: el MLP lo aplana, la LSTM lo usa directamente como secuencia, y la CNN lo permuta a `(n_muestras, 1, window_size)`.

### 4.2 Ventanas deslizantes y split temporal

Para cada distrito se generan ventanas deslizantes de tamaño `window_size`. Una ventana `[t, t+w)` va al conjunto de **entrenamiento** si su año objetivo `t+w` cae dentro del período 1985–2019; va al **test** si cae en 2020–2024.

Esta separación es **estrictamente temporal**: ninguna observación posterior a 2019 interviene en el ajuste del modelo. La separación por ventanas, en lugar de por distritos, aumenta el número de muestras de entrenamiento disponibles ya que los 180 distritos contribuyen cada uno con múltiples ventanas.

### 4.3 Tamaños de ventana explorados

```
DL_WINDOW_VALUES = [3, 4, 5, 6, 7]
```

Se exploran ventanas de 3 a 7 años. Ventanas muy cortas (1–2) no capturan tendencias; ventanas largas (≥8) reducen el número de muestras de entrenamiento disponibles y podrían sobreajustarse a la trayectoria inicial de la serie. El tamaño de ventana óptimo se determina por grid search y forma parte de la especificación del mejor modelo reportado.

---

## 5. Modelo Persistencia — baseline

El modelo de persistencia predice que el valor de cobertura boscosa del próximo año será igual al del año actual. Su propósito es establecer el **piso de rendimiento**: cualquier modelo con RMSE superior al de Persistencia no aporta valor predictivo.

La evaluación sigue el mismo protocolo walk-forward que los modelos complejos, con la diferencia de que "el modelo" es simplemente `ŷ_t = history[-1]`. Este diseño garantiza que la comparación sea metodológicamente válida.

---

## 6. Modelo ARIMA

### 6.1 Selección del orden de diferenciación

El análisis ACF/PACF sobre tres series representativas (máxima variabilidad, mínima variabilidad y mediana) muestra autocorrelaciones significativas en la serie original que desaparecen tras una diferenciación de orden 1. Por ello `d=1` se fija como constante en el grid search.

### 6.2 Estrategia de ventana rodante

En lugar de ajustar ARIMA sobre los 35 años completos, se ajusta sobre una **ventana de los últimos `w` años**. Esto responde a dos consideraciones: (1) las series de deforestación pueden tener rupturas estructurales que invalidan el supuesto de estacionariedad en el largo plazo; (2) el ajuste sobre ventanas cortas es computacionalmente tractable para los 180 distritos × 5 pasos × 81 configuraciones.

```
ARIMA_P_VALUES      = [0, 1, 2]
ARIMA_D_VALUES      = [1]
ARIMA_Q_VALUES      = [0, 1, 2]
ARIMA_WINDOW_VALUES = [3, 4, 5, 6, 7, 10, 15, 20, 25, 30, 35, None]
```

`window=None` equivale a usar el histórico completo disponible en cada paso.

### 6.3 Fallback ante divergencia

Si el ajuste ARIMA no converge para un distrito en un paso dado, la predicción cae de vuelta al valor de persistencia. Este fallback se registra implícitamente en los residuos y puede inflar ligeramente el RMSE del modelo ARIMA en distritos con series irregulares.

---

## 7. Modelo MLP

### 7.1 Representación de la entrada

El MLP recibe la ventana temporal como un **vector plano** de longitud `window_size`. Esta representación no preserva el orden temporal explícitamente — el modelo trata la posición `t-3` y la posición `t-1` como dos features independientes. La capacidad de capturar dependencias temporales depende enteramente de que el optimizador descubra los pesos correctos para cada posición.

### 7.2 Arquitectura

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

### 7.3 Función de pérdida y optimizador

Se usa **MSELoss** como función de pérdida, consistente con la métrica de evaluación (RMSE). El optimizador **Adam** se eligió por su robustez ante diferentes escalas de gradiente y su convergencia más rápida que SGD puro en datasets pequeños. La pérdida de entrenamiento se normaliza por el número total de muestras del dataset (no por el número de batches), lo que hace que el valor reportado sea independiente del `batch_size`.

---

## 8. Modelo LSTM

### 8.1 Representación de la entrada

A diferencia del MLP, la LSTM recibe la ventana como una **secuencia ordenada** `(batch, window_size, 1)`. Las celdas recurrentes procesan un paso temporal a la vez, manteniendo un estado interno que acumula información de los pasos anteriores. Esto le permite capturar tendencias y patrones de cambio interanual sin que el diseñador tenga que especificar explícitamente qué posiciones de la ventana son relevantes.

### 8.2 Activaciones internas — no son hiperparámetros

Las celdas LSTM tienen activaciones fijas por definición matemática: sigmoid en las puertas de olvido, entrada y salida; tanh en el candidato de celda y en la salida del estado oculto. Modificarlas produciría una arquitectura diferente (GRU, MGU, etc.), no una variante de LSTM. Por ello el grid search de LSTM no incluye `activation` como parámetro, a diferencia del MLP y CNN.

### 8.3 Dropout en arquitecturas recurrentes multicapa

El parámetro `dropout` de `nn.LSTM` aplica regularización **entre capas** (no sobre la salida de la última capa). Con `num_layers=1` este mecanismo no tiene efecto matemático, y PyTorch emite una advertencia. La implementación resuelve esto con:

```
lstm_dropout = dropout  si num_layers > 1
lstm_dropout = 0.0      si num_layers = 1
```

El dropout sobre el estado oculto final (antes de la capa lineal de salida) se aplica siempre mediante un módulo `nn.Dropout` separado, independientemente del número de capas.

```
LSTM_HIDDEN_SIZE_VALUES = [16, 32, 64]
LSTM_NUM_LAYERS_VALUES  = [1, 2]
LSTM_DROPOUT_VALUES     = [0.0]
LSTM_EPOCHS_VALUES      = [50]
LSTM_LR_VALUES          = [0.001, 0.0005]
LSTM_BATCH_SIZE_VALUES  = [8, 16]
```

Combinaciones totales: 3 × 2 × 1 × 1 × 2 × 2 = **24**; con 5 ventanas: **120 configuraciones**.

`LSTM_DROPOUT_VALUES = [0.0]` refleja que con las series cortas disponibles (≤35 puntos por distrito) introducir dropout en la LSTM tiende a dificultar la convergencia más que a regularizar.

---

## 9. Modelo CNN

### 9.1 Mecanismo de extracción de características

Una convolución 1D desliza un kernel de tamaño `kernel_size` a lo largo de la dimensión temporal, detectando patrones locales (tendencias cortas, cambios bruscos, rupturas). Con `padding="same"` la longitud temporal de salida es idéntica a la de entrada, lo que permite apilar varias capas convolucionales sin reducir la resolución temporal.

La salida de la última capa convolucional se aplana y se procesa por una capa densa antes de la predicción final:

```
Input(1, window_size) → [Conv1d → Activación → Dropout] × n_capas
                      → Flatten → Linear(dense_size) → Activación → Dropout → Linear(1)
```

### 9.2 Restricción kernel ≤ window_size

Un kernel más grande que la ventana no tiene sentido físico y PyTorch produciría un error. El pipeline omite automáticamente las combinaciones inválidas durante el grid search antes de intentar entrenar, y el constructor de la clase valida el mismo criterio como red de seguridad.

### 9.3 `dense_size` como hiperparámetro adicional

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

### 9.4 Dropout en capas convolucionales

El dropout se aplica tras cada capa convolucional, a diferencia del MLP (una aplicación por capa oculta) y la LSTM (una aplicación antes del FC). Con dos capas conv y dropout=0.1, la CNN experimenta tres operaciones de dropout en total. Esto es comportamiento estándar para CNNs y el grid search seleccionará el valor de dropout que mejor equilibre regularización y capacidad.

---

## 10. Criterios de comparabilidad entre modelos DL

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
| **Métrica de selección** | `rmse_test` sobre ventanas deslizantes |
| **Semilla** | `SEMILLA = 42` fijada al inicio de cada pipeline |

La ausencia de early stopping es deliberada: garantiza que todos los modelos consuman el mismo presupuesto de entrenamiento (en épocas) para la misma configuración, eliminando una fuente de asimetría.

---

## 11. Comparación final

### 11.1 Ranking global

Los cinco modelos se ordenan por RMSE walk-forward global sobre el período 2020–2024. Esta métrica resume el error medio por predicción año-distrito sobre los 900 pares (180 distritos × 5 años).

### 11.2 Criterio de selección de distritos para visualización

Se identifican los 5 distritos mejor predichos y los 5 peor predichos usando un criterio de **consenso entre modelos**:

- **Mejores:** distritos donde el `max(RMSE entre modelos)` es mínimo — todos los modelos aciertan consistentemente. Representan casos de deforestación con patrón predecible.
- **Peores:** distritos donde el `min(RMSE entre modelos)` es máximo — incluso el mejor modelo falla. Representan dinámicas de cobertura que escapan a cualquiera de los enfoques evaluados.

Este criterio evita sesgar la selección hacia los casos donde un modelo en particular sobresale, y ofrece una lectura del comportamiento colectivo del conjunto de modelos.

### 11.3 Visualización

Para cada uno de los 10 distritos se genera un gráfico de panel doble: el panel izquierdo muestra el histórico de cobertura desde el año 2000 hasta 2019 (contexto), y el panel derecho muestra el período 2020–2024 con los valores reales y las predicciones de cada modelo superpuestas. El RMSE individual de cada modelo se incluye en la leyenda.

---

## 12. Inventario de salidas

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
├── persistencia/               ← Métricas globales, por departamento y por distrito
├── arima/
│   ├── analisis_arima/         ← Gráficos ACF/PACF exploratorios
│   └── ...                     ← Grid CSV, mejor config, predicciones .npy
├── mlp/                        ← Grid CSV, mejor .pth, curva, métricas, predicciones .npy
├── lstm/                       ← Ídem
├── cnn/                        ← Ídem
└── comparacion/
    ├── comparacion_modelos.csv ← Ranking de los 5 modelos por RMSE
    ├── mejores_01–05_*.png     ← Gráficos de los 5 distritos mejor predichos
    └── peores_01–05_*.png      ← Gráficos de los 5 distritos peor predichos
```
