# O1 — Decisiones de Diseño Técnico

Documento de decisiones de arquitectura y diseño del módulo de preparación de datos (O1) para el pipeline de pronóstico de deforestación amazónica peruana.

---

## 1. Visión general del módulo

O1 transforma datos satelitales crudos (MapBiomas, 1985–2024) en un panel tabular estructurado, listo para el entrenamiento de modelos predictivos. El módulo resuelve dos problemas fundamentales: la complejidad del dato geoespacial (rasters multi-clase, multi-año, multi-escala) y la necesidad de reducir dimensionalidad sin perder la señal de deforestación.

La arquitectura está dividida en dos etapas lineales e independientes:

- **R1–R2** — Preparación y clasificación binaria del dato raster.
- **R3** — Detección de cambios, zonificación espacial y extracción de series temporales.

---

## 2. Reclasificación binaria (R1–R2)

### Decisión: Reducir a bosque / no-bosque

MapBiomas ofrece ~30 clases de uso de suelo (pasto, agua, cultivos, etc.). La decisión de colapsar esas clases en una distinción binaria —bosque (1) vs. no-bosque (0)— responde a tres razones:

1. **Enfoque en la variable objetivo.** El fenómeno a predecir es pérdida de cobertura forestal, no la transición entre subclases de no-bosque. La granularidad adicional no aporta señal predictiva para ese objetivo.

2. **Consistencia temporal.** Las categorías de MapBiomas han variado entre versiones de la colección. La binarización aisla el pipeline de esos cambios de nomenclatura, garantizando que la comparación 1985–2024 sea metodológicamente coherente.

3. **Reducción de complejidad downstream.** Los modelos de series temporales (LSTM, ARIMA) operan sobre una sola variable continua (% de cobertura forestal), que surge directamente de esta binarización.

### Decisión: NoData = 255 (uint8)

El valor 255 en un raster de 8 bits cumple un rol estructural: marca píxeles no observados (nubes, sombras, bordes) sin introducir ambigüedad. Esta elección es coherente con la convención de MapBiomas, evita conflictos con los valores semánticos 0 y 1, y permite almacenar los rasters en el tipo más compacto disponible.

### Decisión: Depuración de clases inválidas antes de reclasificar

Antes de aplicar la regla bosque/no-bosque se eliminan clases que no tienen representación ecológica válida en la Amazonía peruana (ej. clase 27, "no observado"). Si estas clases pasaran a la reclasificación, contaminarían las métricas de área forestal de años específicos donde la cobertura de nube fue alta.

---

## 3. Detección de cambios (R3.1)

### Decisión: Comparación de años consecutivos (cambio temporal, no acumulado)

El mapa de cambios se construye comparando cada par de años adyacentes, no el raster inicial contra el final. Esto permite detectar tanto deforestación como regeneración (no-bosque → bosque), y preserva la señal de pulsos episódicos de deforestación que se habrían suavizado en una comparación punto a punto de 40 años.

### Decisión: Procesamiento por teselas (5000 × 5000 píxeles)

Cargar el stack temporal completo (~50 000 × 50 000 × 40 capas) en memoria es inviable en hardware convencional. El procesamiento por teselas independientes resuelve esa restricción sin alterar el resultado, porque la detección de cambio es una operación píxel a píxel sin dependencias espaciales entre vecinos.

Este diseño tiene además un beneficio secundario: cada tesela es autocontenida, lo que abre la puerta a paralelización futura sin modificar el algoritmo.

---

## 4. Unidad de análisis: el distrito administrativo (R3.2–R3.3)

### Decisión: Agregar a nivel distrital, no mantener resolución de píxel

El distrito administrativo es la unidad de observación del pipeline. Cada distrito acumula los píxeles de bosque y no-bosque dentro de su geometría, produciendo una serie temporal de cobertura por entidad geográfica estable.

Esta elección tiene dos consecuencias metodológicas directas. Primero, los distritos tienen continuidad temporal garantizada: existen en todos los años del rango 1985–2024, lo que hace posible construir series completas sin imputación. Segundo, la unidad de análisis coincide con la escala a la que operan las decisiones de política forestal en Perú, lo que da relevancia aplicada al pronóstico producido por los modelos downstream.

### Decisión: Estadísticas zonales en lugar de intersección vectorial

Para calcular cuántos píxeles cambiaron dentro de cada distrito se usa estadística zonal sobre el raster, en lugar de intersectar geometrías vectoriales. Las razones son:

- Las operaciones raster son órdenes de magnitud más rápidas que la intersección polígono-polígono sobre 200 distritos y 40 años.
- Las operaciones vectoriales generan artefactos geométricos (slivers) en los bordes de distrito que introducen ruido sin valor analítico.
- El raster de distritos rasterizado es el mismo para todas las consultas temporales, por lo que el costo de rasterización se paga una sola vez.

### Decisión: Selección de los 200 distritos de mayor cambio

El corpus de entrenamiento se restringe a los distritos con mayor densidad de cambio forestal. Esta decisión tiene dos consecuencias metodológicas intencionales:

1. **Concentrar señal.** Los distritos con cambio mínimo no aportan información predictiva útil; incluirlos diluyendo el conjunto de entrenamiento perjudica los modelos.
2. **Separación espacial del conjunto de prueba.** Los distritos no seleccionados forman el conjunto de generalización espacial, que permite evaluar si el modelo generaliza a zonas que no vio durante el entrenamiento —una métrica más exigente que una partición aleatoria.

---

## 5. Estructura del panel de salida (R3.4)

### Decisión: Formato largo (long format), una fila por (distrito, año)

La salida final es un panel en formato largo: cada combinación distrito–año ocupa una fila, con columnas de píxeles de bosque, no-bosque y porcentaje de cobertura. Este formato es nativo para modelos de series temporales panel (LSTM, ARIMA sobre datos de panel) y facilita el filtrado y la validación sin necesidad de transformar la estructura.

### Decisión: Diferir la ingeniería de características al módulo O2

El panel de O1 no incluye rezagos, medias móviles ni diferencias temporales. Esas transformaciones se generan en el módulo siguiente. Esta separación es deliberada: O1 produce el dato observado fiel; O2 produce las representaciones derivadas que los modelos necesitan. Mezclarlos haría el módulo O1 más frágil ante cambios en la estrategia de modelado.

### Decisión: Partición 90 % / 10 % (entrenamiento / generalización espacial)

La partición espacial utiliza el 10 % de los distritos seleccionados como conjunto de generalización. Esta proporción es más conservadora que el 70/30 convencional porque el conjunto de 200 × 40 filas (8 000 observaciones) ya es reducido; un 20 % de prueba recortaría demasiado el entrenamiento. La semilla fija (42) garantiza que la partición sea reproducible en todas las ejecuciones.

---

## 6. Sistema de coordenadas

### Decisión: Doble CRS (EPSG:32718 + EPSG:4326)

El pipeline mantiene dos sistemas de referencia con roles distintos:

| CRS | Uso | Razón |
|-----|-----|-------|
| EPSG:32718 (UTM Zona 18S) | Cálculo de áreas, estadísticas de píxeles | Proyección métrica; error <2% en latitudes peruanas |
| EPSG:4326 (WGS84) | Operaciones con geometrías de distritos | Formato nativo de los shapefiles fuente; evita reproyección |

La alternativa de proyectar todo a un único CRS fue descartada porque reprojecting rasters grandes introduce errores de remuestreo y aumenta significativamente el tiempo de procesamiento sin beneficio en la métrica final (porcentaje, no metros cuadrados absolutos).

---

## 7. Idempotencia y checkpoints

### Decisión: Cada etapa verifica si su salida ya existe

Antes de ejecutar cualquier paso costoso (detección de cambios, zonificación, extracción de series), el pipeline comprueba si el archivo de salida existe. Si existe, omite el paso. Esta propiedad de idempotencia tiene consecuencias prácticas importantes:

- Permite reanudar una ejecución interrumpida sin recomputar desde cero.
- Facilita el desarrollo iterativo: se puede modificar R3.4 sin volver a ejecutar R3.1.
- Hace el pipeline auditable: los archivos intermedios pueden inspeccionarse en cualquier momento.

El costo es que cambios en parámetros upstream no invalidan automáticamente los outputs downstream; la responsabilidad de borrar outputs obsoletos recae en el usuario. Se considera un trade-off aceptable dado el coste computacional de cada etapa.

---

## 8. Trazabilidad de metadatos

Cada etapa genera un CSV de metadatos paralelo al output principal: distribución de clases, recuentos de píxeles, área en km², estadísticas de cambio por distrito. Estos archivos sirven tres propósitos:

1. **Auditoría de calidad** durante el desarrollo, sin necesidad de abrir los rasters.
2. **Trazabilidad** para la tesis: permiten reportar estadísticas descriptivas del dato sin depender de la ejecución del pipeline.
3. **Detección temprana de errores**: una distribución de clases anómala en el CSV de metadatos es una señal de alerta antes de que el error se propague a las series temporales.

---

## 9. Resumen de principios de diseño

| Principio | Manifestación en O1 |
|-----------|---------------------|
| Escalabilidad de memoria | Procesamiento por teselas en R3.1 |
| Alineación con el objetivo | Binarización bosque/no-bosque; unidad distrital |
| Reproducibilidad | Semilla fija, CRS explícitos, NoData canónico |
| Separación de responsabilidades | R1–R2 limpia, R3 agrega, O2 construye características |
| Resiliencia operativa | Checkpoints idempotentes en cada etapa |
| Trazabilidad | Metadatos CSV paralelos a cada output |
