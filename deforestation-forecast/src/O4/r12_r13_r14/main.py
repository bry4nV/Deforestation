"""
Orquestador de O4 — Generalización espacial.

Protocolo de ejecución:
  1. R12 — verifica el dataset de las 20 zonas nuevas (no usadas en
     entrenamiento, 100% completo).
  2. R13 — pronósticos walk-forward del CNN1D extendido (O3/R11, modelo
     final del proyecto) sobre esas 20 zonas, más el pronóstico 2025
     (ancla=2024), sin reentrenar.
  3. R14 — métricas, factores territoriales y gráficos para el informe.

Uso:
    python -m O4.r12_r13_r14.main
"""

import logging

from O3.utils import iniciar_log_archivo

from O4.r12_r13_r14.analisis_generalizacion import analizar_generalizacion
from O4.r12_r13_r14.construir_dataset_generalizacion import construir_dataset_generalizacion
from O4.r12_r13_r14.pipeline_pronosticos import pipeline_pronosticos

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=" * 70)
    logger.info("O4 — GENERALIZACIÓN ESPACIAL (R12, R13, R14)")
    logger.info("=" * 70)

    logger.info("[PASO 1] R12 — Dataset de generalización")
    construir_dataset_generalizacion()

    logger.info("[PASO 2] R13 — Pronósticos en las zonas nuevas")
    pipeline_pronosticos()

    logger.info("[PASO 3] R14 — Análisis y gráficos del informe")
    tabla_metricas = analizar_generalizacion()

    logger.info("=" * 70)
    logger.info("O4 completado.\n" + tabla_metricas.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    iniciar_log_archivo("o4")
    main()
