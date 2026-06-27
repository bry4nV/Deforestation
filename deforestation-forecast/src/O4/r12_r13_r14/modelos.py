"""
Carga de los modelos ya finalizados que se van a generalizar a las 20 zonas
nuevas: ARIMA (O2, sin pesos persistidos — se reajusta por distrito en cada
paso walk-forward) y CNN (O3/R11, pesos entrenados persistidos en .pth).
"""

import logging

import torch

from O3.r11.pipeline_cnn import CNN1D, _parsear_canales

from O4.config import ARIMA_D, ARIMA_P, ARIMA_Q, ARIMA_WINDOW, R11_CNN_MODEL_PTH

logger = logging.getLogger(__name__)


def obtener_orden_arima() -> tuple:
    """(p, d, q), window — hiperparámetros ya finalizados en O2 (ver config.py)."""
    return (ARIMA_P, ARIMA_D, ARIMA_Q), ARIMA_WINDOW


def cargar_cnn_entrenado(ruta_pth: str = R11_CNN_MODEL_PTH) -> tuple:
    """
    Reconstruye el CNN1D de R11 a partir del checkpoint guardado por
    entrenar_config_final_cnn() y carga sus pesos ya entrenados.

    Returns
    -------
    modelo : CNN1D en modo eval()
    config : dict con los hiperparámetros usados (window_size, etc.)
    """
    checkpoint = torch.load(ruta_pth, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    if checkpoint.get("model_type") != "CNN":
        raise ValueError(f"Checkpoint no es de tipo CNN: {checkpoint.get('model_type')}")

    canales_conv = _parsear_canales(config["canales_conv"])
    n_canales_entrada = 7  # COLUMNAS_PREDICTORAS de O3.config

    modelo = CNN1D(
        canales_entrada=n_canales_entrada,
        window_size=int(config["window_size"]),
        canales_conv=canales_conv,
        kernel_size=int(config["kernel_size"]),
        dropout=float(config["dropout"]),
        activacion=config["activacion"],
        tamanio_denso=int(config["tamanio_denso"]),
    )
    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.eval()

    logger.info(f"[OK] CNN cargado desde {ruta_pth} — config: {config['modelo']}")
    return modelo, config
