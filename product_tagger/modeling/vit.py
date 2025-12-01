from typing import Dict, Tuple

import torch.nn as nn
from loguru import logger
from torchvision.models import ViT_B_16_Weights, vit_b_16


def create_vit_model(
    num_classes: int,
    pretrained: bool = True,
) -> Tuple[nn.Module, Dict]:
    """
    Crea un Vision Transformer (ViT-B/16) listo para clasificación.

    Parameters
    ----------
    num_classes : int
        Número de clases de salida.
    pretrained : bool, default=True
        Si True, carga pesos pre-entrenados en ImageNet.

    Returns
    -------
    model : nn.Module
        Modelo ViT con la última capa ajustada a `num_classes`.
    meta : Dict
        Diccionario con información útil (mean, std) para normalización.
    """

    if pretrained:
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)
        # Mean y std documentados en los metadatos de los pesos
        mean = tuple(weights.meta["mean"])
        std = tuple(weights.meta["std"])
    else:
        model = vit_b_16(weights=None)
        # Valores genéricos por defecto si no usamos pesos pre-entrenados
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    logger.info(f"Created ViT-B/16 model with {num_classes} output classes.")

    return model, {"mean": mean, "std": std}


