from typing import Dict, Tuple

from loguru import logger
import torch.nn as nn
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
        Diccionario con información útil (mean, std, image_size) para normalización.
    """

    # Fallbacks seguros de ImageNet (correctos para ViT-B/16)
    default_mean = (0.485, 0.456, 0.406)
    default_std = (0.229, 0.224, 0.225)
    default_size = 224

    if pretrained:
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)

        # Algunos torchvision nuevos no traen meta completo
        meta_raw = getattr(weights, "meta", {}) or {}

        if isinstance(meta_raw, dict):
            mean = tuple(meta_raw.get("mean", default_mean))
            std  = tuple(meta_raw.get("std", default_std))
            image_size = meta_raw.get("image_size", default_size)
        else:
            mean = default_mean
            std = default_std
            image_size = default_size

    else:
        # Sin pesos preentrenados
        model = vit_b_16(weights=None)
        mean = default_mean
        std = default_std
        image_size = default_size

    if isinstance(image_size, (tuple, list)):
        image_size = image_size[0]

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    logger.info(f"Created ViT-B/16 model with {num_classes} output classes.")

    return model, {"mean": mean, "std": std, "image_size": image_size}
