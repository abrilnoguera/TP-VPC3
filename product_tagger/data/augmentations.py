# project_name/data/augmentations.py

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 224):
    """
    Transformaciones estándar y robustas para clasificación.
    - Resize para asegurar tamaño consistente
    - ColorJitter fuerte
    - Ruido
    - Blur
    - Rotación ligera
    - Flip horizontal

    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        
        # Augmentations geométricas y de color
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=12,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.4
        ),
        A.RGBShift(
            r_shift_limit=15,
            g_shift_limit=15,
            b_shift_limit=15,
            p=0.3
        ),

        # Blur / Ruido
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.1),

        # Rotación ligera (no destruye la semántica)
        A.Rotate(limit=12, border_mode=0, p=0.35),

        # Regularización moderna
        A.Cutout(
            num_holes=4,
            max_h_size=image_size // 6,
            max_w_size=image_size // 6,
            fill_value=0,
            p=0.3
        ),
    ])


def get_val_transforms(image_size: int = 224):
    """
    Transformaciones para validación:
    - Sólo resize
    - Sin augmentations destructivas
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        # ToTensorV2 NO va acá
        # El AutoImageProcessor se encarga
    ])

