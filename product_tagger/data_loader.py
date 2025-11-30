from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image
import pandas as pd
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    Dataset genérico para imágenes + etiquetas.

    Parámetros
    ----------
    csv_path : Path
        CSV con columnas ["id", ..., label_col]
    images_dir : Path
        Carpeta donde están las imágenes ya procesadas
    label_col : str
        Columna objetivo
    transform : callable
        Transformaciones (augmentations o normalización)
    class_to_idx : Dict[str, int], optional
        Mappeo de etiqueta de clase a índice entero.
    """

    def __init__(
        self,
        csv_path: Path,
        images_dir: Path,
        label_col: str = "articleType",
        transform=None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):

        self.csv_path = csv_path
        self.images_dir = images_dir
        self.label_col = label_col
        self.transform = transform

        self.data = pd.read_csv(csv_path)

        if label_col not in self.data.columns:
            raise ValueError(f"Label column '{label_col}' not found in CSV.")

        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple:
        row = self.data.iloc[idx]
        img_id = row["id"]
        img_path = self.images_dir / f"{img_id}.jpg"

        img = Image.open(img_path).convert("RGB")
        label = row[self.label_col]

        if self.class_to_idx is not None:
            label = self.class_to_idx[label]

        if self.transform:
            img = self.transform(img)

        return img, label