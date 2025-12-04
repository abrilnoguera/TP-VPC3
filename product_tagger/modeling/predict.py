from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import typer

from product_tagger.config import MODELS_DIR, PROCESSED_DATA_DIR
from product_tagger.data_loader import ImageDataset
from product_tagger.modeling.vit import create_vit_model

app = typer.Typer()


@app.command()
def main(
    test_csv: Path = PROCESSED_DATA_DIR / "test.csv",
    test_images_dir: Path = PROCESSED_DATA_DIR / "images" / "test",
    model_path: Path = MODELS_DIR / "vit_articleType.pt",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    batch_size: int = 64,
    num_workers: int = 4,
    device_str: str = "auto",
):
    """
    Realiza inferencia sobre el split de test utilizando el modelo ViT entrenado.
    """

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    logger.info(f"Using device: {device}")
    logger.info(f"Loading model from: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    # Valores por defecto de ImageNet (mismos que en vit.py)
    # Estos son los valores estándar para modelos preentrenados en ImageNet
    default_meta = {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "image_size": 224,
    }
    meta = checkpoint.get("meta", default_meta)
    num_classes = len(class_to_idx)

    # Cargamos el mismo modelo que en entrenamiento
    model, _ = create_vit_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    mean = meta["mean"]
    std = meta["std"]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # Dataset de test (las etiquetas del CSV no se usan, pero se leen igual)
    dataset = ImageDataset(
        csv_path=test_csv,
        images_dir=test_images_dir,
        label_col="articleType",
        transform=transform,
        class_to_idx=None,  # no necesitamos convertir etiqueta a índice
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    idx_to_class = {int(v): k for k, v in class_to_idx.items()}

    all_ids = []
    all_preds_idx = []
    all_preds_label = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Predict", total=len(loader)):
            # labels aquí son strings, pero no las transformamos
            batch_ids = dataset.data["id"][
                len(all_ids) : len(all_ids) + len(images)
            ].tolist()

            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            preds_idx = preds.cpu().tolist()
            preds_label = [idx_to_class[idx] for idx in preds_idx]

            all_ids.extend(batch_ids)
            all_preds_idx.extend(preds_idx)
            all_preds_label.extend(preds_label)

    # Guardar CSV de predicciones
    import pandas as pd

    df_out = pd.DataFrame(
        {
            "id": all_ids,
            "pred_idx": all_preds_idx,
            "pred_label": all_preds_label,
        }
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(predictions_path, index=False)

    logger.success(f"Saved predictions to {predictions_path}")


if __name__ == "__main__":
    app()
