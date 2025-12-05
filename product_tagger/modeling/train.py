import json
from pathlib import Path
from typing import Dict, Tuple

import mlflow
import mlflow.pytorch
import torch
from loguru import logger
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import typer

from product_tagger.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PROCESSED_DATA_DIR
)
from product_tagger.data_loader import ImageDataset
from product_tagger.modeling.vit import create_vit_model

app = typer.Typer()


def build_dataloaders(
    train_csv: Path,
    val_csv: Path,
    train_images_dir: Path,
    val_images_dir: Path,
    label_col: str = "articleType",
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Construye los DataLoaders de entrenamiento y validación a partir de los CSV
    procesados y las carpetas de imágenes.

    - Lee el CSV de entrenamiento para inferir el conjunto de clases y crear
      un mapeo estable clase -> índice entero (`class_to_idx`).
    - Instancia dos `ImageDataset` (train y val) que aplican las
      transformaciones de entrada y convierten las etiquetas a índices.
    - Devuelve DataLoaders listos para usar en el loop de entrenamiento.
    """

    train_df = ImageDataset(
        csv_path=train_csv,
        images_dir=train_images_dir,
        label_col=label_col,
        transform=None,
    ).data

    # Mapeo estable clase -> índice
    classes = sorted(train_df[label_col].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    logger.info(f"Detected {len(classes)} classes.")

    # Transforms básicos (ToTensor + normalización estándar ImageNet)
    # Los valores se ajustan luego según el meta devuelto por create_vit_model.
    base_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Datasets ya con el mapping
    train_dataset = ImageDataset(
        csv_path=train_csv,
        images_dir=train_images_dir,
        label_col=label_col,
        transform=base_transform,
        class_to_idx=class_to_idx,
    )
    val_dataset = ImageDataset(
        csv_path=val_csv,
        images_dir=val_images_dir,
        label_col=label_col,
        transform=base_transform,
        class_to_idx=class_to_idx,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, class_to_idx


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Ejecuta una época completa de entrenamiento sobre un DataLoader.

    Recorre todos los batches del conjunto de entrenamiento, calcula la
    pérdida, realiza backpropagation y acumula métricas de `loss` y
    `accuracy` promedio.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evalúa el modelo sobre un DataLoader sin actualizar los pesos.

    Se utiliza típicamente para el conjunto de validación, calculando
    la pérdida y la `accuracy` promedio de la época.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


@app.command()
def main(
    train_csv: Path = PROCESSED_DATA_DIR / "train.csv",
    val_csv: Path = PROCESSED_DATA_DIR / "val.csv",
    train_images_dir: Path = PROCESSED_DATA_DIR / "images" / "train",
    val_images_dir: Path = PROCESSED_DATA_DIR / "images" / "val",
    label_col: str = "articleType",
    model_output_path: Path = MODELS_DIR / "vit_articleType.pt",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    seed: int = 42,
    device_str: str = "auto",
):
    """
    Entrena un Vision Transformer (ViT-B/16) sobre las imágenes procesadas
    y registra los experimentos en MLflow.
    """

    # Configuración reproducible / dispositivo
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Using device: {device}")
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")

    # Configuración MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    train_loader, val_loader, class_to_idx = build_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        train_images_dir=train_images_dir,
        val_images_dir=val_images_dir,
        label_col=label_col,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    num_classes = len(class_to_idx)
    model, meta = create_vit_model(num_classes=num_classes, pretrained=True)

    # Normalización según pesos ImageNet
    mean = meta["mean"]
    std = meta["std"]
    logger.info(f"Using normalization mean={mean}, std={std}")

    # Re-definir transforms de los datasets con normalización correcta
    norm_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    train_loader.dataset.transform = norm_transform
    val_loader.dataset.transform = norm_transform

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_acc = 0.0

    with mlflow.start_run(run_name="vit_articleType"):
        # Log de hiperparámetros
        mlflow.log_params(
            {
                "model": "vit_b_16",
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "seed": seed,
                "num_workers": num_workers,
                "label_col": label_col,
            }
        )

        for epoch in range(1, epochs + 1):
            logger.info(f"Epoch [{epoch}/{epochs}]")

            train_loss, train_acc = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

            val_loss, val_acc = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

            logger.info(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

            # Registro de métricas en MLflow
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch,
            )

            # Guardado simple del mejor modelo en disco
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logger.info(
                    f"New best val_acc={best_val_acc:.4f}, saving model to {model_output_path}"
                )

                state = {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "meta": meta,
                    "label_col": label_col,
                }
                torch.save(state, model_output_path)

        # Registrar el modelo (último) en MLflow para exploración
        mlflow.pytorch.log_model(model, artifact_path="model")

        # Guardar mapping clase -> índice como artefacto JSON
        mapping_path = MODELS_DIR / "class_to_idx.json"
        with mapping_path.open("w", encoding="utf-8") as f:
            json.dump(class_to_idx, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(str(mapping_path))

    logger.success("Training complete.")


if __name__ == "__main__":
    app()
