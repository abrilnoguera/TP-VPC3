import json
from pathlib import Path
from typing import Dict, Tuple

from sklearn.metrics import f1_score
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
    PROCESSED_DATA_DIR,
)
from product_tagger.data_loader import ImageDataset
from product_tagger.modeling.vit import create_vit_model_v2

app = typer.Typer()


def build_dataloaders_with_augmentation(
    train_csv: Path,
    val_csv: Path,
    train_images_dir: Path,
    val_images_dir: Path,
    mean,
    std,
    image_size: int = 224,
    label_col: str = "articleType",
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    
    tmp_dataset = ImageDataset(
        csv_path=train_csv,
        images_dir=train_images_dir,
        label_col=label_col,
        transform=None,
    )
    train_df = tmp_dataset.data

    classes = sorted(train_df[label_col].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    logger.info(f"Detected {len(classes)} classes.")

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.1,
                    )
                ],
                p=0.5,
            ),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_dataset = ImageDataset(
        csv_path=train_csv,
        images_dir=train_images_dir,
        label_col=label_col,
        transform=train_transform,
        class_to_idx=class_to_idx,
    )
    val_dataset = ImageDataset(
        csv_path=val_csv,
        images_dir=val_images_dir,
        label_col=label_col,
        transform=val_transform,
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
) -> Tuple[float, float, float, float]:
    """
    Devuelve:
        epoch_loss, epoch_acc, f1_macro, f1_weighted
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

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

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0

    if total > 0:
        f1_macro = f1_score(all_labels, all_preds, average="macro")
        f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    else:
        f1_macro = 0.0
        f1_weighted = 0.0

    return epoch_loss, epoch_acc, f1_macro, f1_weighted


@app.command()
def main(
    train_csv: Path = PROCESSED_DATA_DIR / "train.csv",
    val_csv: Path = PROCESSED_DATA_DIR / "val.csv",
    train_images_dir: Path = PROCESSED_DATA_DIR / "images" / "train",
    val_images_dir: Path = PROCESSED_DATA_DIR / "images" / "val",
    label_col: str = "articleType",
    model_output_path: Path = MODELS_DIR / "vit_articleType_aug.pt",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    seed: int = 42,
    device_str: str = "auto",
    patience: int = 5,
):

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

    # MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Dataset temporal para obtener número de clases
    tmp_ds = ImageDataset(
        csv_path=train_csv,
        images_dir=train_images_dir,
        label_col=label_col,
        transform=None,
    )
    classes = sorted(tmp_ds.data[label_col].unique())
    num_classes = len(classes)

    model, meta = create_vit_model_v2(num_classes=num_classes, pretrained=True)

    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]
    default_size = 224

    if isinstance(meta, dict):
        mean = meta.get("mean", default_mean)
        std = meta.get("std", default_std)
        image_size = meta.get("image_size", default_size)
    else:
        logger.warning(f"meta is not a dict ({type(meta)}). Using default ImageNet stats.")
        mean = default_mean
        std = default_std
        image_size = default_size

    if isinstance(image_size, (tuple, list)):
        image_size = image_size[0]

    logger.info(f"Using normalization mean={mean}, std={std}, image_size={image_size}")

    # DataLoaders con augmentation
    train_loader, val_loader, class_to_idx = build_dataloaders_with_augmentation(
        train_csv=train_csv,
        val_csv=val_csv,
        train_images_dir=train_images_dir,
        val_images_dir=val_images_dir,
        mean=mean,
        std=std,
        image_size=image_size,
        label_col=label_col,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0

    with mlflow.start_run(run_name="vit_articleType_aug"):
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
                "augmentation": "torchvision_flip_colorjitter_rotate",
                "early_stopping_patience": patience,
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

            val_loss, val_acc, val_f1_macro, val_f1_weighted = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

            logger.info(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
                f"val_f1_macro={val_f1_macro:.4f}, val_f1_weighted={val_f1_weighted:.4f}"
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1_macro": val_f1_macro,
                    "val_f1_weighted": val_f1_weighted,
                },
                step=epoch,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                logger.info(
                    f"New best val_loss={best_val_loss:.4f}, saving model to {model_output_path}"
                )
                state = {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "meta": meta,
                    "label_col": label_col,
                }
                torch.save(state, model_output_path)
            else:
                epochs_no_improve += 1
                logger.info(
                    f"No improvement in val_loss for {epochs_no_improve} epoch(s). "
                    f"Patience = {patience}."
                )

            # Condición de corte
            if epochs_no_improve >= patience:
                logger.info(
                    f"Early stopping triggered after {epochs_no_improve} epochs sin mejora "
                    f"en val_loss. Best val_loss={best_val_loss:.4f}."
                )
                break

        best_state = torch.load(model_output_path, map_location=device)
        model.load_state_dict(best_state["model_state_dict"])

        mlflow.pytorch.log_model(model, artifact_path="model")

        mapping_path = MODELS_DIR / "class_to_idx_aug.json"
        with mapping_path.open("w", encoding="utf-8") as f:
            json.dump(class_to_idx, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(str(mapping_path))

    logger.success("Training with augmentation complete.")


if __name__ == "__main__":
    app()
