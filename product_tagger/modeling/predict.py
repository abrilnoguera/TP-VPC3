from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import typer
import mlflow
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

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
    Ejecuta el pipeline de inferencia sobre el split de test.

    - Carga el checkpoint entrenado de ViT-B/16 (pesos, `class_to_idx`, meta y `label_col`).
    - Reconstruye el modelo con el mismo número de clases que en entrenamiento.
    - Prepara un `ImageDataset` de test con las mismas transformaciones de
      resize y normalización usadas en entrenamiento.
    - Genera predicciones para cada imagen y guarda un CSV con
      `id`, índice de clase predicho y etiqueta textual.
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
    label_col = checkpoint.get("label_col", "articleType")

    # Valores por defecto de ImageNet (mismos que en create_vit_model)
    default_meta = {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "image_size": 224,
    }
    meta = checkpoint.get("meta", default_meta)
    num_classes = len(class_to_idx)

    # Cargamos el mismo modelo que en entrenamiento (ViT-B/16)
    model, _ = create_vit_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    mean = meta["mean"]
    std = meta["std"]
    image_size = meta.get("image_size", 224)
    if isinstance(image_size, (tuple, list)):
        image_size = image_size[0]

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # Dataset de test (las etiquetas del CSV se usan para métricas si están disponibles)
    dataset = ImageDataset(
        csv_path=test_csv,
        images_dir=test_images_dir,
        label_col=label_col,
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
    all_true_labels = []
    all_true_idx = []

    has_ground_truth = label_col in dataset.data.columns

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Predict", total=len(loader)):
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

            if has_ground_truth:
                # `labels` son las etiquetas de texto originales
                true_labels_batch = [str(label) for label in labels]
                true_idx_batch = [class_to_idx[lab] for lab in true_labels_batch]
                all_true_labels.extend(true_labels_batch)
                all_true_idx.extend(true_idx_batch)

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

    # Métricas si tenemos ground-truth en el CSV de test
    if has_ground_truth and len(all_true_idx) == len(all_preds_idx) and len(all_true_idx) > 0:
        y_true = all_true_idx
        y_pred = all_preds_idx

        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")

        prec_macro, rec_macro, _, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        prec_weighted, rec_weighted, _, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        logger.info(
            "Test metrics -> "
            f"accuracy={acc:.4f}, "
            f"f1_macro={f1_macro:.4f}, f1_weighted={f1_weighted:.4f}, "
            f"precision_macro={prec_macro:.4f}, recall_macro={rec_macro:.4f}, "
            f"precision_weighted={prec_weighted:.4f}, recall_weighted={rec_weighted:.4f}"
        )

        # Reporte por clase (solo para las clases que aparecen en y_true/y_pred)
        labels = sorted(set(y_true) | set(y_pred))
        target_names = [idx_to_class[i] for i in labels]
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0,
        )
        logger.info("Classification report (per class):\n" + report)

        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{idx_to_class[i]}" for i in range(num_classes)],
            columns=[f"pred_{idx_to_class[i]}" for i in range(num_classes)],
        )
        cm_path = predictions_path.with_name("test_confusion_matrix.csv")
        cm_df.to_csv(cm_path, index=True)
        logger.info(f"Saved confusion matrix to {cm_path}")

        # Log de métricas y artefactos en MLflow
        with mlflow.start_run(run_name="predict_test", nested=True):
            mlflow.log_params(
                {
                    "model_path": str(model_path),
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "label_col": label_col,
                    "image_size": image_size,
                }
            )
            mlflow.log_metrics(
                {
                    "test_accuracy": acc,
                    "test_f1_macro": f1_macro,
                    "test_f1_weighted": f1_weighted,
                    "test_precision_macro": prec_macro,
                    "test_recall_macro": rec_macro,
                    "test_precision_weighted": prec_weighted,
                    "test_recall_weighted": rec_weighted,
                }
            )
            mlflow.log_artifact(str(predictions_path))
            mlflow.log_artifact(str(cm_path))


if __name__ == "__main__":
    app()
