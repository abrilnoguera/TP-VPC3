from pathlib import Path
from typing import Tuple

import mlflow
import mlflow.pytorch
import torch
import typer
from loguru import logger
from mlflow.tracking import MlflowClient

from product_tagger.config import MLFLOW_TRACKING_URI


app = typer.Typer(help="Export a trained MLflow PyTorch model to ONNX and attach it to the run.")


def export_run_to_onnx(
    run_id: str,
    model_artifact_path: str,
    onnx_path: Path,
    dummy_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    opset_version: int = 17,
) -> Path:
    """
    Exporta un modelo PyTorch guardado en MLflow a formato ONNX y lo adjunta
    como artefacto al mismo run.

    Args:
        run_id: ID del run en MLflow donde está guardado el modelo.
        model_artifact_path: Ruta del artefacto del modelo dentro del run (p.ej. "model").
        onnx_path: Ruta de salida del fichero ONNX.
        dummy_shape: Forma del input dummy (batch, C, H, W).
        opset_version: Versión de ONNX opset a usar.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"runs:/{run_id}/{model_artifact_path}"
    logger.info(f"Loading PyTorch model from MLflow URI: {model_uri}")

    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    model.to("cpu")

    logger.info(f"Exporting ONNX model to {onnx_path}")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(*dummy_shape, device="cpu")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path.as_posix(),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=opset_version,
    )

    client = MlflowClient()
    logger.info("Logging ONNX artifact back to MLflow run.")
    client.log_artifact(
        run_id=run_id,
        local_path=onnx_path.as_posix(),
        artifact_path="onnx_model",
    )

    logger.success(f"✅ ONNX attached to run: {run_id} at {onnx_path}")
    return onnx_path


@app.command()
def main(
    run_id: str = typer.Option(..., help="MLflow run_id that contains the trained model."),
    model_artifact_path: str = typer.Option(
        "model",
        help="Artifact path inside the run where the PyTorch model is stored.",
    ),
    output_path: Path = typer.Option(
        Path("product_tagger") / "onnx" / "model_export.onnx",
        help="Output path for the exported ONNX file.",
    ),
    batch: int = typer.Option(1, help="Batch size for the dummy input."),
    channels: int = typer.Option(3, help="Number of channels for the dummy input."),
    height: int = typer.Option(224, help="Height of the dummy input image."),
    width: int = typer.Option(224, help="Width of the dummy input image."),
    opset_version: int = typer.Option(17, help="ONNX opset version to use."),
) -> None:
    """
    CLI para exportar un modelo PyTorch (guardado en MLflow) a ONNX y adjuntarlo al mismo run.
    """
    dummy_shape = (batch, channels, height, width)
    export_run_to_onnx(
        run_id=run_id,
        model_artifact_path=model_artifact_path,
        onnx_path=output_path,
        dummy_shape=dummy_shape,
        opset_version=opset_version,
    )


if __name__ == "__main__":
    app()
