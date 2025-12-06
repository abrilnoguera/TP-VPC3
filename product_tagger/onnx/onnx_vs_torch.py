from pathlib import Path
from typing import Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import onnx
import onnxruntime as ort
import torch
import typer
from loguru import logger

from product_tagger.config import MLFLOW_TRACKING_URI


app = typer.Typer(help="Compare outputs of a PyTorch model from MLflow vs its ONNX export.")


def compare_torch_vs_onnx(
    run_id: str,
    model_artifact_path: str,
    onnx_path: Path,
    dummy_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
) -> float:
    """
    Carga un modelo PyTorch desde MLflow y un modelo ONNX desde disco,
    ejecuta un input dummy y devuelve la diferencia absoluta máxima
    entre ambas salidas.

    Args:
        run_id: ID del run de MLflow donde está el modelo PyTorch.
        model_artifact_path: Ruta del artefacto del modelo dentro del run (p.ej. "model").
        onnx_path: Ruta del fichero ONNX exportado.
        dummy_shape: Forma del input dummy (batch, C, H, W).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"runs:/{run_id}/{model_artifact_path}"
    logger.info(f"Loading PyTorch model from MLflow URI: {model_uri}")
    model = mlflow.pytorch.load_model(model_uri, map_location="cpu").eval()

    logger.info(f"Loading ONNX model from: {onnx_path}")
    onnx_model = onnx.load(onnx_path.as_posix())
    onnx.checker.check_model(onnx_model)

    ort_sess = ort.InferenceSession(onnx_path.as_posix())

    dummy = torch.randn(*dummy_shape, dtype=torch.float32)

    with torch.no_grad():
        torch_out = model(dummy).numpy()

    onnx_out = ort_sess.run(
        None,
        {"input": dummy.numpy()},
    )[0]

    diff = float(np.abs(torch_out - onnx_out).max())
    logger.info(f"Max abs diff between PyTorch and ONNX: {diff}")
    return diff


@app.command()
def main(
    run_id: str = typer.Option(..., help="MLflow run_id that contains the trained model."),
    model_artifact_path: str = typer.Option(
        "model",
        help="Artifact path inside the run where the PyTorch model is stored.",
    ),
    onnx_path: Path = typer.Option(
        Path("product_tagger") / "onnx" / "model_export.onnx",
        help="Path to the exported ONNX file.",
    ),
    batch: int = typer.Option(1, help="Batch size for the dummy input."),
    channels: int = typer.Option(3, help="Number of channels for the dummy input."),
    height: int = typer.Option(224, help="Height of the dummy input image."),
    width: int = typer.Option(224, help="Width of the dummy input image."),
    max_diff_ok: float = typer.Option(
        1e-4,
        help="Threshold for maximum allowed absolute difference. Exits with code 1 if exceeded.",
    ),
) -> None:
    """
    CLI para comparar la salida de un modelo PyTorch en MLflow con su versión ONNX.
    """
    dummy_shape = (batch, channels, height, width)
    diff = compare_torch_vs_onnx(
        run_id=run_id,
        model_artifact_path=model_artifact_path,
        onnx_path=onnx_path,
        dummy_shape=dummy_shape,
    )

    if diff > max_diff_ok:
        logger.error(
            f"Max abs diff {diff} is greater than allowed threshold {max_diff_ok}."
        )
        raise typer.Exit(code=1)

    logger.success(
        f"ONNX vs Torch check passed. Max abs diff {diff} <= threshold {max_diff_ok}."
    )


if __name__ == "__main__":
    app()