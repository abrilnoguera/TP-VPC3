import torch
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from pathlib import Path

from product_tagger.config import MLFLOW_TRACKING_URI  # already defined in your config

# --- Configurar MLflow igual que siempre ---
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Poner acá el run_id del entrenamiento que ya tienes
RUN_ID = "ad484a8bc4194f869d930c7f1706a398"

# 1) Cargar el modelo desde MLflow
model_uri = f"runs:/{RUN_ID}/model"  # o el artifact_path que usaste
model = mlflow.pytorch.load_model(model_uri)
model.eval()
model.to("cpu")



dummy_input = torch.randn(1, 3, 224, 224, device="cpu")  # adapta al input real de tu modelo

onnx_path = Path("model_export.onnx")

torch.onnx.export(
    model,
    dummy_input,
    onnx_path.as_posix(),
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    opset_version=17,
)

# 3) Loguear el ONNX como artifact en *ese mismo run*
client = MlflowClient()
client.log_artifact(
    run_id=RUN_ID,
    local_path=onnx_path.as_posix(),
    artifact_path="onnx_model",  # carpeta dentro de artifacts
)

print("✅ ONNX attached to existing run:", RUN_ID)
