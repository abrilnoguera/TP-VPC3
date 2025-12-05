import torch
import numpy as np
import onnx
import onnxruntime as ort

import mlflow
import mlflow.pytorch
from product_tagger.config import MLFLOW_TRACKING_URI

# --- CONFIG ---
RUN_ID = "ad484a8bc4194f869d930c7f1706a398"
ARTIFACT_PATH = "model"  # el mismo que usaste en addonnx.py
ONNX_PATH = "product_tagger\onnx\model_export.onnx"  # o la ruta donde quedó el ONNX

# 1) Cargar modelo PyTorch desde MLflow (en CPU)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model_uri = f"runs:/{RUN_ID}/{ARTIFACT_PATH}"
model = mlflow.pytorch.load_model(model_uri, map_location="cpu").eval()

# 2) Cargar modelo ONNX y crear sesión
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)

ort_sess = ort.InferenceSession(ONNX_PATH)

# 3) Crear un input de prueba
dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)  # ajusta si tu input es distinto

with torch.no_grad():
    torch_out = model(dummy).numpy()

onnx_out = ort_sess.run(
    None,
    {"input": dummy.numpy()},
)[0]

# 4) Comparar
diff = np.abs(torch_out - onnx_out).max()
print("Max abs diff:", diff)