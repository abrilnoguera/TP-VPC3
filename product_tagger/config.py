import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# ---------------------------------------------------------------------
# Rutas de datos y modelos
# ---------------------------------------------------------------------
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---------------------------------------------------------------------
# Hiperparámetros / constantes de arquitectura
# ---------------------------------------------------------------------
TARGET_SIZE = (224, 224)

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Validación extra: asegurar que suman 1
assert abs((TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT) - 1.0) < 1e-8, \
    "La suma de TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT debe ser 1.0"

# ---------------------------------------------------------------------
# MLflow configuration (Opción B)
# ---------------------------------------------------------------------
import mlflow

# Nombre de experimento: NUEVO para vos, para no reutilizar los meta.yaml viejos
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "vit_aug_v1",
)

# Tracking URI:
# 1) Si hay variable de entorno MLFLOW_TRACKING_URI, se respeta.
# 2) Si no, por defecto: carpeta "mlruns" dentro del proyecto.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    (PROJ_ROOT / "mlruns").as_uri(),  # p.ej. file:///C:/importante/Master/VC3/TP-VPC3/mlruns
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# Logs útiles para verificar en consola que no está usando la ruta de "pedrobarrera"
logger.info(f"[MLflow] tracking_uri = {mlflow.get_tracking_uri()}")
exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
if exp is not None:
    logger.info(
        f"[MLflow] experiment '{MLFLOW_EXPERIMENT_NAME}' -> "
        f"id={exp.experiment_id}, artifact_location={exp.artifact_location}"
    )
else:
    logger.warning(f"[MLflow] experiment '{MLFLOW_EXPERIMENT_NAME}' no encontrado")

# ---------------------------------------------------------------------
# Integración loguru + tqdm (si está instalado)
# ---------------------------------------------------------------------
try:
    from tqdm import tqdm

    try:
        logger.remove(0)
    except ValueError:
        # Si ya no existe el handler base, ignorar el error
        pass

    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
