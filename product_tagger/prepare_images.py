import shutil
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm
from loguru import logger

from product_tagger.config import RAW_DATA_DIR, PROCESSED_DATA_DIR,  TARGET_SIZE


def load_split(csv_name: str) -> pd.DataFrame:
    """Carga uno de los splits procesados (train/val/test)."""
    path = PROCESSED_DATA_DIR / csv_name
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    return pd.read_csv(path)


def ensure_dirs():
    """Crea la estructura de carpetas necesarias."""
    PROCESSED_IMAGES_DIR = PROCESSED_DATA_DIR / "images"
    PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (PROCESSED_IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)


def process_and_copy_image(src_path: Path, dst_path: Path):
    """
    Abre la imagen original, la convierte a RGB,
    la redimensiona a TARGET_SIZE y la guarda.
    """

    try:
        img = Image.open(src_path).convert("RGB")
    except Exception:
        logger.warning(f"Imagen corrupta o inaccesible: {src_path}")
        return False

    # Resize (siempre necesario para ViT)
    img = img.resize(TARGET_SIZE, Image.BICUBIC)

    try:
        img.save(dst_path, format="JPEG", quality=95)
        return True
    except Exception:
        logger.error(f"Error guardando imagen en {dst_path}")
        return False


def prepare_split(df: pd.DataFrame, split_name: str):
    """
    Procesa todas las imágenes de un split y las deja listas para PyTorch.
    - df: DataFrame con IDs del split
    - split_name: "train", "val" o "test"
    """

    logger.info(f"Procesando imágenes para split: {split_name}")

    raw_img_dir = RAW_DATA_DIR / "images"
    out_dir = PROCESSED_DATA_DIR / "images" / split_name

    processed_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row["id"]
        src_path = raw_img_dir / f"{img_id}.jpg"
        dst_path = out_dir / f"{img_id}.jpg"

        ok = process_and_copy_image(src_path, dst_path)
        if ok:
            processed_count += 1

    logger.success(
        f"Split '{split_name}' listo: {processed_count}/{len(df)} imágenes procesadas."
    )


def run_prepare_images():
    logger.info("=== Preparando imágenes procesadas ===")

    ensure_dirs()

    # -------------------------------------------------------
    # LOAD SPLITS
    # -------------------------------------------------------
    train_df = load_split("train.csv")
    val_df = load_split("val.csv")
    test_df = load_split("test.csv")

    # -------------------------------------------------------
    # PROCESS IMAGES
    # -------------------------------------------------------
    prepare_split(train_df, "train")
    prepare_split(val_df, "val")
    prepare_split(test_df, "test")

    logger.success("=== Todas las imágenes procesadas correctamente ===")


if __name__ == "__main__":
    run_prepare_images()