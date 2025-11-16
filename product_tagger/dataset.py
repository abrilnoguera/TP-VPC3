import shutil
from pathlib import Path

import kagglehub
from loguru import logger
from tqdm import tqdm
import typer

from product_tagger.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

def download_dataset_if_needed() -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Ignore .gitkeep
    contents = [p for p in RAW_DATA_DIR.iterdir() if p.name != ".gitkeep"]

    if len(contents) > 0:
        logger.info("RAW_DATA_DIR already contains dataset files. Skipping download.")
        return RAW_DATA_DIR

    logger.info("RAW_DATA_DIR is empty. Downloading dataset from KaggleHub...")

    downloaded_path = Path(
        kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")
    )

    logger.info(f"Dataset downloaded to temporary path: {downloaded_path}")

    # ----------------------------------------------------------
    # 1) Locate the real dataset folder: 'myntradataset'
    # ----------------------------------------------------------
    myntra_folder = None
    for item in downloaded_path.iterdir():
        if item.is_dir() and item.name.lower() == "myntradataset":
            myntra_folder = item
            break

    # If not found, fallback to root
    inner_path = myntra_folder if myntra_folder else downloaded_path

    logger.info(f"Using dataset root (inner_path): {inner_path}")

    # ----------------------------------------------------------
    # 2) Copy ONLY what we need: images/ + styles.csv
    # ----------------------------------------------------------

    for item in inner_path.iterdir():
        # Skip hidden items
        if item.name.startswith("."):
            continue  

        if item.is_dir() and item.name.lower() == "images":
            shutil.copytree(item, RAW_DATA_DIR / "images", dirs_exist_ok=True)

        elif item.is_file() and item.name.lower() == "styles.csv":
            shutil.copy2(item, RAW_DATA_DIR / "styles.csv")

    logger.success("Dataset extracted correctly into RAW_DATA_DIR.")
    return RAW_DATA_DIR

@app.command()
def main(
    input_csv: Path = RAW_DATA_DIR / "styles.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    # Ensure dataset exists (images + CSV)
    dataset_path = download_dataset_if_needed()
    logger.info(f"Dataset available at: {dataset_path}")

    # --------------------------------------------------------------------------
    # TODO: Replace with your real preprocessing logic
    # --------------------------------------------------------------------------
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # --------------------------------------------------------------------------


if __name__ == "__main__":
    app()