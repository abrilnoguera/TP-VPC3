import typer
from loguru import logger

from product_tagger.dataset import run_dataset
from product_tagger.eda import run_eda
from product_tagger.split_dataset import run_split
from product_tagger.prepare_images import run_prepare_images

app = typer.Typer()

@app.command()
def main():
    logger.info("=== PRODUCT TAGGER: DATASET PIPELINE ===")

    run_dataset()
    run_eda()
    run_split()
    run_prepare_images()

    logger.success("Dataset pipeline completed successfully!")

if __name__ == "__main__":
    app()