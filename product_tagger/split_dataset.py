import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple
import shutil
from tqdm import tqdm

from product_tagger.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
from loguru import logger

def collapse_rare_classes(df: pd.DataFrame, label_col: str, min_samples: int = 5) -> pd.DataFrame:
    """
    Collapses classes with fewer than min_samples examples.
    Ensures no stratification failure.
    """
    counts = df[label_col].value_counts()
    rare_classes = counts[counts < min_samples].index

    if len(rare_classes) > 0:
        logger.warning(f"Collapsing {len(rare_classes)} rare classes (< {min_samples}) into 'other'")
        df[label_col] = df[label_col].apply(
            lambda x: "other" if x in rare_classes else x
        )

    return df

def stratified_split(df: pd.DataFrame, label_col: str = "articleType") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stable stratified split: train stratified, val/test random.
    """

    logger.info(f"Original dataset size: {len(df)}")

    # Step 1: collapse ultra-rare classes
    df = collapse_rare_classes(df, label_col, min_samples=5)

    # Step 2: add stratification label
    df["strat_label"] = df[label_col]

    # proportions
    train_size = TRAIN_SPLIT
    temp_size = VAL_SPLIT + TEST_SPLIT
    test_ratio_in_temp = TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT)

    # Step 3 — Train split (STRATIFIED)
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        stratify=df["strat_label"],
        random_state=42
    )

    # Step 4 — Val/Test split (NOT stratified → avoids errors)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio_in_temp,
        shuffle=True,
        random_state=42
    )

    logger.success(f"Train size: {len(train_df)}")
    logger.success(f"Val size:   {len(val_df)}")
    logger.success(f"Test size:  {len(test_df)}")

    return train_df, val_df, test_df

def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

    logger.success("Saved train/val/test CSVs.")

def copy_split_images(df: pd.DataFrame, split_name: str) -> None:
    """
    Copies images referenced in df to:
        data/processed/images/<split_name>/
    """
    src_dir = RAW_DATA_DIR / "images"
    dst_dir = PROCESSED_DATA_DIR / "images" / split_name
    dst_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Copying {len(df)} images to {dst_dir}")

    for pid in tqdm(df["id"].astype(str).tolist()):
        src = src_dir / f"{pid}.jpg"
        dst = dst_dir / f"{pid}.jpg"

        if src.exists():
            shutil.copy(src, dst)
        else:
            logger.warning(f"Image not found: {src}")

def save_all(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    save_splits(train_df, val_df, test_df)

    copy_split_images(train_df, "train")
    copy_split_images(val_df, "val")
    copy_split_images(test_df, "test")

    logger.success("All splits and images successfully saved.")

def run_split() -> None:
    input_path = PROCESSED_DATA_DIR / "dataset_clean.csv"
    logger.info(f"Loading cleaned dataset: {input_path}")

    df = pd.read_csv(input_path)

    train_df, val_df, test_df = stratified_split(df, label_col="articleType")

    save_all(train_df, val_df, test_df)

if __name__ == "__main__":
    run_split()