import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import cv2

from product_tagger.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, TARGET_SIZE
from loguru import logger



# ===============================================================
# 1. Cargar CSV
# ===============================================================
def load_styles():
    csv_path = RAW_DATA_DIR / "styles.csv"
    logger.info(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    return df


# ===============================================================
# 2. Validar imágenes (existencia + apertura)
# ===============================================================
def validate_images(df, images_dir=None):
    if images_dir is None:
        images_dir = RAW_DATA_DIR / "images"

    logger.info("Validating image paths...")

    valid_rows = []
    missing, corrupted = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        pid = row["id"]
        img_path = images_dir / f"{pid}.jpg"

        if not img_path.exists():
            missing += 1
            continue

        try:
            Image.open(img_path)
            valid_rows.append(row)
        except UnidentifiedImageError:
            corrupted += 1

    logger.info(f"Missing images: {missing}")
    logger.info(f"Corrupted images: {corrupted}")
    logger.info(f"Valid images: {len(valid_rows)}")

    return pd.DataFrame(valid_rows)


# ===============================================================
# 3. Plot missing values
# ===============================================================
def plot_missing_values(df):
    missing = df.isna().mean().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing.values, y=missing.index)
    plt.title("Proporción de valores faltantes en styles.csv")
    plt.tight_layout()

    fig_path = FIGURES_DIR / "missing_values.png"
    plt.savefig(fig_path, dpi=300)
    logger.success(f"Saved figure: {fig_path}")
    plt.close()


# ===============================================================
# 4. Plot class distributions
# ===============================================================
def plot_class_distribution(df, column, max_classes=30):
    counts = df[column].value_counts().head(max_classes)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=counts.values, y=counts.index)
    plt.title(f"Distribución de clases para {column}")
    plt.tight_layout()

    fig_path = FIGURES_DIR / f"class_distribution_{column}.png"
    plt.savefig(fig_path, dpi=300)
    logger.success(f"Saved: {fig_path}")
    plt.close()


# ===============================================================
# 5. Random image samples
# ===============================================================
def show_random_images(df, images_dir=None, n=12):
    if images_dir is None:
        images_dir = RAW_DATA_DIR / "images"

    sample = df.sample(n)
    plt.figure(figsize=(15, 10))

    for i, (_, row) in enumerate(sample.iterrows()):
        pid = row["id"]
        img = Image.open(images_dir / f"{pid}.jpg")

        plt.subplot(3, 4, i+1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{row['gender']} - {row['masterCategory']}")

    plt.tight_layout()

    fig_path = FIGURES_DIR / "random_samples.png"
    plt.savefig(fig_path, dpi=300)
    logger.success(f"Saved: {fig_path}")
    plt.close()


# ===============================================================
# 6. Analyze image resolution variability
# ===============================================================
def analyze_image_stats(df, images_dir=None, sample_size=2000):
    if images_dir is None:
        images_dir = RAW_DATA_DIR / "images"

    df_sample = df.sample(min(len(df), sample_size))

    widths, heights = [], []

    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        pid = row["id"]
        img = Image.open(images_dir / f"{pid}.jpg")
        w, h = img.size
        widths.append(w)
        heights.append(h)

    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=widths, y=heights, alpha=0.3)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Variabilidad de tamaños de imagen")
    plt.tight_layout()

    fig_path = FIGURES_DIR / "image_resolution_scatter.png"
    plt.savefig(fig_path, dpi=300)
    logger.success(f"Saved: {fig_path}")
    plt.close()

    mean_w, mean_h = np.mean(widths), np.mean(heights)
    std_w, std_h = np.std(widths), np.std(heights)

    logger.info(f"Mean size: {mean_w:.1f} x {mean_h:.1f}")
    logger.info(f"Std size:  {std_w:.1f} x {std_h:.1f}")

    all_same = (std_w == 0 and std_h == 0)

    if all_same:

        if (mean_w, mean_h) == TARGET_SIZE:
            logger.info("Conclusión: Todas las imágenes tienen el tamaño requerido (224×224).")

        elif mean_w < TARGET_SIZE[0] or mean_h < TARGET_SIZE[1]:
            logger.info(
                f"Conclusión: Todas las imágenes miden {mean_w:.0f}×{mean_h:.0f}, "
                f"pero es menor al tamaño requerido → se requiere upscale."
            )

        elif mean_w > TARGET_SIZE[0] or mean_h > TARGET_SIZE[1]:
            logger.info(
                f"Conclusión: Todas las imágenes miden {mean_w:.0f}×{mean_h:.0f}, "
                f"pero exceden 224×224 → se requiere downscale."
            )

        else:
            logger.info("Conclusión: Tamaño uniforme → se normaliza a 224×224.")

    else:
        logger.info(
            f"Conclusión: Variabilidad de tamaño (std = {std_w:.1f}/{std_h:.1f}) → "
            "normalizar a 224×224."
        )


# ===============================================================
# 7. 🔥 NUEVO: ANALISIS DE CALIDAD DE IMÁGENES
# ===============================================================
def analyze_image_quality(df, images_dir=None, n_samples=500):
    if images_dir is None:
        images_dir = RAW_DATA_DIR / "images"

    logger.info("Analizando calidad de imágenes (brillo, contraste, nitidez, RGB)…")

    df_sample = df.sample(min(n_samples, len(df)))

    brightness, contrast, sharpness = [], [], []
    r_vals, g_vals, b_vals = [], [], []

    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        pid = row["id"]
        img_path = images_dir / f"{pid}.jpg"

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        brightness.append(np.mean(gray))
        contrast.append(np.std(gray))
        sharpness.append(cv2.Laplacian(gray, cv2.CV_64F).var())

        b_vals.append(np.mean(img[:,:,0]))
        g_vals.append(np.mean(img[:,:,1]))
        r_vals.append(np.mean(img[:,:,2]))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0,0].hist(brightness, bins=40, color="orange")
    axes[0,0].set_title("Distribución de brillo")

    axes[0,1].hist(contrast, bins=40, color="purple")
    axes[0,1].set_title("Distribución de contraste")

    axes[1,0].hist(sharpness, bins=40, color="green")
    axes[1,0].set_title("Nitidez (varianza Laplaciana)")

    axes[1,1].boxplot([r_vals, g_vals, b_vals], labels=["Red", "Green", "Blue"])
    axes[1,1].set_title("Medias por canal RGB")

    fig.tight_layout()
    fig_path = FIGURES_DIR / "image_quality.png"
    fig.savefig(fig_path, dpi=300)
    logger.success(f"Saved: {fig_path}")
    plt.close(fig)

    logger.info("--- Conclusiones de calidad ---")
    logger.info(f"Brillo medio: {np.mean(brightness):.1f}")
    logger.info(f"Contraste medio: {np.mean(contrast):.1f}")
    logger.info(f"Nitidez media: {np.mean(sharpness):.1f}")
    logger.info(f"RGB medios: R={np.mean(r_vals):.1f} G={np.mean(g_vals):.1f} B={np.mean(b_vals):.1f}")

    logger.info(
        "Conclusión: Imágenes con fondo blanco, brillo estable, "
        "contraste moderado y nitidez baja por compresión. "
        "→ Añadir augmentations suaves (ColorJitter, GaussianBlur p=0.2)."
    )


# ===============================================================
# 8. Save cleaned dataset
# ===============================================================
def save_clean_dataset(df):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = PROCESSED_DATA_DIR / "dataset_clean.csv"
    df.to_csv(output, index=False)
    logger.success(f"Clean dataset saved at: {output}")


# ===============================================================
# 9. Full EDA Pipeline
# ===============================================================
def run_eda():
    logger.info("=== Starting EDA for Product Tagger ===")

    df = load_styles()

    plot_missing_values(df)

    for col in ["gender", "masterCategory", "subCategory", "articleType"]:
        if col in df.columns:
            plot_class_distribution(df, col)

    df_valid = validate_images(df)

    show_random_images(df_valid)

    analyze_image_stats(df_valid)

    analyze_image_quality(df_valid)   # 🔥🔥🔥 AGREGADO

    save_clean_dataset(df_valid)

    logger.success("=== EDA Completed Successfully ===")


if __name__ == "__main__":
    run_eda()