# Product Tagger

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A Vision Transformer–based system for automatic product attribute tagging to improve eCommerce catalog quality and reduce manual labeling effort.

## Project Organization
```
├── Makefile                 <- Convenience commands like `make data`, `make train`, etc.
├── README.md                <- This file (project overview and instructions)
├── pyproject.toml           <- Project configuration and packaging metadata
├── requirements.txt         <- Pin/describe Python dependencies for reproducibility

├── data/
│   ├── raw/                 <- Original raw inputs (e.g. `raw/styles.csv`, raw images)
│   └── processed/           <- Cleaned and split datasets used by training/eval
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       ├── test_predictions.csv
│       ├── test_confusion_matrix.csv
│       └── images/          <- Prepared images organized for `train`/`val`/`test`

├── documentation/           <- LaTeX source used for documentation
│   └── Informe.pdf         <- Project documentation

├── mlruns/                  <- Local MLflow tracking store with experiment runs

├── models/                  <- Exported model checkpoints and serialized artifacts

├── notebooks/               <- Exploratory and analysis Jupyter notebooks

├── onnx/                    <- Scripts and helpers to export/compare ONNX models

├── product_tagger/          <- Main Python package containing project code
│   ├── __init__.py
│   ├── config.py            <- Configuration (paths, MLflow settings, constants)
│   ├── data_loader.py       <- Helpers to load data and images
│   ├── dataset.py           <- Dataset utilities and dataset classes
│   ├── eda.py               <- Exploratory data analysis utilities
│   ├── features.py          <- Feature engineering and transforms
│   ├── genviz.py            <- Visualization helpers (for reporting / EDA)
│   ├── pipeline.py          <- High-level data preparation pipeline
│   ├── plots.py             <- Plotting helpers for reports and notebooks
│   ├── prepare_images.py    <- Image preprocessing / augmentation helper scripts
│   ├── split_dataset.py     <- Train/val/test split helpers
│   ├── attention/           <- Attention visualization tools
│   └── modeling/            <- Training and inference code
│       ├── __init__.py
│       ├── train.py         <- Training CLI (Typer) for Vision Transformer
│       ├── predict.py       <- Inference CLI to run model predictions
│       └── vit.py           <- Model definitions (ViT and helpers)

├── references/              <- Data dictionaries, bibliographies, and references

├── reports/                 <- Generated analysis outputs (PDFs, HTML, figures)
│   └── figures/

└── product_tagger/onnx_adds  <- (See `product_tagger/onnx`) helper scripts for ONNX

Notes:
- MLflow runs are stored under `mlruns/` in this repo by default (see `product_tagger/config.py`).
- The `data/processed/images/` folder contains image datasets split into `train/`, `val/`, and `test/` subfolders used by the dataloaders.
```

--------

## How to run the project

### 1. Create and activate the environment

From the project root:

```bash
python -m venv .venv
# On Windows PowerShell
.venv\Scripts\Activate.ps1
```

Then install dependencies (only needed once, or when `requirements.txt` changes):

```bash
pip install -r requirements.txt
```

You can also use the Makefile helper:

```bash
make requirements
```

### 2. Run data preparation / pipeline

- **Just build the dataset**:

```bash
make data
```

- **Run the full data pipeline** (dataset + splits + prepared images, etc.):

```bash
make pipeline
```

### 3. Train the ViT model

Use the `train.py` Typer CLI via the Makefile:

```bash
make train
```

By default this trains a ViT-B/16 model and logs the run in MLflow. You can override parameters directly:

```bash
python -m product_tagger.modeling.train --epochs 15 --batch-size 64 --use-augmentation / --no-use-augmentation
```

The best checkpoint is saved to `models/vit_articleType.pt` together with `class_to_idx.json`.

### 4. Run predictions on the test split

After training a model:

```bash
make predict
```

This runs `python -m product_tagger.modeling.predict`, loading `models/vit_articleType.pt` and writing:

- `data/processed/test_predictions.csv`
- `data/processed/test_confusion_matrix.csv` (if test CSV has labels)

It also logs test metrics and artifacts to MLflow.

### 5. Export model to ONNX

After you have a successful training run in MLflow (visible in the UI), you can export the
corresponding PyTorch model to ONNX and attach it as an additional artifact to the same run.

From the project root, with your environment activated and replacing `<RUN_ID>` with a real
run id from MLflow:

```bash
python -m product_tagger.onnx.addonnx --run-id <RUN_ID>
```

Useful options:

- **`--model-artifact-path`**: artifact path inside the run where the PyTorch model is stored (default: `model`).
- **`--output-path`**: where to write the ONNX file (default: `product_tagger/onnx/model_export.onnx`).
- **`--batch`, `--channels`, `--height`, `--width`**: shape of the dummy input used for export.
- **`--opset-version`**: ONNX opset version (default: 17).

### 6. Check ONNX vs PyTorch consistency

To verify that the ONNX export produces numerically similar outputs to the original PyTorch model,
run the comparison CLI using the same `run_id` and the path to the exported ONNX file:

```bash
python -m product_tagger.onnx.onnx_vs_torch --run-id <RUN_ID>
```

Key options:

- **`--model-artifact-path`**: artifact path inside the run where the PyTorch model is stored (default: `model`).
- **`--onnx-path`**: path to the ONNX file (default: `product_tagger/onnx/model_export.onnx`).
- **`--batch`, `--channels`, `--height`, `--width`**: shape of the dummy input used for the check.
- **`--max-diff-ok`**: maximum allowed absolute difference between PyTorch and ONNX outputs
  (default: `1e-4`; process exits with code 1 if exceeded).

## MLflow tracking UI

The project is configured in `config.py` to use a local `mlruns` directory inside the repo as the MLflow tracking store
(`MLFLOW_TRACKING_URI` defaults to that path).

To start the MLflow UI from the project root, with your virtual environment activated:

```bash
mlflow ui --backend-store-uri mlruns --port 5000 --host 127.0.0.1
```

Then open `http://127.0.0.1:5000` in your browser to explore experiments, runs, metrics and artifacts
created by `product_tagger.modeling.train` and `product_tagger.modeling.predict`.

