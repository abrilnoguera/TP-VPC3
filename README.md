# Product Tagger

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A Vision Transformer–based system for automatic product attribute tagging to improve eCommerce catalog quality and reduce manual labeling effort.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         product_tagger and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment
│
├── setup.cfg          <- Configuration file for flake8
│
└── product_tagger   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes product_tagger a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Script to train models (with optional augmentation)
    │
    └── plots.py                <- Code to create visualizations
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

## MLflow tracking UI

The project is configured in `config.py` to use a local `mlruns` directory inside the repo as the MLflow tracking store
(`MLFLOW_TRACKING_URI` defaults to that path).

To start the MLflow UI from the project root, with your virtual environment activated:

```bash
mlflow ui --backend-store-uri mlruns --port 5000 --host 127.0.0.1
```

Then open `http://127.0.0.1:5000` in your browser to explore experiments, runs, metrics and artifacts
created by `product_tagger.modeling.train` and `product_tagger.modeling.predict`.

