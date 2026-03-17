# ML Car Price Prediction

Prediction of car prices using manual ML models and PyCaret, with SHAP/LIME interpretability.

## Quick Access

- [Notebooks](notebooks/)
- [Data](data/)
- [Visualizations](visualizations/)

## Pipeline

| Notebook | Description |
|----------|-------------|
| [01-data-gathering](notebooks/01-data-gathering.ipynb) | Load raw dataset, initial inspection |
| [02-data-cleaning](notebooks/02-data-cleaning.ipynb) | Fix types, typos, extract brand, drop columns |
| [03-eda](notebooks/03-eda.ipynb) | Distributions, correlations, visualizations |
| [04-feature-engineering](notebooks/04-feature-engineering.ipynb) | Create avg_mpg, hp_per_weight, brand_tier |
| [05-preprocessing](notebooks/05-preprocessing.ipynb) | Encode, scale, train/test split |
| [06-train-models](notebooks/06-train-models.ipynb) | Linear Regression, Random Forest, Gradient Boosting |
| [07-evaluation](notebooks/07-evaluation.ipynb) | Test set metrics, residual analysis, feature importance |

## Results

| Model | MAE | RMSE | R2 |
|-------|-----|------|-----|
| Linear Regression | $2,197 | $3,335 | 0.859 |
| Random Forest | $1,533 | $2,195 | 0.939 |
| Gradient Boosting | $1,676 | $2,466 | 0.923 |

Best model: **Random Forest** (R2 = 0.939, MAE = $1,533)

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

### Clone

```bash
git clone https://github.com/krauseannelize/ml-car-price-prediction.git
cd ml-car-price-prediction
```

### Install

```bash
uv sync
```

### Run

Open notebooks interactively:

```bash
uv run jupyter lab
```

Run the full pipeline:

```bash
uv run python run_pipeline.py
```
