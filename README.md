# ML Car Price Prediction

Prediction of car prices using manual ML models and PyCaret, with SHAP/LIME interpretability.

## Tools & Skills Used

![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat&logo=matplotlib&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![PyCaret](https://img.shields.io/badge/PyCaret-1B9E77?style=flat&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat&logoColor=white)
![uv](https://img.shields.io/badge/uv-DE5FE9?style=flat&logo=uv&logoColor=white)

## Quick Access

- [Notebooks](notebooks/)
- [Data](data/)
- [Visualizations](visualizations/)

## Project Overview

This project builds an end-to-end machine learning regression pipeline to predict car prices from vehicle specifications. Three models are trained and evaluated manually (Linear Regression, Random Forest, Gradient Boosting), then compared against PyCaret's automated model selection to assess the value of hands-on feature engineering versus automated ML workflows. SHAP and LIME interpretability analysis is planned as a next step.

Part of the Masterschool AI Enhanced Productivity project series, which explores how automation tools can enhance analytical productivity across different ML problem types.

## Objectives

- Build a complete ML pipeline: data gathering, cleaning, EDA, feature engineering, preprocessing, training, and evaluation
- Train and compare multiple regression models on a real-world car pricing dataset
- Compare manual model performance against PyCaret's automated approach
- Evaluate the trade-offs between hands-on ML workflows and automation tools

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

### Manual Models

| Model | MAE | RMSE | R2 |
|-------|-----|------|-----|
| Linear Regression | $2,197 | $3,335 | 0.859 |
| Random Forest | $1,533 | $2,195 | 0.939 |
| Gradient Boosting | $1,676 | $2,466 | 0.923 |

### PyCaret Comparison

| Model | MAE | RMSE | R2 |
|-------|-----|------|-----|
| LightGBM (PyCaret best) | $2,177 | $3,558 | 0.840 |

Best model: **Random Forest** (R2 = 0.939, MAE = $1,533) — the manual pipeline outperformed PyCaret's automated selection, demonstrating the value of domain-specific feature engineering.

## Setup & Installation

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### 1. Clone the Repository

```bash
git clone https://github.com/krauseannelize/ml-car-price-prediction.git
cd ml-car-price-prediction
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Run the Project

Open notebooks interactively:

```bash
uv run jupyter lab
```

Run the full pipeline:

```bash
uv run python run-pipeline.py
```

Run the PyCaret comparison (output saved to file):

```bash
uv run run-pycaret-comparison.py > pycaret-output.txt 2>&1
```

> **Note:** No `python` prefix needed — the script contains inline metadata
> ([PEP 723](https://peps.python.org/pep-0723/)) that specifies its own
> dependencies and Python version. `uv` reads this directly from the script
> and creates an isolated environment automatically.
