# ML Car Price Prediction

Does a manual ML workflow outperform an automated one? This project answers that question by predicting car prices three ways and comparing the results.

## Tools & Skills Used

![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat&logo=matplotlib&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![PyCaret](https://img.shields.io/badge/PyCaret-1B9E77?style=flat&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat&logoColor=white)
![uv](https://img.shields.io/badge/uv-DE5FE9?style=flat&logo=uv&logoColor=white)

## Project Overview

This project builds an end-to-end machine learning regression pipeline to predict car prices from vehicle specifications. The same dataset is processed through three distinct phases to isolate the impact of manual feature engineering versus automated model selection.

Part of the Masterschool AI Enhanced Productivity project series, which explores how automation tools can enhance analytical productivity across different ML problem types.

## Phases

| Phase | Approach | Description |
|-------|----------|-------------|
| 1 | Manual | User handles preprocessing and trains 3 models (LR, RF, GB) across 7 notebooks |
| 2 | Automated | PyCaret handles preprocessing and trains 18 models on the raw featured dataset |
| 3 | Hybrid | User handles preprocessing, PyCaret trains 18 models on the cleaned data |

## Phase 1 Pipeline

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

| Phase | Approach | Best Model | MAE | RMSE | R2 |
|-------|----------|-----------|-----|------|-----|
| 1 | Manual | Random Forest | $1,533 | $2,195 | 0.939 |
| 2 | Automated | LightGBM | $2,177 | $3,558 | 0.840 |
| **3** | **Hybrid** | **Extra Trees** | **$1,447** | **$2,079** | **0.945** |

### Conclusion

The hybrid approach (Phase 3) produced the best results across all metrics. Manual preprocessing gave PyCaret better inputs to work with, and PyCaret's broader model search found Extra Trees Regressor, a model not tested in the manual phase.

PyCaret on raw data (Phase 2) performed worst, confirming that automated ML benefits significantly from thoughtful data preparation. The fully manual pipeline (Phase 1) landed in the middle with strong preprocessing but only three models to choose from.

The takeaway: automation and domain expertise are most powerful in combination.

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

Run the Phase 1 manual pipeline:

```bash
uv run python run-pipeline.py
```

Run PyCaret Phase 2 - raw data:

```bash
uv run pycaret/run-pycaret-raw.py
```

Run PyCaret Phase 3 - preprocessed data:

```bash
uv run pycaret/run-pycaret-preprocessed.py
```

> **Note:** PyCaret scripts use inline metadata ([PEP 723](https://peps.python.org/pep-0723/))
> to specify their own dependencies and Python version. No `python` prefix needed -
> `uv` reads this directly from the script and creates an isolated environment
> automatically. Output logs are saved in [`pycaret/`](pycaret/).
