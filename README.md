# ML Car Price Prediction

Does a manual ML workflow outperform an automated one? This project answers that question by predicting car prices three ways and comparing the results.

📺 [Watch the project walkthrough on YouTube](https://youtu.be/pPSpYYsdPdM)

## Tools & Skills Used

![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat&logo=matplotlib&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![PyCaret](https://img.shields.io/badge/PyCaret-1B9E77?style=flat&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-4B0082?style=flat&logoColor=white)
![LIME](https://img.shields.io/badge/LIME-32CD32?style=flat&logoColor=white)
![uv](https://img.shields.io/badge/uv-DE5FE9?style=flat&logo=uv&logoColor=white)

## Quick Access

- [Phases overview](#phases) | Summary table of all project phases
- [Notebooks](notebooks/) | Manual pipeline (01-07), white-box interpretation (08), black-box explainability (09)
- [PyCaret scripts](pycaret/) | Phase 2 and 3 automation scripts, logs, and results
- [Results](#phase-4--comparison) | Three-phase metric comparison
- [Presentation video](https://youtu.be/pPSpYYsdPdM) | YouTube walkthrough
- [Setup & Installation](#setup--installation) | Clone, install, and run

## Project Overview

This project builds an end-to-end machine learning regression pipeline to predict car prices from vehicle specifications. The same dataset is processed through three distinct phases to isolate the impact of manual feature engineering versus automated model selection, then interpreted through both white-box (Linear Regression coefficients, Decision Tree splits) and black-box (SHAP, LIME) methods.

Part of the Masterschool AI Enhanced Productivity project series, which explores how automation tools can enhance analytical productivity across different ML problem types.

## Dataset

The project uses the [Car Price dataset on Kaggle](https://www.kaggle.com/datasets/imgowthamg/car-price), containing 205 cars with 24 features (engine size, weight, fuel type, body style, etc.) and `price` as the regression target.

## Phases

| Phase | Approach | Description |
|-------|----------|-------------|
| 1 | [Manual](#phase-1--manual) | User handles preprocessing and trains 3 models (LR, RF, GB) across 7 notebooks |
| 2 | [Automated](#phase-2--automated) | PyCaret handles preprocessing and trains around 18 models on the raw featured dataset |
| 3 | [Hybrid](#phase-3--hybrid) | User handles preprocessing, PyCaret trains around 18 models on the cleaned data |
| 4 | [Comparison](#phase-4--comparison) | Three-phase comparison of metrics and best models |
| 5 | [White-box Interpretation](#phase-5--white-box-interpretation) | Linear Regression coefficients and Decision Tree splits in raw units |
| 6 | [Black-box Explainability](#phase-6--black-box-explainability) | SHAP and LIME analysis of the best-performing ensemble model |
| 7 | [Presentation](#phase-7--presentation) | Project presentation of workflow, findings, and lessons learned |

### Phase 1 — Manual

| Notebook | Description |
|----------|-------------|
| [01-data-gathering](notebooks/01-data-gathering.ipynb) | Load raw dataset, initial inspection |
| [02-data-cleaning](notebooks/02-data-cleaning.ipynb) | Fix types, typos, extract brand, drop columns |
| [03-eda](notebooks/03-eda.ipynb) | Distributions, correlations, visualizations |
| [04-feature-engineering](notebooks/04-feature-engineering.ipynb) | Create avg_mpg, hp_per_weight, brand_tier |
| [05-preprocessing](notebooks/05-preprocessing.ipynb) | Encode, scale, train/test split |
| [06-train-models](notebooks/06-train-models.ipynb) | Linear Regression, Random Forest, Gradient Boosting |
| [07-evaluation](notebooks/07-evaluation.ipynb) | Test set metrics, residual analysis, feature importance |

### Phase 2 — Automated

| Script | Description |
|--------|-------------|
| [run-pycaret-raw.py](pycaret/run-pycaret-raw.py) | PyCaret setup, model comparison, and evaluation on raw featured data |

PyCaret handles all preprocessing (encoding, scaling, imputation) and compares around 18 regression models. The script evaluates all models on the held-out test set in addition to PyCaret's default cross-validation ranking, so they can be compared apples-to-apples with the manual pipeline. Output logs and results are saved in [`pycaret/`](pycaret/).

### Phase 3 — Hybrid

| Script | Description |
|--------|-------------|
| [run-pycaret-preprocessed.py](pycaret/run-pycaret-preprocessed.py) | PyCaret model comparison on manually preprocessed data |

Uses the cleaned and feature-engineered dataset from Phase 1 (notebooks 01-05), then lets PyCaret compare around 18 models with `preprocess=False`. As in Phase 2, all models are evaluated on the same held-out test set used by the manual pipeline. This isolates the benefit of manual data preparation while leveraging PyCaret's broader model search.

### Phase 4 — Comparison

| Phase | Approach | Best Model (test set) | MAE | RMSE | R2 |
|-------|----------|-----------------------|-----|------|-----|
| 1 | Manual | Random Forest | $1,557 | $2,208 | 0.938 |
| 2 | Automated | Random Forest | $1,522 | $2,128 | 0.943 |
| **3** | **Hybrid** | **Random Forest** | **$1,280** | **$1,820** | **0.958** |

#### Why test-set ranking instead of PyCaret's cross-validation pick

PyCaret ranks models by default using cross-validation on the training data. Its CV-best picks were Light Gradient Boosting Machine in Phase 2 and Extra Trees in Phase 3. To compare consistently against the manual pipeline (which is always evaluated on a held-out test set in `07-evaluation.ipynb`), all PyCaret models were re-ranked on the same held-out test set. Random Forest came out on top in every phase.

This reframing is also why PyCaret's "pick" differs from the actual test-set winner — different evaluation criteria yield different rankings.

#### Findings

The hybrid approach (Phase 3) produced the best results across all metrics. Manual preprocessing gave PyCaret better inputs to work with, and PyCaret's broader model search confirmed Random Forest as the strongest fit on this dataset. PyCaret on raw data (Phase 2) and the fully manual pipeline (Phase 1) landed close to each other, both clearly behind the hybrid approach. The takeaway: automation and domain expertise are most powerful in combination.

### Phase 5 — White-box Interpretation

| Notebook | Description |
|----------|-------------|
| [08-interpretation](notebooks/08-interpretation.ipynb) | Linear Regression coefficients and Decision Tree splits in raw units |

Re-fits Linear Regression and a depth-3 Decision Tree on unscaled training data for interpretability. Linear Regression gives directly readable coefficients (e.g., `curbweight`: +$6.36 per pound; `fueltype`: +$12,040 for diesel vs gas). The Decision Tree exposes the most informative splits (root: `enginesize ≤ 182`; top features: `enginesize` 70.6%, `curbweight` 27.9%).

Reading the LR coefficients also caught a redundancy: `fueltype` (diesel) and `fuelsystem_idi` had identical coefficients because they encoded the same information. `fuelsystem` was dropped from the preprocessed feature set, leaving the model leaner and slightly more accurate.

### Phase 6 — Black-box Explainability

| Notebook | Description |
|----------|-------------|
| [09-explainability](notebooks/09-explainability.ipynb) | SHAP and LIME analysis of the Extra Trees model |

Uses SHAP (global feature importance, waterfall plots) and LIME (local instance explanations) to interpret predictions from the Extra Trees model (PyCaret's CV pick on preprocessed data). Both methods agree on the top price drivers: `brand_tier`, `enginesize`, and `curbweight` — the same features the white-box analysis identified. All three engineered features (`brand_tier`, `avg_mpg`, `hp_per_weight`) appear in the SHAP rankings, confirming that manual feature engineering added genuine predictive signal.

The agreement across simple (LR, DT) and complex (Extra Trees + SHAP/LIME) models confirms the model learned genuine patterns, not noise. The model is not just accurate — it is accurate for interpretable reasons.

> **Note:** SHAP and LIME were run on Extra Trees (PyCaret's CV pick on preprocessed data). The test-set winner (Random Forest) was not re-analysed - both are tree ensembles trained on the same features, and the goal of this section is to expose which features matter, not to compare the two models' internals.

### Phase 7 — Presentation

Project presentation summarising the end-to-end workflow, key findings, and lessons learned across all phases.

- 📺 [Watch on YouTube](https://youtu.be/pPSpYYsdPdM)
- 📄 [Presentation slides (PDF)](presentation-car-price-prediction.pdf)

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

> **Note:** PyCaret requires older versions of `pandas` (<2.2) and `scikit-learn` (<1.5) that conflict with the main project's dependencies. The PyCaret scripts use inline metadata ([PEP 723](https://peps.python.org/pep-0723/)) to specify their own dependencies and Python version, which lets `uv` create an isolated environment automatically. This keeps the notebooks on modern versions while letting PyCaret run on the versions it needs. No `python` prefix needed - `uv` reads the metadata directly from the script. Output logs are saved in [`pycaret/`](pycaret/).
