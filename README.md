# Nepal Earthquake Building Damage Prediction

Machine Learning mini-project for predicting the damage level of buildings affected by the 2015 Nepal earthquake.

## Team

- Rohith Karning — PES1UG24AM229
- S Bhuvana Shree — PES1UG24AM235

## Problem

The project studies whether building characteristics collected after the Nepal earthquake can be used to classify structural damage into three levels:

1. Low damage
2. Medium damage
3. Almost complete destruction

The reference work for this project applies KNN, a backpropagation neural network, and Random Forest to the same problem. This implementation reproduces those baseline ideas and extends them with a categorical-aware CatBoost experiment for the geographic-level identifier features.

## Dataset

- Training buildings: 260,601
- Test buildings: 86,868
- Predictive features: 38
- Target: `damage_grade`
- Classes: 1, 2, 3

The raw competition dataset is not committed to Git because of file size. Place the three CSV files in `data/` as described in `data/README.md`.

## Methodology

### Baselines

- K-Nearest Neighbors with 20 selected features and `n_neighbors=7`
- MLP neural network baseline
- Random Forest baseline with 100 estimators

### Improvements / experiments

- MLP architecture comparison using `(32, 16)` hidden layers and lower learning rate
- Small Random Forest hyperparameter search
- CatBoost experiment treating `geo_level_1_id`, `geo_level_2_id`, and `geo_level_3_id` as categorical identifiers

The training/validation split is stratified and uses `random_state=42`. Preprocessing for KNN/MLP/RF is fit only on the training split.

## Model selection

The complete validation split contains 52,121 examples. The final comparison was:

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Random Forest | 71.57% | 0.6578 | 0.7099 |
| CatBoost | **74.67%** | **0.6934** | **0.7414** |

CatBoost was selected as the final candidate because it achieved higher validation Accuracy, Macro F1, and Weighted F1 on the complete validation set.

The 10,000-row common comparison is preserved in `results/model_comparison_10k.csv`; those scores are an intermediate experiment, not the final full-validation headline numbers.

## Final model

`models/final_model.joblib` is the serialized CatBoost classifier trained on all 260,601 labeled training examples with:

- `iterations=300`
- `depth=6`
- `learning_rate=0.05`
- `random_seed=42`
- 11 categorical columns (8 original categorical features + 3 geographic-level identifier features)

The production prediction file is `results/nepal_earthquake_final_predictions.csv` with 86,868 predictions.

## Repository structure

```text
nepal-earthquake-damage-ml/
├── README.md
├── requirements.txt
├── pytest.ini
├── app.py
├── train.py
├── src/
│   └── predict.py
├── tests/
│   └── test_predict.py
├── notebooks/
│   └── Nepal_Earthquake_ML_Project_Final.ipynb
├── models/
│   └── final_model.joblib
├── results/
│   ├── model_comparison_10k.csv
│   ├── model_comparison_full_validation.csv
│   ├── nepal_earthquake_final_predictions.csv
│   └── README.md
├── data/
│   ├── README.md
│   └── sample_input.csv
└── docs/
```

## Requirements

Requirements: Python 3.11 or newer.

## Setup

Create a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data

The dataset comes from the DrivenData competition "Richter's Predictor: Modeling Earthquake Damage", which uses building data from the 2015 Nepal Earthquake Open Data Portal (see `docs/reference_paper_cs229.pdf`, Sections 2 and 4). It consists of three CSV files:

- `train_values.csv` — 260,601 buildings: `building_id` plus 38 features
- `train_labels.csv` — `building_id` and `damage_grade` (1, 2 or 3) for each training building
- `test_values.csv` — 86,868 buildings: `building_id` plus the same 38 features, without labels

Download the three files from the competition (https://www.drivendata.org/competitions/57/nepal-earthquake/) and place them in `data/` (see `data/README.md`). They are gitignored and are only needed for retraining or for batch predictions on the test set; the Streamlit demo and the tests run without them.

## Run the Streamlit demo

```bash
streamlit run app.py
```

The demo has three modes: enter a building's values in a form, upload a CSV, or use a one-row sample input.

## Retrain the final model

Place the raw datasets in `data/` and run:

```bash
python train.py --train-values data/train_values.csv --train-labels data/train_labels.csv --model-out models/final_model.joblib
```

## Generate predictions from a CSV

The prediction utility expects a feature CSV with the same 38 predictive columns used by the model:

```python
from src.predict import predict_csv

predict_csv(
    "data/test_values.csv",
    "models/final_model.joblib",
    "results/predictions.csv",
)
```

## Running tests

After `pip install -r requirements.txt`, run `pytest` from the repo root:

```bash
pytest
```

## Demo note

The serialized model is the CatBoost classifier itself. The inference utility reproduces the model's final preprocessing requirement by dropping `building_id` and converting the 11 categorical columns to strings before prediction.

## Source / reference

Reference paper used for the project framing and baseline methodology:
`docs/reference_paper_cs229.pdf` — CS229 Final Report, Yitian Liang, "Application of Machine Learning Methods to Predict the Level of Buildings Damage in the Nepal Earthquake."
