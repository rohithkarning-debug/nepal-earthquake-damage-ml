from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

GEO_COLUMNS = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
OBJECT_CATEGORICAL_COLUMNS = [
    "land_surface_condition",
    "foundation_type",
    "roof_type",
    "ground_floor_type",
    "other_floor_type",
    "position",
    "plan_configuration",
    "legal_ownership_status",
]
CAT_COLUMNS = OBJECT_CATEGORICAL_COLUMNS + GEO_COLUMNS
TARGET_COLUMN = "damage_grade"
ID_COLUMN = "building_id"


def load_model(model_path: str | Path):
    """Load the serialized CatBoost model."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare raw building records exactly as the final CatBoost model expects."""
    data = df.copy()
    if ID_COLUMN in data.columns:
        data = data.drop(columns=[ID_COLUMN])

    required = set(CAT_COLUMNS)
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required categorical columns: {missing}")

    for col in CAT_COLUMNS:
        data[col] = data[col].astype(str)

    return data


def predict_dataframe(
    df: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """Return building_id and predicted damage_grade for raw building records."""
    features = prepare_features(df)
    predictions = model.predict(features).ravel()
    predictions = pd.Series(predictions, index=df.index).astype(int)

    if not set(predictions.unique()).issubset({1, 2, 3}):
        raise ValueError("Model returned an unexpected damage grade outside {1, 2, 3}.")

    result = pd.DataFrame(index=df.index)
    if ID_COLUMN in df.columns:
        result[ID_COLUMN] = df[ID_COLUMN].values
    else:
        result[ID_COLUMN] = range(len(df))
    result[TARGET_COLUMN] = predictions.values
    return result.reset_index(drop=True)


def predict_csv(
    input_csv: str | Path,
    model_path: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    """Load a raw feature CSV, predict damage grades, and save the result."""
    df = pd.read_csv(input_csv)
    model = load_model(model_path)
    result = predict_dataframe(df, model)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result
