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

# The 38 predictive features in the exact order used to train the final model.
# CatBoost maps DataFrame columns by position, so inputs must follow this order.
# Used as a fallback when the loaded model does not carry its own feature names.
FEATURE_COLUMNS = [
    "geo_level_1_id", "geo_level_2_id", "geo_level_3_id",
    "count_floors_pre_eq", "age", "area_percentage", "height_percentage",
    "land_surface_condition", "foundation_type", "roof_type", "ground_floor_type",
    "other_floor_type", "position", "plan_configuration",
    "has_superstructure_adobe_mud", "has_superstructure_mud_mortar_stone",
    "has_superstructure_stone_flag", "has_superstructure_cement_mortar_stone",
    "has_superstructure_mud_mortar_brick", "has_superstructure_cement_mortar_brick",
    "has_superstructure_timber", "has_superstructure_bamboo",
    "has_superstructure_rc_non_engineered", "has_superstructure_rc_engineered",
    "has_superstructure_other", "legal_ownership_status", "count_families",
    "has_secondary_use", "has_secondary_use_agriculture", "has_secondary_use_hotel",
    "has_secondary_use_rental", "has_secondary_use_institution", "has_secondary_use_school",
    "has_secondary_use_industry", "has_secondary_use_health_post",
    "has_secondary_use_gov_office", "has_secondary_use_use_police",
    "has_secondary_use_other",
]


# Category codes that appear in the training data (the dataset anonymises them as letters).
CATEGORY_CODES = {
    "land_surface_condition": ["n", "o", "t"],
    "foundation_type": ["h", "i", "r", "u", "w"],
    "roof_type": ["n", "q", "x"],
    "ground_floor_type": ["f", "m", "v", "x", "z"],
    "other_floor_type": ["j", "q", "s", "x"],
    "position": ["j", "o", "s", "t"],
    "plan_configuration": ["a", "c", "d", "f", "m", "n", "o", "q", "s", "u"],
    "legal_ownership_status": ["a", "r", "v", "w"],
}


def find_input_problems(df: pd.DataFrame, feature_names: Iterable[str] | None = None) -> list[str]:
    """Return readable problems that would stop a prediction (empty list if the input is fine).

    Checks for missing columns, blank cells, and non-numeric values in numeric columns.
    """
    features = list(feature_names) if feature_names is not None else list(FEATURE_COLUMNS)
    problems: list[str] = []

    missing = [col for col in features if col not in df.columns]
    if missing:
        problems.append(f"Missing {len(missing)} required column(s): {', '.join(missing)}")
        return problems

    for col in features:
        blank_rows = df.index[df[col].isna()].tolist()
        if blank_rows:
            problems.append(f"'{col}' is blank in row(s) {_rows(blank_rows)}")
            continue
        if col not in CAT_COLUMNS:
            numeric = pd.to_numeric(df[col], errors="coerce")
            bad_rows = df.index[numeric.isna()].tolist()
            if bad_rows:
                example = df.loc[bad_rows[0], col]
                problems.append(f"'{col}' must be a number, but row(s) {_rows(bad_rows)} contain values like '{example}'")
    return problems


def _rows(index_values: list) -> str:
    shown = ", ".join(str(int(i) + 1) for i in index_values[:5])
    return shown + (" ..." if len(index_values) > 5 else "")


def load_model(model_path: str | Path):
    """Load the serialized CatBoost model."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def expected_feature_names(model) -> list[str]:
    """Return the feature names, in training order, that the model expects.

    CatBoost exposes these as ``feature_names_`` when trained on a DataFrame. If the
    attribute is absent, empty, or holds only CatBoost's default positional names
    ("0", "1", ...), fall back to the documented training order.
    """
    names = list(getattr(model, "feature_names_", None) or [])
    if names and not all(str(name).isdigit() for name in names):
        return [str(name) for name in names]
    return list(FEATURE_COLUMNS)


def prepare_features(df: pd.DataFrame, feature_names: Iterable[str] | None = None) -> pd.DataFrame:
    """Prepare raw building records exactly as the final CatBoost model expects.

    Selects the model's features in training order, which drops ``building_id``,
    ``damage_grade`` and any other extra columns, and casts categorical columns to str.
    """
    features = list(feature_names) if feature_names is not None else list(FEATURE_COLUMNS)

    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing {len(missing)} required feature column(s): {missing}. "
            f"Expected all {len(features)} model features."
        )

    data = df[features].copy()

    for col in CAT_COLUMNS:
        if col in data.columns:
            data[col] = data[col].astype(str)

    return data


def predict_dataframe(
    df: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """Return building_id and predicted damage_grade for raw building records."""
    features = prepare_features(df, expected_feature_names(model))
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
