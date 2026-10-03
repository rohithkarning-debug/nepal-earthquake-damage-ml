from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier

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


def train(train_values_path: str, train_labels_path: str, model_path: str) -> None:
    train_values = pd.read_csv(train_values_path)
    train_labels = pd.read_csv(train_labels_path)
    df = train_values.merge(train_labels, on="building_id", how="inner", validate="one_to_one")

    X = df.drop(columns=["building_id", "damage_grade"]).copy()
    y = df["damage_grade"].copy()

    for col in CAT_COLUMNS:
        X[col] = X[col].astype(str)

    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        cat_features=CAT_COLUMNS,
        verbose=False,
        random_seed=42,
    )
    model.fit(X, y)

    out = Path(model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)

    print(f"Training rows: {len(df):,}")
    print(f"Feature count: {X.shape[1]}")
    print(f"Saved model: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the final Nepal earthquake CatBoost model.")
    parser.add_argument("--train-values", required=True, help="Path to train_values.csv")
    parser.add_argument("--train-labels", required=True, help="Path to train_labels.csv")
    parser.add_argument("--model-out", default="models/final_model.joblib")
    args = parser.parse_args()
    train(args.train_values, args.train_labels, args.model_out)
