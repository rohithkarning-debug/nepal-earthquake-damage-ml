import pandas as pd
import pytest

from src.predict import prepare_features, predict_dataframe


class DummyModel:
    """Simple model substitute for testing prediction behavior."""

    feature_names_ = ["age", "foundation_type"]

    def predict(self, features):
        return pd.Series([1] * len(features)).to_numpy()


def test_prepare_features_selects_expected_order():
    df = pd.DataFrame({
        "building_id": [101],
        "foundation_type": ["mud"],
        "age": [10],
        "damage_grade": [3],
        "extra_column": ["ignore me"],
    })

    result = prepare_features(
        df,
        feature_names=["age", "foundation_type"],
    )

    assert list(result.columns) == ["age", "foundation_type"]
    assert result.iloc[0]["age"] == 10
    assert result.iloc[0]["foundation_type"] == "mud"


def test_prepare_features_raises_error_for_missing_columns():
    df = pd.DataFrame({"age": [10]})

    with pytest.raises(ValueError, match="missing"):
        prepare_features(
            df,
            feature_names=["age", "foundation_type"],
        )


def test_predict_dataframe_returns_valid_damage_grades():
    df = pd.DataFrame({
        "building_id": [101, 102],
        "age": [10, 20],
        "foundation_type": ["mud", "stone"],
    })

    result = predict_dataframe(df, DummyModel())

    assert list(result.columns) == ["building_id", "damage_grade"]
    assert result["building_id"].tolist() == [101, 102]
    assert result["damage_grade"].tolist() == [1, 1]
    assert result["damage_grade"].isin([1, 2, 3]).all()


def test_predict_dataframe_rejects_invalid_damage_grades():
    class InvalidModel(DummyModel):
        def predict(self, features):
            return pd.Series([4] * len(features)).to_numpy()

    df = pd.DataFrame({
        "building_id": [101],
        "age": [10],
        "foundation_type": ["mud"],
    })

    with pytest.raises(ValueError, match="unexpected damage grade"):
        predict_dataframe(df, InvalidModel())


def test_find_input_problems_reports_blank_and_non_numeric_values():
    from src.predict import find_input_problems

    df = pd.DataFrame({
        "age": ["ten", 5],
        "foundation_type": [None, "r"],
    })

    problems = find_input_problems(df, feature_names=["age", "foundation_type"])

    assert any("'age' must be a number" in p for p in problems)
    assert any("'foundation_type' is blank" in p for p in problems)


def test_find_input_problems_accepts_valid_input():
    from src.predict import find_input_problems

    df = pd.DataFrame({"age": [10], "foundation_type": ["r"]})

    assert find_input_problems(df, feature_names=["age", "foundation_type"]) == []
