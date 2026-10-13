from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from predict import (  # noqa: E402
    CATEGORY_CODES,
    FEATURE_COLUMNS,
    expected_feature_names,
    find_input_problems,
    load_model,
    predict_dataframe,
    prepare_features,
)

MODEL_PATH = ROOT / "models" / "final_model.joblib"
SAMPLE_PATH = ROOT / "data" / "sample_input.csv"
REQUIRED_FEATURES = FEATURE_COLUMNS

DAMAGE_LABELS = {
    1: "Low Damage",
    2: "Medium Damage",
    3: "Almost Complete Destruction",
}

SUPERSTRUCTURE_FLAGS = {
    "has_superstructure_adobe_mud": "Adobe / mud",
    "has_superstructure_mud_mortar_stone": "Mud mortar + stone",
    "has_superstructure_stone_flag": "Stone (flag)",
    "has_superstructure_cement_mortar_stone": "Cement mortar + stone",
    "has_superstructure_mud_mortar_brick": "Mud mortar + brick",
    "has_superstructure_cement_mortar_brick": "Cement mortar + brick",
    "has_superstructure_timber": "Timber",
    "has_superstructure_bamboo": "Bamboo",
    "has_superstructure_rc_non_engineered": "Reinforced concrete (non-engineered)",
    "has_superstructure_rc_engineered": "Reinforced concrete (engineered)",
    "has_superstructure_other": "Other material",
}

SECONDARY_USE_FLAGS = {
    "has_secondary_use_agriculture": "Agriculture",
    "has_secondary_use_hotel": "Hotel",
    "has_secondary_use_rental": "Rental",
    "has_secondary_use_institution": "Institution",
    "has_secondary_use_school": "School",
    "has_secondary_use_industry": "Industry",
    "has_secondary_use_health_post": "Health post",
    "has_secondary_use_gov_office": "Government office",
    "has_secondary_use_use_police": "Police",
    "has_secondary_use_other": "Other use",
}

CATEGORY_LABELS = {
    "land_surface_condition": "Land surface condition",
    "foundation_type": "Foundation type",
    "roof_type": "Roof type",
    "ground_floor_type": "Ground floor type",
    "other_floor_type": "Other floor type",
    "position": "Position",
    "plan_configuration": "Plan configuration",
    "legal_ownership_status": "Legal ownership status",
}

st.set_page_config(page_title="Nepal Earthquake Damage Predictor", page_icon="🏠", layout="wide")

st.title("Nepal Earthquake Building Damage Predictor")
st.caption("CatBoost model with geographic-level identifier features treated as categorical variables.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


model = get_model()


def show_single_prediction(row: pd.DataFrame) -> None:
    """Predict one building and show the grade with class probabilities."""
    problems = find_input_problems(row, expected_feature_names(model))
    if problems:
        st.error("Please fix these values before predicting:")
        for problem in problems:
            st.write(f"- {problem}")
        return
    prediction = predict_dataframe(row, model)
    grade = int(prediction.iloc[0]["damage_grade"])
    probabilities = model.predict_proba(prepare_features(row, expected_feature_names(model)))[0]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Predicted Damage Grade", f"Grade {grade}")
        st.info(DAMAGE_LABELS[grade])
    with col2:
        chart = pd.DataFrame(
            {"Probability": probabilities},
            index=[f"{g} – {DAMAGE_LABELS[g]}" for g in (1, 2, 3)],
        )
        st.bar_chart(chart, horizontal=True)


st.markdown("### Live prediction")
input_mode = st.radio(
    "Choose an input mode",
    ["Enter values manually", "Upload CSV", "Use demo sample"],
    horizontal=True,
)

if input_mode == "Enter values manually":
    if not SAMPLE_PATH.exists():
        st.warning("data/sample_input.csv is missing; it is used to pre-fill the form.")
        st.stop()
    defaults = pd.read_csv(SAMPLE_PATH).iloc[0]
    st.write("The form starts from the demo building. Change any values, then click **Predict**.")

    with st.form("manual_input"):
        values: dict[str, object] = {}

        st.markdown("**Location** (administrative region IDs, from largest to smallest)")
        c1, c2, c3 = st.columns(3)
        values["geo_level_1_id"] = c1.number_input("Geo level 1 ID (0–30)", 0, 30, int(defaults["geo_level_1_id"]))
        values["geo_level_2_id"] = c2.number_input("Geo level 2 ID (0–1427)", 0, 1427, int(defaults["geo_level_2_id"]))
        values["geo_level_3_id"] = c3.number_input("Geo level 3 ID (0–12567)", 0, 12567, int(defaults["geo_level_3_id"]))

        st.markdown("**Size and age**")
        c1, c2, c3, c4, c5 = st.columns(5)
        values["count_floors_pre_eq"] = c1.number_input("Floors", 1, 9, int(defaults["count_floors_pre_eq"]))
        values["age"] = c2.number_input("Age (years)", 0, 995, int(defaults["age"]))
        values["area_percentage"] = c3.number_input("Area (normalised %)", 1, 100, int(defaults["area_percentage"]))
        values["height_percentage"] = c4.number_input("Height (normalised %)", 2, 32, int(defaults["height_percentage"]))
        values["count_families"] = c5.number_input("Families living there", 0, 9, int(defaults["count_families"]))

        st.markdown("**Construction** (the dataset anonymises these categories as letter codes)")
        cols = st.columns(4)
        for i, (col, codes) in enumerate(CATEGORY_CODES.items()):
            default = str(defaults[col])
            values[col] = cols[i % 4].selectbox(
                CATEGORY_LABELS[col], codes, index=codes.index(default) if default in codes else 0
            )

        st.markdown("**Superstructure materials** (tick all that apply)")
        cols = st.columns(4)
        for i, (col, label) in enumerate(SUPERSTRUCTURE_FLAGS.items()):
            values[col] = int(cols[i % 4].checkbox(label, value=bool(defaults[col])))

        st.markdown("**Secondary uses** (tick all that apply)")
        cols = st.columns(5)
        for i, (col, label) in enumerate(SECONDARY_USE_FLAGS.items()):
            values[col] = int(cols[i % 5].checkbox(label, value=bool(defaults[col])))
        values["has_secondary_use"] = int(any(values[col] for col in SECONDARY_USE_FLAGS))

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        show_single_prediction(pd.DataFrame([values])[REQUIRED_FEATURES])

elif input_mode == "Upload CSV":
    uploaded = st.file_uploader("Upload building feature CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        problems = find_input_problems(df, expected_feature_names(model))
        if problems:
            st.error("The uploaded CSV can't be used yet. Please fix:")
            for problem in problems:
                st.write(f"- {problem}")
            st.stop()
        st.write(f"Loaded {len(df):,} buildings.")
        st.dataframe(df.head(10), width="stretch")
        if st.button("Predict damage", type="primary"):
            predictions = predict_dataframe(df, model)
            st.success(f"Generated {len(predictions):,} predictions.")
            st.dataframe(predictions.head(20), width="stretch")
            st.download_button(
                "Download predictions CSV",
                data=predictions.to_csv(index=False).encode("utf-8"),
                file_name="nepal_earthquake_predictions.csv",
                mime="text/csv",
            )

else:
    if not SAMPLE_PATH.exists():
        st.warning("Demo sample file is missing. Add data/sample_input.csv from a single test row.")
        st.stop()
    sample = pd.read_csv(SAMPLE_PATH)
    st.write("Demo input:")
    st.dataframe(sample, width="stretch")
    if st.button("Predict demo building", type="primary"):
        show_single_prediction(sample)

st.divider()
st.markdown("**Model:** CatBoostClassifier  •  **Validation Accuracy:** 74.67%  •  **Validation Macro F1:** 0.6934")
