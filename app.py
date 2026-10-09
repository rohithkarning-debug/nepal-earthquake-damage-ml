from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from predict import CAT_COLUMNS, load_model, predict_dataframe  # noqa: E402

MODEL_PATH = ROOT / "models" / "final_model.joblib"
REQUIRED_FEATURES = [
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

DAMAGE_LABELS = {
    1: "Low Damage",
    2: "Medium Damage",
    3: "Almost Complete Destruction",
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

st.markdown("### Live prediction")
input_mode = st.radio("Choose an input mode", ["Upload CSV", "Use demo sample"], horizontal=True)

if input_mode == "Upload CSV":
    uploaded = st.file_uploader("Upload building feature CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = sorted(set(REQUIRED_FEATURES) - set(df.columns))
        if missing:
            st.error(f"The uploaded CSV is missing {len(missing)} required features.")
            st.write(missing)
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
    sample_path = ROOT / "data" / "sample_input.csv"
    if not sample_path.exists():
        st.warning("Demo sample file is missing. Add data/sample_input.csv from a single test row.")
        st.stop()
    sample = pd.read_csv(sample_path)
    st.write("Demo input:")
    st.dataframe(sample, width="stretch")
    if st.button("Predict demo building", type="primary"):
        prediction = predict_dataframe(sample, model)
        grade = int(prediction.iloc[0]["damage_grade"])
        st.metric("Predicted Damage Grade", f"Grade {grade}")
        st.info(DAMAGE_LABELS[grade])
        st.dataframe(prediction, width="stretch")

st.divider()
st.markdown("**Model:** CatBoostClassifier  •  **Validation Accuracy:** 74.67%  •  **Validation Macro F1:** 0.6934")
