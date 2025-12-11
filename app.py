import streamlit as st
import pandas as pd
import numpy as np
import mlflow.sklearn
import os
import warnings

# ---------------------------------------------------------------
# SIMPLE PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    layout="wide"
)

# ---------------------------------------------------------------
# CACHED MODEL LOADING (Prevents memory crashes)
# ---------------------------------------------------------------
@st.cache_resource
def load_models():
    classification_path = "./models/classification_model"
    regression_path = "./models/regression_model"

    clf, regr = None, None

    try:
        if os.path.exists(classification_path) and os.path.exists(regression_path):
            clf = mlflow.sklearn.load_model(classification_path)
            regr = mlflow.sklearn.load_model(regression_path)
        else:
            st.warning("⚠️ ML models not found.")
    except Exception as e:
        st.error(f"Model load error: {e}")

    return clf, regr


clf, regr = load_models()

# ---------------------------------------------------------------
# CACHED DATA LOADING (Stops repeated huge downloads)
# ---------------------------------------------------------------
@st.cache_data
def load_dataset():
    CSV_URL = "https://drive.google.com/uc?export=download&id=1ocXmSTnawuKzJNHZmEmuWkMFsVt9nLfh"
    try:
        return pd.read_csv(CSV_URL)
    except Exception as e:
        st.error(f"❌ Could not load dataset. {e}")
        return None


df_raw = load_dataset()

# ---------------------------------------------------------------
# BUILD FEATURES FOR MODEL
# ---------------------------------------------------------------
def build_feature_row(raw_row: pd.Series) -> pd.DataFrame:
    FEATURE_COLUMNS = list(clf.feature_names_in_) if clf else list(raw_row.index)
    X = pd.DataFrame(np.zeros((1, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)

    # Numeric columns
    for col in raw_row.index:
        if col in X.columns and isinstance(raw_row[col], (int, float, np.integer, np.floating)):
            X.at[0, col] = raw_row[col]

    # Public transport mapping
    if "Public_Transport_Accessibility" in raw_row.index and "Public_Transport_Accessibility" in X.columns:
        mapping = {"Low": 1, "Medium": 2, "High": 3}
        X.at[0, "Public_Transport_Accessibility"] = mapping.get(str(raw_row["Public_Transport_Accessibility"]), 0)

    return X


# ---------------------------------------------------------------
# UI — HOME SECTION
# ---------------------------------------------------------------
st.title("🏡 Real Estate Investment Advisor")

st.subheader("Welcome 👋")
st.write("This tool helps you check property investment quality and future price estimation.")

if df_raw is not None:
    col1, col2 = st.columns(2)
    col1.metric("Total Properties", f"{len(df_raw)}")
    if "Price_in_Lakhs" in df_raw.columns:
        col2.metric("Avg Price (Lakhs)", f"{df_raw['Price_in_Lakhs'].mean():.2f}")

st.markdown("---")

# ---------------------------------------------------------------
# UI — PREDICTION SECTION
# ---------------------------------------------------------------
st.header("🔍 Predict Investment Quality & Future Price")

if df_raw is None:
    st.warning("Dataset not loaded.")
elif clf is None or regr is None:
    st.warning("Models not loaded.")
else:
    col1, col2, col3, col4 = st.columns(4)

    state = col1.selectbox("State", sorted(df_raw["State"].unique()))
    city = col2.selectbox("City", sorted(df_raw[df_raw["State"] == state]["City"].unique()))
    prop_type = col3.selectbox("Property Type", sorted(df_raw[
        (df_raw["State"] == state) &
        (df_raw["City"] == city)
    ]["Property_Type"].unique()))

    ids = df_raw[
        (df_raw["State"] == state) &
        (df_raw["City"] == city) &
        (df_raw["Property_Type"] == prop_type)
    ]["ID"].tolist()

    if ids:
        prop_id = col4.selectbox("Property ID", sorted(ids))
    else:
        st.warning("No properties available.")
        st.stop()

    st.markdown("---")

    if st.button("Predict"):
        row_raw = df_raw[df_raw["ID"] == prop_id].iloc[0]
        X = build_feature_row(row_raw)

        cls_pred = clf.predict(X)[0]
        reg_pred = regr.predict(X)[0]

        st.success("Prediction Completed!")

        st.subheader("📌 Investment Recommendation")
        st.write("### ✅ Good Investment" if int(cls_pred) == 1 else "### ❌ Not a great investment")

        st.subheader("💰 Estimated Price After 5 Years")
        st.write(f"### ₹ {reg_pred:.2f} Lakhs")
