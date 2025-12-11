import streamlit as st
import pandas as pd
import numpy as np
import mlflow.sklearn
import warnings
import os

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------
# THEME / CSS (Premium Dark Gold)
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #0e1117;
        color: #e6e6e6;
    }
    [data-testid="stSidebar"] {
        background-color: #0f1418;
        border-right: 1px solid #222831;
    }
    .sidebar-title {
        font-size: 20px;
        font-weight: 700;
        color: #f5c542;
        margin-bottom: 12px;
    }
    .pcard {
        background: linear-gradient(180deg, #0f1317, #14181d);
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #2a2f36;
        box-shadow: 0 6px 18px rgba(0,0,0,0.6);
    }
    .metric { color: #f5c542; font-weight: 700; }
    .section-title { color: #f5c542; font-weight: 700; }
    button, .stButton>button {
        background-color: #f5c542 !important;
        color: black !important;
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# LOAD ML MODELS FROM LOCAL MLflow ARTIFACT FOLDERS
# ---------------------------------------------------------------
clf = None
regr = None

classification_path = "./models/classification_model"
regression_path = "./models/regression_model"

try:
    if os.path.exists(classification_path) and os.path.exists(regression_path):
        clf = mlflow.sklearn.load_model(classification_path)
        regr = mlflow.sklearn.load_model(regression_path)
    else:
        st.warning("⚠️ ML models not found. Ensure 'models/classification_model' and 'models/regression_model' exist.")
except Exception as e:
    st.error(f"Model load error: {e}")

# ---------------------------------------------------------------
# LOAD DATASET FROM GOOGLE DRIVE
# ---------------------------------------------------------------
CSV_URL = "https://drive.google.com/uc?export=download&id=1ocXmSTnawuKzJNHZmEmuWkMFsVt9nLfh"

df_raw = None
try:
    df_raw = pd.read_csv(CSV_URL)
except Exception as e:
    st.error(f"❌ Could not load dataset. {e}")

# ---------------------------------------------------------------
# FEATURE BUILDING FUNCTION
# ---------------------------------------------------------------
def build_feature_row(raw_row: pd.Series) -> pd.DataFrame:
    FEATURE_COLUMNS = list(clf.feature_names_in_) if clf else list(raw_row.index)
    X = pd.DataFrame(np.zeros((1, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)

    # Copy numeric values
    for col in raw_row.index:
        if col in X.columns and isinstance(raw_row[col], (int, float, np.integer, np.floating)):
            X.at[0, col] = raw_row[col]

    # Public transport mapping
    if "Public_Transport_Accessibility" in raw_row.index and "Public_Transport_Accessibility" in X.columns:
        X.at[0, "Public_Transport_Accessibility"] = {"Low":1, "Medium":2, "High":3}.get(str(raw_row["Public_Transport_Accessibility"]), 0)

    return X

# ---------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------
st.sidebar.markdown("<div class='sidebar-title'>🏡 Real Estate Advisor</div>", unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", ["Home", "Predict Price"])

# ---------------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------------
if page == "Home":
    st.markdown("<h1 class='section-title'>🏡 Real Estate Investment Advisor</h1>", unsafe_allow_html=True)

    st.markdown("<div class='pcard'>", unsafe_allow_html=True)
    st.markdown("### Welcome 👋")
    st.markdown("This dashboard helps you check property investment quality and future estimated prices.")

    if df_raw is not None:
        col1, col2 = st.columns(2)
        col1.metric("Total Properties", f"{len(df_raw)}")
        if "Price_in_Lakhs" in df_raw.columns:
            col2.metric("Avg Price (Lakhs)", f"{df_raw['Price_in_Lakhs'].mean():.2f}")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# PREDICT PRICE PAGE
# ---------------------------------------------------------------
elif page == "Predict Price":
    st.markdown("<h2 class='section-title'>Predict Property Investment & Future Price</h2>", unsafe_allow_html=True)

    if df_raw is None:
        st.warning("Dataset not loaded.")
    elif clf is None or regr is None:
        st.warning("Models not loaded. Cannot predict.")
    else:
        col1, col2, col3, col4 = st.columns(4)

        state = col1.selectbox("State", sorted(df_raw["State"].unique()))
        city = col2.selectbox("City", sorted(df_raw[df_raw["State"] == state]["City"].unique()))
        prop_type = col3.selectbox("Property Type", sorted(df_raw[(df_raw["State"] == state) & (df_raw["City"] == city)]["Property_Type"].unique()))

        ids = df_raw[
            (df_raw["State"] == state) &
            (df_raw["City"] == city) &
            (df_raw["Property_Type"] == prop_type)
        ]["ID"].tolist()

        if ids:
            prop_id = col4.selectbox("Property ID", sorted(ids))
        else:
            col4.warning("No properties available.")
            st.stop()

        st.divider()

        if st.button("Predict Now"):
            row_raw = df_raw[df_raw["ID"] == prop_id].iloc[0]
            X = build_feature_row(row_raw)

            cls_pred = clf.predict(X)[0]
            reg_pred = regr.predict(X)[0]

            st.success("Prediction Successful!")

            st.subheader("📌 Investment Recommendation")
            if int(cls_pred) == 1:
                st.markdown("### ✅ **Good Investment**")
            else:
                st.markdown("### ❌ **Not a great investment**")

            st.subheader("💰 Estimated Price After 5 Years")
            st.markdown(f"### ₹ **{reg_pred:.2f} Lakhs**")
