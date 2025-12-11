import streamlit as st
import pandas as pd
import numpy as np
import mlflow.sklearn
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import base64
import io
import warnings
import os

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------
# THEME / CSS (Premium Dark Gold)
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    /* App background */
    body, .stApp {
        background-color: #0e1117;
        color: #e6e6e6;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f1418;
        border-right: 1px solid #222831;
    }
    .sidebar-title {
        font-size: 20px;
        font-weight: 700;
        color: #f5c542;
    }
    .pcard {
        background: linear-gradient(180deg, #0f1317, #14181d);
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #2a2f36;
        box-shadow: 0 6px 18px rgba(0,0,0,0.6);
    }
    .metric {
        color: #f5c542;
        font-weight: 700;
    }
    .section-title {
        color: #f5c542;
        font-weight: 700;
    }
    button, .stButton>button {
        background-color: #f5c542 !important;
        color: black !important;
        border-radius: 8px !important;
    }
    .chart-box {
        padding: 12px;
        background: #0f1317;
        border-radius: 10px;
        border: 1px solid #222831;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# OPTIONAL: seaborn if available, else fall back to matplotlib-only
# ---------------------------------------------------------------
USE_SEABORN = True
try:
    import seaborn as sns
    sns.set_theme(style="darkgrid")
except Exception:
    USE_SEABORN = False
    warnings.warn("seaborn not available — falling back to matplotlib-only charts.")

# ---------------------------------------------------------------
# LOAD ML MODELS FROM LOCAL MLflow ARTIFACT FOLDERS (NOT registry)
# ---------------------------------------------------------------
# Expecting these folders to exist in your project root:
# ./models/classification_model
# ./models/regression_model
clf = None
regr = None
model_load_error = None

classification_path = "./models/classification_model"
regression_path = "./models/regression_model"

try:
    if os.path.exists(classification_path) and os.path.exists(regression_path):
        clf = mlflow.sklearn.load_model(classification_path)
        regr = mlflow.sklearn.load_model(regression_path)
    else:
        missing = []
        if not os.path.exists(classification_path):
            missing.append(classification_path)
        if not os.path.exists(regression_path):
            missing.append(regression_path)
        raise FileNotFoundError(f"Model folders not found: {missing}")
except Exception as e:
    model_load_error = e
    # Show a friendly warning but keep app running (visuals still work)
    st.warning(
        "Could not load local MLflow model folders. Make sure you placed the MLflow artifact folders "
        "`models/classification_model` and `models/regression_model` in the project root. "
        f"Details: {e}"
    )

# ---------------------------------------------------------------
# LOAD DATASET FROM GOOGLE DRIVE
# ---------------------------------------------------------------
CSV_URL = "https://drive.google.com/uc?export=download&id=1ocXmSTnawuKzJNHZmEmuWkMFsVt9nLfh"

df_raw = None
try:
    df_raw = pd.read_csv(CSV_URL)
except Exception as e:
    st.error(f"❌ Could not load dataset from Google Drive. Details: {e}")
    df_raw = None


# ---------------------------------------------------------------
# Keep original FEATURE_COLUMNS and helper constants if model loaded
# ---------------------------------------------------------------
if clf is not None:
    try:
        FEATURE_COLUMNS = list(clf.feature_names_in_)
    except Exception:
        FEATURE_COLUMNS = []
else:
    FEATURE_COLUMNS = []  # fallback

AMENITY_COLS   = [c for c in FEATURE_COLUMNS if c.startswith("Amenities_")]
STATE_PREFIX   = "State_"
CITY_PREFIX    = "City_"
LOC_PREFIX     = "Locality_"
PTYPE_PREFIX   = "Property_Type_"
FACE_PREFIX    = "Facing_"
FURN_PREFIX    = "Furnished_Status_"
OWNER_PREFIX   = "Owner_Type_"
AVAIL_PREFIX   = "Availability_Status_"

# ---------------------------------------------------------------
# build_feature_row (kept logically same as your original)
# ---------------------------------------------------------------
def build_feature_row(raw_row: pd.Series) -> pd.DataFrame:
    # If FEATURE_COLUMNS missing (models not loaded), fallback to raw_row columns
    cols = FEATURE_COLUMNS if FEATURE_COLUMNS else list(raw_row.index)
    X = pd.DataFrame(data=np.zeros((1, len(cols)), dtype=float), columns=cols)

    # Copy numeric columns
    for col in raw_row.index:
        if col in X.columns and pd.api.types.is_numeric_dtype(type(raw_row[col])):
            try:
                X.at[0, col] = raw_row[col]
            except Exception:
                X.at[0, col] = 0.0

    # Public transport mapping
    if "Public_Transport_Accessibility" in raw_row.index and "Public_Transport_Accessibility" in X.columns:
        mapping = {"Low": 1, "Medium": 2, "High": 3}
        X.at[0, "Public_Transport_Accessibility"] = mapping.get(str(raw_row["Public_Transport_Accessibility"]), 0)

    # Price_per_SqFt
    if all(c in raw_row.index for c in ["Price_in_Lakhs", "Size_in_SqFt"]):
        try:
            ppsf = raw_row["Price_in_Lakhs"] * 100000.0 / max(raw_row["Size_in_SqFt"], 1)
        except Exception:
            ppsf = 0.0
        if "Price_per_SqFt" in X.columns:
            X.at[0, "Price_per_SqFt"] = ppsf

        if "Nearby_Schools" in raw_row.index and "School_Density_Score" in X.columns:
            try:
                X.at[0, "School_Density_Score"] = raw_row["Nearby_Schools"] / max(raw_row["Size_in_SqFt"], 1)
            except Exception:
                X.at[0, "School_Density_Score"] = 0.0

        if "Nearby_Hospitals" in raw_row.index and "Hospital_Density_Score" in X.columns:
            try:
                X.at[0, "Hospital_Density_Score"] = raw_row["Nearby_Hospitals"] / max(raw_row["Size_in_SqFt"], 1)
            except Exception:
                X.at[0, "Hospital_Density_Score"] = 0.0

    # Age category
    if "Age_of_Property" in raw_row.index and "Property_Age_Category" in X.columns:
        try:
            age = raw_row["Age_of_Property"]
            bins = [0,5,10,20,50,100]
            labels = ["0-5 yrs","5-10 yrs","10-20 yrs","20-50 yrs","50+ yrs"]
            age_label = None
            for i in range(len(bins)-1):
                if bins[i] <= age < bins[i+1]:
                    age_label = labels[i]
                    break
            mapping = {"0-5 yrs":1,"5-10 yrs":2,"10-20 yrs":3,"20-50 yrs":4,"50+ yrs":5}
            X.at[0,"Property_Age_Category"] = float(mapping.get(age_label, 0))
        except Exception:
            X.at[0,"Property_Age_Category"] = 0.0

    # Binary columns
    for col in ["Parking_Space", "Security"]:
        if col in raw_row.index and col in X.columns:
            X.at[0, col] = 1.0 if str(raw_row[col]).strip().lower() == "yes" else 0.0

    # Amenities one-hot
    amenities_str = str(raw_row.get("Amenities", ""))
    raw_amen = [a.strip() for a in amenities_str.split(",") if a.strip()]
    for col in AMENITY_COLS:
        amen_name = col.replace("Amenities_", "")
        if amen_name in raw_amen:
            X.at[0, col] = 1.0

    # One-hot helper
    def set_one_hot(prefix: str, value: str):
        col_name = prefix + str(value)
        if col_name in X.columns:
            X.at[0, col_name] = 1.0

    for prefix, key in [(STATE_PREFIX, "State"), (CITY_PREFIX, "City"), (LOC_PREFIX, "Locality"),
                        (PTYPE_PREFIX, "Property_Type"), (FACE_PREFIX, "Facing"),
                        (FURN_PREFIX, "Furnished_Status"), (OWNER_PREFIX, "Owner_Type"),
                        (AVAIL_PREFIX, "Availability_Status")]:
        if key in raw_row.index:
            set_one_hot(prefix, raw_row[key])

    return X

# ---------------------------------------------------------------
# Chart helper functions (7 charts)
# ---------------------------------------------------------------

def plot_price_distribution(df):
    fig, ax = plt.subplots(figsize=(8,4))
    if "Price_in_Lakhs" not in df.columns:
        st.error("Price_in_Lakhs column missing in dataset.")
        return
    data = df["Price_in_Lakhs"].dropna()
    if USE_SEABORN:
        sns.histplot(data, bins=40, kde=True, ax=ax, color="#f5c542")
    else:
        ax.hist(data, bins=40, alpha=0.9)
        ax.set_ylabel("Count")
    ax.set_title("Price Distribution (in Lakhs)", color="#f5c542")
    ax.set_xlabel("Price (Lakhs)")
    ax.tick_params(colors="white")
    return fig

def plot_avg_price_by_city(df, top_n=20):
    if "City" not in df.columns or "Price_in_Lakhs" not in df.columns:
        st.error("Required columns missing for Avg Price by City.")
        return
    agg = df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10,5))
    if USE_SEABORN:
        sns.barplot(x=agg.values, y=agg.index, ax=ax, palette="crest")
    else:
        ax.barh(agg.index, agg.values)
    ax.set_title(f"Average Price by City (Top {top_n})", color="#f5c542")
    ax.set_xlabel("Average Price (Lakhs)")
    ax.tick_params(colors="white")
    return fig

def plot_property_type_distribution(df):
    if "Property_Type" not in df.columns:
        st.error("Property_Type column missing.")
        return
    counts = df["Property_Type"].value_counts()
    fig, ax = plt.subplots(figsize=(8,5))
    if USE_SEABORN:
        sns.barplot(x=counts.values, y=counts.index, ax=ax)
    else:
        ax.barh(counts.index, counts.values)
    ax.set_title("Property Type Distribution", color="#f5c542")
    ax.set_xlabel("Count")
    ax.tick_params(colors="white")
    return fig

def plot_correlation_heatmap(df):
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        st.error("Not enough numeric columns for correlation heatmap.")
        return
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(10,8))
    if USE_SEABORN:
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", ax=ax, cbar_kws={"shrink":0.6})
    else:
        im = ax.imshow(corr, cmap="coolwarm", aspect="auto", norm=Normalize(vmin=-1, vmax=1))
        ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation Heatmap (numeric features)", color="#f5c542")
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_color("white")
    return fig

def plot_size_vs_price(df):
    if "Size_in_SqFt" not in df.columns or "Price_in_Lakhs" not in df.columns:
        st.error("Required columns missing for Size vs Price scatter.")
        return
    d = df.dropna(subset=["Size_in_SqFt", "Price_in_Lakhs"])
    # sample if huge
    if len(d) > 2000:
        d = d.sample(2000, random_state=42)
    fig, ax = plt.subplots(figsize=(8,5))
    if USE_SEABORN:
        sns.scatterplot(data=d, x="Size_in_SqFt", y="Price_in_Lakhs", ax=ax, alpha=0.6)
        sns.regplot(data=d, x="Size_in_SqFt", y="Price_in_Lakhs", scatter=False, ax=ax, color="#f5c542")
    else:
        ax.scatter(d["Size_in_SqFt"], d["Price_in_LakHs"] if "Price_in_LakHs" in d.columns else d["Price_in_Lakhs"], alpha=0.6)
    ax.set_title("Size (SqFt) vs Price (Lakhs)", color="#f5c542")
    ax.set_xlabel("Size (SqFt)"); ax.set_ylabel("Price (Lakhs)")
    ax.tick_params(colors="white")
    return fig

def plot_price_per_sqft_distribution(df):
    if not all(c in df.columns for c in ["Price_in_Lakhs", "Size_in_SqFt"]):
        st.error("Price_in_Lakhs or Size_in_SqFt missing.")
        return
    d = df.copy()
    d = d[(d["Size_in_SqFt"]>0) & d["Price_in_Lakhs"].notna()]
    d["Price_per_SqFt"] = d["Price_in_Lakhs"]*100000.0 / d["Size_in_SqFt"]
    fig, ax = plt.subplots(figsize=(8,4))
    if USE_SEABORN:
        sns.histplot(d["Price_per_SqFt"].clip(upper=d["Price_per_SqFt"].quantile(0.99)), bins=40, kde=True, ax=ax)
    else:
        ax.hist(d["Price_per_SqFt"].clip(upper=d["Price_per_SqFt"].quantile(0.99)).dropna(), bins=40)
    ax.set_title("Price per SqFt Distribution (INR)", color="#f5c542")
    ax.set_xlabel("Price per SqFt (INR)")
    ax.tick_params(colors="white")
    return fig

def plot_avg_price_by_state(df, top_n=20):
    if "State" not in df.columns or "Price_in_Lakhs" not in df.columns:
        st.error("Required columns missing for Avg Price by State.")
        return
    agg = df.groupby("State")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10,5))
    if USE_SEABORN:
        sns.barplot(x=agg.values, y=agg.index, ax=ax, palette="magma")
    else:
        ax.barh(agg.index, agg.values)
    ax.set_title(f"Average Price by State (Top {top_n})", color="#f5c542")
    ax.set_xlabel("Average Price (Lakhs)")
    ax.tick_params(colors="white")
    return fig

# ---------------------------------------------------------------
# Visual Analytics mapping
# ---------------------------------------------------------------
CHART_MAPPING = {
    "Price Distribution": plot_price_distribution,
    "Average Price by City": plot_avg_price_by_city,
    "Property Type Distribution": plot_property_type_distribution,
    "Correlation Heatmap": plot_correlation_heatmap,
    "Size vs Price Scatterplot": plot_size_vs_price,
    "Price per SqFt Distribution": plot_price_per_sqft_distribution,
    "Average Price by State": plot_avg_price_by_state
}

# ---------------------------------------------------------------
# SIDEBAR NAV
# ---------------------------------------------------------------
st.sidebar.markdown("<div class='sidebar-title'>🏡 Real Estate Advisor</div>", unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", ("Home", "Predict Price", "Visual Analytics"))

# ---------------------------------------------------------------
# VISUAL ANALYTICS (SIDEBAR THUMBNAILS)
# ---------------------------------------------------------------

if page == "Visual Analytics":

    st.markdown("<h2 class='section-title'>📊 Visual Analytics</h2>", unsafe_allow_html=True)

    if df_raw is None:
        st.warning("Dataset not available.")
    else:

        st.sidebar.markdown("### 📊 Visual Analytics")

        # Sidebar list of thumbnails
        for chart_title, chart_func in CHART_MAPPING.items():

            with st.sidebar.expander(f"▶ {chart_title}", expanded=False):

                # Generate thumbnail
                try:
                    fig = chart_func(df_raw)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=40, bbox_inches="tight")
                    buf.seek(0)

                    st.image(buf, use_column_width=True)
                    plt.close(fig)

                    # Expand full chart on click
                    if st.button(f"Expand {chart_title}", key=f"full_{chart_title}"):

                        # Show full chart in main area
                        fig_full = chart_func(df_raw)
                        st.pyplot(fig_full)
                        plt.close(fig_full)

                except Exception as e:
                    st.sidebar.write(f"Thumbnail unavailable: {e}")

    st.stop()

# ---------------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------------
if page == "Home":
    st.markdown("<h1 class='section-title'> 🏡 Real Estate Investment Advisor</h1>", unsafe_allow_html=True)
    st.markdown("<div class='pcard'>", unsafe_allow_html=True)
    st.markdown("### Welcome 👋")
    st.markdown("Use this dashboard to analyze your housing dataset and run investment predictions.")
    if df_raw is not None:
        total_props = len(df_raw)
        avg_price = df_raw["Price_in_Lakhs"].mean() if "Price_in_Lakhs" in df_raw.columns else None
        c1, c2 = st.columns(2)
        c1.metric("Total Properties", f"{total_props}")
        if avg_price is not None:
            c2.metric("Average Price (Lakhs)", f"{avg_price:.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# PREDICTION PAGE
# ---------------------------------------------------------------
elif page == "Predict Price":
    st.markdown("<h2 class='section-title'>Predict Property Investment & Future Price</h2>", unsafe_allow_html=True)
    if df_raw is None:
        st.warning("Dataset required for prediction. Place your CSV at `data/india_housing_prices.csv`.")
    elif clf is None or regr is None:
        st.warning("MLflow models are not loaded. Predictions won't run until models are available.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        # State dropdown
        state_options = sorted(df_raw["State"].dropna().unique())
        state = col1.selectbox("State", state_options)
        city_options = sorted(df_raw[df_raw["State"] == state]["City"].dropna().unique())
        city = col2.selectbox("City", city_options)
        ptype_options = sorted(df_raw[(df_raw["State"] == state) & (df_raw["City"] == city)]["Property_Type"].dropna().unique())
        prop_type = col3.selectbox("Property Type", ptype_options)
        id_options = df_raw[(df_raw["State"] == state) & (df_raw["City"] == city) & (df_raw["Property_Type"] == prop_type)]["ID"].tolist()
        if not id_options:
            col4.warning("No properties found for this combination.")
        else:
            prop_id = col4.selectbox("Property ID", sorted(id_options))
            st.divider()
            if st.button("Predict Now"):
                row_raw = df_raw[df_raw["ID"] == prop_id].iloc[0]
                X_input = build_feature_row(row_raw)
                # enforce feature order if available
                if FEATURE_COLUMNS:
                    X_input = X_input[FEATURE_COLUMNS]
                cls_pred = clf.predict(X_input)[0] if clf is not None else None
                reg_pred = regr.predict(X_input)[0] if regr is not None else None
                st.success("Prediction completed successfully!")
                with st.container():
                    st.subheader("📌 Investment Recommendation")
                    if cls_pred is not None and int(cls_pred) == 1:
                        st.markdown("### ✅ **This is a GOOD Investment**")
                    elif cls_pred is not None:
                        st.markdown("### ❌ **This may NOT be a good investment**")
                    else:
                        st.info("Classification model not available to give recommendation.")
                    st.subheader("💰 Estimated Price After 5 Years")
                    if reg_pred is not None:
                        st.markdown(f"### ₹ **{reg_pred:.2f} Lakhs**")
                    else:
                        st.info("Regression model not available to estimate price.")
                st.info("Note: feature engineering replicates model training pipeline so input columns match model expectation.")

# ---------------------------------------------------------------
# VISUAL ANALYTICS PAGE
# ---------------------------------------------------------------
elif page == "Visual Analytics":
    st.markdown("<h2 class='section-title'>Visual Analytics</h2>", unsafe_allow_html=True)
    if df_raw is None:
        st.warning("Dataset not available. Please add `data/india_housing_prices.csv` to enable visual analytics.")
    else:
        if selected_chart:
            st.markdown(f"### {selected_chart}")
            plot_func = CHART_MAPPING.get(selected_chart)
            if plot_func is None:
                st.error("Chart not implemented.")
            else:
                fig = plot_func(df_raw)
                if fig is not None:
                    # show with Streamlit's matplotlib renderer
                    st.pyplot(fig)
                    plt.close(fig)
        else:
            st.info("Select a chart from the sidebar dropdown to view it.")
            # show small gallery / thumbnails (basic)
            cols = st.columns(3)
            for idx, (name, fn) in enumerate(CHART_MAPPING.items()):
                with cols[idx % 3]:
                    st.markdown(f"**{name}**")
                    # attempt quick thumbnail (small figure)
                    try:
                        thumb = fn(df_raw)
                        if thumb is not None:
                            # render small preview
                            st.pyplot(thumb)
                            plt.close(thumb)
                    except Exception:
                        st.write("Preview unavailable")



