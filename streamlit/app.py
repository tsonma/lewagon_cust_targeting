# streamlit/app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import io
from pathlib import Path
import pandas as pd
import streamlit as st

from eda_profiles import show_profiles          # onglet 👥 Profiles
from predict import show_prediction              # onglet 🔮 Prediction (déjà existant)

# ---------- Expected columns for validation ----------
REQUIRED_COLUMNS = [
    "age", "job", "marital", "education", "default", "balance",
    "housing", "loan", "contact", "day", "month", "campaign",
    "pdays", "previous", "poutcome", "y"
]

@st.cache_data
def load_default_data() -> pd.DataFrame:
    """Load the default bank dataset"""
    data_path = Path(__file__).resolve().parent.parent / "data" / "bank-full.csv"
    try:
        return pd.read_csv(data_path, sep=";")
    except FileNotFoundError:
        st.error(f"Default data file not found: {data_path}")
        return pd.DataFrame()

def validate_csv_structure(df: pd.DataFrame) -> tuple[bool, list]:
    """Validate if CSV has required columns for the investment prediction model"""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return len(missing_cols) == 0, missing_cols

def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Load and validate uploaded CSV file (try common separators)"""
    try:
        content = uploaded_file.getvalue().decode("utf-8")

        for sep in (";", ",", "\t"):
            try:
                df = pd.read_csv(io.StringIO(content), sep=sep)
                if sep != ";" or len(df.columns) > 1:
                    return df
            except Exception:
                pass
        raise ValueError("Could not parse CSV with common separators")
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return pd.DataFrame()

# ---------- Home / Upload ----------
def show_home_page():
    st.title("🏦 Investment Prediction App")
    st.markdown("Upload your customer dataset to get started.")

    st.subheader("📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload your customer dataset in CSV format"
    )
    use_sample = st.button("Use Sample Data", type="secondary")

    if uploaded_file is not None:
        with st.spinner("Loading and validating your dataset..."):
            df = load_uploaded_csv(uploaded_file)

        if not df.empty:
            is_valid, missing_cols = validate_csv_structure(df)
            if is_valid:
                st.success(f"✅ Dataset loaded! ({len(df):,} rows, {len(df.columns)} columns)")
                st.session_state["data"] = df
                st.session_state["data_source"] = f"Uploaded: {uploaded_file.name}"
                with st.expander("Preview Dataset"):
                    st.dataframe(df.head(), use_container_width=True)
                st.success("🎉 Data ready! Use the sidebar to navigate.")
            else:
                st.error("❌ Invalid CSV structure!")
                st.write("Missing columns:", missing_cols)

    elif use_sample:
        with st.spinner("Loading sample dataset..."):
            df = load_default_data()
        if not df.empty:
            st.success(f"✅ Sample dataset loaded! ({len(df):,} rows, {len(df.columns)} columns)")
            st.session_state["data"] = df
            st.session_state["data_source"] = "Sample: bank-full.csv"
            st.success("🎉 Sample data ready! Use the sidebar to navigate.")

    if "data" not in st.session_state or st.session_state["data"] is None:
        st.info("Upload a CSV file or use sample data to get started.")

# ---------- App entry ----------
def main():
    st.set_page_config(page_title="Customer Targeting", layout="wide")

    if "data" not in st.session_state:
        st.session_state["data"] = None

    st.sidebar.header("🧭 Navigation")

    data_available = (
        st.session_state.get("data") is not None
        and not st.session_state["data"].empty
    )
    if st.session_state.get("data_source") and data_available:
        st.sidebar.success(f"📊 Data: {st.session_state['data_source']}")
        st.sidebar.markdown("---")

    # Menu (garder l’ordre établi par l’équipe)
    if data_available:
        page = st.sidebar.selectbox(
            "Choose a page",
            ["🏠 Home", "👥 Profiles", "🔮 Prediction"],
            index=0
        )
    else:
        page = "🏠 Home"
        st.sidebar.info("📁 Please upload data first to access other pages")

    # Routing (on ne modifie que l’appel du tab Profiles)
    if page == "🏠 Home":
        show_home_page()

    elif page == "👥 Profiles":
        if data_available:
            show_profiles(st.session_state["data"])
        else:
            st.warning("Please upload data first on the Home page")
            show_home_page()

    elif page == "🔮 Prediction":
        if data_available:
            show_prediction()
        else:
            st.warning("Please upload data first on the Home page")
            show_home_page()

if __name__ == "__main__":
    main()
