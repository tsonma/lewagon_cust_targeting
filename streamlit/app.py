# streamlit/app.py
import streamlit as st
import pandas as pd
from pathlib import Path

from eda_profiles import show_profiles   # consolidated profiles page (with clustering)
try:
    from predict import show_prediction      # your existing prediction page
    _HAS_PREDICT = True
except Exception:
    _HAS_PREDICT = False

@st.cache_data
def load_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent.parent / "data" / "bank-full.csv"
    try:
        return pd.read_csv(data_path, sep=";")
    except FileNotFoundError:
        st.error(f"Data file not found: {data_path}")
        return pd.DataFrame()

def main():
    st.set_page_config(page_title="Customer Targeting", layout="wide")
    st.sidebar.header("Navigation")

    pages = ["Profiles"]
    if _HAS_PREDICT:
        pages.append("Prediction")

    page = st.sidebar.selectbox("Choose a page", pages, index=0)

    df = load_data()
    if df.empty:
        st.warning("Dataset is empty. Please add data to /data/bank-full.csv")
        return

    if page == "Profiles":
        show_profiles(df)
    elif page == "Prediction" and _HAS_PREDICT:
        show_prediction()

if __name__ == "__main__":
    main()
