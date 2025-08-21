# streamlit/app.py
import streamlit as st
import pandas as pd
from pathlib import Path

from eda_profiles import show_profiles   # consolidated profiles page (with clustering)
from predict import show_prediction      # your existing prediction page

@st.cache_data
def load_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent.parent / "data" / "bank-full.csv"
    try:
        return pd.read_csv(data_path, sep=";")
    except FileNotFoundError:
        st.error(f"Data file not found: {data_path}")
        return pd.DataFrame()

def main():
    st.sidebar.header("Navigation")
    # Landing page = Profiles
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Profiles", "Prediction"],
        index=0
    )

    df = load_data()

    if page == "Profiles":
        show_profiles(df)
    elif page == "Prediction":
        show_prediction()

if __name__ == "__main__":
    main()
