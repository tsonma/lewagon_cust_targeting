import streamlit as st
import pandas as pd
from streamliteda import show_eda
import plotly.express as px
from pathlib import Path
from predict import show_prediction
@st.cache_data
def load_data():
    try:
        data_path = Path(__file__).resolve().parent.parent / "data" / "bank-full.csv"
        return pd.read_csv(data_path, sep=";")
    except FileNotFoundError:
        st.error(f"Data file not found at {data_path}")
        return pd.DataFrame()

def main():
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["EDA", "Prediction"])

    data = load_data()

    if page == "EDA":
        st.title("Exploratory Data Analysis")
        if not data.empty:
            show_eda(data)
        else:
            st.warning("No data available to display EDA.")

    elif page == "Prediction":
        show_prediction()

if __name__ == "__main__":
    main()
