import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
from pathlib import Path

from eda_profiles import show_profiles   # consolidated profiles page (with clustering)
from predict import show_prediction      # your existing prediction page
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ---------- Profile cards (images + descriptions) ----------
PROFILE_INFO = {
    "Young & Students": {
        "img": "streamlit/assets/profile_young_students.jpg",
        "desc": (
            "Early‑career or students (20–35). Digital‑first, lower balances, "
            "curious and reactive to concise, mobile‑friendly campaigns. "
            "Good targets for entry products and trial offers."
        ),
    },
    "Retired / Stable": {
        "img": "streamlit/assets/profile_retired_stable.jpg",
        "desc": (
            "50+ retired or close to retirement. Higher average balances, "
            "risk‑averse, prefer clear and trustworthy communication. "
            "Respond to safe, income/guarantee‑oriented products."
        ),
    },
    "Middle-age with Loans": {
        "img": "streamlit/assets/profile_middle_loans.jpg",
        "desc": (
            "Mid‑career households with housing/personal loans. Budget‑conscious, "
            "value clear ROI and terms. Improve conversion with tailored value props "
            "and cross‑sell around repayment planning."
        ),
    },
    "High-balance Professionals": {
        "img": "streamlit/assets/profile_high_balance.jpg",
        "desc": (
            "Smaller segment with high balances/income. Busy schedules, prefer premium "
            "offers and concise outreach. High upside per conversion."
        ),
    },
}

# ---------- Data loader (WORKING version with KMeans profiles) ----------
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
