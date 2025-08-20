import streamlit as st
import pandas as pd
from pathlib import Path

from streamliteda import show_eda
from predict import show_prediction
from eda_profiles import show_profiles  # donut + filters

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from theme import set_theme
set_theme()

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
    try:
        data_path = Path(__file__).resolve().parent.parent / "data" / "bank-full.csv"
        df = pd.read_csv(data_path, sep=";")

        # If profiles already exist, keep them
        if "profile" in df.columns:
            return df

        # 1) Minimal numeric set for clustering (fast/robust)
        num_cols = [c for c in ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
                    if c in df.columns]
        if len(num_cols) < 2:
            # Not enough numeric columns → return raw data (dashboard will warn)
            return df

        # 2) Scale + KMeans (4 profiles)
        X = df[num_cols].copy()
        X = X.fillna(X.median(numeric_only=True))
        X_scaled = StandardScaler().fit_transform(X)

        km = KMeans(n_clusters=4, n_init=10, random_state=42)
        df["cluster"] = km.fit_predict(X_scaled)

        # 3) Map clusters → readable names (adjust freely if your data maps differently)
        profile_map = {
            0: "Young & Students",
            1: "Retired / Stable",
            2: "Middle-age with Loans",
            3: "High-balance Professionals",
        }
        df["profile"] = df["cluster"].map(profile_map).astype(str)

        return df

    except FileNotFoundError:
        st.error("Data file not found: data/bank-full.csv")
        return pd.DataFrame()

# ---------- App ----------
def main():
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["EDA", "Profiles", "Prediction"])

    data = load_data()

    if page == "EDA":
        st.title("Exploratory Data Analysis")
        if not data.empty:
            show_eda(data)
        else:
            st.warning("No data available to display EDA.")

    elif page == "Profiles":
        st.title("👥 Customer Profiles Overview")

        # 1) Profile gallery (pictures + rich descriptions)
        cols = st.columns(2)
        for i, (name, info) in enumerate(PROFILE_INFO.items()):
            with cols[i % 2]:
                try:
                    st.image(info["img"], caption=name, use_container_width=True)
                except Exception:
                    st.info(f"Add image at: {info['img']}")
                st.markdown(f"**{name}**")
                st.write(info["desc"])
                st.markdown("---")

        # 2) Interactive dashboard (donut by y=yes/no + variable filters)
        st.subheader("📊 Interactive Dashboard")
        show_profiles(data)

    elif page == "Prediction":
        show_prediction()

if __name__ == "__main__":
    main()
