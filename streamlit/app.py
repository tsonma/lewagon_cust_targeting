import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
from pathlib import Path
import io

from eda_profiles import show_profiles
from predict import show_prediction

# ---------- Expected columns for validation ----------
REQUIRED_COLUMNS = [
    'age', 'job', 'marital', 'education', 'default', 'balance',
    'housing', 'loan', 'contact', 'day', 'month', 'campaign',
    'pdays', 'previous', 'poutcome', 'y'
]

# ---------- Data loader and validation ----------
@st.cache_data
def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Load and validate uploaded CSV file"""
    try:
        content = uploaded_file.getvalue().decode('utf-8')
        try:
            df = pd.read_csv(io.StringIO(content), sep=";")
            if len(df.columns) > 1:
                return df
        except:
            pass
        try:
            df = pd.read_csv(io.StringIO(content), sep=',')
            return df
        except:
            pass
        try:
            df = pd.read_csv(io.StringIO(content), sep='\t')
            return df
        except:
            pass
        raise ValueError("Could not parse CSV with common separators")
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        return pd.DataFrame()

def validate_csv_structure(df: pd.DataFrame) -> tuple[bool, list]:
    """Validate if CSV has required columns for the investment prediction model"""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return len(missing_cols) == 0, missing_cols

def show_home_page():
    """Home page with file upload functionality for the Profiles page."""
    st.title("Predictr")
    st.markdown("Upload your customer dataset to analyze customer profiles.")

    # File upload section
    st.subheader("📁 Upload Dataset for Profiles")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        key="home_uploader",
        help="This dataset will be used for the Profiles and EDA pages."
    )

    # Initialize a flag to control button visibility
    if 'file_uploaded' not in st.session_state:
        st.session_state['file_uploaded'] = False
        st.session_state['data'] = None

    if uploaded_file is not None:
        with st.spinner("Loading and validating your dataset..."):
            df = load_uploaded_csv(uploaded_file)
            if not df.empty:
                is_valid, missing_cols = validate_csv_structure(df)
                if is_valid:
                    st.session_state['data'] = df
                    st.session_state['data_source'] = f"Uploaded: {uploaded_file.name}"
                    st.session_state['file_uploaded'] = True
                    st.success(f"✅ Dataset loaded! ({len(df):,} rows, {len(df.columns)} columns)")

                    with st.expander("Preview Dataset"):
                        st.dataframe(df.head())

                else:
                    st.error("❌ Invalid CSV structure!")
                    st.write("Missing columns:", missing_cols)
                    st.session_state['data'] = None
                    st.session_state['file_uploaded'] = False
            else:
                st.session_state['data'] = None
                st.session_state['file_uploaded'] = False

    # Only show the button if a valid file has been successfully uploaded
    if st.session_state.get('file_uploaded'):
        st.success("🎉 Data ready! Click 'Start Analysis' to proceed to the Profiles page.")
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🚀 Start Analysis", use_container_width=True):
                st.session_state['current_page'] = "👥 Profiles"
                st.rerun()
    elif uploaded_file is None:
        st.info("Upload a CSV file to analyze customer profiles.")

def main():
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "🏠 Home"

    # Sidebar navigation - always show all options
    st.sidebar.header("🧭 Navigation")

    page_options = ["🏠 Home", "👥 Profiles", "🔮 Prediction"]
    try:
        current_index = page_options.index(st.session_state['current_page'])
    except ValueError:
        current_index = 0

    # The only line that has been changed
    page = st.sidebar.selectbox(
        "Choose a page",
        page_options,
        index=current_index,
        key="page_selector"
    )

    st.session_state['current_page'] = page

    # Show data source status for the profiles page
    if st.session_state.get('data') is not None and not st.session_state['data'].empty:
        st.sidebar.success(f"📊 Profiles Data: {st.session_state['data_source']}")
    st.sidebar.markdown("---")

    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "👥 Profiles":
        show_profiles(st.session_state.get('data'))
    elif page == "🔮 Prediction":
        show_prediction()

if __name__ == "__main__":
    main()
