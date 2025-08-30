import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
from pathlib import Path
import io

from eda_profiles import show_profiles
from predict import show_prediction

# ---------- Custom CSS Injection for Fonts ----------
def local_css(file_name):
    """Function to load and inject local CSS file."""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Call the function with the path to your CSS file
local_css(".streamlit/style.css")

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

def add_sidebar_logo():
    """Add company logo to the sidebar"""
    logo_path = Path(__file__).parent / "assets" / "company_logo_2.png"

    if os.path.exists(logo_path):
        # Option 1: Simple logo at the top of sidebar
        st.sidebar.image(str(logo_path), width=500)
        st.sidebar.markdown("---")
    else:
        # Fallback: Company name as text
        st.sidebar.markdown("### Your Company Name")
        st.sidebar.markdown("---")

def show_home_page():
    # --- Centered Logo using columns ---
    # logo_path = Path(__file__).parent / "assets" / "company_logo.png"

    # col_spacer_left, col_logo, col_spacer_right = st.columns([1, 4, 1])

    # with col_logo:
    #     if os.path.exists(logo_path):
    #         # Use fixed width to avoid pixelation
    #         st.image(str(logo_path), width=500)
    #     else:
    #         st.warning("Logo image not found. Please save it to 'streamlit/assets/logo.png'")

    # --- New: Welcoming Message and Description ---
    st.markdown(
        """
        ##
        Your AI-powered partner in helping you make better decisions and smarter predictions.
        """
    )

    st.divider()

    # --- File Upload and Default Dataset Section ---
    st.subheader("📁 Upload a dataset to generate Profiles")

    # This is the original file uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        key="home_uploader",
        help="This dataset will be used for the Profiles page."
    )

    # Original logic for processing an uploaded file
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

                    # --- RE-ADDED: Preview Dataset Section ---
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

    # This is the section that renders the button. It should be the only place.
    if st.session_state.get('file_uploaded'):
        st.divider()
        st.success("Data ready! Click 'Start Analysis' to proceed to the Profiles page.")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Start Analysis", use_container_width=True):
                st.session_state['current_page'] = "Profiles"
                st.rerun()

def handle_page_change():
    st.session_state['current_page'] = st.session_state['page_selector']

def main():
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "Home"

    # Add logo to sidebar first
    add_sidebar_logo()

    # Sidebar navigation
    st.sidebar.header("Navigation")
    page_options = ["Home", "Profiles", "Prediction"]

    # Use on_change to explicitly handle the page switch
    st.sidebar.selectbox(
        "Select Page",
        page_options,
        index=page_options.index(st.session_state['current_page']),
        key="page_selector",
        on_change=handle_page_change
    )

    # Show data source status for the profiles page
    if (
        st.session_state.get('data') is not None
        and not st.session_state['data'].empty
        and st.session_state['current_page'] != "Prediction"
    ):
        st.sidebar.success(f"📊 Profiles Data: {st.session_state['data_source']}")

    st.sidebar.markdown("---")

    # The page routing logic now depends directly on the single source of truth in session state.
    if st.session_state['current_page'] == "Home":
        show_home_page()
    elif st.session_state['current_page'] == "Profiles":
        show_profiles(st.session_state.get('data'))
    elif st.session_state['current_page'] == "Prediction":
        show_prediction()

if __name__ == "__main__":
    main()
