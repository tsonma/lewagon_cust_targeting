import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
from pathlib import Path
import io

from eda_profiles import show_profiles   # consolidated profiles page (with clustering)
from predict import show_prediction      # your existing prediction page

# ---------- Expected columns for validation ----------
REQUIRED_COLUMNS = [
    'age', 'job', 'marital', 'education', 'default', 'balance',
    'housing', 'loan', 'contact', 'day', 'month', 'campaign',
    'pdays', 'previous', 'poutcome', 'y'
]

# ---------- Data loader and validation ----------
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
    """Load and validate uploaded CSV file"""
    try:
        # Try different separators
        content = uploaded_file.getvalue().decode('utf-8')

        # First try semicolon separator (like bank-full.csv)
        try:
            df = pd.read_csv(io.StringIO(content), sep=';')
            if len(df.columns) > 1:  # Successfully parsed with semicolon
                return df
        except:
            pass

        # Try comma separator
        try:
            df = pd.read_csv(io.StringIO(content), sep=',')
            return df
        except:
            pass

        # Try tab separator
        try:
            df = pd.read_csv(io.StringIO(content), sep='\t')
            return df
        except:
            pass

        # If all fail, raise error
        raise ValueError("Could not parse CSV with common separators")

    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        return pd.DataFrame()

def show_home_page():
    """Home page with file upload functionality"""
    st.title("🏦 Investment Prediction App")
    st.markdown("Upload your customer dataset to get started.")

    # File upload section
    st.subheader("📁 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your customer dataset in CSV format"
    )

    # Handle file upload
    if uploaded_file is not None:
        with st.spinner("Loading and validating your dataset..."):
            df = load_uploaded_csv(uploaded_file)

            if not df.empty:
                # Validate structure
                is_valid, missing_cols = validate_csv_structure(df)

                if is_valid:
                    st.success(f"✅ Dataset loaded! ({len(df):,} rows, {len(df.columns)} columns)")

                    # Store in session state
                    st.session_state['data'] = df
                    st.session_state['data_source'] = f"Uploaded: {uploaded_file.name}"

                    # Show preview
                    with st.expander("Preview Dataset"):
                        st.dataframe(df.head())

                    st.success("🎉 Data ready! Use the sidebar to navigate.")

                    # Start Analysis button
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col2:
                        if st.button("Start Analysis", use_container_width=True):
                            st.session_state['current_page'] = "👥 Profiles"
                            st.rerun()

                else:
                    st.error("❌ Invalid CSV structure!")
                    st.write("Missing columns:", missing_cols)

    # Instructions when no data is loaded
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.info("Upload a CSV file to get started.")

def main():
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "🏠 Home"

    # Sidebar navigation
    st.sidebar.header("Navigation")

    # Show data source if available
    if 'data_source' in st.session_state and st.session_state['data'] is not None:
        st.sidebar.success(f"📊 Data: {st.session_state['data_source']}")
        st.sidebar.markdown("---")

    # Navigation menu - check if we actually have data loaded
    data_available = (st.session_state.get('data') is not None and
                     not st.session_state['data'].empty)

    if data_available:
        # Get the index of current page for selectbox
        page_options = ["🏠 Home", "👥 Profiles", "🔮 Prediction"]
        try:
            current_index = page_options.index(st.session_state['current_page'])
        except ValueError:
            current_index = 0  # Default to Home if current_page is invalid

        page = st.sidebar.selectbox(
            "Choose a page",
            page_options,
            index=current_index,
            key="page_selector"
        )

        # Update current page in session state
        st.session_state['current_page'] = page

    else:
        page = "🏠 Home"
        st.session_state['current_page'] = "🏠 Home"
        st.sidebar.info("📁 Please upload data first to access other pages")

    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "👥 Profiles":
        if data_available:
            show_profiles(st.session_state['data'])
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
