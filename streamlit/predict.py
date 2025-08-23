import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pathlib
import streamlit as st
import plotly.express as px
import pickle
import pandas as pd
import openai
from dotenv import load_dotenv
from src.getdata_utils import load_data
from faker import Faker

load_dotenv()

openai.api_key = os.getenv("openai_api_key")
# openai.api_key = st.secrets["openai_api_key"]

# Add project root to sys.path
project_root = pathlib.Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

@st.cache_resource
def load_model():
    """Load the trained model once and cache it."""
    file_path = project_root / "models" / "log_pipeline_20250809.pkl"
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

def color_prediction(pred):
    """Return only check or cross depending on prediction."""
    return "✅" if pred == "Will Invest" else "❌"

# ---------- UI for Single Client Prediction ----------
def single_client_ui(model):
    st.write("### Input Client Information")

    # Client name input (for script personalization only)
    client_name = st.text_input("Client Name", placeholder="Enter client's name for personalized script")

    col1, col2 = st.columns(2)
    age = col1.number_input("Age", min_value=18, max_value=100, value=30)
    job = col2.selectbox("Job", [
        "admin.", "blue-collar", "technician", "services", "management", "retired",
        "unemployed", "self-employed", "entrepreneur", "housemaid", "student"
    ])

    col1, col2 = st.columns(2)
    balance = col1.number_input("Balance", value=0)
    housing = col2.selectbox("Has housing loan?", ["yes", "no"])

    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        marital = col1.selectbox("Marital Status", ["married", "single", "divorced"])
        education = col2.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])

        col1, col2, col3 = st.columns(3)
        default = col1.selectbox("Has credit default?", ["yes", "no"])
        loan = col2.selectbox("Has personal loan?", ["yes", "no"])
        contact = col3.selectbox("Contact communication type", ["cellular", "telephone", "unknown"])

        col1, col2, col3 = st.columns(3)
        month = col1.selectbox("Last contact month", [
            "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
        ])
        day = col2.number_input("Last contact day of month", min_value=1, max_value=31, value=5)
        campaign = col3.number_input("Number of contacts during campaign", min_value=1, max_value=50, value=1)

        col1, col2, col3 = st.columns(3)
        pdays = col1.number_input("Days passed since last contact (-1 means never)", value=-1)
        previous = col2.number_input("Number of contacts before this campaign", min_value=0, max_value=50, value=0)
        poutcome = col3.selectbox("Outcome of previous campaign", ["failure", "unknown", "success"])

    # Store inputs
    input_data = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }])

    # Prediction button
    if st.button("Predict for Single Client"):
        proba = model.predict_proba(input_data)[0][1]

        # Store input data, prediction result, AND client name in session_state
        st.session_state["last_input"] = input_data
        st.session_state["last_prediction"] = proba
        st.session_state["client_name"] = client_name

    # Display prediction result if it exists (even after rerun)
    if "last_prediction" in st.session_state:
        st.info(f"Probability of Investing: {st.session_state['last_prediction']:.2%}")

    # Script button (only shows up if prediction has been made)
    if "last_input" in st.session_state:
        if st.button("Generate Personalized Script"):
            df_str = st.session_state["last_input"].to_csv(index=False)

            system_prompt = """
            You are a highly experienced and successful bank representative,
            specialized in investments, with a proven track record of helping clients achieve
            their financial goals while always acting in their best interest.
            You are empathetic, trustworthy, and persuasive in your communication.
            The name of our bank is 'LeWagon'.
            """

            user_prompt = f"""
            Here is the customer data:
            {df_str}

            Client name: {st.session_state.get('client_name', 'N/A')}

            Please generate a short personalized call script (2 paragraphs) for this client,
            highlighting their situation and suggesting why an investment is a good fit.
            If a client name is provided, please use it naturally in the script.
            """

            try:
                response = openai.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=300
                )
                script = response.choices[0].message.content
                st.subheader("Personalized Script")
                st.write(script)

            except Exception as e:
                st.error(f"Error generating script: {e}")

# ---------- UI for Bulk CSV Prediction ----------
def bulk_csv_ui(model, threshold):
    uploaded_file = st.file_uploader("Upload a CSV file for the Prediction of Multiple Clients", type=["csv"])
    if uploaded_file:

        # Load data
        X, y = load_data(filepath=uploaded_file)
        probabilities = model.predict_proba(X)[:, 1]
        predictions = ["Will Invest" if p >= threshold else "Will Not Invest" for p in probabilities]

        X["Prediction"] = predictions
        X["Probability"] = probabilities  # keep numeric for sorting

        # --- Generate random names & phone numbers (CACHED) ---
        data_hash = hash(str(X.index.tolist() + X.columns.tolist()))
        if f"names_{data_hash}" not in st.session_state:
            fake = Faker()
            fake.seed_instance(42)
            st.session_state[f"names_{data_hash}"] = [fake.name() for _ in range(len(X))]
            st.session_state[f"phones_{data_hash}"] = [f"(514)-{fake.random_int(100, 999)}-{fake.random_int(1000, 9999)}" for _ in range(len(X))]

        X["Name"] = st.session_state[f"names_{data_hash}"]
        X["Phone"] = st.session_state[f"phones_{data_hash}"]

        # Sort descending by probability
        X.sort_values(by="Probability", ascending=False, inplace=True)

        # Probability of at least one success
        prob_no_investment = 1
        for prob in probabilities:
            prob_no_investment *= (1 - prob)
        prob_at_least_one = 1 - prob_no_investment

        # Additional metrics
        total_customers = len(X)
        predicted_investors = sum([1 for pred in predictions if pred == "Will Invest"])
        avg_probability = probabilities.mean()

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Probability of Success", f"{prob_at_least_one:.1%}")
        with col2:
            st.metric("Investors Above Threshold", f"{predicted_investors}/{total_customers}")
        with col3:
            st.metric("Average Probability", f"{avg_probability:.1%}")

        # Format for display
        df_display = X.copy()
        df_display["Prediction"] = [color_prediction(pred) for pred in df_display["Prediction"]]
        df_display["Probability"] = df_display["Probability"].apply(lambda p: f"{p:.2%}")

        st.write("### 🎯 Results")
        st.dataframe(df_display[["Name", "Phone", "Probability"]], hide_index=True)

        # Download button
        csv_download = X[["Name","Phone","age","job","marital","education","balance","housing","loan","Probability"]].to_csv(index=False).encode('utf-8')
        st.download_button(label="💾 Download Predictions", data=csv_download, file_name="predictions.csv", mime="text/csv")

        # Optional ChatGPT scripts generation
        if st.button(" Generate Personalized Scripts", type="primary"):
            st.markdown("""
            <style>
            div.stButton > button:first-child {
                background-color: #28a745;
                color: white;
                border-color: #28a745;
            }
            div.stButton > button:first-child:hover {
                background-color: #218838;
                border-color: #1e7e34;
            }
            </style>
            """, unsafe_allow_html=True)

            scripts_key = f"scripts_{data_hash}_{threshold}"
            if scripts_key not in st.session_state:
                st.write("### Generating Personalized Scripts...")
                scripts = []

                system_prompt = """
                You are a friendly and professional bank representative from 'LeWagon',
                specialized in investments, with a genuine desire to help clients achieve
                their financial goals. You are empathetic, trustworthy, and warm in your communication.
                Use light, friendly humor that's appropriate and respectful - think gentle wit rather than
                anything that could be perceived as making fun of the customer. Keep the tone positive,
                encouraging, and supportive throughout.
                """

                progress_bar = st.progress(0)
                for i, (idx, row) in enumerate(X.iterrows()):
                    customer_data = row[["age","job","marital","education","housing","loan"]].to_dict()
                    customer_str = ", ".join([f"{k}: {v}" for k,v in customer_data.items()])
                    user_prompt = f"""
                    Here is the customer data for {row['Name']}:
                    {customer_str}

                    Please generate a short personalized call script (2 paragraphs) for this client,
                    highlighting their situation and suggesting why an investment opportunity might be beneficial for them.
                    Use gentle, friendly humor that's warm and respectful.
                    Use the customer's name: {row['Name']}.
                    IMPORTANT: Do not mention account balances, specific dollar amounts, or financial details.
                    """

                    try:
                        response = openai.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            max_tokens=300,
                            temperature=0.7
                        )
                        scripts.append(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"Error generating script for {row['Name']}: {e}")
                        scripts.append(f"Script generation failed: {str(e)}")

                    progress_bar.progress((i+1)/len(X))

                st.session_state[scripts_key] = scripts
            else:
                st.success("Using cached scripts!")

            X["Script"] = st.session_state[scripts_key]

            st.write("#### Top 3 Prospects")
            top_3_display = df_display.head(3)
            for i, (idx, row) in enumerate(top_3_display.iterrows()):
                with st.expander(f"Script for {row['Name']} ({row['Prediction']} - {row['Probability']})"):
                    st.write(X.loc[idx,"Script"])

            csv_download_with_scripts = X.to_csv(index=False).encode('utf-8')
            st.download_button(label="💾 Download ALL Predictions with Scripts",
                               data=csv_download_with_scripts,
                               file_name="all_predictions_with_scripts.csv",
                               mime="text/csv")

        if st.button("🗑️ Clear Cached Data"):
            keys_to_remove = [key for key in st.session_state.keys() if str(data_hash) in key]
            for key in keys_to_remove:
                del st.session_state[key]
            st.success("Cached data cleared! Refresh to generate new names.")
            st.experimental_rerun()

# ---------- Main Prediction Page ----------
def show_prediction():
    model = load_model()

    # Sidebar selection
    mode = st.sidebar.selectbox("Select Mode", ["Single Client", "Multiple Clients"])

    # Dynamic title based on selected mode
    if mode == "Single Client":
        st.title("Prediction - Single Client")
    else:
        st.title("Prediction - Multiple Clients")

    # Only show threshold slider for Multiple Clients
    threshold = None
    if mode == "Multiple Clients":
        threshold = st.sidebar.slider(
            "Adjust investment threshold",
            0.0, 0.10, 0.035, 0.01
        )

    if mode == "Single Client":
        single_client_ui(model)
    else:
        bulk_csv_ui(model, threshold)
