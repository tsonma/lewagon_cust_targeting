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

def bulk_csv_ui(model, threshold):
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:

        # Load data
        X,y = load_data(filepath=uploaded_file)
        probabilities = model.predict_proba(X)[:, 1]
        predictions = ["Will Invest" if p >= threshold else "Will Not Invest" for p in probabilities]

        X["Prediction"] = predictions
        X["Probability"] = probabilities

        # FILTER CUSTOMERS BASED ON THRESHOLD
        total_customers_original = len(X)
        mask_above_threshold = X["Probability"] >= threshold
        X_filtered = X[mask_above_threshold].copy()
        probabilities_filtered = probabilities[mask_above_threshold]

        # Sort descending by probability
        X_filtered.sort_values(by="Probability", ascending=False, inplace=True)

        # User input for desired number of successes
        max_prospects = len(probabilities_filtered) if len(probabilities_filtered) > 0 else 1
        desired_successes = st.number_input(
            "How many successful investments do you want to achieve?",
            min_value=1,
            max_value=max_prospects,
            value=min(1, max_prospects),
            help="Select the number of successful investments you're targeting"
        )

        # Calculate expected number of successes and probability
        if len(probabilities_filtered) > 0:
            expected_successes = sum(probabilities_filtered)
            avg_prob = sum(probabilities_filtered) / len(probabilities_filtered)
            n_prospects = len(probabilities_filtered)

            from math import comb
            prob_at_least_desired = sum(
                comb(n_prospects, k) * (avg_prob ** k) * ((1 - avg_prob) ** (n_prospects - k))
                for k in range(desired_successes, n_prospects + 1)
            )
        else:
            prob_at_least_desired = 0
            expected_successes = 0

        # Additional metrics
        total_customers_filtered = len(X_filtered)
        predicted_investors = len(X_filtered)
        avg_probability = probabilities_filtered.mean() if len(probabilities_filtered) > 0 else 0

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Prospects", f"{total_customers_filtered}",
                     delta=f"-{total_customers_original - total_customers_filtered} filtered out")
        with col2:
            st.metric(f"Probability of ≥ {desired_successes} Success{'es' if desired_successes > 1 else ''}",
                     f"{prob_at_least_desired:.1%}")
        with col3:
            st.metric("Average Probability", f"{avg_probability:.1%}")

        # Show threshold info
        st.info(f"📊 Showing only customers with probability ≥ {threshold:.1%} | "
                f"Filtered out: {total_customers_original - total_customers_filtered} customers")

        if len(X_filtered) == 0:
            st.warning(f"⚠️ No customers meet the threshold of {threshold:.1%}. Try lowering the threshold.")
            return

        # Format for display
        df_display = X_filtered.copy()
        df_display["Prediction"] = [color_prediction(pred) for pred in df_display["Prediction"]]
        df_display["Probability"] = df_display["Probability"].apply(lambda p: f"{p:.2%}")

        st.write("### Results")
        st.dataframe(df_display[["name", "phone number", "Probability"]], hide_index=True)

        # Download button (using filtered data)
        csv_download = X_filtered[["name","phone number","age","job","marital","education","balance","housing","loan","Probability"]].to_csv(index=False).encode('utf-8')
        st.download_button(label="💾 Download Predictions", data=csv_download, file_name="predictions.csv", mime="text/csv")

        # Optional ChatGPT scripts generation (using filtered data)
        if st.button("🤖 Generate Personalized Scripts"):
            data_hash = hash(str(X_filtered.index.tolist() + X_filtered.columns.tolist()))
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
                for i, (idx, row) in enumerate(X_filtered.iterrows()):
                    customer_data = row[["age","job","marital","education","housing","loan"]].to_dict()
                    customer_str = ", ".join([f"{k}: {v}" for k,v in customer_data.items()])
                    user_prompt = f"""
                    Here is the customer data for {row['name']}:
                    {customer_str}

                    Please generate a short personalized call script (2 paragraphs) for this client,
                    highlighting their situation and suggesting why an investment opportunity might be beneficial for them.
                    Use gentle, friendly humor that's warm and respectful.
                    Use the customer's name: {row['name']}.
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
                        st.error(f"Error generating script for {row['name']}: {e}")
                        scripts.append(f"Script generation failed: {str(e)}")

                    progress_bar.progress((i+1)/len(X_filtered))

                st.session_state[scripts_key] = scripts
            else:
                st.success("Using cached scripts!")

            X_filtered["Script"] = st.session_state[scripts_key]

            st.write("#### Top 3 Prospects")
            top_3_display = df_display.head(3)
            for i, (idx, row) in enumerate(top_3_display.iterrows()):
                with st.expander(f"Script for {row['name']} ({row['Prediction']} - {row['Probability']})"):
                    st.write(X_filtered.loc[idx,"Script"])

            csv_download_with_scripts = X_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(label="💾 Download Predictions with Scripts",
                               data=csv_download_with_scripts,
                               file_name="all_predictions_with_scripts.csv",
                               mime="text/csv")

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
