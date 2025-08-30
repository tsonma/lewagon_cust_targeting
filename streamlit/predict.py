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

# ---------- UI for Single Client Prediction ----------
def single_client_ui(model):
    st.write("### Input Client Information")

    col1, col2 = st.columns(2)
    age = col1.number_input("Age", min_value=18, max_value=100, value=30)
    job = col2.selectbox("Job", [
        "admin.", "blue-collar", "technician", "services", "management", "retired",
        "unemployed", "self-employed", "entrepreneur", "housemaid", "student"
    ])

    col1, col2 = st.columns(2)
    balance = col1.number_input("Balance", value=5000)
    housing = col2.selectbox("Housing Loan?", ["no", "yes"])

    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        marital = col1.selectbox("Marital Status", ["married", "single", "divorced"])
        education = col2.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])

        col1, col2, col3 = st.columns(3)
        default = col1.selectbox("Credit Default?", ["yes", "no"])
        loan = col2.selectbox("Personal Loan?", ["no", "yes"])
        contact = col3.selectbox("Contact Communication type", ["cellular", "telephone", "unknown"])

        col1, col2, col3 = st.columns(3)
        month = col1.selectbox("Last Contact Month", [
            "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
        ])
        day = col2.number_input("Last Contact Day of Month", min_value=1, max_value=31, value=5)
        campaign = col3.number_input("Number of Contacts During Campaign", min_value=1, max_value=50, value=1)

        col1, col2, col3 = st.columns(3)
        pdays = col1.number_input("Days Passed Since Last Contact (-1 Means Never)", value=-1)
        previous = col2.number_input("Number of Contacts Before This Campaign", min_value=0, max_value=50, value=0)
        poutcome = col3.selectbox("Outcome of Previous Campaign", ["failure", "unknown", "success"])

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

    # Display prediction result if it exists (even after rerun)
    if "last_prediction" in st.session_state:
        st.info(f"Probability of Investing: {st.session_state['last_prediction']:.2%}")

    # Script button (only shows up if prediction has been made)
    if "last_input" in st.session_state:
        if st.button("Generate Sales Pitch"):
            df_str = st.session_state["last_input"].to_csv(index=False)

            # Extract key financial indicators for context
            client_balance = st.session_state["last_input"]["balance"].iloc[0]
            has_housing_loan = st.session_state["last_input"]["housing"].iloc[0] == "yes"
            has_personal_loan = st.session_state["last_input"]["loan"].iloc[0] == "yes"
            client_job = st.session_state["last_input"]["job"].iloc[0]

            # Create financial context for the prompt
            financial_context = f"""
            Financial Context:
            - Current account balance: {'High' if client_balance > 10000 else 'Moderate' if client_balance > 1000 else 'Limited'} liquidity position
            - Housing loan status: {'Currently has housing loan obligations' if has_housing_loan else 'No current housing loan'}
            - Personal loan status: {'Has personal loan commitments' if has_personal_loan else 'No personal loan obligations'}
            - Employment: {client_job}
            """

            system_prompt = """
            You are a highly experienced and successful bank representative,
            specialized in investments, with a proven track record of helping clients achieve
            their financial goals while always acting in their best interest.
            You are empathetic, trustworthy, and persuasive in your communication.

            Always consider the client's current financial obligations and liquidity position
            when providing advice. Be sensitive to their debt situation and cash flow constraints.
            """

            user_prompt = f"""
            Here is the customer data:
            {df_str}

            {financial_context}

            You are advising a financial advisor who works at an investment company that currently offers only one investment product.
            The advisor wants to build stronger, more personal relationships with clients during consultations. Do not talk about marital status or if they have kids.
            Please mention the field of work that client is in.

            Based on the client's financial situation (including their liquidity position and loan obligations),
            provide 3 specific conversation topics that will help the advisor:

            1. Connect personally with clients beyond just discussing the investment product
            2. Understand the client's individual financial situation and goals, considering their current financial commitments
            3. Build trust and rapport for long-term relationships while being sensitive to their financial constraints

            Format each topic as a concise bullet point with 1-2 personalized sentences.
            Be tactful and avoid directly mentioning specific account balances or loan amounts.
            Consider whether the client may have limited disposable income due to loan obligations or should focus on building emergency funds first.
            Focus on topics that any advisor can easily incorporate into their meetings, regardless of their experience level.
            """

            # Add progress bar for single client script generation
            progress_bar = st.progress(0)

            try:
                # Update progress to show we're starting the API call
                progress_bar.progress(25)

                response = openai.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=600,
                    temperature=0.7
                )

                script = response.choices[0].message.content

                # Complete the progress bar
                progress_bar.progress(100)

                st.write(script)

                # Remove the progress bar after completion
                progress_bar.empty()

            except Exception as e:
                progress_bar.empty()  # Remove progress bar on error
                st.error(f"Error generating script: {e}")

# ---------- UI for Multiple Client Prediction ----------
def bulk_csv_ui(model):
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        # Load data
        X, y = load_data(filepath=uploaded_file)
        probabilities = model.predict_proba(X)[:, 1]

        max_prospects = len(probabilities)

        # ---- Capture Filters (logic first) ----
        threshold_percent = st.session_state.get("threshold_percent", 0)
        desired_successes = st.session_state.get("desired_successes", min(1, max_prospects))

        # Convert % back to decimal
        threshold = threshold_percent / 100

        # ---- Apply filters based on threshold ----
        predictions = ["Will Invest" if p >= threshold else "Will Not Invest" for p in probabilities]
        X["Prediction"] = predictions
        X["Probability"] = probabilities

        total_customers_original = len(X)
        mask_above_threshold = X["Probability"] >= threshold
        X_filtered = X[mask_above_threshold].copy()
        probabilities_filtered = probabilities[mask_above_threshold]
        X_filtered.sort_values(by="Probability", ascending=False, inplace=True)

        # ---- Calculate Metrics ----
        if len(probabilities_filtered) > 0:
            avg_probability = probabilities_filtered.mean()
            n_prospects = len(probabilities_filtered)
            from math import comb
            prob_at_least_desired = sum(
                comb(n_prospects, k) * (avg_probability ** k) * ((1 - avg_probability) ** (n_prospects - k))
                for k in range(desired_successes, n_prospects + 1)
            )
        else:
            avg_probability = 0
            prob_at_least_desired = 0

        total_customers_filtered = len(X_filtered)

        # ---- Results Header ----
        st.write("### Results")

        # ---- Metrics Section ----
        col1, col2 = st.columns(2)
        with col1:
            delta_value = None
            if total_customers_original > total_customers_filtered:
                delta_value = f"-{total_customers_original - total_customers_filtered} filtered out"
            st.metric("Total Prospects", f"{total_customers_filtered}", delta=delta_value)
        with col2:
            st.metric(f"Probability of ≥ {desired_successes} Success{'es' if desired_successes > 1 else ''}",
                     f"{prob_at_least_desired:.1%}")

        # ---- Filters Section (rendered here, but already applied above) ----
        col1, col2 = st.columns(2)
        with col1:
            threshold_percent = st.number_input(
                "Probability Threshold (%)",
                min_value=0,
                max_value=20,
                value=threshold_percent,
                step=1,
                key="threshold_percent"
            )
            threshold = threshold_percent / 100
        with col2:
            desired_successes = st.number_input(
                "Desired # of Success",
                min_value=1,
                max_value=max_prospects,
                value=desired_successes,
                key="desired_successes"
            )

        # ---- Table Section ----
        df_display = X_filtered.copy()
        df_display["Probability"] = df_display["Probability"].apply(lambda p: f"{p:.2%}")
        st.dataframe(df_display[["name", "phone number", "Probability"]], hide_index=True)

        # ---- Download ----
        csv_download = X_filtered[["name","phone number","age","job","marital","education","balance","housing","loan","Probability"]].to_csv(index=False).encode('utf-8')
        st.download_button(label="💾 Download Predictions", data=csv_download, file_name="predictions.csv", mime="text/csv")

        # ---- Sales Pitch Generation ----
        if st.button("Generate Sales Pitch"):
            data_hash = hash(str(X_filtered.index.tolist() + X_filtered.columns.tolist()))
            scripts_key = f"scripts_{data_hash}_{threshold}"
            if scripts_key not in st.session_state:
                st.write("### Generating Personalized Scripts...")
                scripts = []

                system_prompt = """
                You are a highly experienced and successful bank representative,
                specialized in investments, with a proven track record of helping clients achieve
                their financial goals while always acting in their best interest.
                You are empathetic, trustworthy, and persuasive in your communication.
                """

                progress_bar = st.progress(0)
                for i, (idx, row) in enumerate(X_filtered.iterrows()):
                    client_balance = row["balance"]
                    has_housing_loan = row["housing"] == "yes"
                    has_personal_loan = row["loan"] == "yes"
                    client_job = row["job"]
                    client_name = row["name"]

                    financial_context = f"""
                    Financial Context for {client_name}:
                    - Liquidity: {'High' if client_balance > 10000 else 'Moderate' if client_balance > 1000 else 'Limited'}
                    - Housing loan: {'Yes' if has_housing_loan else 'No'}
                    - Personal loan: {'Yes' if has_personal_loan else 'No'}
                    - Employment: {client_job}
                    """

                    user_prompt = f"Here is the customer data for {client_name}:\n{financial_context}\nProvide 3 bullet points for conversation topics."

                    try:
                        response = openai.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            max_tokens=600,
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
                with st.expander(f"Sales Pitch for {row['name']} - {row['Probability']})"):
                    st.write(X_filtered.loc[idx,"Script"])

            csv_download_with_scripts = X_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(label="💾 Download Predictions with Sales Pitch",
                               data=csv_download_with_scripts,
                               file_name="all_predictions_with_sales_pitch.csv",
                               mime="text/csv")

# ---------- Main Prediction Page ----------
def show_prediction():
    model = load_model()

    # Sidebar selection
    mode = st.sidebar.selectbox("Select Prediction Type", ["Single Client", "Multiple Clients"])

    # Dynamic title based on selected mode
    if mode == "Single Client":
        st.title("Prediction - Single Client")
    else:
        st.title("Prediction - Multiple Clients")

    if mode == "Single Client":
        single_client_ui(model)
    else:
        bulk_csv_ui(model)
