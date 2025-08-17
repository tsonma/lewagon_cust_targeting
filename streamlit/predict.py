import sys
import pathlib
import streamlit as st
import plotly.express as px
import pickle
import pandas as pd
import os
import openai
from dotenv import load_dotenv
from src.getdata_utils import load_data
from faker import Faker

load_dotenv()
openai.api_key = os.getenv("openai_api_key")  # store your key in env var


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
def single_client_ui(model, threshold):
    st.write("### ✏️ Enter Single Customer Information")

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

    if st.button("Predict for Single Client"):
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

        proba = model.predict_proba(input_data)[0][1]

        if proba >= threshold:
            st.markdown("<h3 style='color:green;'>✅ Will Invest</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:red;'>❌ Will Not Invest</h3>", unsafe_allow_html=True)

        st.info(f"Probability of Investing: {proba:.2%}")
        st.info(f"Threshold used: {threshold:.2%}")

 # ---- ChatGPT Integration ----
        df_str = input_data.to_csv(index=False)

        system_prompt = """
        You are a highly experienced and successful bank representative,
        specialized in investments, with a proven track record of helping clients achieve
        their financial goals while always acting in their best interest.
        You are empathetic, trustworthy, and persuasive in your communication. The name of our bank is 'LeWagon'.
        """

        user_prompt = f"""
        Here is the customer data:
        {df_str}

        Please generate a short personalized call script (2 paragraphs) for this client,
        highlighting their situation and suggesting why an investment is a good fit.
        Add a touch of humor to keep it engaging. Can you please assign a random name to the customer?
        """

        try:
            response = openai.chat.completions.create(
                model="gpt-4.1-mini",  # faster + cheaper
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300
            )
            script = response.choices[0].message.content
            st.subheader("📞 Personalized Call Script")
            st.write(script)

        except Exception as e:
            st.error(f"Error generating script: {e}")


# ---------- UI for Bulk CSV Prediction ----------

# ---------- UI for Bulk CSV Prediction ----------

def bulk_csv_ui(model, threshold):
    st.write("### 📂 Upload CSV for Bulk Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:

        # Load data
        X, y = load_data(filepath=uploaded_file)
        probabilities = model.predict_proba(X)[:, 1]
        predictions = ["Will Invest" if p >= threshold else "Will Not Invest" for p in probabilities]

        X["Prediction"] = predictions
        X["Probability"] = probabilities  # keep numeric for sorting

        # --- Generate random names & phone numbers ---
        fake = Faker()
        X["Name"] = [fake.name() for _ in range(len(X))]
        X["Phone"] = [f"(514)-{fake.random_int(100, 999)}-{fake.random_int(1000, 9999)}" for _ in range(len(X))]

        # Sort descending by probability
        X.sort_values(by="Probability", ascending=False, inplace=True)

        # Calculate probability of convincing at least 1 customer
        # P(at least 1 success) = 1 - P(all failures)
        prob_no_investment = 1
        for prob in probabilities:
            prob_no_investment *= (1 - prob)

        prob_at_least_one = 1 - prob_no_investment

        # Calculate some additional metrics
        total_customers = len(X)
        predicted_investors = sum([1 for pred in predictions if pred == "Will Invest"])
        avg_probability = probabilities.mean()

        # Display metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Probability of Success",
                f"{prob_at_least_one:.1%}"            )

        with col2:
            st.metric(
                "Investors Above Threshold",
                f"{predicted_investors}/{total_customers}",
                f"{(predicted_investors/total_customers):.1%}"
            )

        with col3:
            st.metric(
                "Average Probability",
                f"{avg_probability:.1%}"
            )

        # Format the probability for display
        df_display = X.copy()
        df_display["Prediction"] = [color_prediction(pred) for pred in df_display["Prediction"]]
        df_display["Probability"] = df_display["Probability"].apply(lambda p: f"{p:.2%}")

        # Display results
        st.write("### 📊 Results")
        st.dataframe(
            df_display[["Name", "Phone", "Prediction", "Probability"]],
            hide_index=True
        )

        # Download button
        csv_download = X[["Name", "Phone", "age", "job", "marital", "education", "balance", "housing", "loan", "Prediction", "Probability"]].to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv_download,
            file_name="predictions.csv",
            mime="text/csv"
        )

        # --- Optional ChatGPT Integration for Script Generation ---
        if st.button("🤖 Generate Personalized Call Scripts", type="primary"):
            # Add custom CSS for green button
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

            st.write("### 🤖 Generating personalized call scripts...")
            scripts = []

            system_prompt = """
            You are a friendly and professional bank representative from 'LeWagon',
            specialized in investments, with a genuine desire to help clients achieve
            their financial goals. You are empathetic, trustworthy, and warm in your communication.
            Use light, friendly humor that's appropriate and respectful - think gentle wit rather than
            anything that could be perceived as making fun of the customer. Keep the tone positive,
            encouraging, and supportive throughout.
            """

            # Generate script for ALL customers
            progress_bar = st.progress(0)

            for i, (idx, row) in enumerate(X.iterrows()):
                # Create customer data string for this specific row (excluding sensitive balance info)
                customer_data = row[["age", "job", "marital", "education", "housing", "loan"]].to_dict()
                customer_str = ", ".join([f"{k}: {v}" for k, v in customer_data.items()])

                user_prompt = f"""
                Here is the customer data for {row['Name']}:
                {customer_str}

                Please generate a short personalized call script (2 paragraphs) for this client,
                highlighting their situation and suggesting why an investment opportunity might be beneficial for them.
                Use gentle, friendly humor that's warm and respectful - avoid anything that could sound
                condescending or make light of their financial situation. The humor should be lighthearted
                and build rapport, not critique the customer. Keep it professional yet personable.
                Use the customer's name: {row['Name']}.

                IMPORTANT: Do not mention account balances, specific dollar amounts, or financial details.
                Focus on their life situation, career, and general financial wellness.
                """

                try:
                    response = openai.chat.completions.create(
                        model="gpt-4o-mini",  # Fixed model name
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=300,
                        temperature=0.7  # Add some creativity variation
                    )
                    script = response.choices[0].message.content
                    scripts.append(script)

                except Exception as e:
                    st.error(f"Error generating script for {row['Name']}: {e}")
                    scripts.append(f"Script generation failed: {str(e)}")

                # Update progress bar
                progress_bar.progress((i + 1) / len(X))

            # Add scripts to ALL customers
            X["Script"] = scripts

            # Show scripts for top 3 prospects only on Streamlit
            st.write("#### 📞 Personalized Call Scripts (Top 3 Prospects)")
            top_3_display = df_display.head(3)
            for i, (idx, row) in enumerate(top_3_display.iterrows()):
                with st.expander(f"Script for {row['Name']} ({row['Prediction']} - {row['Probability']})"):
                    st.write(X.loc[idx, "Script"])

            # Download button with scripts for ALL customers
            csv_download_with_scripts = X.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download All Predictions with Scripts as CSV",
                data=csv_download_with_scripts,
                file_name="all_predictions_with_scripts.csv",
                mime="text/csv"
            )

# ---------- Main Prediction Page with Tabs ----------

def show_prediction():
    st.title("Customer Investment Prediction")

    model = load_model()

    # --- Session state to remember selected tab ---
    if "selected_tab" not in st.session_state:
        st.session_state.selected_tab = 0  # default to first tab

    # Tab switcher (emulates tabs but preserves selection on rerun)
    tab_labels = ["✏️ Single Customer", "📂 Bulk CSV Upload"]
    st.session_state.selected_tab = st.radio(
        "Choose view:",
        range(len(tab_labels)),
        format_func=lambda i: tab_labels[i],
        index=st.session_state.selected_tab
    )

    # Show content based on selected tab
    if st.session_state.selected_tab == 0:
        # Threshold slider only for this tab
        threshold = st.slider("Adjust investment threshold", 0.0, 0.10, 0.05, 0.10)
        single_client_ui(model, threshold)
    else:
        # Threshold slider only for bulk CSV
        threshold = st.slider("Adjust investment threshold", 0.0, 0.10, 0.05, 0.10)
        bulk_csv_ui(model, threshold)
