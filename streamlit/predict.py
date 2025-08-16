import sys
import pathlib
import streamlit as st
import plotly.express as px
import pickle
import pandas as pd
import os
import openai
from dotenv import load_dotenv

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
    if pred == "Will Invest":
        return "✅"
    else:
        return "❌"

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
def bulk_csv_ui(model, threshold):
    st.write("### 📂 Upload CSV for Bulk Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:

        X, y = load_data(filepath=uploaded_file)
        probabilities = model.predict_proba(X)[:, 1]
        predictions = ["Will Invest" if p >= threshold else "Will Not Invest" for p in probabilities]

        df["Prediction"] = predictions
        df["Probability"] = [f"{p:.2%}" for p in probabilities]

        df_display = df.copy()
        df_display["Prediction"] = [color_prediction(pred) for pred in df["Prediction"]]

        st.write("### Results")

        st.write(df_display[["age", "job", "marital", "education", "balance", "housing", "loan", "Prediction", "Probability"]])

        csv_download = X.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv_download,
            file_name="predictions.csv",
            mime="text/csv"
        )

# ---------- Main Prediction Page with Tabs ----------
def show_prediction():
    st.title("Customer Investment Prediction")

    model = load_model()

    # Shared threshold slider
    threshold = st.slider("Adjust investment threshold", 0.0, 1.0, 0.5, 0.01)

    # Tabs for Single Customer vs Bulk Upload
    tab1, tab2 = st.tabs(["✏️ Single Customer", "📂 Bulk CSV Upload"])

    with tab1:
        single_client_ui(model, threshold)

    with tab2:
        bulk_csv_ui(model, threshold)
