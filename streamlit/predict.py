import sys
import pathlib
import streamlit as st
import pickle
import pandas as pd

# Add project root to sys.path
project_root = pathlib.Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

@st.cache_resource
def load_model():
    file_path = '../models/log_pipeline_20250807.pkl'
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

def color_prediction(pred):
    """Return HTML span with color for prediction."""
    if pred == "Will Invest":
        return f"<span style='color:green; font-weight:bold;'>✅ {pred}</span>"
    else:
        return f"<span style='color:red; font-weight:bold;'>❌ {pred}</span>"

def show_prediction():
    st.title("Customer Investment Prediction")

    # Load model once
    model = load_model()

    # Batch CSV Upload
    st.write("### 📂 Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]

        df["Prediction"] = ["Will Invest" if p == 1 else "Will Not Invest" for p in predictions]
        df["Probability"] = [f"{p:.2%}" for p in probabilities]

        # Color-code predictions
        df_display = df.copy()
        df_display["Prediction"] = [
            color_prediction(pred) for pred in df["Prediction"]
        ]

        st.write("### Results")
        st.write(df_display.to_html(escape=False), unsafe_allow_html=True)
        return  # Skip single entry mode if CSV uploaded

    # Single Prediction Form
    st.write("### ✏️ Enter Single Customer Information")

    # Main visible fields
    col1, col2 = st.columns(2)
    age = col1.number_input("Age", min_value=18, max_value=100, value=30)
    job = col2.selectbox("Job", [
        "admin.", "blue-collar", "technician", "services", "management", "retired",
        "unemployed", "self-employed", "entrepreneur", "housemaid", "student"
    ])

    col1, col2 = st.columns(2)
    balance = col1.number_input("Balance", value=0)
    housing = col2.selectbox("Has housing loan?", ["yes", "no"])

    # Advanced options
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

    # Prediction Button
    if st.button("Predict"):
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

        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.markdown("<h3 style='color:green;'>✅ Will Invest</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:red;'>❌ Will Not Invest</h3>", unsafe_allow_html=True)

        st.info(f"Probability of Investing: {proba:.2%}")
a
