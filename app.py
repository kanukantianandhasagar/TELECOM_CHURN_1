# ============================================================
# TELECOM CUSTOMER CHURN PREDICTION
# Developed by K. Anandha Sagar
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📱",
    layout="wide"
)

# ------------------------------------------------------------
# Load Model
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("telecom_churn_model.pkl")

model = load_model()

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

@st.cache_data
def load_dataset():
    return pd.read_csv("telecom_churn_cleaned_for_project.csv")

df = load_dataset()

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.title("📱 Navigation")

page = st.sidebar.selectbox(
    "Select Page",
    [
        "🏠 Home",
        "📊 Dataset",
        "🤖 Prediction",
        "👨‍💻 About"
    ]
)
# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.title("📱 Telecom Customer Churn Prediction")

    st.markdown("---")

    st.write("""
This application predicts whether a telecom customer is likely to **Churn** or **Stay**
using a Machine Learning model.

### 🚀 Project Features

✅ Customer Churn Prediction

✅ Data Cleaning & Preprocessing

✅ Exploratory Data Analysis (EDA)

✅ Feature Engineering

✅ Machine Learning Model Training

✅ Model Performance Comparison

✅ Multiple Machine Learning Algorithms

✅ Real-Time Customer Churn Prediction

✅ Interactive Streamlit Dashboard

✅ Telecom Customer Dataset Analysis

✅ User-Friendly Interface

---

### 🛠 Technologies Used

🐍 Python

📊 Pandas

🔢 NumPy

📈 Matplotlib

📉 Seaborn

🤖 Scikit-Learn

💾 Joblib

🌐 Streamlit

📓 Google Colab

🐙 GitHub

☁ Streamlit Community Cloud

📂 CSV Dataset

---

### 📋 Project Workflow

1️⃣ Data Collection

2️⃣ Data Cleaning

3️⃣ Data Preprocessing

4️⃣ Exploratory Data Analysis

5️⃣ Feature Engineering

6️⃣ Train-Test Split

7️⃣ Model Training

8️⃣ Model Evaluation

9️⃣ Model Selection

🔟 Model Deployment

Use the sidebar to navigate through the application.

### Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Joblib

Use the sidebar to navigate through the application.
""")

    st.info("Navigate using the left sidebar.")
# ============================================================
# DATASET PAGE
# ============================================================

elif page == "📊 Dataset":

    st.title("📊 Telecom Dataset")

    st.write("Dataset Shape")

    st.write(df.shape)

    st.write("Dataset Preview")

    st.dataframe(df.head(20))

    st.write("Column Information")

    st.dataframe(pd.DataFrame(df.dtypes, columns=["Data Type"]))

    st.write("Missing Values")

    st.dataframe(df.isnull().sum())
# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "🤖 Prediction":

    st.title("🤖 Customer Churn Prediction")

    st.markdown("### Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox("Gender", ["Male", "Female"])

        senior = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=100,
            value=12
        )

        phone = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    with col2:

        protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )

        total = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=1000.0
        )
    st.markdown("---")

    if st.button("🔍 Predict Churn", use_container_width=True):

        input_data = pd.DataFrame({

            "gender":[gender],
            "SeniorCitizen":[senior],
            "Partner":[partner],
            "Dependents":[dependents],
            "tenure":[tenure],
            "PhoneService":[phone],
            "MultipleLines":[multiple],
            "InternetService":[internet],
            "OnlineSecurity":[security],
            "OnlineBackup":[backup],
            "DeviceProtection":[protection],
            "TechSupport":[support],
            "StreamingTV":[tv],
            "StreamingMovies":[movies],
            "Contract":[contract],
            "PaperlessBilling":[paperless],
            "PaymentMethod":[payment],
            "MonthlyCharges":[monthly],
            "TotalCharges":[total]

        })

        prediction = model.predict(input_data)[0]

        try:
            probability = model.predict_proba(input_data)[0]
            churn_probability = probability[1] * 100
            stay_probability = probability[0] * 100
        except Exception:
            churn_probability = None
            stay_probability = None

        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("⚠️ Customer is Likely to Churn")

        else:

            st.success("✅ Customer is Likely to Stay")

        if churn_probability is not None:

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Stay Probability",
                    f"{stay_probability:.2f}%"
                )

            with col2:
                st.metric(
                    "Churn Probability",
                    f"{churn_probability:.2f}%"
                )

        st.markdown("---")

        st.write("Customer Details Used For Prediction")

        st.dataframe(input_data)
# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "👨‍💻 About":

    st.title("👨‍💻 About Project")

    st.markdown("""
## Telecom Customer Churn Prediction

### Project Objective

The objective of this project is to predict whether a telecom customer is likely to leave the company (churn) using Machine Learning.

### 🤖 Machine Learning Models Evaluated

✔ Logistic Regression

✔ K-Nearest Neighbors (KNN)

✔ Decision Tree Classifier

✔ Random Forest Classifier

✔ Support Vector Machine (SVM)

### 🏆 Final Selected Model

✔ Logistic Regression

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Joblib

### Dataset

Telecom Customer Churn Dataset

### Developed By

**Nitish singh Rajput**

M.Sc Data Science

University College of Science,
Saifabad, Hyderabad
""")

    st.success("Thank you for using the Telecom Customer Churn Prediction App.")
