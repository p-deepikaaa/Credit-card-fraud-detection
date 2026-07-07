import streamlit as st
import pandas as pd

from src.data_loader import load_csv
from src.preprocessing import preprocess_data
from src.prediction import load_model, predict
from src.visualization import (
    create_pie_chart,
    create_bar_chart
)

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# ---------------------------------------------------
# Header
# ---------------------------------------------------

st.title("💳 Credit Card Fraud Detection Dashboard")

st.write(
    "Upload a transaction dataset and identify potentially fraudulent transactions using a trained Machine Learning model."
)

st.divider()

# ---------------------------------------------------
# Upload CSV
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "📂 Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Load Dataset
    df = load_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Preprocess
    prediction_df = preprocess_data(df)

    # Prediction
    predictions = predict(model, prediction_df)

    result_df = prediction_df.copy()

    result_df["Prediction"] = predictions

    result_df["Prediction"] = result_df["Prediction"].replace({
        0: "Legitimate",
        1: "Fraud"
    })

    st.divider()

    # ---------------------------------------------------
    # Metrics
    # ---------------------------------------------------

    total_transactions = len(result_df)
    fraud_transactions = (
        result_df["Prediction"] == "Fraud"
    ).sum()

    legitimate_transactions = (
        result_df["Prediction"] == "Legitimate"
    ).sum()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "💳 Total Transactions",
        total_transactions
    )

    c2.metric(
        "🚨 Fraud Transactions",
        fraud_transactions
    )

    c3.metric(
        "✅ Legitimate Transactions",
        legitimate_transactions
    )

    st.divider()

    # ---------------------------------------------------
    # Charts
    # ---------------------------------------------------

    left, right = st.columns(2)

    with left:
        st.plotly_chart(
            create_pie_chart(result_df),
            use_container_width=True
        )

    with right:
        st.plotly_chart(
            create_bar_chart(result_df),
            use_container_width=True
        )

    st.divider()

    # ---------------------------------------------------
    # Prediction Table
    # ---------------------------------------------------

    st.subheader("📊 Prediction Results")

    ordered_columns = [
        "Prediction"
    ] + [
        column
        for column in result_df.columns
        if column != "Prediction"
    ]

    st.dataframe(
        result_df[ordered_columns],
        use_container_width=True
    )

    st.divider()

    # ---------------------------------------------------
    # Download
    # ---------------------------------------------------

    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Prediction Results",
        csv,
        "fraud_predictions.csv",
        "text/csv"
    )

else:

    st.info("👆 Upload a CSV file to start fraud detection.")