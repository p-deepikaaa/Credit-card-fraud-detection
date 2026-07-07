import plotly.express as px


def create_pie_chart(df):
    return px.pie(
        df,
        names="Prediction",
        title="Fraud vs Legitimate Transactions",
        hole=0.45
    )


def create_bar_chart(df):
    return px.histogram(
        df,
        x="Prediction",
        color="Prediction",
        title="Prediction Distribution"
    )