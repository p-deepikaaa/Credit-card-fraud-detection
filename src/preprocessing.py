def preprocess_data(df):
    processed_df = df.copy()

    if "Class" in processed_df.columns:
        processed_df = processed_df.drop(columns=["Class"])

    return processed_df