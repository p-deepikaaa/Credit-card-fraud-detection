import joblib
from src.config import MODEL_PATH


def load_model():
    return joblib.load(MODEL_PATH)


def predict(model, dataframe):
    return model.predict(dataframe)