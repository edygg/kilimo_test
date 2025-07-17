import os
import pickle
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


class AvailableModels:
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"


def get_model(model_name: str, storage_dir: str) -> RandomForestRegressor | LinearRegression:
    if model_name == AvailableModels.RANDOM_FOREST:
        file_path = os.path.join(storage_dir, "random_forest_model.pkl")
    elif model_name == AvailableModels.LINEAR_REGRESSION:
        file_path = os.path.join(storage_dir, "linear_regression_model.pkl")
    else:
        raise ValueError(f"Model {model_name} is not available.")

    with open(file_path, "rb") as f:
        model = pickle.load(f)
        return model


def get_label_encoder(storage_dir: str) -> LabelEncoder:
    file_path = os.path.join(storage_dir, "label_encoder.pkl")
    with open(file_path, "rb") as f:
        label_encoder = pickle.load(f)
        return label_encoder


def get_predict_obj(payload: dict) -> pd.DataFrame:
    return pd.DataFrame([
        payload,
    ])