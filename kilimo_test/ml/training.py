import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

from kilimo_test.logging import logger


MODELS_LOCAL_STORAGE = os.getenv("MODELS_LOCAL_STORAGE") or "/.models"


def get_models_local_storage():
    return MODELS_LOCAL_STORAGE


def df_label_encoded(df: pd.DataFrame):
    categorical_columns = df.select_dtypes(include=['category']).columns.tolist()
    label_encoder = LabelEncoder()
    df_copy = df.copy(deep=True)

    all_categories = np.concatenate(
        [df_copy[column].unique() for column in categorical_columns]
    )
    label_encoder.fit(all_categories)

    for column in categorical_columns:
        df_copy[column] = label_encoder.transform(df_copy[column])

    return label_encoder, df_copy


def save_label_encoder(label_encoder, storage_dir: str):
    file_path = os.path.join(storage_dir, "label_encoder.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(label_encoder, f, pickle.HIGHEST_PROTOCOL)


def save_corr_matrix(df: pd.DataFrame, storage_dir: str):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='PuOr')
    plt.tight_layout()
    plt.savefig(os.path.join(storage_dir, "correlation_heatmap.png"), dpi=300)
    plt.close()


def get_training_and_test_sets(df: pd.DataFrame):
    x = df.drop("hg_ha_yield", axis=1)  # Features
    y = df["hg_ha_yield"]  # Goal variable

    # Divide the test and train data
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42  # Put a literal for replicate results
    )

    return x_train, x_test, y_train, y_test


def evaluate_model(y_true, y_pred, model_type):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "model_type": model_type,
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R^2_SCORE": round(r2, 4)
    }


def training_random_forest_model(x_train, y_train):
    # Hyperparameters optimization
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    random_forest_model = RandomForestRegressor(
        random_state=42 # Put a literal for replicate results
    )

    random_forest_grid_search = GridSearchCV(
        estimator=random_forest_model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    # Train the model
    random_forest_grid_search.fit(x_train, y_train)

    # Result best model found
    best_random_forest_model = random_forest_grid_search.best_estimator_

    logger.info("Best hyperparameters for Random Forest model")
    logger.info(random_forest_grid_search.best_params_)

    return best_random_forest_model


def training_linear_regression_model(x_train, y_train):
    linear_regression_model = LinearRegression()

    # Train the model
    linear_regression_model.fit(x_train, y_train)

    return linear_regression_model


def get_predictions(model, x_test):
    return model.predict(x_test)


def save_metrics_for_models(
        y_test,
        random_forest_y_predict,
        linear_regression_model_y_predict,
        storage_dir: str
):
    model_evaluation_results = [
        evaluate_model(y_test, random_forest_y_predict, "random_forest_model"),
        evaluate_model(y_test, linear_regression_model_y_predict, "linear_regression_model")
    ]

    results_df = pd.DataFrame(model_evaluation_results)
    results_df.to_csv(os.path.join(storage_dir, "model_evaluation_results.csv"), index=False)

    metrics_to_plot = ["MSE", "RMSE", "MAE", "R^2_SCORE"]

    for metric in metrics_to_plot:
        plt.figure(figsize=(6, 4))
        plt.bar(results_df["model_type"], results_df[metric], alpha=0.7)
        plt.title(f"Model comparison - {metric}")
        plt.ylabel(metric)
        plt.xticks(rotation=15)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(storage_dir, f"model_comparison_{metric.lower()}.png"), dpi=300)
        plt.close()


def save_model_to_pkl(model, model_name: str, storage_dir: str):
    model_file_path = os.path.join(storage_dir, f"{model_name}.pkl")

    with open(model_file_path, "wb") as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
