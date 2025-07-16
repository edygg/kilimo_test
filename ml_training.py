import click

from kilimo_test.ml.training import save_corr_matrix, get_models_local_storage, get_training_and_test_sets, \
    training_random_forest_model, training_linear_regression_model, save_metrics_for_models, save_model_to_pkl
from kilimo_test.pipelines.crop_yield import get_crop_yield_processed_data, get_local_dir_storage
from kilimo_test.logging import logger


@click.command()
def execute_ml_training():
    logger.info("Loading crop yield dataset")
    crop_yield_df = get_crop_yield_processed_data(
        storage_dir=get_local_dir_storage()
    )
    logger.info("%s", crop_yield_df.info(verbose=True))
    save_corr_matrix(
        crop_yield_df,
        get_models_local_storage()
    )

    logger.info("Getting training and test sets")
    x_train, x_test, y_train, y_test = get_training_and_test_sets(crop_yield_df)

    logger.info("Training random forest model")
    rf_model = training_random_forest_model(
        x_train,
        y_train,
    )

    logger.info("Training linear regression model")
    ln_model = training_linear_regression_model(
        x_train,
        y_train,
    )

    logger.info("Calc test results for metrics")
    rf_model_predictions = rf_model.predict(x_test)
    ln_model_predictions = ln_model.predict(x_test)

    logger.info("Saving metrics for models")
    save_metrics_for_models(
        y_test=y_test,
        random_forest_y_predict=rf_model_predictions,
        linear_regression_model_y_predict=ln_model_predictions,
        storage_dir=get_models_local_storage(),
    )

    logger.info("Saving models")
    save_model_to_pkl(
        rf_model,
        "random_forest_model",
        get_models_local_storage(),
    )
    save_model_to_pkl(
        ln_model,
        "linear_regression_model",
        get_models_local_storage(),
    )



if __name__ == "__main__":
    execute_ml_training()