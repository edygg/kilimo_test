import os.path

import click

from kilimo_test.pipelines.crop_yield import download_crop_yield_data, get_running_dir_storage, read_crop_yield_data, \
    process_crop_yield_data, store_crop_yield_data, get_local_dir_storage
from kilimo_test.logging import logger


@click.command()
def execute_crop_yield_pipeline():
    logger.info("Downloading crop yield kaggle dataset")
    running_id_dir = get_running_dir_storage()
    extracted_on_dir = download_crop_yield_data(running_id_dir)

    logger.info(
        f"Crop yield dataset downloaded and extracted to %s",
        os.path.abspath(extracted_on_dir)
    )
    pesticides_df, rainfall_df, temp_df, yield_df = read_crop_yield_data(extracted_on_dir)
    crop_yield_df = process_crop_yield_data(
        pesticides_df=pesticides_df,
        rainfall_df=rainfall_df,
        temp_df=temp_df,
        yield_df=yield_df,
    )
    logger.info("%s", crop_yield_df.info(verbose=True))
    logger.info("%s", crop_yield_df.head(5))

    dst_file_path = store_crop_yield_data(
        df=crop_yield_df,
        storage_dir=get_local_dir_storage(),
    )
    logger.info(
        "Crop yield dataset stored at %s",
        os.path.abspath(dst_file_path),
    )


if __name__ == "__main__":
    execute_crop_yield_pipeline()