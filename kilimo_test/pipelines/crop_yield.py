import io
import os
import requests
import zipfile
import pandas as pd
import duckdb

from uuid import uuid4
from typing import Optional


CROP_YIELD_FILE_URLS = [
    "https://www.kaggle.com/api/v1/datasets/download/patelris/crop-yield-prediction-dataset"
]


LOCAL_DIR_STORAGE = os.getenv("LOCAL_DIR_STORAGE") or "/.data"


def get_crop_yield_files_urls():
    return CROP_YIELD_FILE_URLS


def get_local_dir_storage():
    return LOCAL_DIR_STORAGE


def get_running_dir_storage(running_id: Optional[str] = None):
    running_id_dir = running_id if running_id else str(uuid4())

    return os.path.join(
        get_local_dir_storage(),
        running_id_dir,
    )


def download_crop_yield_data(storage_dir):
    files_to_download = get_crop_yield_files_urls()

    for file_url in files_to_download:
        response = requests.get(file_url, stream=True)  # Stream cuz is a zip file
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        zip_file.extractall(storage_dir)

    return storage_dir


def read_crop_yield_data(storage_dir):
    return (
        pd.read_csv(os.path.join(storage_dir, "pesticides.csv")),
        pd.read_csv(os.path.join(storage_dir, "rainfall.csv")),
        pd.read_csv(os.path.join(storage_dir, "temp.csv")),
        pd.read_csv(os.path.join(storage_dir, "yield.csv")),
    )


def process_crop_yield_data(
        pesticides_df: pd.DataFrame,
        rainfall_df: pd.DataFrame,
        temp_df: pd.DataFrame,
        yield_df: pd.DataFrame,
) -> pd.DataFrame:
    all_dfs = [
        pesticides_df,
        rainfall_df,
        temp_df,
        yield_df,
    ]

    all_dfs = [df.copy(deep=True) for df in all_dfs]

    for df in all_dfs:
        df.dropna(inplace=True)

    cleaned_pesticides_df, cleaned_rainfall_df, cleaned_temp_df, cleaned_yield_df = all_dfs

    crop_yield_df: pd.DataFrame = duckdb.sql(
    """
        WITH 
          temperatues_by_country AS (
            SELECT
              t.year
              , t.country
              , t.avg_temp AS avg_temp_in_celsius
            FROM cleaned_temp_df AS t
          )
          , pesticides_by_country AS (
            SELECT
              p.Area AS country
              , p.Year AS year
              , p.Value AS pesticide_tonnes
            FROM cleaned_pesticides_df AS p
          )
          , rainfall_by_country AS (
            SELECT
              r." Area" AS country
              , r.Year AS year
              , r.average_rain_fall_mm_per_year
            FROM cleaned_rainfall_df AS r
          )
          , yields AS (
            SELECT
              y.Area AS country
              , y.Year AS year
              , y.Item AS element
              , y.Value AS hg_ha_yield
            FROM cleaned_yield_df AS y
          )
        SELECT
          y.country
          , y.year
          , y.element
          , y.hg_ha_yield
          , rc.average_rain_fall_mm_per_year
          , pc.pesticide_tonnes
          , tc.avg_temp_in_celsius
        FROM yields AS y
          LEFT JOIN rainfall_by_country AS rc ON (y.country = rc.country AND y.year = rc.year)
          LEFT JOIN pesticides_by_country AS pc ON (y.country = pc.country AND y.year = pc.year)
          LEFT JOIN temperatues_by_country AS tc ON (y.country = tc.country AND y.year = tc.year)
    """
    ).df()

    crop_yield_df.dropna(inplace=True)

    # Fix col types
    crop_yield_df["country"] = crop_yield_df["country"].astype("category")
    crop_yield_df["year"] = crop_yield_df["year"].astype("int16")
    crop_yield_df["element"] = crop_yield_df["element"].astype("category")
    crop_yield_df["hg_ha_yield"] = crop_yield_df["hg_ha_yield"].astype("int32")
    crop_yield_df["pesticide_tonnes"] = crop_yield_df["pesticide_tonnes"].astype("float")
    crop_yield_df["avg_temp_in_celsius"] = crop_yield_df["avg_temp_in_celsius"].astype("float")

    def str_to_int(x):
        try:
            return int(x)
        except:
            return -1

    crop_yield_df["average_rain_fall_mm_per_year"] = crop_yield_df["average_rain_fall_mm_per_year"].apply(str_to_int).astype("int32")
    crop_yield_df = crop_yield_df[crop_yield_df["average_rain_fall_mm_per_year"] != -1].copy(deep=True)

    return crop_yield_df


def store_crop_yield_data(df: pd.DataFrame, storage_dir: str):
    file_path = os.path.join(storage_dir, "crop_yield_processed.parquet")
    df.to_parquet(file_path, index=False)
    return file_path


def get_crop_yield_processed_data(storage_dir: str):
    return pd.read_parquet(os.path.join(storage_dir, "crop_yield_processed.parquet"))