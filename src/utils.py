"""
Project : Predictive Analytics Using Historical Data
File    : utils.py
Description : Utility Functions
"""

import os
import time
import logging
from pathlib import Path

import joblib
import pandas as pd


# ==========================================
# LOGGER CONFIGURATION
# ==========================================

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)

logger = logging.getLogger(__name__)


# ==========================================
# DIRECTORY FUNCTIONS
# ==========================================

def create_directory(directory):

    directory = Path(directory)

    if not directory.exists():

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        logger.info(
            f"Directory created : {directory}"
        )

    else:

        logger.info(
            f"Directory already exists : {directory}"
        )


# ==========================================
# FILE VALIDATION
# ==========================================

def file_exists(file_path):

    file_path = Path(file_path)

    return file_path.exists()


def validate_csv(file_path):

    if not file_exists(file_path):

        raise FileNotFoundError(

            f"{file_path} not found."

        )

    if file_path.suffix != ".csv":

        raise ValueError(

            "Only CSV files are supported."

        )

    return True


# ==========================================
# CSV OPERATIONS
# ==========================================

def load_csv(file_path):

    validate_csv(
        Path(file_path)
    )

    dataframe = pd.read_csv(
        file_path
    )

    logger.info(
        "CSV Loaded Successfully"
    )

    return dataframe


def save_csv(
        dataframe,
        output_path
):

    dataframe.to_csv(

        output_path,

        index=False

    )

    logger.info(

        f"CSV Saved : {output_path}"

    )


# ==========================================
# DATASET INFORMATION
# ==========================================

def dataset_summary(dataframe):

    print("\nDataset Summary")

    print("-" * 50)

    print(

        dataframe.describe()

    )


def dataset_shape(dataframe):

    rows, columns = dataframe.shape

    print(

        f"Rows : {rows}"

    )

    print(

        f"Columns : {columns}"

    )


def dataset_columns(dataframe):

    print("\nColumns\n")

    for column in dataframe.columns:

        print(column)


# ==========================================
# DATA VALIDATION
# ==========================================

def missing_values(dataframe):

    print("\nMissing Values\n")

    print(

        dataframe.isnull().sum()

    )


def duplicate_values(dataframe):

    duplicates = dataframe.duplicated().sum()

    print(

        f"\nDuplicate Rows : {duplicates}"

    )


# ==========================================
# TIMER
# ==========================================

class ExecutionTimer:

    def __init__(self):

        self.start_time = None

    def start(self):

        self.start_time = time.time()

    def stop(self):

        elapsed = (

            time.time()

            -

            self.start_time

        )

        print(

            f"\nExecution Time : {elapsed:.2f} seconds"

        )


# ==========================================
# MODEL FUNCTIONS
# ==========================================

def save_model(

        model,

        file_path

):

    joblib.dump(

        model,

        file_path

    )

    logger.info(

        "Model Saved Successfully."

    )


def load_model(file_path):

    return joblib.load(

        file_path

    )
# ==========================================
# REPORT GENERATION
# ==========================================

def write_report(

        report_path,

        title,

        content

):

    with open(

            report_path,

            "w",

            encoding="utf-8"

    ) as file:

        file.write(title)

        file.write("\n")

        file.write("=" * 60)

        file.write("\n\n")

        file.write(content)

    logger.info(

        f"Report Generated : {report_path}"

    )


# ==========================================
# CONSOLE UTILITIES
# ==========================================

def print_title(title):

    print("\n")

    print("=" * 70)

    print(title)

    print("=" * 70)


def print_subtitle(title):

    print("\n")

    print("-" * 60)

    print(title)

    print("-" * 60)


def success(message):

    print(f"[SUCCESS] {message}")


def warning(message):

    print(f"[WARNING] {message}")


def error(message):

    print(f"[ERROR] {message}")


# ==========================================
# DATAFRAME UTILITIES
# ==========================================

def dataframe_head(

        dataframe,

        rows=5

):

    print(dataframe.head(rows))


def dataframe_tail(

        dataframe,

        rows=5

):

    print(dataframe.tail(rows))


def dataframe_info(dataframe):

    print(dataframe.info())


def unique_values(

        dataframe,

        column

):

    print(

        dataframe[column].unique()

    )


# ==========================================
# STATISTICS
# ==========================================

def basic_statistics(dataframe):

    print_title(

        "Basic Statistics"

    )

    print(

        dataframe.describe()

    )


def correlation_matrix(dataframe):

    print_title(

        "Correlation Matrix"

    )

    print(

        dataframe.corr(

            numeric_only=True

        )

    )


# ==========================================
# FEATURE VALIDATION
# ==========================================

def validate_columns(

        dataframe,

        required_columns

):

    missing = []

    for column in required_columns:

        if column not in dataframe.columns:

            missing.append(column)

    if len(missing) > 0:

        raise ValueError(

            f"Missing Columns : {missing}"

        )

    success(

        "Column validation completed."

    )


# ==========================================
# PREDICTION EXPORT
# ==========================================

def export_predictions(

        dataframe,

        output_path

):

    dataframe.to_csv(

        output_path,

        index=False

    )

    success(

        f"Prediction exported : {output_path}"

    )


# ==========================================
# EXCEPTION HANDLER
# ==========================================

def safe_execute(function):

    try:

        function()

    except Exception as exception:

        logger.error(

            exception

        )

        print(

            f"Unexpected Error : {exception}"

        )


# ==========================================
# DATE & TIME
# ==========================================

from datetime import datetime


def current_date():

    return datetime.now().strftime(

        "%d-%m-%Y"

    )


def current_time():

    return datetime.now().strftime(

        "%H:%M:%S"

    )


# ==========================================
# END MESSAGE
# ==========================================

def project_completed():

    print("\n")

    print("=" * 70)

    print(

        "Predictive Analytics Project Executed Successfully"

    )

    print(

        f"Date : {current_date()}"

    )

    print(

        f"Time : {current_time()}"

    )

    print("=" * 70)