"""
Project : Predictive Analytics Using Historical Data
File    : config.py
Author  : Abhishek Kumar
Version : 1.0
"""

from pathlib import Path

# =====================================================
# PROJECT INFORMATION
# =====================================================

PROJECT_NAME = "Predictive Analytics Using Historical Data"

PROJECT_VERSION = "1.0.0"

AUTHOR = "Abhishek Rajput"

DESCRIPTION = (
    "Machine Learning based Predictive Analytics Project"
)

# =====================================================
# ROOT DIRECTORY
# =====================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

# =====================================================
# DATA DIRECTORY
# =====================================================

DATA_DIR = ROOT_DIR / "data"

RAW_DATA = DATA_DIR / "historical_data.csv"

PROCESSED_DATA = DATA_DIR / "processed_data.csv"

# =====================================================
# MODEL DIRECTORY
# =====================================================

MODEL_DIR = ROOT_DIR / "models"

MODEL_FILE = MODEL_DIR / "regression_model.pkl"

# =====================================================
# REPORT DIRECTORY
# =====================================================

REPORT_DIR = ROOT_DIR / "reports"

PREDICTION_REPORT = REPORT_DIR / "prediction_report.txt"

PREPROCESSING_REPORT = (
    REPORT_DIR / "preprocessing_report.txt"
)

MODEL_REPORT = (
    REPORT_DIR / "model_performance.txt"
)

# =====================================================
# VISUALIZATION DIRECTORY
# =====================================================

IMAGE_DIR = ROOT_DIR / "images"

SALES_GRAPH = IMAGE_DIR / "sales_trend.png"

PREDICTION_GRAPH = IMAGE_DIR / "prediction_graph.png"

COMPARISON_GRAPH = IMAGE_DIR / "comparison_graph.png"

# =====================================================
# DATASET SETTINGS
# =====================================================

TARGET_COLUMN = "Sales"

FEATURE_COLUMNS = [

    "Month",

    "Customers",

    "Marketing_Spend"

]

TEST_SIZE = 0.20

RANDOM_STATE = 42

# =====================================================
# MODEL SETTINGS
# =====================================================

MODEL_NAME = "Linear Regression"

ENABLE_MODEL_SAVE = True

ENABLE_MODEL_LOAD = True

# =====================================================
# PREPROCESSING SETTINGS
# =====================================================

REMOVE_DUPLICATES = True

HANDLE_MISSING_VALUES = True

STANDARDIZE_COLUMNS = True

DROP_EMPTY_ROWS = True

# =====================================================
# REPORT SETTINGS
# =====================================================

REPORT_TITLE = (
    "Predictive Analytics Performance Report"
)

DATE_FORMAT = "%d-%m-%Y"

TIME_FORMAT = "%H:%M:%S"

# =====================================================
# LOGGING SETTINGS
# =====================================================

LOG_LEVEL = "INFO"

LOG_FILE = "project.log"

# =====================================================
# CHART SETTINGS
# =====================================================

FIGURE_WIDTH = 10

FIGURE_HEIGHT = 6

DPI = 120

# =====================================================
# METRICS
# =====================================================

METRICS = [

    "MAE",

    "MSE",

    "RMSE",

    "R2 Score"

]

# =====================================================
# FILE VALIDATION
# =====================================================

SUPPORTED_FILE_TYPES = [

    ".csv"

]

# =====================================================
# CONSOLE MESSAGES
# =====================================================

SUCCESS = "Operation Completed Successfully"

FAILED = "Operation Failed"

LOADING = "Loading Dataset..."

TRAINING = "Training Model..."

SAVING = "Saving Model..."

PREDICTING = "Generating Predictions..."

REPORT = "Generating Report..."

# =====================================================
# END
# =====================================================