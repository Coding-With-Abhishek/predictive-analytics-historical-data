"""
Project : Predictive Analytics Using Historical Data
File    : evaluation.py
Author  : Abhishek Kumar
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from config import (
    MODEL_FILE,
    PROCESSED_DATA,
    TARGET_COLUMN,
    FEATURE_COLUMNS
)

from utils import (
    load_csv,
    load_model,
    print_title,
    print_subtitle,
    success,
    validate_columns
)


class ModelEvaluation:

    """
    Evaluate trained regression model
    """

    def __init__(self):

        self.model = None

        self.data = None

        self.features = None

        self.target = None

        self.predictions = None

        self.metrics = {}

    # ===========================================
    # LOAD MODEL
    # ===========================================

    def load_regression_model(self):

        print_title(
            "Loading Regression Model"
        )

        self.model = load_model(
            MODEL_FILE
        )

        success(
            "Model Loaded Successfully."
        )

    # ===========================================
    # LOAD DATASET
    # ===========================================

    def load_dataset(self):

        print_subtitle(
            "Loading Dataset"
        )

        self.data = load_csv(
            PROCESSED_DATA
        )

        validate_columns(
            self.data,
            FEATURE_COLUMNS + [TARGET_COLUMN]
        )

        success(
            "Dataset Loaded Successfully."
        )

    # ===========================================
    # PREPARE FEATURES
    # ===========================================

    def prepare_dataset(self):

        print_subtitle(
            "Preparing Evaluation Dataset"
        )

        self.features = self.data[
            FEATURE_COLUMNS
        ]

        self.target = self.data[
            TARGET_COLUMN
        ]

        print(
            f"Total Samples : {len(self.features)}"
        )

        success(
            "Dataset Prepared."
        )

    # ===========================================
    # PREDICT VALUES
    # ===========================================

    def predict(self):

        print_subtitle(
            "Generating Predictions"
        )

        self.predictions = self.model.predict(
            self.features
        )

        self.data["Predicted_Sales"] = np.round(
            self.predictions,
            2
        )

        success(
            "Prediction Completed."
        )
# ===========================================
    # CALCULATE EVALUATION METRICS
    # ===========================================

    def calculate_metrics(self):

        print_subtitle(
            "Model Evaluation Metrics"
        )

        mae = mean_absolute_error(
            self.target,
            self.predictions
        )

        mse = mean_squared_error(
            self.target,
            self.predictions
        )

        rmse = np.sqrt(
            mse
        )

        r2 = r2_score(
            self.target,
            self.predictions
        )

        self.metrics = {

            "Mean Absolute Error": round(
                mae,
                4
            ),

            "Mean Squared Error": round(
                mse,
                4
            ),

            "Root Mean Squared Error": round(
                rmse,
                4
            ),

            "R2 Score": round(
                r2,
                4
            )

        }

        print("\nPerformance Metrics\n")

        for key, value in self.metrics.items():

            print(

                f"{key} : {value}"

            )

        success(
            "Metrics Calculated Successfully."
        )

    # ===========================================
    # RESIDUAL ANALYSIS
    # ===========================================

    def residual_analysis(self):

        print_subtitle(
            "Residual Analysis"
        )

        self.data["Residual"] = (

            self.target

            -

            self.predictions

        )

        print(

            self.data[

                [

                    TARGET_COLUMN,

                    "Predicted_Sales",

                    "Residual"

                ]

            ].head()

        )

        success(
            "Residual Analysis Completed."
        )

    # ===========================================
    # ERROR ANALYSIS
    # ===========================================

    def error_statistics(self):

        print_subtitle(
            "Prediction Error Statistics"
        )

        absolute_error = np.abs(

            self.target

            -

            self.predictions

        )

        print(

            f"Maximum Error : {absolute_error.max():.2f}"

        )

        print(

            f"Minimum Error : {absolute_error.min():.2f}"

        )

        print(

            f"Average Error : {absolute_error.mean():.2f}"

        )

        success(
            "Error Analysis Completed."
        )

    # ===========================================
    # MODEL SCORE
    # ===========================================

    def model_score(self):

        print_subtitle(
            "Model Accuracy"
        )

        score = self.model.score(

            self.features,

            self.target

        )

        self.metrics["Model Score"] = round(

            score,

            4

        )

        print(

            f"Model Score : {score:.4f}"

        )

        success(
            "Model Score Generated."
        )
# ===========================================
    # GENERATE EVALUATION REPORT
    # ===========================================

    def generate_report(self):

        print_subtitle(
            "Generating Evaluation Report"
        )

        report_path = "reports/evaluation_report.txt"

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as report:

            report.write(
                "MODEL EVALUATION REPORT\n"
            )

            report.write("=" * 70)

            report.write("\n\n")

            report.write(
                f"Total Records : {len(self.data)}\n\n"
            )

            report.write(
                "Performance Metrics\n"
            )

            report.write("-" * 70)

            report.write("\n")

            for metric, value in self.metrics.items():

                report.write(
                    f"{metric} : {value}\n"
                )

            report.write("\n")

            report.write(
                "Residual Statistics\n"
            )

            report.write("-" * 70)

            report.write("\n")

            report.write(
                self.data[
                    [
                        TARGET_COLUMN,
                        "Predicted_Sales",
                        "Residual"
                    ]
                ].describe().to_string()
            )

            report.write("\n\n")

            report.write(
                "Evaluation Completed Successfully.\n"
            )

        success(
            "Evaluation Report Generated."
        )

    # ===========================================
    # COMPLETE EVALUATION PIPELINE
    # ===========================================

    def execute(self):

        print_title(
            "Predictive Analytics Model Evaluation"
        )

        self.load_regression_model()

        self.load_dataset()

        self.prepare_dataset()

        self.predict()

        self.calculate_metrics()

        self.residual_analysis()

        self.error_statistics()

        self.model_score()

        self.generate_report()

        success(
            "Evaluation Pipeline Completed."
        )


# ===========================================
# MAIN PROGRAM
# ===========================================

if __name__ == "__main__":

    evaluator = ModelEvaluation()

    evaluator.execute()

    print("\n")

    print("=" * 70)

    print("MODEL EVALUATION COMPLETED")

    print("=" * 70)

    print("Generated Files")

    print("------------------------------")

    print("✔ evaluation_report.txt")

    print("✔ Residual Analysis")

    print("✔ Performance Metrics")

    print("✔ Model Accuracy Report")

    print("=" * 70)