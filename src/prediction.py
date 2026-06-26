"""
Project : Predictive Analytics Using Historical Data
File    : prediction.py
Author  : Abhishek Kumar
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from config import (
    MODEL_FILE,
    PROCESSED_DATA,
    FEATURE_COLUMNS
)

from utils import (
    load_csv,
    load_model,
    print_title,
    print_subtitle,
    success,
    warning,
    validate_columns
)


class PredictionEngine:

    """
    Prediction Engine
    """

    def __init__(self):

        self.model = None

        self.data = None

        self.features = None

        self.predictions = None

    # ===========================================
    # LOAD TRAINED MODEL
    # ===========================================

    def load_trained_model(self):

        print_title(

            "Loading Trained Regression Model"

        )

        self.model = load_model(

            MODEL_FILE

        )

        success(

            "Regression Model Loaded."

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

            FEATURE_COLUMNS

        )

        success(

            "Dataset Loaded Successfully."

        )

    # ===========================================
    # PREPARE FEATURES
    # ===========================================

    def prepare_features(self):

        print_subtitle(

            "Preparing Prediction Features"

        )

        self.features = self.data[

            FEATURE_COLUMNS

        ]

        print(

            f"Prediction Samples : {len(self.features)}"

        )

        success(

            "Features Ready."

        )

    # ===========================================
    # GENERATE PREDICTIONS
    # ===========================================

    def generate_predictions(self):

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

            "Predictions Generated."

        )

    # ===========================================
    # DISPLAY SAMPLE OUTPUT
    # ===========================================

    def preview_predictions(self):

        print_subtitle(

            "Prediction Preview"

        )

        print(

            self.data.head()

        )
# ===========================================
    # FUTURE FORECAST
    # ===========================================

    def forecast_future_sales(self):

        print_subtitle(
            "Forecasting Future Sales"
        )

        last_month = int(
            self.data["Month"].max()
        )

        average_customers = int(
            self.data["Customers"].mean()
        )

        average_marketing = int(
            self.data["Marketing_Spend"].mean()
        )

        future_data = pd.DataFrame({

            "Month": [
                last_month + 1,
                last_month + 2,
                last_month + 3
            ],

            "Customers": [
                average_customers,
                average_customers + 20,
                average_customers + 40
            ],

            "Marketing_Spend": [
                average_marketing,
                average_marketing + 200,
                average_marketing + 400
            ]

        })

        forecast = self.model.predict(
            future_data
        )

        future_data["Predicted_Sales"] = np.round(
            forecast,
            2
        )

        self.future_predictions = future_data

        print("\nFuture Forecast\n")

        print(self.future_predictions)

        success(
            "Future Forecast Generated."
        )

    # ===========================================
    # EXPORT PREDICTIONS
    # ===========================================

    def export_predictions(self):

        print_subtitle(
            "Exporting Predictions"
        )

        self.data.to_csv(

            "reports/predicted_sales.csv",

            index=False

        )

        self.future_predictions.to_csv(

            "reports/future_forecast.csv",

            index=False

        )

        success(
            "Prediction Files Saved."
        )

    # ===========================================
    # GENERATE REPORT
    # ===========================================

    def generate_report(self):

        print_subtitle(
            "Generating Prediction Report"
        )

        with open(

            "reports/prediction_report.txt",

            "w",

            encoding="utf-8"

        ) as report:

            report.write(
                "PREDICTIVE ANALYTICS REPORT\n"
            )

            report.write("=" * 60)

            report.write("\n\n")

            report.write(
                f"Total Records : {len(self.data)}\n"
            )

            report.write(
                f"Average Predicted Sales : {self.data['Predicted_Sales'].mean():.2f}\n"
            )

            report.write(
                f"Maximum Prediction : {self.data['Predicted_Sales'].max():.2f}\n"
            )

            report.write(
                f"Minimum Prediction : {self.data['Predicted_Sales'].min():.2f}\n"
            )

            report.write("\nFuture Forecast\n")

            report.write("-" * 60)

            report.write("\n")

            report.write(

                self.future_predictions.to_string(

                    index=False

                )

            )

        success(
            "Prediction Report Generated."
        )

    # ===========================================
    # COMPLETE PIPELINE
    # ===========================================

    def execute(self):

        self.load_trained_model()

        self.load_dataset()

        self.prepare_features()

        self.generate_predictions()

        self.preview_predictions()

        self.forecast_future_sales()

        self.export_predictions()

        self.generate_report()

        success(
            "Prediction Pipeline Completed."
        )


# ===========================================
# MAIN PROGRAM
# ===========================================

if __name__ == "__main__":

    engine = PredictionEngine()

    engine.execute()

    print("\n")

    print("=" * 70)

    print("PREDICTIVE ANALYTICS COMPLETED")

    print("=" * 70)

    print("Generated Files")

    print("------------------------------")

    print("✔ prediction_report.txt")

    print("✔ predicted_sales.csv")

    print("✔ future_forecast.csv")

    print("=" * 70)