"""
Project : Predictive Analytics Using Historical Data
File    : visualization.py
Author  : Abhishek Kumar
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from config import (
    PROCESSED_DATA,
    MODEL_FILE,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)

from utils import (
    load_csv,
    load_model,
    print_title,
    print_subtitle,
    success,
    validate_columns
)


class Visualization:

    """
    Data Visualization Module
    """

    def __init__(self):

        self.model = None

        self.data = None

        self.features = None

        self.predictions = None

    # ===========================================
    # LOAD MODEL
    # ===========================================

    def load_model_file(self):

        print_title(

            "Loading Trained Model"

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

            FEATURE_COLUMNS + [TARGET_COLUMN]

        )

        self.features = self.data[

            FEATURE_COLUMNS

        ]

        success(

            "Dataset Loaded Successfully."

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

            "Prediction Completed."

        )

    # ===========================================
    # ACTUAL VS PREDICTED
    # ===========================================

    def actual_vs_predicted(self):

        print_subtitle(

            "Actual vs Predicted"

        )

        plt.figure(

            figsize=(10,6)

        )

        plt.plot(

            self.data["Month"],

            self.data[TARGET_COLUMN],

            marker="o",

            label="Actual"

        )

        plt.plot(

            self.data["Month"],

            self.data["Predicted_Sales"],

            marker="s",

            label="Predicted"

        )

        plt.title(

            "Actual vs Predicted Sales"

        )

        plt.xlabel(

            "Month"

        )

        plt.ylabel(

            "Sales"

        )

        plt.grid(True)

        plt.legend()

        plt.savefig(

            "reports/actual_vs_predicted.png",

            dpi=300

        )

        plt.close()

        success(

            "Actual vs Predicted Chart Saved."

        )
# ===========================================
    # SALES TREND VISUALIZATION
    # ===========================================

    def sales_trend(self):

        print_subtitle(
            "Sales Trend Analysis"
        )

        plt.figure(
            figsize=(10,6)
        )

        plt.plot(
            self.data["Month"],
            self.data[TARGET_COLUMN],
            marker="o",
            linewidth=2
        )

        plt.title(
            "Historical Sales Trend"
        )

        plt.xlabel(
            "Month"
        )

        plt.ylabel(
            "Sales"
        )

        plt.grid(True)

        plt.savefig(
            "reports/sales_trend.png",
            dpi=300
        )

        plt.close()

        success(
            "Sales Trend Chart Saved."
        )

    # ===========================================
    # RESIDUAL PLOT
    # ===========================================

    def residual_plot(self):

        print_subtitle(
            "Residual Plot"
        )

        residual = (

            self.data[TARGET_COLUMN]

            -

            self.data["Predicted_Sales"]

        )

        plt.figure(
            figsize=(10,6)
        )

        plt.scatter(
            self.data["Predicted_Sales"],
            residual
        )

        plt.axhline(
            y=0,
            linestyle="--"
        )

        plt.title(
            "Residual Analysis"
        )

        plt.xlabel(
            "Predicted Sales"
        )

        plt.ylabel(
            "Residual"
        )

        plt.grid(True)

        plt.savefig(
            "reports/residual_plot.png",
            dpi=300
        )

        plt.close()

        success(
            "Residual Plot Saved."
        )

    # ===========================================
    # FEATURE IMPORTANCE
    # ===========================================

    def feature_importance(self):

        print_subtitle(
            "Feature Importance"
        )

        importance = pd.DataFrame({

            "Feature": FEATURE_COLUMNS,

            "Coefficient": self.model.coef_

        })

        importance = importance.sort_values(

            by="Coefficient",

            ascending=False

        )

        plt.figure(
            figsize=(8,6)
        )

        plt.bar(

            importance["Feature"],

            importance["Coefficient"]

        )

        plt.title(
            "Feature Importance"
        )

        plt.xlabel(
            "Features"
        )

        plt.ylabel(
            "Coefficient"
        )

        plt.grid(True)

        plt.savefig(
            "reports/feature_importance.png",
            dpi=300
        )

        plt.close()

        print("\n")

        print(importance)

        success(
            "Feature Importance Chart Saved."
        )
# ===========================================
    # CORRELATION MATRIX
    # ===========================================

    def correlation_matrix(self):

        print_subtitle(
            "Correlation Matrix"
        )

        correlation = self.data.corr(
            numeric_only=True
        )

        plt.figure(
            figsize=(8,6)
        )

        plt.imshow(
            correlation,
            cmap="coolwarm",
            interpolation="nearest"
        )

        plt.colorbar()

        plt.xticks(
            range(len(correlation.columns)),
            correlation.columns,
            rotation=45
        )

        plt.yticks(
            range(len(correlation.columns)),
            correlation.columns
        )

        plt.title(
            "Correlation Matrix"
        )

        plt.tight_layout()

        plt.savefig(
            "reports/correlation_matrix.png",
            dpi=300
        )

        plt.close()

        success(
            "Correlation Matrix Saved."
        )

    # ===========================================
    # FORECAST VISUALIZATION
    # ===========================================

    def forecast_visualization(self):

        print_subtitle(
            "Forecast Visualization"
        )

        forecast = self.data.copy()

        plt.figure(
            figsize=(10,6)
        )

        plt.plot(
            forecast["Month"],
            forecast["Predicted_Sales"],
            marker="o",
            linewidth=2,
            label="Predicted Sales"
        )

        plt.title(
            "Predicted Sales Forecast"
        )

        plt.xlabel(
            "Month"
        )

        plt.ylabel(
            "Sales"
        )

        plt.grid(True)

        plt.legend()

        plt.savefig(
            "reports/predicted_sales_forecast.png",
            dpi=300
        )

        plt.close()

        success(
            "Forecast Chart Saved."
        )

    # ===========================================
    # EXECUTE VISUALIZATION
    # ===========================================

    def execute(self):

        print_title(
            "Predictive Analytics Visualization"
        )

        self.load_model_file()

        self.load_dataset()

        self.generate_predictions()

        self.actual_vs_predicted()

        self.sales_trend()

        self.residual_plot()

        self.feature_importance()

        self.correlation_matrix()

        self.forecast_visualization()

        success(
            "Visualization Pipeline Completed."
        )


# ===========================================
# MAIN PROGRAM
# ===========================================

if __name__ == "__main__":

    visualizer = Visualization()

    visualizer.execute()

    print("\n")

    print("=" * 70)

    print("VISUALIZATION COMPLETED")

    print("=" * 70)

    print("Generated Files")

    print("------------------------------")

    print("✔ actual_vs_predicted.png")

    print("✔ sales_trend.png")

    print("✔ residual_plot.png")

    print("✔ feature_importance.png")

    print("✔ correlation_matrix.png")

    print("✔ predicted_sales_forecast.png")

    print("=" * 70)