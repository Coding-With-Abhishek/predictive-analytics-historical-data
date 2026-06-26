"""
Project : Predictive Analytics Using Historical Data
File    : model_training.py
Author  : Abhishek Kumar
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from config import (
    PROCESSED_DATA,
    MODEL_FILE,
    TEST_SIZE,
    RANDOM_STATE,
    TARGET_COLUMN,
    FEATURE_COLUMNS
)

from utils import (
    load_csv,
    save_model,
    print_title,
    print_subtitle,
    success,
    warning,
    validate_columns
)


class ModelTrainer:

    """
    Model Training Class
    """

    def __init__(self):

        self.data = None

        self.model = None

        self.X = None

        self.y = None

        self.X_train = None

        self.X_test = None

        self.y_train = None

        self.y_test = None

        self.predictions = None

        self.results = {}

    # ===========================================
    # LOAD DATASET
    # ===========================================

    def load_dataset(self):

        print_title(
            "Loading Processed Dataset"
        )

        self.data = load_csv(
            PROCESSED_DATA
        )

        validate_columns(
            self.data,
            FEATURE_COLUMNS + [TARGET_COLUMN]
        )

        success(
            "Processed dataset loaded successfully."
        )

    # ===========================================
    # PREPARE FEATURES
    # ===========================================

    def prepare_features(self):

        print_subtitle(
            "Preparing Features"
        )

        self.X = self.data[
            FEATURE_COLUMNS
        ]

        self.y = self.data[
            TARGET_COLUMN
        ]

        print(
            f"Feature Shape : {self.X.shape}"
        )

        print(
            f"Target Shape  : {self.y.shape}"
        )

        success(
            "Features prepared."
        )

    # ===========================================
    # TRAIN TEST SPLIT
    # ===========================================

    def split_dataset(self):

        print_subtitle(
            "Train Test Split"
        )

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test

        ) = train_test_split(

            self.X,

            self.y,

            test_size=TEST_SIZE,

            random_state=RANDOM_STATE

        )

        print(
            f"Training Samples : {len(self.X_train)}"
        )

        print(
            f"Testing Samples  : {len(self.X_test)}"
        )

        success(
            "Dataset split completed."
        )
# ===========================================
    # TRAIN LINEAR REGRESSION MODEL
    # ===========================================

    def train_linear_regression(self):

        print_subtitle(
            "Training Linear Regression Model"
        )

        self.model = LinearRegression()

        self.model.fit(

            self.X_train,

            self.y_train

        )

        success(
            "Linear Regression Model Trained."
        )

    # ===========================================
    # TRAIN RIDGE REGRESSION
    # ===========================================

    def train_ridge_regression(self):

        print_subtitle(
            "Training Ridge Regression Model"
        )

        ridge = Ridge(

            alpha=1.0

        )

        ridge.fit(

            self.X_train,

            self.y_train

        )

        ridge_score = ridge.score(

            self.X_test,

            self.y_test

        )

        self.results["Ridge Score"] = round(

            ridge_score,

            4

        )

        success(
            "Ridge Regression Completed."
        )

    # ===========================================
    # TRAIN LASSO REGRESSION
    # ===========================================

    def train_lasso_regression(self):

        print_subtitle(
            "Training Lasso Regression Model"
        )

        lasso = Lasso(

            alpha=0.1

        )

        lasso.fit(

            self.X_train,

            self.y_train

        )

        lasso_score = lasso.score(

            self.X_test,

            self.y_test

        )

        self.results["Lasso Score"] = round(

            lasso_score,

            4

        )

        success(
            "Lasso Regression Completed."
        )

    # ===========================================
    # MAKE PREDICTIONS
    # ===========================================

    def predict(self):

        print_subtitle(
            "Generating Predictions"
        )

        self.predictions = self.model.predict(

            self.X_test

        )

        success(
            "Predictions Generated."
        )

    # ===========================================
    # EVALUATION METRICS
    # ===========================================

    def evaluate_model(self):

        print_subtitle(
            "Evaluating Model"
        )

        mae = mean_absolute_error(

            self.y_test,

            self.predictions

        )

        mse = mean_squared_error(

            self.y_test,

            self.predictions

        )

        rmse = np.sqrt(

            mse

        )

        r2 = r2_score(

            self.y_test,

            self.predictions

        )

        self.results["MAE"] = round(

            mae,

            4

        )

        self.results["MSE"] = round(

            mse,

            4

        )

        self.results["RMSE"] = round(

            rmse,

            4

        )

        self.results["R2 Score"] = round(

            r2,

            4

        )

        print("\nModel Performance\n")

        for metric, value in self.results.items():

            print(

                f"{metric} : {value}"

            )

        success(
            "Model Evaluation Completed."
        )
# ===========================================
    # CROSS VALIDATION
    # ===========================================

    def cross_validation(self):

        print_subtitle(
            "Cross Validation"
        )

        scores = cross_val_score(

            self.model,

            self.X,

            self.y,

            cv=5,

            scoring="r2"

        )

        self.results["Cross Validation Mean"] = round(

            scores.mean(),

            4

        )

        self.results["Cross Validation Std"] = round(

            scores.std(),

            4

        )

        print(
            f"Average CV Score : {scores.mean():.4f}"
        )

        success(
            "Cross Validation Completed."
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

        print(importance)

        self.feature_scores = importance

        success(
            "Feature Importance Generated."
        )

    # ===========================================
    # SAVE TRAINED MODEL
    # ===========================================

    def save_trained_model(self):

        print_subtitle(
            "Saving Trained Model"
        )

        save_model(

            self.model,

            MODEL_FILE

        )

        success(
            "Regression Model Saved."
        )

    # ===========================================
    # MODEL SUMMARY
    # ===========================================

    def model_summary(self):

        print_subtitle(
            "Model Summary"
        )

        print(

            f"Model : {type(self.model).__name__}"

        )

        print(

            f"Training Samples : {len(self.X_train)}"

        )

        print(

            f"Testing Samples : {len(self.X_test)}"

        )

        print("\nPerformance Metrics\n")

        for key, value in self.results.items():

            print(

                f"{key} : {value}"

            )

        success(
            "Model Summary Generated."
        )
# ===========================================
    # GENERATE PERFORMANCE REPORT
    # ===========================================

    def generate_report(self):

        print_subtitle(
            "Generating Performance Report"
        )

        report_path = "reports/model_performance.txt"

        with open(

            report_path,

            "w",

            encoding="utf-8"

        ) as report:

            report.write(

                "MODEL PERFORMANCE REPORT\n"

            )

            report.write(

                "=" * 60

            )

            report.write("\n\n")

            report.write(

                f"Model : {type(self.model).__name__}\n\n"

            )

            report.write(

                "Performance Metrics\n"

            )

            report.write(

                "-" * 60

            )

            report.write("\n")

            for metric, value in self.results.items():

                report.write(

                    f"{metric} : {value}\n"

                )

            report.write("\n")

            report.write(

                "Feature Importance\n"

            )

            report.write(

                "-" * 60

            )

            report.write("\n")

            report.write(

                self.feature_scores.to_string(

                    index=False

                )

            )

        success(

            "Performance Report Generated."

        )

    # ===========================================
    # COMPLETE TRAINING PIPELINE
    # ===========================================

    def execute(self):

        print_title(

            "Predictive Analytics Model Training"

        )

        self.load_dataset()

        self.prepare_features()

        self.split_dataset()

        self.train_linear_regression()

        self.train_ridge_regression()

        self.train_lasso_regression()

        self.predict()

        self.evaluate_model()

        self.cross_validation()

        self.feature_importance()

        self.save_trained_model()

        self.model_summary()

        self.generate_report()

        success(

            "Model Training Pipeline Completed."

        )


# ===========================================
# MAIN PROGRAM
# ===========================================

if __name__ == "__main__":

    trainer = ModelTrainer()

    trainer.execute()

    print("\n")

    print("=" * 70)

    print("PREDICTIVE ANALYTICS MODEL TRAINING COMPLETED")

    print("=" * 70)

    print("\nGenerated Files")

    print("------------------------------")

    print("✔ regression_model.pkl")

    print("✔ model_performance.txt")

    print("✔ Prediction Model Ready")

    print("=" * 70)