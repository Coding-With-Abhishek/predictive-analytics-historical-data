"""
Project : Predictive Analytics Using Historical Data
File    : data_preprocessing.py
Author  : Abhishek Kumar
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from config import (
    RAW_DATA,
    PROCESSED_DATA
)

from utils import (
    load_csv,
    save_csv,
    dataset_summary,
    dataset_shape,
    dataset_columns,
    missing_values,
    duplicate_values,
    print_title,
    print_subtitle,
    success,
    warning,
    validate_columns
)


class DataPreprocessor:

    """
    Historical Data Preprocessing Class
    """

    def __init__(self):

        self.data = None

        self.scaler = StandardScaler()

        self.required_columns = [

            "Month",

            "Sales",

            "Customers",

            "Marketing_Spend",

            "Profit"

        ]

    # ===========================================
    # LOAD DATA
    # ===========================================

    def load_dataset(self):

        print_title(

            "Loading Historical Dataset"

        )

        self.data = load_csv(

            RAW_DATA

        )

        success(

            "Dataset Loaded Successfully."

        )

        return self.data

    # ===========================================
    # VALIDATE DATA
    # ===========================================

    def validate_dataset(self):

        print_subtitle(

            "Dataset Validation"

        )

        validate_columns(

            self.data,

            self.required_columns

        )

        dataset_shape(

            self.data

        )

        dataset_columns(

            self.data

        )

        dataset_summary(

            self.data

        )

    # ===========================================
    # HANDLE MISSING VALUES
    # ===========================================

    def handle_missing_values(self):

        print_subtitle(

            "Handling Missing Values"

        )

        missing_values(

            self.data

        )

        numeric_columns = self.data.select_dtypes(

            include=np.number

        ).columns

        for column in numeric_columns:

            self.data[column].fillna(

                self.data[column].mean(),

                inplace=True

            )

        success(

            "Missing Values Replaced."

        )

    # ===========================================
    # REMOVE DUPLICATES
    # ===========================================

    def remove_duplicates(self):

        print_subtitle(

            "Removing Duplicate Records"

        )

        duplicate_values(

            self.data

        )

        before = len(

            self.data

        )

        self.data.drop_duplicates(

            inplace=True

        )

        after = len(

            self.data

        )

        removed = before - after

        print(

            f"Duplicates Removed : {removed}"

        )

        success(

            "Duplicate Removal Completed."

        )

    # ===========================================
    # STANDARDIZE COLUMN NAMES
    # ===========================================

    def standardize_columns(self):

        print_subtitle(

            "Standardizing Columns"

        )

        self.data.columns = [

            column.strip()

            .replace(

                " ",

                "_"

            )

            .title()

            for column in self.data.columns

        ]

        success(

            "Column Names Standardized."

        )
# ===========================================
    # REMOVE OUTLIERS
    # ===========================================

    def remove_outliers(self):

        print_subtitle(

            "Outlier Detection"

        )

        numeric_columns = [

            "Sales",

            "Customers",

            "Marketing_Spend",

            "Profit"

        ]

        for column in numeric_columns:

            q1 = self.data[column].quantile(0.25)

            q3 = self.data[column].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - (1.5 * iqr)

            upper = q3 + (1.5 * iqr)

            self.data = self.data[

                (self.data[column] >= lower)

                &

                (self.data[column] <= upper)

            ]

        success(

            "Outliers Removed."

        )

    # ===========================================
    # FEATURE ENGINEERING
    # ===========================================

    def feature_engineering(self):

        print_subtitle(

            "Feature Engineering"

        )

        self.data["Revenue_Per_Customer"] = (

            self.data["Sales"]

            /

            self.data["Customers"]

        )

        self.data["Profit_Margin"] = (

            self.data["Profit"]

            /

            self.data["Sales"]

        ) * 100

        self.data["Marketing_Efficiency"] = (

            self.data["Sales"]

            /

            self.data["Marketing_Spend"]

        )

        success(

            "Features Created."

        )

    # ===========================================
    # FEATURE SCALING
    # ===========================================

    def feature_scaling(self):

        print_subtitle(

            "Scaling Numerical Features"

        )

        columns = [

            "Customers",

            "Marketing_Spend",

            "Profit",

            "Revenue_Per_Customer",

            "Profit_Margin",

            "Marketing_Efficiency"

        ]

        self.data[columns] = self.scaler.fit_transform(

            self.data[columns]

        )

        success(

            "Scaling Completed."

        )

    # ===========================================
    # SORT DATA
    # ===========================================

    def sort_dataset(self):

        print_subtitle(

            "Sorting Dataset"

        )

        self.data.sort_values(

            by="Month",

            inplace=True

        )

        self.data.reset_index(

            drop=True,

            inplace=True

        )

        success(

            "Dataset Sorted."

        )

    # ===========================================
    # PREPROCESS PIPELINE
    # ===========================================

    def preprocess(self):

        self.load_dataset()

        self.validate_dataset()

        self.handle_missing_values()

        self.remove_duplicates()

        self.standardize_columns()

        self.remove_outliers()

        self.feature_engineering()

        self.feature_scaling()

        self.sort_dataset()

        return self.data
# ===========================================
    # SAVE PROCESSED DATASET
    # ===========================================

    def save_processed_dataset(self):

        print_subtitle(

            "Saving Processed Dataset"

        )

        save_csv(

            self.data,

            PROCESSED_DATA

        )

        success(

            "Processed Dataset Saved."

        )

    # ===========================================
    # GENERATE PREPROCESSING REPORT
    # ===========================================

    def generate_report(self):

        print_subtitle(

            "Generating Preprocessing Report"

        )

        print("\n")

        print("=" * 60)

        print("PREPROCESSING SUMMARY")

        print("=" * 60)

        print(

            f"Total Records : {len(self.data)}"

        )

        print(

            f"Total Columns : {len(self.data.columns)}"

        )

        print(

            "\nColumns"

        )

        for column in self.data.columns:

            print(

                f"• {column}"

            )

        print("\nStatistics\n")

        print(

            self.data.describe()

        )

        success(

            "Preprocessing Report Generated."

        )

    # ===========================================
    # EXECUTE COMPLETE PIPELINE
    # ===========================================

    def execute(self):

        try:

            print_title(

                "Predictive Analytics Data Preprocessing"

            )

            self.preprocess()

            self.save_processed_dataset()

            self.generate_report()

            success(

                "Data Preprocessing Completed Successfully."

            )

            return self.data

        except Exception as error:

            print(

                f"\nError : {error}"

            )

            raise


# ===========================================
# MAIN PROGRAM
# ===========================================

if __name__ == "__main__":

    processor = DataPreprocessor()

    processed_data = processor.execute()

    print("\n")

    print("=" * 60)

    print("FIRST FIVE RECORDS")

    print("=" * 60)

    print(

        processed_data.head()

    )

    print("\n")

    print("=" * 60)

    print("LAST FIVE RECORDS")

    print("=" * 60)

    print(

        processed_data.tail()

    )

    print("\n")

    print("=" * 60)

    print("DATA PREPROCESSING FINISHED")

    print("=" * 60)