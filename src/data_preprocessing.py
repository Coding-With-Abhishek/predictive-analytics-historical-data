import pandas as pd

class DataPreprocessor:

    def __init__(self, file_path):

        self.file_path = file_path

    def load_data(self):

        return pd.read_csv(
            self.file_path
        )

    def remove_duplicates(
        self,
        dataframe
    ):

        return dataframe.drop_duplicates()

    def handle_missing_values(
        self,
        dataframe
    ):

        dataframe.fillna(
            dataframe.mean(
                numeric_only=True
            ),
            inplace=True
        )

        return dataframe

    def standardize_columns(
        self,
        dataframe
    ):

        dataframe.columns = [

            column.strip()

            .replace(
                " ",
                "_"
            )

            .title()

            for column in dataframe.columns

        ]

        return dataframe

    def preprocess_data(self):

        df = self.load_data()

        df = self.remove_duplicates(df)

        df = self.handle_missing_values(df)

        df = self.standardize_columns(df)

        return df

if __name__ == "__main__":

    processor = DataPreprocessor(

        "data/historical_data.csv"

    )

    processed = processor.preprocess_data()

    processed.to_csv(

        "data/processed_data.csv",

        index=False

    )

    print(

        "Historical dataset processed successfully."

    )