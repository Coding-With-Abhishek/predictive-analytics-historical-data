# Predictive Analytics Using Historical Data

A machine learning project that predicts future sales using historical business data. The project performs data preprocessing, feature engineering, model training, prediction, evaluation, and visualization using Python and Scikit-learn.

---

## Features

- Historical data preprocessing
- Missing value handling
- Duplicate removal
- Outlier detection
- Feature engineering
- Feature scaling
- Linear Regression model
- Ridge Regression model
- Lasso Regression model
- Model evaluation
- Future sales prediction
- Performance reports
- Data visualization
- Modular Python architecture

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

---

## Project Structure

```text
predictive-analytics-historical-data
│
├── data
│   ├── historical_data.csv
│   └── processed_data.csv
│
├── src
│   ├── config.py
│   ├── utils.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── prediction.py
│   ├── evaluation.py
│   └── visualization.py
│
├── reports
│   ├── preprocessing_report.txt
│   ├── model_performance.txt
│   ├── prediction_report.txt
│   ├── evaluation_report.txt
│   ├── predicted_sales.csv
│   └── future_forecast.csv
│
├── models
│   └── README.md
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## Dataset

The project uses a historical sales dataset containing:

- Month
- Sales
- Customers
- Marketing Spend
- Profit

The preprocessing pipeline generates additional features:

- Revenue Per Customer
- Profit Margin
- Marketing Efficiency

---

## Machine Learning Workflow

1. Load historical dataset
2. Validate dataset
3. Handle missing values
4. Remove duplicate records
5. Detect outliers
6. Feature engineering
7. Feature scaling
8. Train regression model
9. Evaluate model performance
10. Predict future sales
11. Generate reports

---

## Installation

```bash
git clone https://github.com/your-username/predictive-analytics-historical-data.git

cd predictive-analytics-historical-data

pip install -r requirements.txt
```

---

## Run Project

Preprocess data

```bash
python src/data_preprocessing.py
```

Train model

```bash
python src/model_training.py
```

Generate predictions

```bash
python src/prediction.py
```

Evaluate model

```bash
python src/evaluation.py
```

Generate visualizations

```bash
python src/visualization.py
```

---

## Generated Outputs

After running the project, the following files are generated automatically:

### Reports

- preprocessing_report.txt
- model_performance.txt
- prediction_report.txt
- evaluation_report.txt
- predicted_sales.csv
- future_forecast.csv

### Generated Files

- regression_model.pkl
- project.log

> These files are generated during project execution and are not included in the repository.

---

## Future Improvements

- Random Forest Regression
- XGBoost Integration
- Hyperparameter Tuning
- Interactive Dashboard
- REST API Deployment
- Streamlit Web Application

---

## Author

**Abhishek Kumar**

B.Tech CSE (Artificial Intelligence)

---

## License

This project is licensed under the MIT License.