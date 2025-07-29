# Fraud Detection Project

This project focuses on detecting fraudulent transactions using real-world datasets. It involves data preprocessing, feature engineering, exploratory data analysis, and training machine learning models for fraud detection.

## 📁 Project Structure

├── data
│   ├── processed
│   └── raw
│       ├── creditcard.csv
│       ├── Fraud_Data.csv
│       └── IpAddress_to_Country.csv
├── models
│   └── xgb_fraud_model.pkl
├── notebooks
│   └── EDA_and_Preprocessing.ipynb
├── README.md
├── reports
├── requirements.txt
└── scripts
    └── preprocessing.py


## 📊 Datasets

- **creditcard.csv**: Credit card transaction data with labeled fraud cases.
- **Fraud_Data.csv**: E-commerce fraud records with user and transaction info.
- **IpAddress_to_Country.csv**: Mapping of IP ranges to country codes.

## 🚀 Tasks Completed

- Data cleaning, merging, and handling missing values
- Feature engineering (e.g., time-based and frequency features)
- Handling class imbalance using resampling techniques
- Model training using:
  - Logistic Regression
  - XGBoost
- Evaluation using metrics suitable for imbalanced data (F1, AUC-PR, Confusion Matrix)

## 🛠 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## 📌 Usage

Run the main preprocessing logic:

```bash
python scripts/preprocessing.py
```

Explore the data and results in the notebook:

```bash
notebooks/EDA_and_Preprocessing.ipynb
```
