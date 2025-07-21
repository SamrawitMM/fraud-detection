# scripts/preprocessing.py

import pandas as pd
import numpy as np
import ipaddress
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# -------------------------------
# Load Data
# -------------------------------
fraud_df = pd.read_csv("data/raw/Fraud_Data.csv")
ip_country_df = pd.read_csv("data/raw/IpAddress_to_Country.csv")
creditcard_df = pd.read_csv("data/raw/creditcard.csv")

# -------------------------------
# Clean & Preprocess fraud_df
# -------------------------------

# Fix datatypes
fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'])
fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'])
fraud_df['ip_address'] = fraud_df['ip_address'].astype(np.uint32)

# Remove duplicates
fraud_df.drop_duplicates(inplace=True)

# Clean IP table
ip_country_df['lower_bound_ip_address'] = ip_country_df['lower_bound_ip_address'].astype(np.uint32)
ip_country_df['upper_bound_ip_address'] = ip_country_df['upper_bound_ip_address'].astype(np.uint32)

# Map IP to country
def map_ip_to_country(ip, ip_table):
    match = ip_table[
        (ip_table['lower_bound_ip_address'] <= ip) & 
        (ip_table['upper_bound_ip_address'] >= ip)
    ]
    return match['country'].values[0] if not match.empty else 'Unknown'

fraud_df['country'] = fraud_df['ip_address'].apply(lambda x: map_ip_to_country(x, ip_country_df))

# Feature Engineering
fraud_df['hour_of_day'] = fraud_df['purchase_time'].dt.hour
fraud_df['day_of_week'] = fraud_df['purchase_time'].dt.dayofweek
fraud_df['time_since_signup'] = (fraud_df['purchase_time'] - fraud_df['signup_time']).dt.total_seconds()
fraud_df['user_transaction_count'] = fraud_df.groupby('user_id')['purchase_time'].transform('count')

# One-hot encode categorical
cat_cols = ['source', 'browser', 'sex', 'country']
fraud_df = pd.get_dummies(fraud_df, columns=cat_cols, drop_first=True)

# Scale numerical
scaler = StandardScaler()
num_cols = ['purchase_value', 'age', 'time_since_signup', 'user_transaction_count']
fraud_df[num_cols] = scaler.fit_transform(fraud_df[num_cols])

# Prepare data for modeling
drop_cols = ['signup_time', 'purchase_time', 'ip_address', 'device_id', 'user_id']
X_fraud = fraud_df.drop(columns=drop_cols + ['class'])
y_fraud = fraud_df['class']

X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(
    X_fraud, y_fraud, test_size=0.2, stratify=y_fraud, random_state=42
)

# SMOTE oversampling
sm = SMOTE(random_state=42)
X_train_fraud_res, y_train_fraud_res = sm.fit_resample(X_train_fraud, y_train_fraud)

# Save processed fraud data
pd.DataFrame(X_train_fraud_res, columns=X_fraud.columns).to_csv("data/processed/X_train_fraud.csv", index=False)
pd.DataFrame(X_test_fraud, columns=X_fraud.columns).to_csv("data/processed/X_test_fraud.csv", index=False)
pd.Series(y_train_fraud_res, name='class').to_csv("data/processed/y_train_fraud.csv", index=False)
pd.Series(y_test_fraud, name='class').to_csv("data/processed/y_test_fraud.csv", index=False)

print("✅ Fraud dataset preprocessed and saved.")

# -------------------------------
# Preprocess creditcard_df
# -------------------------------

# Scale Time and Amount
creditcard_df[['Time', 'Amount']] = StandardScaler().fit_transform(creditcard_df[['Time', 'Amount']])

X_credit = creditcard_df.drop(columns=['Class'])
y_credit = creditcard_df['Class']

X_train_cc, X_test_cc, y_train_cc, y_test_cc = train_test_split(
    X_credit, y_credit, test_size=0.2, stratify=y_credit, random_state=42
)

X_train_cc_res, y_train_cc_res = SMOTE(random_state=42).fit_resample(X_train_cc, y_train_cc)

# Save processed credit card data
pd.DataFrame(X_train_cc_res, columns=X_credit.columns).to_csv("data/processed/X_train_credit.csv", index=False)
pd.DataFrame(X_test_cc, columns=X_credit.columns).to_csv("data/processed/X_test_credit.csv", index=False)
pd.Series(y_train_cc_res, name='Class').to_csv("data/processed/y_train_credit.csv", index=False)
pd.Series(y_test_cc, name='Class').to_csv("data/processed/y_test_credit.csv", index=False)

print("✅ Credit card dataset preprocessed and saved.")
