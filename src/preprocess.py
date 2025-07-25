# src/preprocess.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load the raw data
raw_path = os.path.join("data", "raw", "b2b_data.csv")
df = pd.read_csv(raw_path)

# 2. Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# 3. Handle missing values
df.fillna(method="ffill", inplace=True)

# 4. Encode categorical features
categorical_cols = df.select_dtypes(include='object').columns.tolist()
categorical_cols.remove("churn")  # We'll handle target separately

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Encode target column ("churn")
df["churn"] = df["churn"].map({"No": 0, "Yes": 1})

# 5. Scale numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols.remove("churn")  # Don’t scale the target!

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 6. Save the processed data
processed_path = os.path.join("data", "processed", "cleaned_data.csv")
os.makedirs(os.path.dirname(processed_path), exist_ok=True)
df.to_csv(processed_path, index=False)

print("✅ Preprocessing complete. Cleaned data saved to:", processed_path)
