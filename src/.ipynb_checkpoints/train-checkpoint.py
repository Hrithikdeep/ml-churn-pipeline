

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 1. Load cleaned data
data_path = os.path.join("data", "processed", "cleaned_data.csv")
df = pd.read_csv(data_path)

# 2. Split features & target
X = df.drop("churn", axis=1)
y = df["churn"]

# 3. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Define models
models = {
    "logreg": LogisticRegression(max_iter=1000),
    "randomforest": RandomForestClassifier(n_estimators=100, random_state=42),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# 5. Train, evaluate and save
for name, model in models.items():
    print(f"\n🔧 Training: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)

    print(f" Accuracy: {acc:.4f}")
    print(f" ROC-AUC: {roc:.4f}")
    print(" Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    model_path = os.path.join("models", f"{name}_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
