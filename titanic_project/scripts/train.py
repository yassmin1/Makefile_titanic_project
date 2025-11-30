# scripts/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data/features.csv")

# Separate features and label
X = df.drop(columns=["Survived"])
y = df["Survived"]

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/model.pkl")

print("Model trained and saved to models/model.pkl")
print(f"Validation accuracy: {model.score(X_val, y_val):.4f}")
