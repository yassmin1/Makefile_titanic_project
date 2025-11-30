#scripts/clean.py
import pandas as pd

# Load raw Kaggle training dataset
df = pd.read_csv("data/train.csv")

# Drop irrelevant columns
df = df.drop(columns=["Cabin", "Ticket", "Name"])

# Fill missing Age with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked with most common value
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Encode Sex as numeric
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Save clean file
df.to_csv("data/clean.csv", index=False)

print("Cleaned dataset saved to data/clean.csv")
