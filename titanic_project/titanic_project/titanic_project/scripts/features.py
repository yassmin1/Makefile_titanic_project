# scripts/features.py

import pandas as pd

df = pd.read_csv("data/clean.csv")

# Create new features
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

# One-hot encode Embarked
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

df.to_csv("data/features.csv", index=False)

print("Feature dataset saved to data/features.csv")
