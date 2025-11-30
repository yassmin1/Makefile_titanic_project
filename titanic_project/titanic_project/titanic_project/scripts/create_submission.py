# scripts/create_submission.py
import pandas as pd
import joblib

# Load model
model = joblib.load("models/model.pkl")

# Load test dataset
test = pd.read_csv("data/test.csv")

# Prepare the test data the same way we prepared training data
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})

test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
test["IsAlone"] = (test["FamilySize"] == 1).astype(int)
test["FarePerPerson"] = test["Fare"] / test["FamilySize"]

# One-hot encode Embarked
test = pd.get_dummies(test, columns=["Embarked"], drop_first=True)

# Align columns with training data
train_cols = pd.read_csv("data/features.csv").drop(columns=["Survived"]).columns
test = test.reindex(columns=train_cols, fill_value=0)

# Predictions
pred = model.predict(test)

# Build submission
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": pred
})

submission.to_csv("submission/submission.csv", index=False)

print("Submission saved to submission/submission.csv")
