# Makefile
# Define Python command
PY=python3

# 1. Download dataset from Kaggle
download:
	kaggle competitions download -c titanic -p data/


# 2. Extract downloaded zip file
extract:
	unzip data/titanic.zip -d data/

# 3. Clean raw data
clean: data/train.csv
	$(PY) scripts/clean.py

# 4. Create ML-ready features
features: data/clean.csv
	$(PY) scripts/features.py

# 5. Train a model and save model.pkl
train: data/features.csv
	$(PY) scripts/train.py

# 6. Generate Kaggle submission file
submit: models/model.pkl
	$(PY) scripts/create_submission.py

# Run the whole pipeline
all: download extract clean features train submit
