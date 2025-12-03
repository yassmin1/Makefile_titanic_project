
# Titanic ML Automated with Makefile

This project demonstrates a complete, end-to-end Machine Learning workflow using the **Titanic Kaggle dataset**, fully automated using a **Makefile pipeline**.

The workflow includes:

- Downloading the dataset from Kaggle  
- Extracting the ZIP file  
- Cleaning the dataset  
- Engineering features  
- Training a model  
- Creating a Kaggle submission  

Run the entire pipeline with a single command:

```bash
make all
````

---

## Project Structure

```
titanic_project/
│
├── data/
│   ├── raw.csv
│   ├── train.csv        
│   ├── test.csv         
│   ├── clean.csv
│   ├── features.csv
│
├── models/
│   └── model.pkl
│
├── submission/
│   └── submission.csv
│
├── scripts/
│   ├── clean.py
│   ├── features.py
│   ├── train.py
│   └── create_submission.py
│
├── Makefile
└── README.md
```

# How to Use This Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Set Up Your Kaggle API Token (New API Method)

Kaggle now provides a **single environment variable token** (not a kaggle.json file).

After generating your token from Kaggle, set it:

```bash
export KAGGLE_API_TOKEN=YOUR_TOKEN_HERE
```

To make this permanent, add it to your `~/.bashrc`:

```bash
echo 'export KAGGLE_API_TOKEN=YOUR_TOKEN_HERE' >> ~/.bashrc
source ~/.bashrc
```

Test if authentication works:

```bash
kaggle competitions list
```

---

### 3. Run the Entire Pipeline

```bash
make all
```

This will sequentially:

1. Download the Titanic dataset
2. Extract the ZIP file
3. Clean the dataset
4. Build ML features
5. Train a RandomForest model
6. Generate a submission file

---

## Makefile Overview

```make
download:
	kaggle competitions download -c titanic -p data/

extract:
	unzip data/titanic.zip -d data/

clean: data/train.csv
	python3 scripts/clean.py

features: data/clean.csv
	python3 scripts/features.py

train: data/features.csv
	python3 scripts/train.py

submit: models/model.pkl
	python3 scripts/create_submission.py

all: download extract clean features train submit
```

