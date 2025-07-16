# 🌍 IBRD Loan Classification with Machine Learning

This project analyzes the International Bank for Reconstruction and Development (IBRD) loan dataset to predict the Loan Status using various machine learning classification models. The workflow includes data cleaning, feature engineering, model training, and performance comparison.

## 📂 Dataset

* Source: World Bank IBRD statement of loans and guarantees (latest snapshot)

* Format: CSV

* Rows: ~10,000+

* Columns: 24,000+ after encoding

* Target Variable: Loan_Status


## 🔍 Objective

The goal is to:

* Clean and preprocess the data

* Compare performance across multiple ML classifiers

* Identify important features contributing to predictions


## 🛠️ Tech Stack

| Category         | Tools Used                                                                 |
|------------------|----------------------------------------------------------------------------|
| Language         | Python 3.10+                                                               |
| Libraries        | pandas, numpy, seaborn, matplotlib, scikit-learn                           |
| ML Models        | Random Forest, Logistic Regression, Decision Tree, SVM, Gradient Boosting  | 
| Notebook/Script  | .ipynb and .py supported                                                   | 



## 📊 Workflow

### ✅ Data Preprocessing

* Renamed columns, handled missing values

* Converted and extracted date features (loan duration, year, month)

* One-hot encoded categorical variables

* Removed outliers using Z-score filtering

* Scaled numerical features using StandardScaler

* Feature selection with SelectKBest (top 100)

### ✅ Model Training

* Trained and evaluated the following classifiers:

* Random Forest

* Logistic Regression

* Decision Tree

* SVM (Linear)

* Gradient Boosting

### ✅ Evaluation Metrics

* For each model:

* Accuracy Score

* Confusion Matrix

* Classification Report (Precision, Recall, F1)

### ✅ Visualizations

* Number of loans by region

* Loan distribution by status code

* Accuracy comparison of models with custom color bar chart (inspired by uploaded image)

* Top 20 most important features (Random Forest)


## 🖼️ Example Visualizations

* 📊 Model Comparison Graph (Bar chart with accuracy for each model)

* 🌍 Region-wise Loan Distribution

* 📋 Loan Status Code Count Plot

* 🌟 Top 20 Feature Importances from Random Forest


## 🚀 How to Run

### 🔹 Option 1: Python Script
```
python final_project.py
```

### 🔹 Option 2: Jupyter Notebook
```
jupyter notebook data-cleaning.ipynb
```


## 📌 Requirements

Install dependencies with:
```
pip install pandas numpy matplotlib seaborn scikit-learn
```
