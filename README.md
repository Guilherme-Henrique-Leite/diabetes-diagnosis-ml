# Diabetes Prediction Model - Machine Learning 🚀

This project demonstrates the application of machine learning techniques to predict the likelihood of diabetes in individuals using the **Pima Indians Diabetes Database**. The goal is to predict whether a person has diabetes based on features such as age, glucose level, BMI, and other medical data. The project leverages two popular models: **Logistic Regression** and **Random Forest**, along with data preprocessing techniques to enhance model performance.

## Table of Contents
- [About the Project](#about-the-project)
- [Technologies](#technologies-%EF%B8%8F)
- [Project Structure](#project-structure-)
- [How to Run](#how-to-run)
- [Results](#results)

## About the Project 
Diabetes is one of the leading causes of death worldwide, and early detection can save lives. This project explores the use of machine learning to classify diabetes risk based on health metrics like BMI, glucose level, and age.

### Key Techniques Used:
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Used to balance the imbalanced dataset.
- **Random Forest**: Applied as a robust classification model for diabetes prediction.
- **Logistic Regression**: Implemented as a simpler, interpretable model for comparison.
- **Standardization**: Ensures all features have similar scales for better model performance.

The project uses Python libraries such as `Pandas`, `Scikit-learn`, and `Imbalanced-learn` to preprocess, train, and evaluate the models.

## Technologies 🛠️

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-orange?logo=jupyter)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24.2-orange)
![Pandas](https://img.shields.io/badge/Pandas-1.5.2-blue?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Seaborn-brightgreen)
![Imbalanced-learn](https://img.shields.io/badge/Imbalanced--learn-0.9.1-red)


## Project Structure 📁

The project is organized as follows:
```
diabetes-prediction/
├── notebooks/
│   └── diabetes_diagnosis_ml.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── random_forest_model.py
│   ├── regression_model.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

### 1. Clone the Repository

To start working with the project, clone this repository to your local machine:

```
git clone https://github.com/Guilherme-Henrique-Leite/diabetes-diagnosis-ml.git
cd diabetes-prediction
```

### 2. Set up the Environment
It is recommended to use a virtual environment to manage dependencies. If you're using pip, you can set up a virtual environment by running the following commands:
  ```
  python3 -m venv venv
  - source venv/bin/activate  # On Windows, use venv\Scripts\activate
  ```

Install the required dependencies:
  ```
  - pip install -r requirements.txt
  ```

### 3. Run the Project
You can either run the Jupyter notebook for model exploration or use the Python scripts for data preprocessing and model training:

Option 1: Open the notebook notebooks/diabetes_diagnosis_ml.ipynb and run each cell to explore the model's performance.

Option 2: Use the Python scripts in the src/ folder to preprocess the data and train the models:
```
Run src/data_preprocessing.py to preprocess the data and handle missing values.
```

## Results 📊

### Model Performance
| Model                | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
|----------------------|----------|-----------------------|-------------------|---------------------|
| Logistic Regression  | 75.3%    | 0.75                  | 0.75              | 0.75                |
| Random Forest        | 76.0%    | 0.78                  | 0.76              | 0.76                |
