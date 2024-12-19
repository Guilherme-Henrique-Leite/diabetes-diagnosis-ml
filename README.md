# Diabetes Prediction Model - Machine Learning

This project demonstrates the application of machine learning techniques to predict the likelihood of diabetes in individuals using the **Pima Indians Diabetes Database**. The goal is to predict whether a person has diabetes based on features such as age, glucose level, BMI, and other medical data. The project leverages two popular models: **Logistic Regression** and **Random Forest**, along with data preprocessing techniques to enhance model performance.

## Table of Contents
- [Overview](#overview)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Results](#results)
- [Conclusion](#conclusion)
- [Contact](#contact)

## Overview

Diabetes is one of the leading causes of death worldwide, and early detection is crucial for effective management. This project explores the use of machine learning algorithms to classify whether a person has diabetes based on various health metrics.

### Key Techniques Used:
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Used to balance the imbalanced dataset.
- **Random Forest**: Applied as a robust classification model for diabetes prediction.
- **Logistic Regression**: Implemented as a simpler, interpretable model for comparison.
- **Standardization**: Ensures all features have similar scales for better model performance.

The project uses Python libraries such as `Pandas`, `Scikit-learn`, and `Imbalanced-learn` to preprocess, train, and evaluate the models.

## Technologies

- **Python** (v3.7+)
- **Jupyter Notebook** (for model experimentation)
- **Scikit-learn** (for machine learning algorithms)
- **Imbalanced-learn** (for SMOTE oversampling)
- **Pandas** (for data manipulation)
- **Matplotlib/Seaborn** (for data visualization)

## Project Structure

The project is organized as follows:

### Files Breakdown:
- `data/`: Contains the raw dataset for the diabetes prediction.
- `notebooks/`: Jupyter notebooks for data exploration, preprocessing, model training, and evaluation.
- `src/`: Contains Python scripts for data preprocessing (`preprocessing.py`), models training and any utility functions.
- `requirements.txt`: Lists the required Python libraries and dependencies to run the project.
- `.gitignore`: Specifies files and directories that should not be tracked by Git, such as virtual environments and cache files.

## How to Run

### 1. Clone the Repository

To start working with the project, clone this repository to your local machine:

```
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
```

### 2. Set up the Environment
It is recommended to use a virtual environment to manage dependencies. If you're using pip, you can set up a virtual environment by running the following commands:
  ```
  python3 -m venv venv
  - source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
  ```

Install the required dependencies:
  ```
  - pip install -r requirements.txt
  ```

# Diabetes Prediction Model - Machine Learning

This project demonstrates the application of machine learning techniques to predict the likelihood of diabetes in individuals using the **Pima Indians Diabetes Database**. The goal is to predict whether a person has diabetes based on features such as age, glucose level, BMI, and other medical data. The project leverages two popular models: **Logistic Regression** and **Random Forest**, along with data preprocessing techniques to enhance model performance.

## Table of Contents
- [Overview](#overview)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Results](#results)
- [Conclusion](#conclusion)
- [Contact](#contact)

## Overview

Diabetes is one of the leading causes of death worldwide, and early detection is crucial for effective management. This project explores the use of machine learning algorithms to classify whether a person has diabetes based on various health metrics. 

### Key Techniques Used:
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Used to balance the imbalanced dataset.
- **Random Forest**: Applied as a robust classification model for diabetes prediction.
- **Logistic Regression**: Implemented as a simpler, interpretable model for comparison.
- **Standardization**: Ensures all features have similar scales for better model performance.

The project uses Python libraries such as `Pandas`, `Scikit-learn`, and `Imbalanced-learn` to preprocess, train, and evaluate the models.

## Technologies

- **Python** (v3.12+)
- **Jupyter Notebook** (for model experimentation)
- **Scikit-learn** (for machine learning algorithms)
- **Imbalanced-learn** (for SMOTE oversampling)
- **Pandas** (for data manipulation)
- **Matplotlib/Seaborn** (for data visualization)

## Project Structure

### Files Breakdown:
- `data/`: Contains the raw dataset for the diabetes prediction.
- `notebooks/`: Jupyter notebooks for data exploration, preprocessing, model training, and evaluation.
- `src/`: Contains Python scripts for data preprocessing (`preprocessing.py`), model training (`model.py`), and any utility functions.
- `requirements.txt`: Lists the required Python libraries and dependencies to run the project.
- `.gitignore`: Specifies files and directories that should not be tracked by Git, such as virtual environments and cache files.

## How to Run

### 1. Clone the Repository

To start working with the project, clone this repository to your local machine:

```
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
```

2. Set up the Environment
It is recommended to use a virtual environment to manage dependencies. If you're using pip, you can set up a virtual environment by running the following commands:

python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required dependencies:
```
- pip install -r requirements.txt
```

3. Run the Project
You can either run the Jupyter notebook for model exploration or use the Python scripts for data preprocessing and model training:

Option 1: Open the notebook notebooks/diabetes_prediction_notebook.ipynb and run each cell to explore the model's performance.
Option 2: Use the Python scripts in the src/ folder to preprocess the data and train the models:
```
Run src/preprocessing.py to preprocess the data and handle missing values.
```

Logistic Regression: The model achieved an accuracy of 75.3%, with a precision, recall, and F1-score that showed a good balance between the two classes.
Random Forest: The model achieved an accuracy of 76%, with improved performance on classifying diabetic cases.
Confusion Matrix and Classification Report:
Random Forest:
  - Accuracy: 0.76
  - Precision: 0.78 (weighted average)
  - Recall: 0.76 (weighted average)
  - F1-Score: 0.76 (weighted average)


Conclusion
The project demonstrates the effectiveness of machine learning techniques in predicting diabetes. While the models performed well, further improvements could be made by exploring additional feature engineering techniques or hyperparameter tuning. The use of SMOTE was essential in balancing the imbalanced dataset, which resulted in better performance on minority class prediction.
