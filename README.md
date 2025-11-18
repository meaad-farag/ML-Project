# 🤖 Student Performance Predictor: A Machine Learning Classification Project

[![GitHub](https://img.shields.io/badge/GitHub-Project-blue?style=for-the-badge&logo=github)](link_to_your_github_repo)
[![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Model_Training-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-FF6A00?style=for-the-badge&logo=jupyter)](https://jupyter.org/)

## 🌟 Project Overview

This project implements a **binary classification** machine learning model to predict the academic outcome of students—specifically, whether a student is likely to **Pass or Fail** their final examination. Utilizing data science and key ML algorithms, the goal is to provide a robust predictive tool for educational institutions to facilitate **early intervention** and support for at-risk students.

### 🎯 Key Machine Learning Skills Demonstrated

* **Data Preprocessing and Feature Engineering:** Handling mixed data types, encoding categorical variables, and managing data quality.
* **Addressing Class Imbalance:** Applying techniques like **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the target variable for reliable model training.
* **Model Selection and Evaluation:** Training, comparing, and fine-tuning multiple classification models using industry-standard metrics.
* **Model Interpretability:** Analyzing **Feature Importance** to understand the key drivers of student performance.

---

## 💻 Technologies and Libraries

The project is developed entirely in **Python** and relies on the following key libraries:

| Category | Library | Purpose |
| :--- | :--- | :--- |
| **Data Manipulation** | `Pandas`, `NumPy` | Data loading, cleaning, transformation, and numerical operations. |
| **Machine Learning** | `Scikit-learn` | Model implementation (Logistic Regression, Decision Tree, SVM), data splitting, and performance metrics (Accuracy, F1, AUC). |
| **Data Imbalance** | `imblearn` (SMOTE) | Handling the imbalanced nature of the target variable. |
| **Visualization** | `Matplotlib`, `Seaborn` | Exploratory Data Analysis (EDA) and visualizing model results (e.g., ROC Curves, Feature Importance plots). |
| **Development** | `Jupyter Notebook` | Interactive development and step-by-step documentation of the entire ML pipeline. |

---

## ⚙️ ML Pipeline Steps

The project follows a rigorous machine learning workflow documented step-by-step in `code.ipynb`.

### 1. Data Loading and Exploration (EDA)
* Initial inspection of the dataset structure and features (demographics, study habits, past grades).
* Handling initial missing values (if any) and identifying categorical vs. numerical features.

### 2. Data Preprocessing and Feature Engineering
* **Categorical Encoding:** Converting non-numerical features (e.g., 'sex', 'school') into a numerical format suitable for ML algorithms (e.g., **Label Encoding** or **One-Hot Encoding**).
* **Feature Scaling:** Applying **StandardScaler** to numerical features to standardize the input range, which is critical for distance-based algorithms like SVM and Logistic Regression. 
* **Target Variable Preparation:** Converting the continuous final grade into a binary classification outcome (Pass/Fail).

### 3. Addressing Class Imbalance (SMOTE)
* Analyzing the distribution of the target class (Pass/Fail).
* Applying **SMOTE** on the training data to synthetically create samples for the minority class, ensuring the models do not have a prediction bias towards the majority class.

### 4. Model Training and Comparison
* The dataset is split into training and testing subsets (`train_test_split`).
* Three different classification models were trained and benchmarked:
    * **Logistic Regression:** A linear model used to predict the probability of a binary outcome.
    * **Decision Tree:** A non-linear, highly interpretable model.
    * **Support Vector Machine (SVM):** Utilized for effective boundary separation in the feature space.

### 5. Model Evaluation and Improvement
* Performance metrics were calculated for each model on the test set, including:
    * **Accuracy**
    * **Precision** and **Recall**
    * **F1-Score** (Crucial metric for classification on imbalanced data)
    * **ROC AUC Score** and plotting the **ROC Curve**. 

[Image of ROC Curve for Logistic Regression]

* **Hyperparameter Tuning** was used to achieve the final optimized performance metrics (e.g., **Accuracy: 79.75%**, **F1 Score: 0.82**).

### 6. Feature Importance Analysis
* Analyzing the coefficients of the **Logistic Regression** model and the feature importance scores from the **Decision Tree** to identify the most significant factors affecting student performance (e.g., **past failures** and **absences** were key predictors).

---

## 🚀 Execution Guide (Anaconda Environment)

To replicate the results and run the `code.ipynb` notebook:

### Prerequisites

You must have **Anaconda** installed to manage the Python environment and dependencies.

### 1. Setup the Environment

Open your terminal or **Anaconda Prompt** and execute the following commands to create and activate a dedicated environment:

```bash
# Create a new environment named 'student_env'
conda create -n student_env python=3.9

# Activate the environment
conda activate student_env
