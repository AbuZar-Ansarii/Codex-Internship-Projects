## CODEX INTERNSHIP PROJECTS
A collection of machine learning projects demonstrating classification and regression techniques on classic datasets.

##  📁 Projects Overview
1. Iris Flower Classification
Type: Multi-class Classification
Algorithm: Various classifiers (LogisticRegressionCV, KNN, Decision Trees)
Description: Classification of iris flowers into three species (Setosa, Versicolour, Virginica) based on sepal and petal measurements.

2. Boston House Price Prediction
Type: Regression
Algorithm: Linear Regression.
Description: Predicting housing prices in Boston based on various features like crime rate, number of rooms, and accessibility to highways.

3. Spam Mail Classification
Type: Binary Classification
Algorithm: Naive Bayes, Logistic RegresionCV
Description: Classifying emails as spam or ham (non-spam) using natural language processing techniques.

 
## 🚀 Technologies Used
Python 

Scikit-learn - Machine learning algorithms

Pandas - Data manipulation and analysis

NumPy - Numerical computing

Matplotlib & Seaborn - Data visualization

Jupyter Notebook - Interactive development

## 📊 Datasets
Iris Dataset: Publicly available data on Kaggle

Boston Housing Dataset: Publicly available data on Kaggle

Spam Collection Dataset: Publicly available SMS spam collection Kaggle

## 📈 Results Summary

# Iris Flower Classification
Best Model: Decision Tree

Decision Tree ACCURACY =  100.0

Key Features: Sepal length, sepal width, petal length, petal width

# Spam Mail Classification
Best Model: Naive Bayes 

Accuracy: 98.20627802690582

Precision: 
Ham : 0.98,  
Spam : 0.98

Recall:
Ham : 1.00, 
Spam : 0.86

F1 Score:
Ham : 0.99, 
Spam : 0.91

Key Features: Word frequency, special characters, message length

# Boston House Price Prediction
Best Model: LinearRegression

mean_squared_error: 24.40482518814649

mean_absolute_error: 3.2064039639003847

Key Features: RM (number of rooms), LSTAT (lower status population), PTRATIO (pupil-teacher ratio)

# 📝 Key Learnings
Data preprocessing and cleaning techniques

Feature engineering and selection

Model evaluation and hyperparameter tuning

Cross-validation strategies

Handling imbalanced datasets (for spam classification)

Interpretation of model results and feature importance

## 📋 Project Structure
```sh

CODEX INTERNSHIP ML-Projects/
│
├── Iris-Classification/
│   ├── iris_classification.ipynb
│   ├── iris_classification.py
│   └── requirements.txt
│
├── Boston-House-Price-Prediction/
│   ├── boston_housing.ipynb
│   ├── boston_housing.py
│   └── requirements.txt
│
├── Spam-Classification/
│   ├── spam_classification.ipynb
│   ├── spam_classification.py
│   └── requirements.txt
│
└── README.md

