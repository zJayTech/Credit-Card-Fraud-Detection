# Credit Card Fraud Detection Project

## Project Overview

In today's digital economy, credit card fraud is a pervasive and costly problem for financial institutions and consumers alike. The rapid increase in online transactions necessitates robust systems capable of identifying and preventing fraudulent activities in real-time. This project delves into the challenging domain of credit card fraud detection, focusing on the application of data analysis and machine learning techniques to uncover suspicious patterns.

The core objective is to analyze a dataset of real-world credit card transactions, understand the underlying characteristics that differentiate legitimate transactions from fraudulent ones, and build a baseline predictive model. A significant challenge in this field is the extreme class imbalance, where fraudulent transactions represent a minuscule fraction of the total. This project specifically addresses how to approach such imbalanced datasets, which is a critical skill for any data professional.

Through this analysis, I aim to demonstrate proficiency in:

- Exploratory Data Analysis (EDA): Uncovering insights and patterns within complex transactional data.
- Data Preprocessing: Preparing raw data for machine learning models, including handling scaling.
- Handling Imbalanced Data: Employing strategies to build effective models when target classes are disproportionately represented.
- Classification Modeling: Building and evaluating a predictive model to identify fraudulent transactions.
- Result Interpretation & Communication: Translating technical findings into actionable insights.

This repository serves as a practical demonstration of a data science workflow applied to a real-world financial problem, showcasing the steps from raw data to actionable insights and a preliminary predictive model.

## Dataset

The dataset utilized for this project is the Credit Card Fraud Detection dataset, generously provided by ULB (Université Libre de Bruxelles) and available on Kaggle.

Key characteristics of the dataset include:

- Approximately 285,000 transactions, predominantly from European cardholders, occurring over two days in September 2013.
- Features V1 through V28: These are the result of a PCA (Principal Component Analysis) transformation, a common technique used to protect sensitive user information while retaining the predictive power of the original features. This anonymization means the exact nature of these features is unknown, requiring a focus on their statistical properties and relationships.
- Time feature: Represents the seconds elapsed between each transaction and the first transaction in the dataset.
- Amount feature: The transaction amount.
- Class feature (Target Variable): Binary indicator where 0 signifies a legitimate transaction and 1 signifies a fraudulent transaction.
- Severe Imbalance: A crucial aspect is that only about 0.172% of all transactions are fraudulent, posing a significant challenge for model training and evaluation.

## Getting Started

To replicate this analysis, please ensure you have the creditcard.csv dataset downloaded from the Kaggle link above and placed in the appropriate directory (as referenced in the code, typically the same folder as your Jupyter Notebook or Python script).

The following section outlines the initial steps involved in setting up the environment and loading the data.

### 1. Importing Libraries and Loading Data

This foundational step initiates our analysis by bringing in all the necessary Python libraries that equip us with tools for data manipulation, visualization, and machine learning. Following the environment setup, the credit card transaction dataset is loaded into a Pandas DataFrame, making it accessible for subsequent processing and analysis.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Specify the path to your downloaded dataset
# IMPORTANT: Update this path to where you have saved 'creditcard.csv' on your local machine.
df = pd.read_csv(r"C:\Users\Custom\Documents\pandas\datasets\creditcard.csv")
```
### 2. Initial Data Overview

Once the dataset is loaded, the first crucial step in any data analysis project is to get a preliminary understanding of its structure, content, and quality. This involves examining the first few rows, checking data types, reviewing basic statistical summaries, and identifying any missing values. This step helps in forming initial hypotheses and planning subsequent data cleaning and preprocessing steps.

```python
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())  # Use print() to ensure it appears in Markdown

print("\nDescriptive Statistics for Numerical Columns:")
print(df.describe())

print("\nMissing Values per Column:")
print(df.isnull().sum())  # Confirm no missing values
```
### 3. Class Imbalance Analysis

Credit card fraud detection is a classic example of an imbalanced classification problem. This means that one class (legitimate transactions) vastly outnumbers the other class (fraudulent transactions). Understanding this imbalance is critical, as it directly impacts how we build and evaluate our machine learning models. If not properly addressed, models might simply predict the majority class, leading to high accuracy but failing to detect the rare, yet critical, fraudulent cases.

This section quantifies and visualizes this imbalance.

```python
print("\nClass Distribution:")
print(df['Class'].value_counts())
print(df['Class'].value_counts(normalize=True) * 100)

sns.countplot(x='Class', data=df)
plt.title('Distribution of Transactions (0: Legitimate, 1: Fraud)')
plt.savefig("Distribution-of-Transactions.png")  # Saving the plot
plt.show()  # Displaying the plot
