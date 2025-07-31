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
```
![#](https://github.com/zJayTech/Credit-Card-Fraud-Detection/blob/main/Distribution-of-Transactions.png?raw=true)

### 4. Transaction Time and Amount Distributions by Class

Understanding how transaction time and amounts vary between legitimate and fraudulent transactions can help uncover patterns that aid in detection. This section visualizes the distributions for both the `Time` and `Amount` features by class.

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Time distribution for each class
sns.histplot(df[df['Class'] == 0]['Time'], bins=50, ax=axes[0], color='blue', label='Legitimate')
sns.histplot(df[df['Class'] == 1]['Time'], bins=50, ax=axes[0], color='red', label='Fraud')
axes[0].set_title('Transaction Time Distribution by Class')
axes[0].set_xlabel('Time (seconds elapsed from first transaction)')
axes[0].legend()

# Amount distribution for each class
sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, ax=axes[1], color='blue', label='Legitimate')
sns.histplot(df[df['Class'] == 1]['Amount'], bins=50, ax=axes[1], color='red', label='Fraud')
axes[1].set_title('Transaction Amount Distribution by Class')
axes[1].set_xlabel('Amount')
axes[1].set_yscale('log')  # Use log scale due to skewed distribution
axes[1].legend()

plt.tight_layout()
plt.savefig("Transaction-Time_and_Amount")  # Save the combined plot
plt.show()

# Describe Amount for each class
print("\nAmount Statistics for Legitimate Transactions:")
print(df[df['Class'] == 0]['Amount'].describe())

print("\nAmount Statistics for Fraudulent Transactions:")
print(df[df['Class'] == 1]['Amount'].describe())
```
![#](https://github.com/zJayTech/Credit-Card-Fraud-Detection/blob/main/Transaction-Time_and_Amount.png?raw=true)

### 5. Feature Scaling

Many machine learning algorithms perform better or converge faster when numerical input features are scaled to a standard range. The PCA-transformed features (V1 through V28) in this dataset are already scaled. However, the Time and Amount columns are not, and their raw values could disproportionately influence the model due to their larger magnitudes compared to the PCA features.

To address this, we apply StandardScaler to both Time and Amount features. Standardization (or Z-score normalization) scales the data such that it has a mean of 0 and a standard deviation of 1. This ensures all features contribute equally to the distance calculations in algorithms like Logistic Regression.

After scaling, we create a new DataFrame that includes these scaled features and excludes the original Time and Amount columns to avoid redundancy and potential multicollinearity.

```python
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Drop original 'Time' and 'Amount' if you prefer to use scaled versions
df_scaled = df.drop(['Time', 'Amount'], axis=1)
print(df_scaled.head())
```

### 6. PCA Feature Exploration

The V features (V1 through V28) in this dataset are the result of a Principal Component Analysis (PCA) transformation. This anonymization is typical in sensitive domains like finance to protect privacy. While we don't know the original meaning of these features, their distributions can still reveal important patterns, especially when comparing legitimate and fraudulent transactions. Features that show distinct differences in their distributions between the two classes are likely strong indicators of fraud.

This section demonstrates how to explore these PCA-transformed features by visualizing their distributions for both legitimate (Class 0) and fraudulent (Class 1) transactions. This helps identify features that are potentially highly discriminative for fraud detection.

```python
# Example: Plotting V1 for both classes
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(df_scaled[df_scaled['Class'] == 0]['V1'], bins=50, ax=ax, color='blue', label='Legitimate', kde=True)
sns.histplot(df_scaled[df_scaled['Class'] == 1]['V1'], bins=50, ax=ax, color='red', label='Fraud', kde=True)
ax.set_title('Distribution of V1 by Class')
ax.set_xlabel('V1')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.savefig("Distribution-of-V1.png")  # Saving the plot
plt.show()  # Displaying the plot

# You would repeat this for several 'V' features (e.g., V1, V2, V3, V4, V10, V12, V14, V16, V17, V18 etc.)
# These are often cited as being important in fraud detection research.
```
![#](https://github.com/zJayTech/Credit-Card-Fraud-Detection/blob/main/Distrubution-of-V1.png?raw=true)

### 7. Data Splitting for Modeling

Before training any machine learning model, it's a critical best practice to split your dataset into distinct training and testing sets. This step ensures that we can evaluate how well our model generalizes to new, unseen data, rather than just how well it memorizes the data it was trained on (which leads to overfitting).

For highly imbalanced datasets like credit card fraud, a standard random split isn't enough. We need to use stratified sampling to ensure that both the training and testing sets maintain the same proportion of legitimate and fraudulent transactions as the original dataset. This prevents scenarios where, for example, the test set might contain too few (or even zero) fraudulent cases, leading to unreliable evaluation metrics.

```python
X = df_scaled.drop('Class', axis=1)  # Features (independent variables)
y = df_scaled['Class']               # Target variable (dependent variable)

# Split data into training and testing sets (80% train, 20% test)
# stratify=y is CRUCIAL for imbalanced datasets to ensure both train/test sets
# have similar class distribution as the original dataset.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train Class Distribution:\n", y_train.value_counts(normalize=True))
print("Test Class Distribution:\n", y_test.value_counts(normalize=True))
```

### 8. Model Training (Logistic Regression Baseline)

With the data preprocessed and split, we are now ready to train our first machine learning model. For this project, we start with Logistic Regression as a baseline classifier. Logistic Regression is a simple, yet powerful and interpretable algorithm, making it an excellent choice for an initial model.

Crucially, given the extreme class imbalance in our credit card fraud dataset, simply training a standard Logistic Regression model would likely lead to poor fraud detection performance. The model would tend to overfit to the majority class (legitimate transactions) and ignore the rare fraud cases. To address this, we leverage the `class_weight='balanced'` parameter.

```python
# Train a Logistic Regression model as a baseline
model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
# Using class_weight='balanced' is one way to handle imbalance directly within the model

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class (fraud)
```
### 9. Model Evaluation and Interpretation

After training our Logistic Regression model, the next critical step is to evaluate its performance on the unseen test data. For imbalanced datasets like fraud detection, simply looking at overall accuracy can be misleading (as a model predicting "no fraud" all the time would still be >99% accurate). Therefore, we focus on specific metrics that provide a more nuanced understanding of how well our model identifies the minority class (fraud). We also analyze feature importance to understand which factors contribute most to the model's predictions.

```python
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score: {roc_auc:.4f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve for Fraud Detection')
plt.legend()
plt.savefig("ROC-Curve-for-Fraud.png")  # Ensure .png extension
plt.show()

# Analyze feature coefficients/importance (for Logistic Regression, coefficients are importance proxy)
# Note: For tree-based models, you'd use model.feature_importances_
coefficients = pd.DataFrame({'feature': X_train.columns, 'coefficient': model.coef_[0]})
coefficients['abs_coefficient'] = np.abs(coefficients['coefficient'])
coefficients = coefficients.sort_values(by='abs_coefficient', ascending=False)
print("\nTop 10 Most Important Features (by absolute coefficient):")
print(coefficients.head(10))
```
![#](https://github.com/zJayTech/Credit-Card-Fraud-Detection/blob/main/ROC-Curve-for-Fraud.png?raw=true)

## Conclusion

This project demonstrates a full end-to-end pipeline for detecting credit card fraud using real-world, imbalanced data. By applying key data preprocessing techniques, such as feature scaling and stratified sampling, and building a baseline Logistic Regression model with class weighting, we were able to identify patterns that differentiate fraudulent transactions from legitimate ones.

While Logistic Regression serves as a strong interpretable starting point, future work can include exploring more complex models like Random Forests, XGBoost, or neural networks, as well as experimenting with advanced techniques such as SMOTE for oversampling, anomaly detection methods, or feature engineering to boost performance.

Ultimately, fraud detection remains a high-stakes application of machine learning where model accuracy, interpretability, and robustness must be carefully balanced. This project lays a strong foundation for further experimentation and improvement in this space.

---

**Connect with me on LinkedIn:** [Zack Jones](https://www.linkedin.com/in/zjaytech/)
**Email:** zackjones.dataanalyst@gmail.com
