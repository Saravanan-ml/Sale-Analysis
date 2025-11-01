# 🛒 E-commerce Sales Prediction using Machine Learning

## 📖 Overview
This project aims to **analyze and predict e-commerce sales** using machine learning techniques.  
It includes **data cleaning, visualization, feature engineering, and predictive modeling** using both **Linear Regression** and **Random Forest Regressor** models.

The goal is to understand key factors influencing sales and predict future sales performance based on historical data.

---

## 📂 Project Structure

Ecommerce-Sales-Prediction/

│
├── 📄 Ecommerce_Sales_Prediction.py           # Main Python script

├── 📊 Ecommerce_Sales_Prediction_Dataset.csv  # Dataset file

├── 📦 requirements.txt                        # Required dependencies

└── 📝 README.md                               # Project documentation



## 🧠 Key Objectives
- Perform **data preprocessing and feature engineering** on raw e-commerce data.
- Explore data through **visualization** (trends, product categories, customer segments).
- Train and evaluate **machine learning models** to predict total sales.
- Compare performance between **Linear Regression** and **Random Forest Regressor**.
- Predict sales for **new data entries** using the trained model.

---

## ⚙️ Technologies Used
- **Python 3**
- **Pandas** – Data manipulation  
- **NumPy** – Numerical computations  
- **Matplotlib & Seaborn** – Data visualization  
- **Scikit-learn** – Machine learning modeling and preprocessing  

---

## 🧩 Workflow Summary

### 1. Data Loading & Inspection
- Loads CSV data with proper error handling.
- Displays initial rows and dataset info.

### 2. Data Preprocessing & Feature Engineering
- Converts `Date` column to datetime.
- Drops unnecessary columns (`Unnamed` or duplicate date columns).
- Handles missing values.
- Creates new features:
  - `Year`, `Month`, `DayOfWeek`, `DayOfYear`
- Calculates `Total_Sales = Price × Units_Sold`.

### 3. Data Visualization
- **Sales Trend Over Time**
- **Sales by Product Category**
- **Sales by Customer Segment**

### 4. Model Building
- Defines features and target variable (`Total_Sales`).
- Preprocesses data using:
  - `OneHotEncoder` for categorical features
  - `StandardScaler` for numerical features
- Splits data (80% training, 20% testing).

### 5. Model Training & Evaluation
#### a. Linear Regression
- Trains a baseline regression model.
- Evaluates using R² score.

#### b. Random Forest Regressor
- Trains an ensemble model for better accuracy.
- Evaluates using:
  - Mean Absolute Error (MAE)
  - R² score

### 6. Visualization of Model Results
- Plots **Actual vs Predicted** sales for Random Forest.

### 7. Prediction on New Data
- Predicts sales for a new hypothetical input.

---

## 📊 Sample Output

<img width="950" height="650" alt="image" src="https://github.com/user-attachments/assets/17248cfa-c9a9-4109-b810-e2b558459b41" />
<img width="950" height="650" alt="image" src="https://github.com/user-attachments/assets/e5097f39-c697-49eb-9747-36f2231f0c09" />
 

