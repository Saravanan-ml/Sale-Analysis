import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. CONFIGURATION AND INITIAL SETUP ---
# Set a style for the plots for better aesthetics
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

# The name of the file. Assuming it's in the same directory as the script.
# Using a raw string (r"...") or forward slashes (/) is the correct way to handle
# Windows file paths to avoid the "invalid escape sequence" error.
file_path = r'D:\DataAnalysis work\Ecommerce_Sales_Prediction_Dataset.csv'

# --- 2. DATA LOADING AND INITIAL INSPECTION ---
def load_data(path):
    """Loads the dataset from a CSV file with error handling."""
    try:
        df = pd.read_csv(path)
        print("Dataset loaded successfully!")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
        return None

if __name__ == "__main__":
    df = load_data(file_path)

    # Proceed only if the dataset was loaded successfully
    if df is not None:
        print("\n--- Initial Data Inspection ---")
        print(df.head())
        print("\n--- Data Information ---")
        df.info()

        # --- 3. DATA PREPROCESSING AND FEATURE ENGINEERING ---
        print("\n--- Preprocessing Data and Creating New Features ---")

        # Convert 'Date' column to datetime objects, coercing errors
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

        # Drop columns with a significant number of missing values (more than a certain threshold)
        # We also explicitly drop 'Unnamed' columns and the duplicate 'date' column
        cols_to_drop = [col for col in df.columns if 'Unnamed:' in col or col == 'date']
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        print(f"Dropped extra columns: {cols_to_drop}")

        # Drop rows with any remaining missing values
        df.dropna(inplace=True)
        print("Removed remaining rows with missing values.")

        # Create new time-based features from the 'Date' column
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek # Monday=0, Sunday=6
        df['DayOfYear'] = df['Date'].dt.dayofyear

        # Calculate the total sales (our target variable)
        df['Total_Sales'] = df['Price'] * df['Units_Sold']

        print("\n--- Data after Preprocessing and Feature Engineering ---")
        print(df.head())
        print("\n--- Summary Statistics ---")
        print(df.describe())
        print("\n--- Data Information after all cleaning ---")
        df.info()

        # --- 4. DATA VISUALIZATION FOR INSIGHTS ---
        print("\n--- Data Visualization ---")

        # Sales trend over time
        plt.figure(figsize=(15, 7))
        df.groupby('Date')['Total_Sales'].sum().plot()
        plt.title('Daily Total Sales Trend Over Time')
        plt.xlabel('Date')
        plt.ylabel('Total Sales ($)')
        plt.show()

        # Sales by Product Category
        plt.figure(figsize=(12, 6))
        # Changed ci=None to errorbar=None to fix the FutureWarning
        sns.barplot(x='Product_Category', y='Total_Sales', data=df, estimator=sum, errorbar=None)
        plt.title('Total Sales by Product Category')
        plt.xlabel('Product Category')
        plt.ylabel('Total Sales ($)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Sales by Customer Segment
        plt.figure(figsize=(12, 6))
        # Changed ci=None to errorbar=None to fix the FutureWarning
        sns.barplot(x='Customer_Segment', y='Total_Sales', data=df, estimator=sum, errorbar=None)
        plt.title('Total Sales by Customer Segment')
        plt.xlabel('Customer Segment')
        plt.ylabel('Total Sales ($)')
        plt.show()

        # --- 5. MODEL PREPARATION ---
        # Define features (X) and target (y)
        features = ['Product_Category', 'Discount', 'Customer_Segment', 'Marketing_Spend',
                    'Year', 'Month', 'DayOfWeek', 'DayOfYear']
        target = 'Total_Sales'

        X = df[features]
        y = df[target]

        # Identify categorical and numerical columns for preprocessing
        categorical_features = ['Product_Category', 'Customer_Segment']
        numerical_features = ['Discount', 'Marketing_Spend', 'Year', 'Month', 'DayOfWeek', 'DayOfYear']
        
        # Create a preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                ('num', StandardScaler(), numerical_features)
            ],
            remainder='passthrough'
        )

        # Split the data into training and testing sets (80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("\n--- Data Split for Training and Testing ---")
        print(f"Training set shape: {X_train.shape}")
        print(f"Testing set shape: {X_test.shape}")

        # --- 6. MODEL TRAINING & EVALUATION (Linear Regression) ---
        # Create and train the Linear Regression pipeline
        linear_model = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('regressor', LinearRegression())])

        print("\n--- Training the Linear Regression Model ---")
        linear_model.fit(X_train, y_train)
        print("Model training complete.")

        # Evaluate Linear Regression model's performance
        y_pred_linear = linear_model.predict(X_test)
        r2_linear = r2_score(y_test, y_pred_linear)
        print("\n--- Linear Regression Model Evaluation ---")
        print(f"R-squared (R²): {r2_linear:.2f}")

        # --- 7. MODEL TRAINING & EVALUATION (Random Forest Regressor) ---
        # Create and train a more complex model: RandomForestRegressor
        forest_model = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

        print("\n--- Training the Random Forest Regressor Model ---")
        forest_model.fit(X_train, y_train)
        print("Model training complete.")

        # Evaluate Random Forest model's performance
        y_pred_forest = forest_model.predict(X_test)
        mae_forest = mean_absolute_error(y_test, y_pred_forest)
        r2_forest = r2_score(y_test, y_pred_forest)
        print("\n--- Random Forest Regressor Model Evaluation ---")
        print(f"Mean Absolute Error (MAE): {mae_forest:.2f}")
        print(f"R-squared (R²): {r2_forest:.2f}")

        # Visualize the actual vs. predicted values for the Random Forest model
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_forest, alpha=0.6, edgecolors='w', linewidths=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
        plt.title('Actual vs. Predicted Total Sales (Random Forest)')
        plt.xlabel('Actual Sales')
        plt.ylabel('Predicted Sales')
        plt.legend()
        plt.show()

        # --- 8. MAKING A PREDICTION ON NEW DATA ---
        print("\n--- Making a Prediction on New Data using Random Forest Model ---")
        new_data = pd.DataFrame([{
            'Product_Category': 'Toys',
            'Discount': 10.5,
            'Customer_Segment': 'Premium',
            'Marketing_Spend': 5000,
            'Year': 2025,
            'Month': 8,
            'DayOfWeek': 4, # Friday
            'DayOfYear': 242 # approx
        }])

        # Make the prediction
        predicted_sales = forest_model.predict(new_data)
        print(f"Predicted sales for the new data point: ${predicted_sales[0]:.2f}")
