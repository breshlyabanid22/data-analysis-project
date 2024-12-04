import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from io import StringIO

# Set page layout
st.set_page_config(page_title="Housing Price Prediction", layout="wide")

# App title and introduction
st.title("🏠 Housing Price Prediction in Delhi")
st.markdown(
    """
    Welcome to the **Housing Price Prediction** app! 🎉  
    This app leverages **machine learning models** to predict housing prices based on various features like area, bedrooms, bathrooms, and location factors.
    The dataset is sourced from Kaggle and contains over 1,000 rows, offering rich insights for analysis.
    """
)
st.sidebar.title("🔧 App Settings")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Housing.csv")
    df = pd.get_dummies(df, drop_first=True)  # Convert categorical to numeric
    return df

st.sidebar.subheader("Dataset Overview")
df = load_data()

# Display dataset preview
if st.sidebar.checkbox("Show Dataset Preview", value=True):
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10))

# Dataset Info
buffer = StringIO()
df.info(buf=buffer)
dataset_info = buffer.getvalue()

if st.sidebar.checkbox("Show Dataset Information"):
    st.subheader("ℹ️ Dataset Information")
    st.text(dataset_info)

# Summary Statistics
if st.sidebar.checkbox("Show Summary Statistics"):
    st.subheader("📈 Summary Statistics")
    st.write(df.describe())

# Feature Scaling and Engineering
st.sidebar.subheader("Data Preprocessing")
scaler = StandardScaler()
numerical_cols = ["area", "bedrooms", "bathrooms", "stories", "parking"]
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
df["bedrooms_area"] = df["bedrooms"] * df["area"]

# Splitting data
X = df.drop(columns=["price"])
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection
st.sidebar.subheader("Model Selection")
models = {
    "Ridge Regression": Ridge(alpha=0.1),
    "Lasso Regression": Lasso(alpha=0.1),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
}
selected_models = st.sidebar.multiselect(
    "Select Models to Train", options=list(models.keys()), default=list(models.keys())
)

# Train and Evaluate Models
st.subheader("🚀 Model Training and Comparison")
results = []
for name in selected_models:
    model = models[name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({"Model": name, "MSE": mse, "MAE": mae, "R-Squared": r2})

results_df = pd.DataFrame(results)
st.dataframe(results_df)

# Best Model Visualization
if "Random Forest Regressor" in selected_models:
    best_model = models["Random Forest Regressor"]
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    st.divider()
    st.subheader("📊 Visualizations")

    # Actual vs Predicted
    st.markdown("### 🎯 Actual vs Predicted Prices")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Predicted Prices")
    ax.set_title("Actual vs Predicted Housing Prices")
    st.pyplot(fig)

    # Residuals Plot
    residuals = y_test - y_pred
    st.markdown("### 📉 Residuals Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, residuals, alpha=0.7)
    ax.axhline(y=0, color="red", linestyle="--")
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals Plot")
    st.pyplot(fig)


