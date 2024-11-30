import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from io import StringIO

st.title('Data Analysis and Techniques')
st.write('''For our data analysis project, we have chosen the Housing Price Prediction dataset from /n 
         Kaggle. This dataset contains detailed information on housing prices and various features such as the
number of bedrooms, square footage, and location factors. The dataset comprises over 1,000
rows, making it robust for analysis and modeling''')


st.subheader('Dataset')
st.write('Housing Price in Delhi')
def load_data():
    df = pd.read_csv('Housing.csv')
    df = pd.get_dummies(df, drop_first=True)  # Convert categorical to numeric
    return df

df = load_data()

st.dataframe(df.head(100))

# Get a summary of the dataset
buffer = StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.divider()

st.subheader("Dataset Information")
st.text(s)
st.divider()

st.subheader('Summary Statistics')
st.write(df.describe())


# Feature scaling
scaler = StandardScaler()
numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Feature engineering: add interaction terms
df['bedrooms_area'] = df['bedrooms'] * df['area']

st.divider()
st.subheader('Model Training and Comparison')
# Split data
X = df.drop(columns=['price'])
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": Ridge(alpha=0.1),  # Ridge Regression as an improvement
    "Lasso Regression": Lasso(alpha=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append((name, mse, mae, r2))

# Display results
results_df = pd.DataFrame(results, columns=["Model", "MSE", "MAE", "R-squared"])
st.write("Model Comparison:", results_df)

st.divider()
st.subheader('Visualizations')

# Visualize actual vs predicted for best-performing model
best_model = RandomForestRegressor(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred)
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
ax.set_xlabel("Actual Prices")
ax.set_ylabel("Predicted Prices")
ax.set_title("Actual vs Predicted Housing Prices")
st.pyplot(fig)

# Residuals plot
residuals = y_test - y_pred
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, residuals)
ax.axhline(y=0, color='red', linestyle='--')
ax.set_xlabel("Actual Prices")
ax.set_ylabel("Residuals")
ax.set_title("Residuals Plot")
st.pyplot(fig)
