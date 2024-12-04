import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(page_title="Housing Price Prediction", layout="wide")

st.sidebar.title('🏠 Housing Price Prediction in Delhi')
page = st.sidebar.radio('Go to', ['Home', 'Data Exploration and Preparation', 'Analysis and Insights', 'Conclusion and Recommendations'])

# Load Data Function (Housing.csv)
def load_data():
    df = pd.read_csv('Housing.csv')
    df = pd.get_dummies(df, drop_first=True)  
    return df
df = load_data()

if page == 'Home':
    st.title('Data Analysis and Techniques')
    st.write('''For our data analysis project, we have chosen the Housing Price Prediction dataset from
            Kaggle. This dataset contains detailed information on housing prices and various features such as the
            number of bedrooms, square footage, and location factors. The dataset comprises over 1,000
            rows, making it robust for analysis and modeling''')
    st.subheader('Dataset')
    st.subheader('The dataset contains the following columns:')
    st.markdown(' [ ' + ' ], [ '.join(df.columns) + ' ] ')
    st.title('Housing Price in Delhi')
    
    if st.checkbox('Show Dataset Preview', value= True):
        df = load_data()
        st.subheader('📊 Dataset of House Pricing')
        st.dataframe(df.head(200))


    st.divider()

    col1, col2 = st.columns(2)
    
    with col1:
        if st.checkbox('Show Data Information'):
            st.subheader("Dataset Information")
            buffer = StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

    with col2:
        if st.checkbox('Show Summary Statistics'):
            st.subheader('📈 Summary Statistics')
            st.write(df.describe())
        
    st.divider()
elif page == 'Data Exploration and Preparation':
    st.title('Data Exploration and Preparation')
    st.subheader('Data Exploration and Preparation')

    st.write('Cleaning data, handling missing values, and preparing for analysis...')
    st.write('Shape of dataset:', df.shape)
    st.write('Number of missing values:', df.isnull().sum().sum())
    st.write('Number of duplicate rows:', df.duplicated().sum())

    st.divider()
    
    
    st.subheader('Data Cleaning and Preparation')
    col1, col2 = st.columns(2)

   
    with col1:
        st.subheader('Data Exploration')
        st.write('Before cleaning:')
        st.write(df.isnull().sum())

    
    with col2:
        st.subheader('Data Preparation')
        df = df.drop_duplicates()
        df = df.dropna()  
        st.write('After cleaning:')
        st.write(df.isnull().sum())

    st.divider()
    st.subheader('Histograms for numerical columns:')
    numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

    col1, col2 = st.columns(2)
    columns = [col1, col2]

    for i, col in enumerate(numerical_cols):
        with columns[i % 2]:
            if st.checkbox(f'Show Histogram for {col}'):
                st.bar_chart(df[col].value_counts())

    st.divider()
    
    st.subheader('Correlation Matrix')
    with st.expander('Correlation Matrix'):
        corr = df[numerical_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)        
        st.pyplot(fig)

elif page == 'Analysis and Insights':

    scaler = StandardScaler()
    numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    df['bedrooms_area'] = df['bedrooms'] * df['area']

    st.divider()
        
    st.subheader('Model Training and Comparison')

    X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    models = {
        "Linear Regression": Ridge(alpha=0.1),  
        "Lasso Regression": Lasso(alpha=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write("Regression Results:")
    st.write(f'R^2 Score: {r2_score(y_test, y_pred)}')
    st.write(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
    st.write(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}')

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
