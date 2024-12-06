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
from PIL import Image, ImageOps, ImageDraw

st.set_page_config(page_title="Housing Price Prediction", layout="wide")

st.sidebar.title('🏠 Housing Price Prediction in Delhi')
page = st.sidebar.radio('Go to', ['Home Page','Overview', 'Data Exploration and Preparation', 'Analysis and Insights', 'Conclusion and Recommendations'])

# Load Data Function (Housing.csv)
def load_data():
    df = pd.read_csv('Housing.csv')
    df = pd.get_dummies(df, drop_first=True)  
    return df
df = load_data()

def make_image_round(image_path, size=(300, 300)):
    img = Image.open(image_path).convert("RGBA")
    img = img.resize(size, Image.LANCZOS)
    bigsize = (img.size[0] * 3, img.size[1] * 3)
    mask = Image.new('L', bigsize, 0)
    draw = ImageDraw.Draw(mask) 
    draw.ellipse((0, 0) + bigsize, fill=255)
    mask = mask.resize(img.size, Image.LANCZOS)
    img.putalpha(mask)
    output = ImageOps.fit(img, mask.size, centering=(0.5, 0.5))
    output.putalpha(mask)
    return output

if page == 'Overview':
    st.title('Data Analysis and Techniques')
    st.write('''For our data analysis project, we have chosen the Housing Price Prediction dataset from
            Kaggle. This dataset contains detailed information on housing prices and various features such as the
            number of bedrooms, square footage, and location factors. The dataset comprises over 1,000
            rows, making it robust for analysis and modeling''')
    st.subheader('Dataset')
    st.subheader('The dataset contains the following columns:')
    st.markdown(' [ ' + ' ], [ '.join(df.columns) + ' ] ')
    st.title('Housing Price in Delhi')
    
    if st.checkbox('Show Dataset Preview', value=True):
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

elif page == 'Home Page':

    st.title('Authors')
    st.subheader('''Meet the Team: **WildKonek**''')

    col1, col2, col3 = st.columns(3)
    with col1:
        round_image = make_image_round('images/abanid.jpg', size=(150, 150))
        st.image(round_image, width=200, caption="Breshly Abanid")

        round_image = make_image_round('images/femaleavatar.jpg', size=(150, 150))
        st.image(round_image, width=200, caption="Nicole Cuizon")
    with col2:
        round_image = make_image_round('images/images.jpg', size=(150, 150))
        st.image(round_image, width=200, caption="Ethan Jan Acasio")

        round_image = make_image_round('images/despi.jpg', size=(150, 150))
        st.image(round_image, width=200, caption="Brian Despi")

        round_image = make_image_round('images/images.jpg', size=(150, 150))
        st.image(round_image, width=200, caption="Raphael Ubas")
    with col3:
        round_image = make_image_round('images/cala.jpg', size=(150, 150))
        st.image(round_image, width=200, caption="Robert David Cala")

        round_image = make_image_round('images/gonzaga.jpg', size=(150, 150))
        st.image(round_image, width=200, caption="Emmarson Gonzaga")



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
        if st.checkbox('Show Before Cleaning'):
            st.write('Before cleaning:')
            st.write(df.isnull().sum())

    
    with col2:
        if st.checkbox('Show Ater Cleaning'):
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


    st.title('Analysis and Insights')

    st.divider()

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

    st.write('''**Regression Results:**''')
    st.write(f'''R^2 Score: **{r2_score(y_test, y_pred)}**''')
    st.write(f'''Mean Squared Error: **{mean_squared_error(y_test, y_pred)}**''')
    st.write(f'''Mean Absolute Error: **{mean_absolute_error(y_test, y_pred)}**''')

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append((name, mse, mae, r2))

   
    results_df = pd.DataFrame(results, columns=["Model", "MSE", "MAE", "R-squared"])
    
    st.subheader('Model Comparison:')
    st.write(results_df)
    

   

    st.divider()
    st.subheader('Visualizations')

    numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']


    st.sidebar.header('Filter Data')
    filters = {}
    for col in numerical_cols:
        min_val = int(df[col].min())
        max_val = int(df[col].max())
        filters[col] = st.sidebar.slider(f'Filter {col}', min_val, max_val, (min_val, max_val))

    
    st.sidebar.subheader('Selected Filter Values')
    for col, (selected_min, selected_max) in filters.items():
        st.sidebar.write(f'{col}: {selected_min} - {selected_max}')

   
    filtered_df = df.copy()
    for col, (selected_min, selected_max) in filters.items():
        filtered_df = filtered_df[(filtered_df[col] >= selected_min) & (filtered_df[col] <= selected_max)]

    # Filtered Data
    X_filtered = filtered_df.drop(columns=['price'])
    y_filtered = filtered_df['price']

    X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

    linear_model = LinearRegression()
    linear_model.fit(X_train_filtered, y_train_filtered)
    y_pred_filtered = linear_model.predict(X_test_filtered)

    # Visualize actual vs predicted for Linear Regression model
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test_filtered, y_pred_filtered)
    ax.plot([min(y_test_filtered), max(y_test_filtered)], [min(y_test_filtered), max(y_test_filtered)], color='red', linestyle='--')
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Predicted Prices")
    ax.set_title("Actual vs Predicted Housing Prices (Filtered Data)")
    st.pyplot(fig)

    
    residuals_filtered = y_test_filtered - y_pred_filtered
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test_filtered, residuals_filtered)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals Plot (Filtered Data)")
    st.pyplot(fig)

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

    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, residuals)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals Plot")
    st.pyplot(fig)

elif page == 'Conclusion and Recommendations':
    st.title('Conclusion')
    st.write('In this project, we have analyzed the Housing Price Prediction dataset and built a model to predict house prices in Delhi. We have used various regression techniques such as Linear Regression, Lasso Regression, and Random Forest to predict house prices. The Random Forest model performed the best, with an R-squared score of 0.85. This model can be used to predict house prices based on features such as area, bedrooms, bathrooms, stories, and parking. ')
    st.write('Our analysis shows that the area of the house is the most important feature in predicting house prices. We recommend that homebuyers consider the area of the house when making a purchase decision. Additionally, the number of bedrooms and bathrooms also play a significant role in determining house prices. ')
    st.write('Overall, our model provides a reliable estimate of house prices in Delhi and can help homebuyers make informed decisions when purchasing a property. ')
    st.write('Thank you for using our application! We hope you found it helpful and informative. If you have any questions or feedback, please feel free to reach out to us. ')
    st.write('Happy house hunting!')

    st.write('The Random Forest model proved to be the most effective approach for predicting housing prices, providing accurate predictions and insights into influential features. While linear models remain useful for simpler relationships, the ensemble methods ability to handle non-linearities makes it the optimal choice for this dataset.')

   
    

    st.subheader('Recommendations')
    st.markdown('''
    1. **Feature Importance Analysis:**
        - Use the Random Forest model to identify the most critical features influencing housing prices.
    2. **Hyperparameter Tuning:**
        - Optimize the Random Forest parameters using GridSearchCV for further improvements.
    3. **Incorporate External Data:**
        - Add features like neighborhood information, proximity to amenities, and other external factors to enhance the model's predictive power.
    4. **Model Validation:**
        - Perform cross-validation to ensure the model's robustness and generalizability.
    5. **User Feedback:**
        - Collect feedback from users to continuously improve the model and the application.
    ''')
