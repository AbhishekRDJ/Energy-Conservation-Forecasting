import streamlit as st
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the trained model
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit interface for file upload
st.title('Energy Price Prediction using Linear Regression')
st.write('Upload your CSV file for prediction')

# File uploader for the CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the uploaded file into a DataFrame
    data = pd.read_csv(uploaded_file)

    # Check the first few rows of the dataset to understand its structure
    st.write(data.head())

    # Convert 'time' column to datetime if it exists
    if 'time' in data.columns:
        data['time'] = pd.to_datetime(data['time'], utc=True)

        # Extract datetime features
        data['year'] = data['time'].dt.year
        data['month'] = data['time'].dt.month
        data['day'] = data['time'].dt.day
        data['hour'] = data['time'].dt.hour
        data['minute'] = data['time'].dt.minute

        # Drop the original 'time' column
        data = data.drop(['time'], axis=1)

    # Remove columns with all NaN values
    data = data.dropna(axis=1, how='all')

    # Ensure the data has the same columns as the training data
    # Get the list of columns the model was trained on
    model_columns = [col for col in data.columns if col != 'price actual']  # Exclude target column
    X = data[model_columns]

    # Handle missing values using mean imputation for the remaining NaN values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Check if the number of features in the uploaded data matches the model's expected number of features
    if X_imputed.shape[1] != model.coef_.shape[0]:
        st.error(f"Mismatch in the number of features. Expected {model.coef_.shape[0]} features, but the data has {X_imputed.shape[1]} features.")
        st.stop()

    # Perform prediction
    prediction = model.predict(X_imputed)

    # Display the predictions
    st.write("Predictions:", prediction)

    # Calculate metrics if 'price actual' is in the dataset
    if 'price actual' in data.columns:
        y = data['price actual']
        mse = mean_squared_error(y, prediction)
        r2 = r2_score(y, prediction)

        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R2 Score: {r2}")
