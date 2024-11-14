import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('spain_energy_data.csv')  # Replace with the actual file path

# Convert the 'time' column to datetime with UTC to avoid timezone warnings
data['time'] = pd.to_datetime(data['time'], utc=True)

# Extract datetime features
data['year'] = data['time'].dt.year
data['month'] = data['time'].dt.month
data['day'] = data['time'].dt.day
data['hour'] = data['time'].dt.hour
data['minute'] = data['time'].dt.minute

# Drop the original 'time' column
data = data.drop(['time'], axis=1)

# Forward fill missing values for time series continuity
data.ffill(inplace=True)

# Set features and target variable
X = data.drop(['price actual'], axis=1)  # Use all columns except 'price actual' as features
y = data['price actual']                 # Set the target variable as 'price actual'

# Handle any remaining NaN values in X using mean imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Save the trained model to a .pkl file
with open('linear_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as linear_regression_model.pkl")
