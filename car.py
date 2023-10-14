# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_absolute_error, r2_score

# Load your car price dataset (replace 'car_data.csv' with your dataset)
data = pd.read_csv("C:\\Users\\KIIT\\Downloads\\CarPrice_Assignment.csv")

# Explore your dataset to understand its structure and features
# print(data.head())

# Select relevant features and target variable (price)
X = data[['wheelbase','carlength','cylindernumber', 'stroke','boreratio','compressionratio','peakrpm', 'citympg', 'highwaympg', 'enginesize', 'horsepower']]  # Features
y = data['price']  # Target variable (Price)

X = pd.get_dummies(X, columns=['cylindernumber'], drop_first=True)# categorial column to binary set

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'R-squared (R2) Score: {r2:.2f}')







