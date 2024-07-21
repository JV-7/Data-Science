import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load the original data
data = pd.read_csv(r'C:\Users\jayavaradhan.olivu\OneDrive - DISYS\Documents\Data Science\Data Science - Python\Python '
                   r'- Data Source\Walmart_sales.csv')

data = data.drop(columns=['Date'])
data['Weekly_Sales'] = data['Weekly_Sales'].str.replace('$', '').str.replace(',', '').astype(float)

# Prepare the features and target variable
x = data[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']]
y = data['Weekly_Sales']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize and fit the model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions on the test set
test_predictions = model.predict(x_test)
print(f'predictions: {test_predictions}\n')

# Calculate metrics on the test set
mse = mean_squared_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)

print(f'Original Data Metrics:\nMean Squared Error (MSE): {mse}\nR^2 Score: {r2}\n')

# Calculate the correlation matrix for the sample data
correlation_matrix_sample = data.corr()

print("\nCorrelation Matrix for Sample Data:")
print(correlation_matrix_sample)

''' correlation '''

correlation_output = np.corrcoef(y_test, test_predictions)

print(f'''\n correlation_output:
      {correlation_output}''')

# Example with our own data point for prediction
print('Example with our own data point for prediction')

# Sample data for a different prediction task
data = {'Year': [2020, 2021, 2022, 2023],
        'Tax': [100, 300, 490, 780],
        'House_Price': [1200, 3400, 5300, 9000]}

df = pd.DataFrame(data)

# Prepare the features and target variable
x = df[['Year', 'Tax']]
y = df['House_Price']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize and fit the model
model = LinearRegression()
model.fit(x_train, y_train)

# Make a prediction with our own data point
new_data = pd.DataFrame([[2028, 700]], columns=['Year', 'Tax'])
custom_prediction = model.predict(new_data)
coefficient = model.coef_
interception = model.intercept_

# Calculate the correlation matrix for the sample data
correlation_matrix_sample = df.corr()

print("\nCorrelation Matrix for Sample Data:")
print(correlation_matrix_sample)

''' correlation '''

correlation_output = np.corrcoef(y_test, custom_prediction)

print(f'''\n correlation_output:
      {correlation_output}''')

# Calculate metrics on the test set
test_predictions = model.predict(x_test)
mse_sample = mean_squared_error(y_test, test_predictions)
r2_sample = r2_score(y_test, test_predictions)

print(f'''
Custom prediction: {custom_prediction}\n
Coefficient: {coefficient}\n
Intercept: {interception}\n
Sample Data Metrics:\n 
Mean Squared Error (MSE): {mse_sample}
R^2 Score: {r2_sample}\n
''')

''' if NaN came as a output, which indicate that you have less data '''
