""" Linear regression - Supervised model """

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option('display.max_column', None)
pd.set_option('display.max_row', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: "%.3f" % x)

df = pd.read_csv(r'C:\Users\jayavaradhan.olivu\OneDrive - DISYS\Documents\Data Science\Data Science - Python\Python - '
                 r'Data Source\Walmart_sales.csv')

df['Weekly_Sales'] = df['Weekly_Sales'].replace(r'[\$,]', '', regex=True).astype(float)

# The first argument [\$,] is the regex pattern to match any dollar signs or commas
'''The second argument '' is the 
replacement string, indicating that any matched characters should be replaced with nothing (i.e., removed).'''

x, y = df[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']], df['Weekly_Sales']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
'''
test_size denotes that 80% of data for train and 20% for test purpose
without random_state model will take data randomly from the data source - to keep this is constant we are using
random_state as constant

Why 42?
The choice of 42 as the default value for random_state
'''

assert x_train.shape[0] == y_train.shape[0], "Mismatched number of samples in x_train and y_train"

model = LinearRegression()
model.fit(x_train, y_train)

y_prediction = model.predict(x_test)

y_prediction_format = [f'${pred:.2f}' for pred in y_prediction]

print('Y_Prediction:\n')
for i, prediction in enumerate(y_prediction_format, start=0):
    print(i, prediction)

''' below check denotes that whether our predictions are correct '''
mean_squared_error_check = mean_squared_error(y_test, y_prediction)
r2_score_check = r2_score(y_test, y_prediction)

print(f'************************ \n'
      f'Mean Squared Check: \n'
      f'${mean_squared_error_check}\n'
      f'**********************')
print(f'r2 Score Check: '
      f'\n${round(r2_score_check,2)}')

df_new = pd.DataFrame(x_test)
df_new['Actual'] = y_test.values
df_new['Predicted'] = y_prediction
df_new['Squared_Error'] = (df_new['Actual'] - df_new['Predicted']) * 2
df_new['Error'] = df_new['Actual'] - df_new['Predicted']
df_new['MSA'] = mean_squared_error_check

print('\n')
print(df_new)

''' Mean Square Error Detailing
 '''
data = {
    'Temperature': [60, 70, 80, 90, 100],
    'Fuel_Price': [3.5, 3.7, 3.6, 3.8, 4.0],
    'CFI': [220, 221, 222, 223, 224],
    'Unemployment': [5.0, 5.2, 5.1, 5.3, 5.4],
    'Weekly_Sales': [20000, 21000, 22000, 23000, 24000]
}

df = pd.DataFrame(data)

x, y = df[['Temperature', 'Fuel_Price', 'CFI', 'Unemployment']], df['Weekly_Sales']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_prediction = model.predict(x_test)

mean_squared_error_check = mean_squared_error(y_test, y_prediction)

df_new = pd.DataFrame(x_test)

df_new['Actual'] = y_test.values
df_new['Predicted'] = y_prediction
df_new['Squared_Error'] = (df_new['Actual'] - df_new['Predicted']) * 2
df_new['Error'] = df_new['Actual'] - df_new['Predicted']
df_new['MSA'] = mean_squared_error_check
print('\n')
print(mean_squared_error_check)
print('\n')
print(df_new)
print('\n')
print(df)
