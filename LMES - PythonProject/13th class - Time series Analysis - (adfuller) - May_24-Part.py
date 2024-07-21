import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

pd.set_option('display.max_column', None)
pd.set_option('display.max_row', None)
pd.set_option('display.width', 1000)

data = pd.read_csv('Apple_Stock.csv', parse_dates=['Date'])

stationary_data_check = adfuller(data['Date'])
print(stationary_data_check)

print(stationary_data_check[0])  # p value
print(stationary_data_check[1])

P = 5  # Auto regressive Order
D = 1  # Differential Order
Q = 0  # Moving Order

Model = ARIMA(data['Date'], order=(P, D, Q))
Model_Fit = Model.fit()

Forcast_Steps = 10
Final_Data = Model_Fit.forecast(steps=Forcast_Steps)


print(Final_Data)

'''this code is my ChatGPT'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import fuller, adfuller

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

# Load the data
data = pd.read_csv('Apple_Stock.csv', parse_dates=['Date'], index_col='Date')

# Ensure 'Close' column is cleaned of any non-numeric characters and convert to float
data["Close"] = data["Close"]

# Check for missing values and handle them
data["Close"].fillna(method='ffill', inplace=True)  # Forward fill for missing values

# Ensure 'Close' column is of a numeric type
data["Close"] = pd.to_numeric(data["Close"], errors='coerce')

# Drop any remaining NaN values
data.dropna(subset=["Close"], inplace=True)

# Perform the Augmented Dickey-Fuller test for stationary
stationary_data_check = adfuller(data['Close'])
print(f'ADF Statistic: {stationary_data_check[0]}')
print(f'p-value: {stationary_data_check[1]}')
print('Critical Values:')
for key, value in stationary_data_check[4].items():
    print(f'\t{key}: {value}')

# ARIMA model parameters
P = 5  # Auto regressive Order
D = 1  # Differential Order
Q = 0  # Moving Order

# Fit the ARIMA model
model = ARIMA(data['Close'], order=(P, D, Q))
model_fit = model.fit()

# Forecast future values
forecast_steps = 10
forecast = model_fit.forecast(steps=forecast_steps)

# Print forecasted values
print(forecast)

# Plot the original data and forecast
plt.figure(figsize=(10, 8))
plt.plot(data['Close'], label='Historical Data')
plt.plot(forecast, label='Forecast', color='red')
plt.title('Apple Stock Close Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


















