""" Time series """

"""
parse_dates

>> This is to let python knows which column to pick as date time columns. If not we may have to manually do to t
from string to date time

>> Conversion during the parse_dates is more efficient and time saving process

>> This will lead to better usage various pandas based date time functions

index_col

>> This will convert that particular column in the row(index). By doing so we can be in better position to man
apply few functions.

"""

''' with CSV '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 100)

data = pd.read_csv('Apple_Stock.csv', parse_dates=['Date'], index_col='Date')
# data = data.rename(columns={'Low': 'Low_1'})

print(data.head())
print(data.index)

plt.figure(figsize=(10, 8))

for column in data.columns:
    plt.plot(data.index, data[column], label=column)

plt.title('Time Series Data')
plt.xlabel('Data')
plt.ylabel('Value')

plt.figure(figsize=(10, 8))
plt.plot(data['Close'])
plt.show()

''' Note: If only one values is given inside the plot, then  '''

''' Without CSV '''

print('------------------Without CSV--------------------------')

data_1 = {
    'Date': ['2024-01-03', '2024-01-01', '2024-01-02'],
    'Open': [183.25, 182.71, 182.952],
    'High': [184.73, 183.24, 183.74],
    'Low': [182.62, 180.62, 181.21],
    'Close': [184.82, 182.81, 182.47],
    'Volume': [25871300, 33556700, 28907300]
}

df_1 = pd.DataFrame(data_1)

df_1['Date'] = pd.to_datetime(df_1['Date'])
df_1.set_index('Date', inplace=True)
print(df_1)
''' Sorting the data Frame by its index to ensure the DatetimeIndex is monotonic 
Both approaches are same since we are the date as index - second is best
'''

# df_1 = df_1.sort_values(['Date'], ascending=True)
df_1 = df_1.sort_index()

df_1.sort_index(inplace=True)  # if we do the sort function directly to the data set - inplace= must be true because we
# aren't storing in any variable.

''' Slicing operation 
 Fetching data in between range '''

subset = df_1['2024-01-01':'2024-01-02']
print(subset)
