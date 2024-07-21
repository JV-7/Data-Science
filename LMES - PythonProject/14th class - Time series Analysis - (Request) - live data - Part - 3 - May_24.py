import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 10000000)

url = 'https://api.coingecko.com/api/v3/coins/markets'

parameter = {'vs_currency': 'usd',
             'order': 'market_cap_desc',
             'per_page': 100,
             'page': 1,
             'sparkline': False}

response = requests.get(url, params=parameter)
data = response.json()
data_df = pd.DataFrame(data)

print(data_df.head())

data_df.dropna(inplace=True)
data_df['market_cap'] = data_df['market_cap'].astype(int)
print(data_df.head())


df_describe = data_df.describe()
print(df_describe)

plt.figure(figsize=(10, 8))
sns.histplot(data_df['market_cap'], bins=50, kde=True)
plt.show()




