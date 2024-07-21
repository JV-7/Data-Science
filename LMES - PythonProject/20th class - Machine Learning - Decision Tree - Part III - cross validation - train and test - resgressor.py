import numpy as np
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
# pd.set_option('display.float_format', lambda x_1: "%.3f" % x_1)
# pd.options.display.float_format = '{:.3f}'.format
np.set_printoptions(suppress=True, formatter={'float_kind': '{:.3f}'.format})

# california_housing = fetch_california_housing()
#
# x = california_housing.data
# y = california_housing.target
#
# df = pd.DataFrame(x, columns=california_housing.feature_names)
#
# print(df)
''' this inbuilt data set 
below is the external data sets'''

housing_df = pd.read_csv(r'C:\Users\jayavaradhan.olivu\OneDrive - DISYS\Documents\Data Science\Data Science - Python\Python - Data Source\housing.csv')

x = housing_df.drop(['median_house_value'], axis=1)
y_regression = housing_df['median_house_value']  # this is y-axis only not particularly for regression alone

LE = LabelEncoder()
x['ocean_proximity'] = LE.fit_transform(x['ocean_proximity'])
x = x.fillna(x['ocean_proximity'].mean())
'''Since x - 'ocean_proximity' contains strings so we are using LE to convert it to int'''

x_train, x_test, y_train, y_test = train_test_split(x, y_regression, test_size=0.2, random_state=42)

regressor = DecisionTreeRegressor()

cross_val = cross_val_score(regressor, x_train, y_train, cv=5, scoring='neg_mean_squared_error')
cross_mean_regressor = -cross_val.mean()
cross_std_regressor = cross_val.std()

print(f'cross_val: {cross_val}')
print(f'cross_mean_regressor: {cross_mean_regressor}')
print(f'cross_std_regressor: {cross_std_regressor}')

regressor.fit(x_train, y_train)
prediction = regressor.predict(x_test)
print(f'prediction: {prediction}')




