import numpy as np
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

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

housing_df = pd.read_csv(
    r'C:\Users\jayavaradhan.olivu\OneDrive - DISYS\Documents\Data Science\Data Science - Python\Python - '
    r'Data Source\housing.csv')

x = housing_df.drop(['median_house_value'], axis=1)
y_regression = housing_df['median_house_value']  # this is y-axis only not particularly for regression alone

LE = LabelEncoder()
x['ocean_proximity'] = LE.fit_transform(x['ocean_proximity'])
x = x.fillna(x['ocean_proximity'].mean())
'''Since x - 'ocean_proximity' contains strings so we are using LE to convert it to int'''

''' This decision Tree Regressor '''
x_train, x_test, y_train, y_test = train_test_split(x, y_regression, test_size=0.2, random_state=42)

regressor = DecisionTreeRegressor()
regressor.fit(x_train, y_train)
y_prediction_regressor = regressor.predict(x_test)

mse = mean_squared_error(y_test, y_prediction_regressor)

print(f'y_prediction_regressor: {y_prediction_regressor}')
print(f'mse: {mse}')

''' This is decision Tress Classification '''
x_train, x_test, y_train, y_test = train_test_split(x, y_regression, train_size=0.8, test_size=0.2, random_state=42)

Classifier = DecisionTreeClassifier()
Classifier.fit(x_train, y_train)
y_prediction_classifier = Classifier.predict(x_test)

ae = accuracy_score(y_test, y_prediction_classifier)

print(f'y_prediction_classifier: {y_prediction_classifier}')
print(f'accuracy_score: {ae}')

''' below lines to convert actual input of regression into classification '''

bins = [100000, 200000, 300000, 400000, np.inf]
labels = ['A', 'B', 'C', 'D']

''' Bins
"Binning" is a way to group a range of continuous values into discrete intervals or categories.
 Labels
The labels array defines the categorical labels assigned to each bin

y_classification - in series not in dataframe so use below step for filtering the values'''

y_classification = pd.cut(y_regression, bins=bins, labels=labels)

filtered = y_classification.loc[y_classification == 'A']

print(filtered.head())
print(y_classification.head())
print(y_regression[:15])
