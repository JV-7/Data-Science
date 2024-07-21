import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

''' finding the null values '''

data = pd.read_csv('Titanic-Dataset.csv')
print(data.head(10))
print(type(data))
print(f'Unique values of Embarked {data['Embarked'].unique()}')
my_df = data.copy()  #shallow copy
null_df = my_df.isnull().sum()
print(f' Null Data \n{null_df}')

''' simple imputer used to fill the None data, only in the integer columns '''
''' using imputer method using the fillna - default value that gets filled in the empty row is MEAN values'''

from sklearn.impute import SimpleImputer
''' for single column '''
imputer = SimpleImputer()
data['Age'] = imputer.fit_transform(data[['Age']])
print(f' null data after using the SimpleImputer in Age columns \n{data.isnull().sum()}')
print(data.loc[data['Age'] == 'missing_value'])

''' multiple columns '''

data[['Age', 'Embarked']] = imputer.fit_transform(data[['Age', 'Embarked']])
print(f' null data after using the SimpleImputer in Age, Embarked columns \n{data.isnull().sum()}')  # this will be
# ended with error because simple Imputer works only in integer column

''' the default value MEAN also can be changeable 
 using the strategy and fill_value we can fill the null values with Missing_value'''

imputer = SimpleImputer(strategy='constant', fill_value='Missing_value')

data[['Age', 'Embarked']] = imputer.fit_transform(data[['Age', 'Embarked']])
print(data.loc[data[['Embarked'] == 'missing_value']])
print(data)
''' since Embarked contains string value it's not possible to convert into MEAN '''
''' In order to overcome we are using the label encoding method '''

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()
data['Encoder_Embarked'] = LE.fit_transform(data[['Embarked']])
print(f'unique values of Embarked \n{data['Embarked'].unique()}')
print(f'unique values of Encoder_Embarked \n{data['Encoder_Embarked'].unique()}')
print(data.loc[data['Encoder_Embarked'] == 3])


data.replace({'Encoder_Embarked': 3}, 'Missing Values', inplace=True)

print(data)

