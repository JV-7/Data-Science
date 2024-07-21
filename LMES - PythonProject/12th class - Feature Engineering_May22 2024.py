""" Feature Engineering also Pre-Processing
creating a new feature
mainly time-based data
Feature selection
Feature scaling and transformation
"""
import pandas as pd

data = {'Festival_date': ['2024-01-01', '2024-02-19', '2024-03-24'],
        'Festival_time': ['2024-01-01 09:00:00', '2024-02-19 12:30:00', '2024-03-24 10:00:00']}

df = pd.DataFrame(data)

df['Festival_date'] = pd.to_datetime(df['Festival_date'])
df['Festival_time'] = pd.to_datetime(df['Festival_time'])

df['Day_of_week'] = df['Festival_date'].dt.dayofweek  # 0 - monday likewise for remaining weekdays
df['month'] = df['Festival_date'].dt.month  # 0 - monday likewise for remaining weekdays
df['quarter'] = df['Festival_date'].dt.quarter  # 0 - monday likewise for remaining weekdays

print(df)
print('\n')
''' ------------------Dimensionality Reduction------------------------- '''
print('------------------Dimensionality Reduction-------------------------')
print('\n')

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
df = pd.read_csv('train.csv')

df_1 = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

print(df_1.head())

df_1['Age'] = df_1['Age'].fillna(df_1['Age'].median())
df_1['Embarked'] = df_1['Embarked'].fillna(df_1['Embarked'].mode()[0])

print(df_1.head())

df_1 = pd.get_dummies(df_1, columns=['Sex', 'Embarked'], drop_first=True)  ## similar to label encoding

print(df_1.head(5))

x = df_1.drop(columns=['Survived'])
y = df_1['Survived']
print(x)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

'''
cov_matrix = np.cov(x_scaled.T)  # T means if transforms the input on (n_feature and n_samples)
eigen_values, eigen_vector = np.linalg.eig(cov_matrix)

sorted_indices = np.argsort(eigen_values)[::-1]
sorted_eigen_values = eigen_values[sorted_indices]
sorted_eigen_vector = eigen_vector[:, sorted_indices]

n_components = 2
top_eigenvectors = sorted_eigen_vector[:, :n_components]

x_pca = x_scaled.dot(top_eigenvectors)
print(x_pca)

$ Instead of writing these codes we can using the PCA library to achieve the below steps
 
>> step1 standardization
>> step2 co-variance matrix
>> Eigenvalue decomposition

'''
# new library
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

x_pca = pca.fit_transform(x_scaled)

print(f'components \n{pca.components_}')

plt.figure(figsize=(10, 7))
sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=y, palette='viridis')
plt.legend('Survived')
plt.show()










