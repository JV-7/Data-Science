from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

my_data = pd.read_csv('train.csv')

numerical_columns = ['Age', 'Fare', 'Parch', 'SibSp']
categorical_columns = ['Embarked', 'Sex', 'Pclass']

x = my_data[numerical_columns + categorical_columns]
y = my_data['Survived']  # assigning target variable to y

number_transformer = SimpleImputer(strategy='constant')
categorical_data_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                               ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

date_pre_processing = ColumnTransformer(transformers=[('num', number_transformer, numerical_columns),
                                                      ('cat', categorical_data_transformer, categorical_columns)
                                                      ])

post_transformation = date_pre_processing.fit_transform(x)

cat_columns = date_pre_processing.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()

final_column_names = numerical_columns + cat_columns.tolist()

final_df = pd.DataFrame(post_transformation, columns=final_column_names)
#
# print(post_transformation)
# print('-------------------')
# print(cat_columns)
# print('-------------------')
# print(final_column_names)
# print('-------------------')
# print()
# print('-----------------------')
# print(x.head(5))
# print('-------------------')
print(final_df.head(10))
#

date_pre_processing.set_output(transform='pandas')
a = date_pre_processing.fit_transform(my_data)
print(a.head())





































