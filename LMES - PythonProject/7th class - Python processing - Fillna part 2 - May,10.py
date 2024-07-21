import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

my_data = {'Name': ['Dhoni', 'Ashwin', 'Raina', None],
           'Age': [42, 38, None, 22]}

my_data_df = pd.DataFrame(my_data)
# print(my_data_df)

'''100 will store in all the places wherever data is not found'''

# my_data_df.fillna(100, inplace=True)

''' substitute the mean into the place where data is not found '''

# my_data_df['Age'].fillna(my_data_df['Age'].mean(), inplace=True)
#
# my_data_df.fillna({'Age': my_data_df['Age'].mean()}, inplace=True)

# print(my_data_df)

''' Another for ffill - forward fill and bfill - backward fill '''

# print("Original DataFrame:")
# print(my_data_df)
# print("\nNaN values per column:")
# print(my_data_df.isna().sum())
# my_data_df.bfill(inplace=True)
# print(my_data_df)

# print('Original DataFrame:')
# print(my_data_df)
# print('\n Nan Value per columns:')
# print(my_data_df.isna().sum())
# my_data_df.ffill(inplace=True)
# print(my_data_df)

''' Replace function for the null or none values '''

# my_data_df.replace(to_replace=np.nan, value=90, inplace=True)

# print(my_data_df)

''' --------------------------Label Encoding--------------------------- '''

my_data_1 = {'Team': ['CKK', 'RR', 'RCB', 'KKK', 'LSG', 'DC', 'PKBS', 'MI']}

my_data_1_df = pd.DataFrame(my_data_1)

label_encoder = LabelEncoder()  #it is inbuilt class for producing label encoder
# label_encoder is the object for the class LabelEncoder()
my_data_1_df['Labels'] = label_encoder.fit_transform(my_data_1_df['Team'])

print(my_data_1_df)

# ''' Label encoding without using any library '''
#
# my_data_1_df['Labels'] = my_data_1_df['Team'].map({'CKK': 50, 'RR': 9, 'RCB': 32, 'KKK': 45, 'LSG': 78, 'DC': 13, 'PKBS': 2, 'MI': 78})
#
# print(my_data_1_df)

'''------------ONE HOT ENCODING-----------------------'''

''' without library '''

my_data_2 = {'Degree': ['Masters', 'PHD', 'Bachelor', 'Masters', 'PHD'],
             'Name': ['Kohil', 'Ashwin', 'Rahul', 'raina', 'dhoni']}

my_data_2_df = pd.DataFrame(my_data_2)
# #
# print(my_data_2_df)
#
# encoder_df = pd.get_dummies(my_data_2['Degree'], prefix='')
#
# print(encoder_df)
#
# final_df = pd.concat([my_data_2_df, encoder_df], axis=1)
#
# print(final_df)

''' with library '''
#
OHE = OneHotEncoder(sparse_output=False)  # Basically df will be in Matrix, so it's not possible to convert into
# fit_transform, so we're using sparse_output to converting into array
encoder_date = OHE.fit_transform(my_data_2_df[['Degree']])
print(encoder_date)
print(type(encoder_date))

df_encoder_1 = pd.DataFrame(encoder_date, columns=OHE.get_feature_names_out(['Degree']))
print(df_encoder_1)

final_encoder = pd.concat([my_data_2_df, df_encoder_1], axis=1)
print(final_encoder)


























