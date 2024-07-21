"""---------MinMax Scalar and StandardScalar---------------"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

my_data = {'Score': [100, 20, 30, 40, 50],
           'Average': [0.1, 0.2, 0.3, 0.4, 0.5]}

df = pd.DataFrame(my_data)
print(df)
#
# MMS = MinMaxScaler()
# scaled_min_max = MMS.fit_transform(df)
# print(scaled_min_max)
#
SS = StandardScaler()
scaled_standard = SS.fit_transform(df)
print(scaled_standard)
# #
# scaled_min_max_df = pd.DataFrame(scaled_min_max, columns=df.columns)
# scaled_standard_df = pd.DataFrame(scaled_standard, columns=df.columns)
#
# final = pd.concat([df, scaled_min_max_df, scaled_standard_df], axis=1)
#
# print(final)

''' Applying for specific column '''
'''-----------Pipeline--------------'''
#
# from sklearn.compose import ColumnTransformer
#
# col_transfer = ColumnTransformer([('minmax', MinMaxScaler(), ['Score'])], remainder='passthrough')
# min_max_scalar = col_transfer.fit_transform(df)
# df_min_max_scalar = pd.DataFrame(min_max_scalar, columns=df.columns)
# print(f'min_max \n {df_min_max_scalar}')
#
# col_transfer_1 = ColumnTransformer([('A', StandardScaler(), ['Average'])], remainder='passthrough')
# std_clm = col_transfer_1.fit_transform(df)
# df_std_clm = pd.DataFrame(std_clm, columns=['Average', 'Score'])
# print(f'std \n{df_std_clm}')
#
# final_1 = pd.concat([df, df_min_max_scalar, df_std_clm], axis=1)
# print(final_1)


















