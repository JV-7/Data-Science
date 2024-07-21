import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_column', None)
pd.set_option('display.max_row', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: "%.3f" % x)

df = pd.read_csv(r'C:\Users\jayavaradhan.olivu\OneDrive - DISYS\Documents\Data Science\Data Science - Python\Python - '
                 r'Data Source\Salary.csv', encoding='ISO-8859-1')

df['Salary'] = df['Salary'].replace(r'[\$,]', '', regex=True).astype(float)

x, y = df[['YearsExperience']], df['Salary']

data = {'YearsExperience': [11.2, 11.5]}

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

df_1 = pd.DataFrame(data)

model = LinearRegression()
model.fit(x_train, y_train)
y_prediction = model.predict(x_test)
y_prediction_1 = model.predict(df_1)
df_1['Salary'] = y_prediction_1

mean_squared_error_check = mean_squared_error(y_test, y_prediction)
r2_score_check = r2_score(y_test, y_prediction)

df_new = pd.DataFrame(x_test)
df_new['Actual'] = y_test.values
df_new['Predicted'] = y_prediction
df_new['Squared_Error'] = (df_new['Actual'] - df_new['Predicted']) * 2
df_new['Error'] = df_new['Actual'] - df_new['Predicted']
df_new['MSA'] = mean_squared_error_check

df_2 = pd.concat([df, df_1], axis=0, ignore_index=True)

print(df_1)

print('\n')
print(df_2)
print(df_new)


