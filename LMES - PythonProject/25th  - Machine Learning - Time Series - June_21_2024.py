import calendar
from dateutil.parser import parse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mp

plt.rcParams.update({'figure.figsize': (10, 7), 'figure.dpi': 120})

data = pd.read_csv(r'C:\Users\jayavaradhan.olivu\OneDrive - DISYS\Documents\Data Science\Data Science - Python\Python '
                   r'- Data Source\Electric_Production.csv', parse_dates=['DATE'], index_col=['DATE'])


def draw_plot(data, x, y, title, xlabel, ylabel, dpi):
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


draw_plot(data=data, x=data.index, y=data.values, title='Time Series', xlabel='Date', ylabel='Value', dpi='')

''' Time series - 2 '''

data.reset_index(inplace=True)

data['Year'] = [i.year for i in data['DATE']]
data['Month'] = [calendar.month_abbr[i.month] for i in data['DATE']]

np.random.seed(100)
unique_years = data['Year'].unique()
my_color = np.random.choice(list(mp.colors.XKCD_COLORS.keys()), len(unique_years), replace=False)

print(unique_years)
print(data.columns)

A = data.loc[data['Year'] == 1986, :].shape[0]-.9
B = data.loc[data['Year'] == 1987, 'IPG2211A2N'][-1:].values[0]

plt.figure(figsize=(16, 12), dpi=80)
for i, y in enumerate(unique_years):
    print(i, y)
    if i > 0:
        plt.plot('Month', 'IPG2211A2N', data=data.loc[data['Year'] == y, :], color=my_color[i], label=y)
        plt.text(data.loc[data['Year'] == y, :].shape[0] - .9, data.loc[data['Year'] == y, 'IPG2211A2N'][-1:].values[0], y,
                 fontsize=12, color=my_color[i])

plt.show()
print(data.head(10))

print(f'''A Value:
      {A}''')

print(f'''B Value:
      {B}''')

'''A 
Explanation:
data.loc[data['Year'] == 1986, :]:

data: This is your DataFrame.
.loc[]: This is a method used to access a group of rows and columns by labels or a boolean array.
data['Year'] == 1986: This creates a boolean mask where True indicates rows where the Year column is 1986.
data.loc[data['Year'] == 1986, :]: This selects all rows where the Year is 1986 and all columns (: means all columns).
.shape[0]:

.shape returns a tuple representing the dimensionality of the DataFrame. The first element of the tuple (at index 0) is the number of rows.
.shape[0] thus gives the number of rows where the Year is 1986.
- .9:

This subtracts 0.9 from the number of rows obtained.

B -
Explanation:
data.loc[data['Year'] == 1987, 'IPG2211A2N']:

data: This is your DataFrame.
.loc[]: This is a method used to access a group of rows and columns by labels or a boolean array.
data['Year'] == 1987: This creates a boolean mask where True indicates rows where the Year column is 1987.
data.loc[data['Year'] == 1987, 'IPG2211A2N']: This selects the rows where the Year is 1987 and the column 'IPG2211A2N'.
[-1:]:

This selects the last row from the filtered DataFrame/Series. The [-1:] slice is used to get the last element.
.values:

.values returns the underlying numpy array of the Series. This converts the selected pandas Series to a numpy array.
[0]:

This accesses the first (and only) element of the numpy array.
'''
