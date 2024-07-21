import pandas as pd
import numpy as np

'''One Dimension'''

var = ['a', 'b', 'c', 'd']
print(var)
print(type(var))
my_df = pd.Series(var)
print(my_df)
print(type(my_df))

'''Two Dimension'''

var1 = ['a', 'b', 'c', 'd']
print(var1)
print(type(var1))
my_df1 = pd.DataFrame(var1)
print(my_df1)
print(type(my_df1))

'''Converting into CSV file concept'''

var3 = {'Name': ['JV', 'Sreeram', 'Chandra'], 'Team': ['Business Operation', 'Business Operation', None]}
print(var3)
print(type(var3))
my_df2 = pd.DataFrame(var3, index=['r1', 'r2', 'r3'])  #changing the indexing value
print(my_df2)
print(type(my_df2))

my_df2.to_csv('Team Details.csv', index=False)

'''going read the CSV file'''

# my_data = pd.read_csv("AUTO LOAD TIMESHEETS ALCON_05-02-2024.csv")
# print(my_data)

my_data1 = pd.read_csv("Team Details.csv")
print(my_data1)

'''removing the unnamed column name'''

my_date2 = pd.read_csv("Team Details.csv", index_col=False)

print(my_date2)

'''NumPy'''

print()
print('-----------Numpy-------------')
print()

my_list = np.array([2, 4, 6, 8, 10])

print(my_list)  # Array

my_list1 = np.arange(start=100, stop=112)  #
print(my_list1)

'''Below tow lines is to check the shape and dimension of the give data'''

print(f'Shape {my_list1.shape}')
print(f'Dimension {my_list1.ndim}')

'''Below code is change the shape of the data'''

reshape_array = my_list1.reshape(3, 4)  # is use for convert one single rows to columns based on the total

print(reshape_array)
print(type(reshape_array))
print(f'Shape {reshape_array.shape}')
print(f'Dimension {reshape_array.ndim}')

'''One dimension to two dimension'''

reshape_array = my_list1.reshape(-1, 1)

print(reshape_array)
print(type(reshape_array))
print(f'Shape {reshape_array.shape}')
print(f'Dimension {reshape_array.ndim}')

print('')
print('Done')
