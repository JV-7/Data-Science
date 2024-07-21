import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

poke_df = pd.read_csv(r'C:\Users\jayavaradhan.olivu\OneDrive - DISYS\Documents\Data Science\Data Science - Python\Python - Data Source\pokemon_data.csv')


# poke_df_1 = pd.read_excel(
#     'C:\\Users\\jayavaradhan.olivu\\OneDrive - DISYS\\Documents\\Data Science\\Data Science - Python\\Pandas - '
#     'Practice\\pokemon_data.xlsx')
#
# poke_df_2 = pd.read_csv(
#     'C:\\Users\\jayavaradhan.olivu\\OneDrive - DISYS\\Documents\\Data Science\\Data Science - Python\\Pandas - '
#     'Practice\\pokemon_data.txt',
#     delimiter='\t')

# print(poke_df.head(3))  #top 3 rows

# print(poke_df.tail(3))  #bottom 3 rows

# print(poke_df_1.head())

# print(poke_df_2)

'''Reading the columns(Headers)'''

# print(poke_df_2.columns)

'''Read each column'''

# print(poke_df[['Name', 'Type 1', 'Type 2']])

''' Read each rows '''

# print(poke_df.iloc[0:5])

''' we can also fetch the specific row(value) using iloc - function '''

# print(poke_df.iloc[2, 1])

# print(poke_df['Name'])

# for index, row in poke_df.iterrows():
#     print(index, row['Name'])

# for i, j in poke_df.iterrows():
#     print(i, j)

# for row, index in poke_df.iterrows():
#     print(row, index)

'''filtering - getting the data on specific condition'''

# var = poke_df.loc[poke_df['Type 1'] == "Fire"]
#
# print(var)

# print(poke_df.iloc[[5]])

'''sorting describing'''
# var = poke_df.sort_values(['Type 1', 'HP'], ascending=True)
#
# print(var)

'''Making changes to the data'''

# print(poke_df)

# poke_df['Total'] = poke_df['HP'] + poke_df['Attack'] + poke_df['Defense'] + poke_df['Sp. Atk'] + poke_df['Sp. Def']
# + \                poke_df['Speed']
'''once the column added to the dataframe it'll have a default memory. To remove 'Total' column'''

'''Another way of adding column'''

''' Iloc parameter rows and column iloc[rows, columns]'''

# poke_df['Total'] = poke_df.iloc[:, 4:10].sum(axis=1)

# print(poke_df.head(5))

# colm = poke_df.columns.to_list()

''' rearranging the columns '''

# poke_df = poke_df[colm[0:4] + colm[12:13] + colm[4:12]]

''' saving the modified data into csv, xlsx and txt '''

# poke_df.to_csv('modified.csv', index=False)
# poke_df.to_excel('modified.xlsx', index=False)
# poke_df.to_csv('modified.txt', index=False, sep='\t')
# print(poke_df.head(5))

# print(poke_df.head(10))

''' filtering the data '''

# new_poke_df = poke_df.loc[(poke_df['Type 1'] == 'Grass') & (poke_df['Type 2'] == 'Poison') & (poke_df['HP'] > 7)]
#
# new_poke_df.reset_index(drop= True, inplace=True)
#
# new_poke_df.to_csv(r'C:\Users\jayavaradhan.olivu\OneDrive - DISYS\Documents\Data Science\Data Science - Python\Pandas - Practice\Filtered.csv', index=False)
#
# print(new_poke_df)
#
# new_poke_df_1 = poke_df.loc[poke_df['Name'].str.contains('Mega')]
#
# new_poke_df_1 = poke_df.loc[~poke_df['Name'].str.contains('~Mega')]  # this will ignore the Mega
#
# new_poke_df_1 = poke_df.loc[poke_df['Type 1'].str.contains('Fire|Grass', regex=True)]  # | means 'OR'
#
# new_poke_df_1 = poke_df.loc[poke_df['Type 1'].str.contains('fire|grass', flags=re.I, regex=True)]
# # this will help to remove the case-sensitive 'flags=re.I
#
# new_poke_df_1 = poke_df.loc[poke_df['Name'].str.contains('pi[a-z]', flags=re.I, regex=True)]
# # Names which contains 'Pi' and the next letters with any alphabets
#
# new_poke_df_1 = poke_df.loc[poke_df['Name'].str.contains('^pi[a-z]', flags=re.I, regex=True)] # Name begin with pi
#
# print(new_poke_df_1)

''' Aggregate statistics '''

# print(poke_df)

# poke_df_3 = pd.read_csv(r'C:\Users\jayavaradhan.olivu\OneDrive - DISYS\Documents\Data Science\Data Science - Python\Pandas - Practice\modified.csv')

# poke_df_3_1 = poke_df_3.groupby(['Type 1']).sum()
#
# print(poke_df_3_1[['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary']])

# poke_df_3['Count'] = 1
#
# poke_df_3_2 = poke_df_3.groupby(['Type 1', 'Type 2']).count()['Count']
#
# print(poke_df_3_2)
#
# poke_df_3_3 = poke_df_3.groupby(['Type 1']).count()['Count']
#
# print(poke_df_3_3)


'''
a = [{'p': 2, 'q': 4, 'r': 6, 's': 8},
{'a': 200, 'b': 400, 'c': 600, 'd': 800},
{'p': 2000, 'q': 4000, 'r': 6000, 's': 8000}]
info = pd.DataFrame(a)
print(type(a))
print(a)
print(info.iloc[0])
# <class 'pandas.core.series.Series'>
'''
