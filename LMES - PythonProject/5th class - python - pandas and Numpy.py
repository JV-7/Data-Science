import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.expand_frame_repr', False)

data = {"Name": ["a", "b", None, "Kohli"],
        "Age": [4, 3, 26, None]}

my_df = pd.DataFrame(data)

# print(my_df)

''' To find the empty rows - Isnull'''
empty_data = my_df.isnull()

# print(empty_data)

# print(empty_data.loc[empty_data["Name"] == True])

'''Reading the CSV file'''

my_df1 = pd.read_csv(
    r"C:\Users\jayavaradhan.olivu\OneDrive - DISYS\Documents\Data Science\Data Science - Python\Pandas - Practice\match_data.csv")

# print(my_df1.head(10))

# empty_data1 = my_df1['other_player_dismissed'].isnull()
#
# print(empty_data1.count())

# my_df1['other_player_dismissed'].fillna(200, inplace=True)
#
# # print(my_df1.head(100))

''' Replacing the Null values into providing values'''

# my_df1['other_player_dismissed'] = my_df1['other_player_dismissed'].fillna(200)
#
# print(my_df1)

'''Below code - end with error because all the columns has least one none value '''

# my_df1.dropna(inplace=True)
#
# print(my_df1)
#
# delete1 = my_df1.dropna(how="all")
#
# print(delete1)
