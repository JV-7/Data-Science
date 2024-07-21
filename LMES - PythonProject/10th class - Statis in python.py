import pandas as pd

pd.set_option('display.max_column', None)
pd.set_option('display.max_row', None)
pd.set_option('display.width', 500)

df = pd.read_csv('train.csv')

# print(df.head(10))

''' Descriptive Statistics '''
descriptive_statis = df.describe()
print(descriptive_statis)

''' Mean, Mode, Median '''
Mean_df = df['Age'].mean()
print(Mean_df)

Mode_df = df['Age'].mod(other=6)
print(Mode_df.head(10))

Median_df = df['Age'].median()
print(Median_df)

''' variance and standard deviation '''

var_stat = df['Age'].var()
print(var_stat)

std_stat = df['Age'].std()
print(std_stat)

''' Overall statis '''

df.info()

''' quantile - '"probability distribution into equal parts"'. '''

first_quantile = df['Age'].quantile([0.25, 0.75, 0.5])
print(f'quantile \n{first_quantile}')
''' in simple words data fall below 25%, 75%, 50% '''

''' co-relation and co-variance '''

co_relation = df[['Age', 'Survived']].corr()
print(co_relation)

# co-relation is the mathematical feature that can be applied between two columns
''' 1 value of 0 indicates No - correlation
2 value of 1 positive co-relation
3 value of -1 negative co-relation
if the co-relation is either 1 or -1 is good for our model

----------------- Detailed view of co-relation--------------------------------

1. correlation of 1 or -1 is perfect model for linear relationship. For every increase in one variable there is a 
increase of y or decrease of y
2. when we get either 1 or -1, means our data is highly consistent

Note:
    
    above correlation of 1 or -1 may increase "Multi collinearity", "Overfitting", "Duplication"

'''
my_data = {"Height": [100, 120, 150, 140, 190],
           "Weight": [40, 40, 50, 60, 70],
           "Age": [22, 33, 44, 55, 66]}

df_1 = pd.DataFrame(my_data)

co_relation_1 = df_1[["Height", "Weight"]].corr()

print(co_relation_1)
print(co_relation_1.iloc[1, 0])

co_relation_1 = df_1[["Height", "Weight"]].corr().iloc[1, 0]
print(co_relation_1)

co_relation_1 = df_1["Height"].corr(df_1["Weight"])
print(co_relation_1)
#
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.regplot(x='Height', y='Weight', data=df_1)
plt.title("scatter for finding co-relation")
plt.xlabel('Height(cm)')
plt.ylabel('Weight(kg)')
plt.show()
'''---Below code is avoid multi co-linearity in our given dataset '''

my_data = {"Height": [100, 120, 130, 140, 150],
           "Weight": [40, 50, 60, 70, 80],
           "Age": [22, 33, 44, 55, 66]}

df_1 = pd.DataFrame(my_data)

co_relation_1 = df_1[["Height", "Weight"]].corr()

print(co_relation_1)
print(co_relation_1.iloc[0, 1])

co_relation_1 = df_1[["Height", "Weight"]].corr().iloc[1, 0]
print(co_relation_1)

from statsmodels.stats.outliers_influence import variance_inflation_factor

x = df_1[['Height', 'Weight']]

my_vif_df = pd.DataFrame()
my_vif_df['Feature'] = x.columns
my_vif_df['VIF'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
print(my_vif_df)

''' NOT VIF (Variance) value above 10 is generally said to be data with high multi co-linearity
 '''
print('------------')
print(x.shape)
print(x.values)
















