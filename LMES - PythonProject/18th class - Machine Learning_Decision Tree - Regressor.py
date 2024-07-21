from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt
import pandas as pd
import statistics

data = {
    'Size': [1, 2, 3, 4, 5],
    'Price': [10, 20, 30, 40, 50],
    'QTY': [5, 6, 2, 4, 3]
}

df = pd.DataFrame(data)

variance = statistics.variance(df['Size'])
variance_1 = statistics.variance(df['Price'])
variance_2 = statistics.variance(df['QTY'])

print(variance)
print(variance_1)
print(variance_2)
print(df.describe())

x = df[['Size', 'QTY']]
y = df['Price']

regressor = DecisionTreeRegressor()
regressor.fit(x, y)

plt.figure(figsize=(10, 8))
plot_tree(regressor, feature_names=['Size', 'QTY'], filled=True)
plt.title('Decision Tree Regressor')
plt.show()
