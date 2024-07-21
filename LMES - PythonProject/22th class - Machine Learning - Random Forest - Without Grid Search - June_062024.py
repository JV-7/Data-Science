from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

california_housing = fetch_california_housing()
x = california_housing.data
y = california_housing.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=10, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                              random_state=42, verbose=2)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error (y_test, y_pred)

print(mse)








