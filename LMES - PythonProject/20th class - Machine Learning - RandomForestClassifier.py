from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder

housing_df = pd.read_csv(
    r'C:\Users\jayavaradhan.olivu\OneDrive - DISYS\Documents\Data Science\Data Science - Python\Python - Data '
    r'Source\housing.csv')

x = housing_df.drop(['median_house_value'], axis=1)
y = housing_df['median_house_value']  # this is y-axis only not particularly for regression alone
LE = LabelEncoder()
x['ocean_proximity'] = LE.fit_transform(x['ocean_proximity'])
x = x.fillna(x['ocean_proximity'].mean())
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

random_forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_regressor.fit(x_train, y_train)

print(random_forest_regressor.feature_importances_)

prediction = random_forest_regressor.predict(x_test)
print(prediction)






















