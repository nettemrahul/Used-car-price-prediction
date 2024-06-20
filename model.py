
import pandas as pd

df = pd.read_csv('car data.csv')

df.head()

df.shape

newdata = df.drop('company', axis=1)
s
newdata.head()

newdata.shape

newdata.isnull().sum()

newdata.describe()

newdata.columns

newdata.drop(['Car_Name'], axis=1, inplace=True)

newdata.head()

newdata['current_year'] = 2023

newdata.head()

newdata['no_years'] = newdata['current_year'] - newdata['Year']

newdata.head()

newdata.drop(['Year', 'current_year'], axis=1, inplace=True)

newdata.head()

newdata = pd.get_dummies(newdata, drop_first=True)

newdata.head()



x = newdata.iloc
y = newdata.iloc

x = newdata.iloc[:, 1:]
y = newdata.iloc[:, 0]

x['Owner'].unique()

x.head()

y.head()

from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor()
model.fit(x, y)


print(model.feature_importances_)

from sklearn.model_selection import train_test_split

x_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators= 1000,min_samples_split= 2, min_samples_leaf= 1,max_features= 'sqrt', max_depth= 2)
regressor.fit(x_train, y_train)

import pickle

pickle.dump(regressor, open('random_forest_regression_model.pkl', 'wb'))




