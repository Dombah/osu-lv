from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.metrics as metrics
import pandas as pd
import math


data = pd.read_csv('data_C02_emission.csv')

data = data.drop(['Make', 'Model'], axis = 1)


input_variables = ['Fuel Consumption City (L/100km)',
                   'Fuel Consumption Hwy (L/100km)',
                   'Fuel Consumption Comb (L/100km)',
                   'Fuel Consumption Comb (mpg)',
                   'Engine Size (L)',
                   'Cylinders'
                   ]

output_variable = ['CO2 Emissions (g/km)']
X = data[input_variables].to_numpy()
y = data[output_variable].to_numpy()

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, random_state = 1)

plt.scatter(X_train[:,3], y_train, color = 'blue')
plt.scatter(X_test[:,3], y_test, color = 'red')
plt.legend(['Train', 'Test'])
plt.show()

sc = MinMaxScaler ()
X_train_n = sc.fit_transform (X_train)
X_test_n = sc.transform (X_test)


plt.hist(X_train[:,2])
plt.hist(X_train_n[:,2])
plt.legend(['Before scaling', 'After scaling'])
plt.show()

linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)

print(linearModel.coef_) # parametri modela 

y_test_p = linearModel.predict (X_test_n)
plt.scatter(y_test, y_test_p, color = 'blue')
plt.show()

MAE = metrics.mean_absolute_error(y_test, y_test_p)
MSE = metrics.mean_squared_error(y_test, y_test_p)
RMSE = math.sqrt(metrics.mean_squared_error(y_test, y_test_p))
MAPE = metrics.mean_absolute_percentage_error(y_test, y_test_p)
R2 = metrics.r2_score(y_test, y_test_p)

print(f'MAE: {MAE}')
print(f'MSE: {MSE}')
print(f'RMSE: {RMSE}')
print(f'MAPE: {MAPE}')
print(f'R2: {R2}')