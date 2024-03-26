from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
import math


data = pd.read_csv('data_C02_emission.csv')

input_variables = ['Fuel Consumption City (L/100km)',
                   'Fuel Consumption Hwy (L/100km)',
                   'Fuel Consumption Comb (L/100km)',
                   'Fuel Consumption Comb (mpg)',
                   'Engine Size (L)',
                   'Cylinders',
                   'Fuel Type'
                   ]

output_variable = ['CO2 Emissions (g/km)']
X = data[input_variables].to_numpy()
y = data[output_variable].to_numpy()

ohe = OneHotEncoder ()
X_encoded = ohe.fit_transform(data[['Fuel Type']]).toarray()

X = np.hstack((X[:,:-1], X_encoded))

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, random_state = 1)

linearModel = lm.LinearRegression()
linearModel.fit(X_train, y_train)

print(linearModel.coef_) # parametri modela 

y_test_p = linearModel.predict (X_test)

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

max_error = max(np.abs(y_test - y_test_p))
print('Max error: ', max_error)

max_error_index = np.argmax(np.abs(y_test-y_test_p))
car_model = data.iloc[max_error_index]['Model']

print(f'Car model with the max error: {car_model}')