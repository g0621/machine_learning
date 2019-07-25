import pandas as pd, numpy as np

data = pd.read_csv('Salary_Data.csv')

Y = data.iloc[:, -1].values
X = data.iloc[:, :-1].values

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

"""
THE LABIRARY WILL TAKE CARE OF SCALING 

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

Y_train = sc_Y.fit_transform(Y_train)
Y_test = sc_Y.transform(Y_test) 

"""
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_predict = regressor.predict(X_test)

# from sklearn.metrics import mean_absolute_error
# print(mean_absolute_error(Y_test,Y_predict))

import matplotlib.pyplot as plt

# plotting the train 00data
plt.scatter(X_train, Y_train, c='red')
plt.plot(X_train, regressor.predict(X_train), c='blue')  # will plot the reg. line of model
plt.title('Salary vcs Exp (Training)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# plotting the test 00data
plt.scatter(X_test, Y_test, c='red')
plt.plot(X_train, regressor.predict(X_train), c='blue')  # will plot the reg. line of model that will be predicted
plt.title('Salary vcs Exp (Test)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
