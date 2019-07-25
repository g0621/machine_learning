import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('Position_Salaries.csv')
X = data.iloc[:, 1:2].values
Y = data.iloc[:, -1].values

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression

lin_regrettor = LinearRegression()
lin_regrettor.fit(X, Y)

# used to add more features columns like X^2, X^3,X^4 etc
from sklearn.preprocessing import PolynomialFeatures

# increasing the degree may fit the curve better
pol_regrettor = PolynomialFeatures(degree=4)
X_poly = pol_regrettor.fit_transform(X)

lin2_regrettor = LinearRegression()
lin2_regrettor.fit(X_poly, Y)

# Visualizing

# pridctions by linaer model
import matplotlib.pyplot as plt
plt.scatter(X, Y,c='red')
plt.plot(X,lin_regrettor.predict(X),c='blue')
plt.title('Truth or Bluff(Linear)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# making own custom input
X_grid = np.arange(min(X),max(X),0.1) # this will give an vector but we need matrix
X_grid = X_grid.reshape(len(X_grid),1)

# predictions by polynomial models
plt.scatter(X, Y,c='red')
plt.plot(X_grid,lin2_regrettor.predict(pol_regrettor.fit_transform(X_grid)),c='blue')
plt.title('Truth or Bluff(Poly)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# predicting single value
lin_regrettor.predict([[6.5]])
lin2_regrettor.predict(pol_regrettor.transform([[6.5]]))