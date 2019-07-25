import pandas as pd, numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Position_Salaries.csv')
X = data.iloc[:, 1:2].values
y = data.iloc[:, -1].values

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()

regressor.fit(X, y)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, c='red')
plt.plot(X_grid, regressor.predict(X_grid), c='blue')
plt.title('truth or bluff Decision tree')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
