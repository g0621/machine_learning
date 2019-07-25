import pandas as pd, numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Position_Salaries.csv')
X = data.iloc[:, 1:2].values
y = data.iloc[:, -1].values

from sklearn.ensemble import RandomForestRegressor

# increasing the number of estimators/trees we can increase the accuracy
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)

y_predict = regressor.predict(X)

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, c='red')
plt.plot(X_grid, regressor.predict(X_grid), c='blue')
plt.title('Truth or bluff (Random forest')
plt.xlabel('job levels')
plt.ylabel('salary')
plt.show()
