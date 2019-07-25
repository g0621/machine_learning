import numpy as np, pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Position_Salaries.csv')
X = data.iloc[:, 1:2].values
y = data.iloc[:, -1].values

# svr doesnt supports feature scaling on its own

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_Y = StandardScaler()
y = sc_Y.fit_transform(y.reshape(-1,1))

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

regretor = SVR(kernel='rbf')
regretor.fit(X, y)

# As model is fitted with scalled value so to predict certain value we will use

sc_Y.inverse_transform(regretor.predict(sc_X.transform([[6.5]])))

# level 10 considered as outlier bcz 05SVR model uses penalty parameters to allow soft boundry
Y_pred = regretor.predict(X)
plt.scatter(X, y, c='red')
plt.plot(X, regretor.predict(X), c='blue')
plt.title('truth or bluff svr')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
