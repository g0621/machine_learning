import pandas as pd,numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
data = pd.read_csv('csv_name.csv')
X = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# sc_X = StandardScaler()
# sc_y = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# y_train = sc_y.fit_transform(y_train)

from sklearn.preprocessing import PolynomialFeatures

# making own custom input
X_grid = np.arange(min(X),max(X),0.1) # this will give an vector but we need matrix
X_grid = X_grid.reshape(len(X_grid),1)

# increasing the degree may fit the curve better
pol_regrettor = PolynomialFeatures(degree=4)
X_poly = pol_regrettor.fit_transform(X)
lin2_regrettor = LinearRegression()
lin2_regrettor.fit(X_poly, y)