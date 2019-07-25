import pandas as pd, numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm


def backwardElimination(x, y, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(len(x[0])):
                if regressor_OLS.pvalues[j].astype(float) == maxVar:
                    x = np.delete(x, j, 1)
        regressor_OLS.summary()
    return x


def backwardElimination1(x, y, SL):
    numVars = len(x[0])
    temp = np.zeros((50, 6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if regressor_OLS.pvalues[j].astype(float) == maxVar:
                    temp[:, j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if adjR_before >= adjR_after:
                        x_rollback = np.hstack((x, temp[:, [0, j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print(regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
        regressor_OLS.summary()
    return x


data = pd.read_csv('50_Startups.csv')

Y = data.iloc[:, -1].values
X = data.iloc[:, :-1].values

lableEncoder_X = LabelEncoder()
X[:, 3] = lableEncoder_X.fit_transform(X[:, 3])

ohe = OneHotEncoder(categorical_features=[3])
X = ohe.fit_transform(X).toarray()

# dummy variable trap but this model automatically takes care of that

X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# multiple variable model can also be handled by this model
regretor = LinearRegression()
regretor.fit(X_train, y_train)

Y_predict = regretor.predict(X_test)

""" Backwards Elimination starts  """

# appending original X over the 1's column will add as OSL doesnt consider d0 to be added automatically
X = np.append(np.ones((50, 1)).astype(int), values=X, axis=1)

X = backwardElimination(X, Y, 0.05)

X_opt = X[:, [0, 1, 2, 3, 4, 5]]  # will only contain the features that are significant

# Oridinary least square similarly claculate the summation of difference, it is  also a model similar to
# LinearRegression
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()

# x2 has highest p value .90 so remove it
# then x1 .94
#  then x2(index 4) .64
# we could have removed 5 b as its greater than signifance value.
X_opt = X[:, [0, 3, 5]]  # will only contain the features that are significant
# Oridinary least square similarly claculate the summation of difference, it is  also a model similar to
# LinearRegression
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()
