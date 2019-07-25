import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""Reading the dataset 
    1. iloc .values removes the column and row labels """
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1]
"""Removing the missing values strategy can be mean, median, most_frequent"""
from sklearn.preprocessing.imputation import Imputer
# from sklearn.impute import SimpleImputer

SI = Imputer(missing_values=np.nan, strategy='mean')

"""when we fit a model with 00data it calculates important parameters like mean etc from 
   given 00data , then when we transform another set using that model then it utilizes that 
   previous model. """
SI = SI.fit(X[:, 1:3])
X[:, 1:3] = SI.transform(X[:, 1:3])

"""we cant use english labels so we change it to 1,2,3 but it can give different weight 
    to columns so we change it to n different columns were n is number of types of entries 
    in categorical column"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])

oneHotEncoder = OneHotEncoder(categorical_features=[0])
X = oneHotEncoder.fit_transform(X).toarray()
labelEncoder_Y = LabelEncoder()
Y = labelEncoder_Y.fit_transform(Y)

# splitting into test train
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)  # as we don't want to know the values for test


