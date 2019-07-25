import pandas as pd,numpy as np

data = pd.read_csv('Market_Basket_Optimisation.csv',header=None)
transactions = []
for i in range(7501):
    transactions.append([str(data.values[i,j]) for j in range(20)])

from association_rules.apyori import apriori
# min support is bought 3 times a day so total in week is 7*3
# so support is 7*3 / 7500
rules = apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3)
results = list(rules)