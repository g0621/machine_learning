import pandas as pd, numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Mall_Customers.csv')
X = data.iloc[:, [3, 4]].values


# using the sypy for dendogram

import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.title('Dendogram')
plt.show()

# fitting with 5 clusters as found optimal in above step to aglomerative clustering
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=5)
y_predict = model.fit_predict(X)

plt.scatter(X[y_predict == 0, 0], X[y_predict == 0, 1], s=100, c='red', label='cluster 1')
plt.scatter(X[y_predict == 1, 0], X[y_predict == 1, 1], s=100, c='blue', label='cluster 2')
plt.scatter(X[y_predict == 2, 0], X[y_predict == 2, 1], s=100, c='green', label='cluster 31')
plt.scatter(X[y_predict == 3, 0], X[y_predict == 3, 1], s=100, c='cyan', label='cluster 4')
plt.scatter(X[y_predict == 4, 0], X[y_predict == 4, 1], s=100, c='magenta', label='cluster 5')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow')
plt.title('Cluster od clients')
plt.xlabel('Annual inclome k$')
plt.ylabel('Spending 1-100')
plt.legend()
plt.show()