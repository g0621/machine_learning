import pandas as pd, numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Mall_Customers.csv')

X = data.iloc[:, [3, 4]].values

# using the elbow method to determine the number of clusters
from sklearn.cluster import KMeans

WCSS = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)
plt.plot(range(1, 11), WCSS)
plt.title('The elbow method')
plt.xlabel('Number of cluster')
plt.ylabel('WCSS')
plt.show()

# initlizign with perfect values
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_predict = kmeans.fit_predict(X)

plt.scatter(X[y_predict == 0, 0], X[y_predict == 0, 1], s=100, c='red', label='cluster 1')
plt.scatter(X[y_predict == 1, 0], X[y_predict == 1, 1], s=100, c='blue', label='cluster 2')
plt.scatter(X[y_predict == 2, 0], X[y_predict == 2, 1], s=100, c='green', label='cluster 31')
plt.scatter(X[y_predict == 3, 0], X[y_predict == 3, 1], s=100, c='cyan', label='cluster 4')
plt.scatter(X[y_predict == 4, 0], X[y_predict == 4, 1], s=100, c='magenta', label='cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow')
plt.title('Cluster od clients')
plt.xlabel('Annual inclome k$')
plt.ylabel('Spending 1-100')
plt.legend()
plt.show()
