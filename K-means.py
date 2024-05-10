from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100, 2)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.title('K-means Clustering')
plt.show()
