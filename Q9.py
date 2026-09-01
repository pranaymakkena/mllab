# 9. Implementation of K-Means Clustering

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_blobs(
    n_samples=200,
    centers=3,
    cluster_std=1.0,
    random_state=42
)

# Create K-Means model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Train model
kmeans.fit(X)

# Cluster labels
labels = kmeans.labels_

# Cluster centers
centers = kmeans.cluster_centers_

print("Cluster Centers:")
print(centers)

# Visualization
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels
)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=200,
    label="Centroids"
)

plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

plt.show()
