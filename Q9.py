# 9. Implementation of K-Means Clustering

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

n = int(
    input("Enter number of data points: ")
)

X = []

print(
    "\nEnter X and Y coordinates:"
)

for i in range(n):

    print(f"\nPoint {i + 1}")

    x = float(
        input("X coordinate: ")
    )

    y = float(
        input("Y coordinate: ")
    )

    X.append([
        x,
        y
    ])

X = np.array(X)

k = int(
    input("\nEnter number of clusters: ")
)

# Create K-Means model
model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

# Train model
model.fit(X)

labels = model.labels_

centers = model.cluster_centers_

print(
    "\n--- Cluster Assignments ---"
)

for i in range(n):

    print(
        "Point",
        i + 1,
        "belongs to Cluster",
        labels[i] + 1
    )

print(
    "\n--- Cluster Centers ---"
)

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
    s=200
)

plt.xlabel("X")
plt.ylabel("Y")

plt.title(
    "K-Means Clustering"
)

plt.show()