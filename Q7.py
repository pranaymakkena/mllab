# 7. Implementation of KNN using sklearn

import numpy as np

from sklearn.model_selection import (
    train_test_split
)

from sklearn.preprocessing import (
    StandardScaler
)

from sklearn.neighbors import (
    KNeighborsClassifier
)

from sklearn.metrics import (
    accuracy_score
)

n = int(
    input("Enter number of samples: ")
)

X = []
y = []

print(
    "\nEnter two features and class:"
)

for i in range(n):

    print(f"\nSample {i + 1}")

    feature1 = float(
        input("Feature 1: ")
    )

    feature2 = float(
        input("Feature 2: ")
    )

    label = int(
        input("Class (0 or 1): ")
    )

    X.append([
        feature1,
        feature2
    ])

    y.append(label)

X = np.array(X)
y = np.array(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)

k = int(
    input("\nEnter value of K: ")
)

# Create KNN model
model = KNeighborsClassifier(
    n_neighbors=k
)

# Train model
model.fit(
    X_train,
    y_train
)

# Prediction
prediction = model.predict(
    X_test
)

print(
    "\nAccuracy:",
    accuracy_score(
        y_test,
        prediction
    )
)

# New sample
print("\nEnter new sample:")

feature1 = float(
    input("Feature 1: ")
)

feature2 = float(
    input("Feature 2: ")
)

new_sample = scaler.transform(
    [[feature1, feature2]]
)

result = model.predict(
    new_sample
)

print(
    "Predicted Class:",
    result[0]
)