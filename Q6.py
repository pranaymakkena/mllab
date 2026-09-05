# 6. Implementation of Decision tree using sklearn and its parameter tuning

import numpy as np

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.tree import (
    DecisionTreeClassifier
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

# Decision Tree
model = DecisionTreeClassifier(
    random_state=42
)

model.fit(
    X_train,
    y_train
)

prediction = model.predict(
    X_test
)

print(
    "\nAccuracy before tuning:",
    accuracy_score(
        y_test,
        prediction
    )
)

# Parameter tuning
parameters = {

    "criterion": [
        "gini",
        "entropy"
    ],

    "max_depth": [
        2,
        3,
        4,
        5
    ],

    "min_samples_split": [
        2,
        5,
        10
    ]
}

grid = GridSearchCV(
    DecisionTreeClassifier(
        random_state=42
    ),
    parameters,
    cv=3
)

grid.fit(
    X_train,
    y_train
)

print(
    "\nBest Parameters:"
)

print(
    grid.best_params_
)

best_model = grid.best_estimator_

prediction = best_model.predict(
    X_test
)

print(
    "\nAccuracy after tuning:",
    accuracy_score(
        y_test,
        prediction
    )
)