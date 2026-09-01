# 6. Implementation of Decision tree using sklearn and its parameter tuning

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Decision Tree
model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy before tuning:",
      accuracy_score(y_test, y_pred))

# Parameters for tuning
parameters = {
    "max_depth": [2, 3, 4, 5],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"]
}

# Grid Search
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    parameters,
    cv=3
)

grid.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid.best_params_)

# Best model
best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

print("\nAccuracy after tuning:",
      accuracy_score(y_test, y_pred))
