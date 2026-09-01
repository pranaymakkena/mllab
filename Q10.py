# 10. Performance analysis of Classification Algorithms on a specific dataset (Mini Project)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Classification models
models = {
    "Logistic Regression":
        LogisticRegression(max_iter=5000),

    "KNN":
        KNeighborsClassifier(n_neighbors=5),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
}

accuracy_results = {}

# Train and evaluate each model
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    accuracy_results[name] = accuracy

    print(name)
    print("Accuracy:", accuracy)
    print()

# Find best model
best_model = max(
    accuracy_results,
    key=accuracy_results.get
)

print("Best Performing Model:", best_model)

# Visualization
plt.figure(figsize=(9, 5))

plt.bar(
    accuracy_results.keys(),
    accuracy_results.values()
)

plt.xlabel("Classification Algorithms")
plt.ylabel("Accuracy")
plt.title("Performance Comparison of Classification Algorithms")

plt.ylim(0, 1)

plt.show()
