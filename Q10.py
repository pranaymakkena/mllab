# 10. Performance analysis of Classification Algorithms on a specific dataset (Mini Project)

import pandas as pd

from sklearn.model_selection import (
    train_test_split
)

from sklearn.preprocessing import (
    StandardScaler
)

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.neighbors import (
    KNeighborsClassifier
)

from sklearn.tree import (
    DecisionTreeClassifier
)

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.metrics import (
    accuracy_score
)

# Get CSV file from user
filename = input(
    "Enter CSV file name: "
)

# Read dataset
df = pd.read_csv(filename)

print(
    "\n--- Dataset ---"
)

print(df)

# Features
X = df.iloc[:, :-1]

# Target
y = df.iloc[:, -1]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# Classification models
models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=5000
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
}

results = {}

# Train models
for name, model in models.items():

    if name in [
        "Decision Tree",
        "Random Forest"
    ]:

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(
            X_test
        )

    else:

        model.fit(
            X_train_scaled,
            y_train
        )

        prediction = model.predict(
            X_test_scaled
        )

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    results[name] = accuracy

    print(
        "\n" + name
    )

    print(
        "Accuracy:",
        accuracy
    )

# Find best model
best_model = max(
    results,
    key=results.get
)

print(
    "\n--- Performance Analysis ---"
)

for name, accuracy in results.items():

    print(
        name,
        ":",
        accuracy
    )

print(
    "\nBest Performing Algorithm:",
    best_model
)

print(
    "Best Accuracy:",
    results[best_model]
)