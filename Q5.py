# 5. Implementation of Multiple Linear Regression for House Price Prediction using sklearn

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

n = int(
    input("Enter number of houses: ")
)

X = []
y = []

print("\nEnter house details:")

for i in range(n):

    print(f"\nHouse {i + 1}")

    area = float(
        input("Area in sq.ft: ")
    )

    bedrooms = float(
        input("Number of bedrooms: ")
    )

    age = float(
        input("Age of house: ")
    )

    price = float(
        input("House price: ")
    )

    X.append([
        area,
        bedrooms,
        age
    ])

    y.append(price)

X = np.array(X)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(
    X_train,
    y_train
)

# Predict test data
y_pred = model.predict(
    X_test
)

print("\n--- Model Results ---")

print(
    "Mean Squared Error:",
    mean_squared_error(
        y_test,
        y_pred
    )
)

print(
    "R2 Score:",
    r2_score(
        y_test,
        y_pred
    )
)

# Predict new house
print("\nEnter details of new house")

area = float(
    input("Area in sq.ft: ")
)

bedrooms = float(
    input("Number of bedrooms: ")
)

age = float(
    input("Age of house: ")
)

new_house = [
    [area, bedrooms, age]
]

price = model.predict(
    new_house
)

print(
    "Predicted House Price:",
    price[0]
)