# 4. Write a Python program to implement Simple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

n = int(
    input("Enter number of data points: ")
)

x_values = []
y_values = []

print("\nEnter X values:")

for i in range(n):
    x = float(input(f"X{i + 1}: "))
    x_values.append(x)

print("\nEnter Y values:")

for i in range(n):
    y = float(input(f"Y{i + 1}: "))
    y_values.append(y)

# Convert to NumPy arrays
X = np.array(x_values).reshape(-1, 1)
y = np.array(y_values)

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict training values
y_pred = model.predict(X)

print("\n--- Regression Results ---")

print("Slope:",
      model.coef_[0])

print("Intercept:",
      model.intercept_)

print("\nPredicted Values:")

for i in range(n):

    print(
        "X =", x_values[i],
        "Actual =", y_values[i],
        "Predicted =", y_pred[i]
    )

# New prediction
new_x = float(
    input("\nEnter X value for prediction: ")
)

new_prediction = model.predict(
    [[new_x]]
)

print(
    "Predicted Y:",
    new_prediction[0]
)

# Graph
plt.scatter(
    X,
    y,
    label="Actual Data"
)

plt.plot(
    X,
    y_pred,
    label="Regression Line"
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Simple Linear Regression")
plt.legend()

plt.show()