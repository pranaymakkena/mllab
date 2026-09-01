# 4. Write a Python program to implement Simple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Independent variable
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)

# Dependent variable
y = np.array([45, 50, 55, 60, 65, 70])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Prediction
y_pred = model.predict(X)

# Display values
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

hours = [[7]]
prediction = model.predict(hours)

print("Predicted value for 7:", prediction[0])

# Visualization
plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, label="Regression Line")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Simple Linear Regression")
plt.legend()

plt.show()
