# 5. Implementation of Multiple Linear Regression for House Price Prediction using sklearn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample house dataset
data = {
    "Area": [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800],
    "Bedrooms": [2, 2, 3, 3, 4, 4, 4, 5],
    "Age": [10, 8, 7, 5, 4, 3, 2, 1],
    "Price": [3000000, 3500000, 4500000, 5200000,
              6000000, 6500000, 7500000, 8500000]
}

df = pd.DataFrame(data)

# Features and target
X = df[["Area", "Bedrooms", "Age"]]
y = df["Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Actual Prices:")
print(y_test.values)

print("\nPredicted Prices:")
print(y_pred)

print("\nMean Squared Error:",
      mean_squared_error(y_test, y_pred))

print("R2 Score:",
      r2_score(y_test, y_pred))

# Predict a new house
new_house = pd.DataFrame(
    [[2100, 4, 3]],
    columns=["Area", "Bedrooms", "Age"]
)

price = model.predict(new_house)

print("\nPredicted House Price:", price[0])
