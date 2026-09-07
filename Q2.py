# 2. Study of Python Basic Libraries such as Statistics, Math, Numpy and Scipy

import statistics as st
import math
import numpy as np
from scipy import stats

n = int(input("Enter number of values: "))

data = []

print("Enter the values:")

for i in range(n):
    value = float(input(f"Value {i + 1}: "))
    data.append(value)

array = np.array(data)

print("\n--- STATISTICS LIBRARY ---")

print("Mean:", st.mean(data))
print("Median:", st.median(data))

if len(data) > 1:
    print("Standard Deviation:",
          st.stdev(data))

print("\n--- MATH LIBRARY ---")

number = float(input("Enter a number for Math operations: "))

print("Square Root:", math.sqrt(number))
print("Square:", math.pow(number, 2))
print("Factorial:",
      math.factorial(int(number)))

print("\n--- NUMPY LIBRARY ---")

print("Array:", array)
print("Mean:", np.mean(array))
print("Maximum:", np.max(array))
print("Minimum:", np.min(array))
print("Variance:", np.var(array))
print("Standard Deviation:", np.std(array))

print("\n--- SCIPY LIBRARY ---")

result = stats.describe(array)

print("Number of observations:",
      result.nobs)

print(
    "Minimum and Maximum:",
    (float(result.minmax[0]),
     float(result.minmax[1]))
)

print("Mean:", result.mean)
print("Variance:", result.variance)
