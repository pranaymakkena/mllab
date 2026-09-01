# 2. Study of Python Basic Libraries such as Statistics, Math, Numpy and Scipy

import statistics
import math
import numpy as np
from scipy import stats

data = [10, 20, 30, 40, 50]

# Statistics library
print("STATISTICS LIBRARY")
print("Mean:", statistics.mean(data))
print("Median:", statistics.median(data))
print("Standard Deviation:", statistics.stdev(data))

# Math library
print("\nMATH LIBRARY")
print("Square root of 25:", math.sqrt(25))
print("Factorial of 5:", math.factorial(5))
print("Value of pi:", math.pi)
print("Power:", math.pow(2, 3))

# NumPy library
print("\nNUMPY LIBRARY")
array = np.array(data)
print("Array:", array)
print("Mean:", np.mean(array))
print("Maximum:", np.max(array))
print("Minimum:", np.min(array))

# SciPy library
print("\nSCIPY LIBRARY")
result = stats.describe(data)

print("Number of observations:", result.nobs)
print("Minimum and Maximum:", result.minmax)
print("Mean:", result.mean)
print("Variance:", result.variance)
