# 1. Write a python program to compute Central Tendency Measures: Mean, Median, Mode, Measure of Dispersion: Variance, Standard Deviation

import statistics as st

n = int(input("Enter number of values: "))

data = []

print("Enter the values:")

for i in range(n):
    value = float(input(f"Value {i + 1}: "))
    data.append(value)

print("\nData:", data)

# Central Tendency
mean = st.mean(data)
median = st.median(data)
mode = st.mode(data)

# Dispersion
variance = st.variance(data)
std_dev = st.stdev(data)

print("\n--- Central Tendency ---")
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)

print("\n--- Dispersion ---")
print("Variance:", variance)
print("Standard Deviation:", std_dev)
