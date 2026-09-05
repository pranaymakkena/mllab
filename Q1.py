# 1. Write a python program to compute Central Tendency Measures: Mean, Median, Mode, Measure of Dispersion: Variance, Standard Deviation

import statistics

n = int(input("Enter number of values: "))

data = []

print("Enter the values:")

for i in range(n):
    value = float(input(f"Value {i + 1}: "))
    data.append(value)

print("\nData:", data)

# Central Tendency
mean = statistics.mean(data)
median = statistics.median(data)
mode = statistics.mode(data)

# Dispersion
variance = statistics.variance(data)
std_dev = statistics.stdev(data)

print("\n--- Central Tendency ---")
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)

print("\n--- Dispersion ---")
print("Variance:", variance)
print("Standard Deviation:", std_dev)