# 3. Study of Python Libraries for ML application such as Pandas and Matplotlib

import pandas as pd
import matplotlib.pyplot as plt

# Creating a dataset
data = {
    "Student": ["A", "B", "C", "D", "E"],
    "Marks": [75, 82, 90, 65, 88]
}

# Create DataFrame
df = pd.DataFrame(data)

print("Dataset:")
print(df)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Plotting
plt.figure(figsize=(8, 5))

plt.bar(df["Student"], df["Marks"])

plt.xlabel("Students")
plt.ylabel("Marks")
plt.title("Student Marks")

plt.show()
