# 3. Study of Python Libraries for ML application such as Pandas and Matplotlib

import pandas as pd
import matplotlib.pyplot as plt

n = int(input("Enter number of students: "))

names = []
marks = []

for i in range(n):

    name = input(
        f"Enter name of student {i + 1}: "
    )

    mark = float(
        input(f"Enter marks of {name}: ")
    )

    names.append(name)
    marks.append(mark)

# Create DataFrame
data = {
    "Student": names,
    "Marks": marks
}

df = pd.DataFrame(data)

print("\n--- Student Data ---")
print(df)

print("\n--- Statistical Summary ---")
print(df.describe())

# Plot
plt.figure(figsize=(8, 5))

plt.bar(
    df["Student"],
    df["Marks"]
)

plt.xlabel("Students")
plt.ylabel("Marks")
plt.title("Student Marks")

plt.show()