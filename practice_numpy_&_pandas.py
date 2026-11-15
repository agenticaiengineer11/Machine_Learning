
import numpy as np
import pandas as pd
df = pd.read_csv("students.csv")

print("Original Dataset:")
print(df)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())
df["Marks"] = df["Marks"].fillna(df["Marks"].mean())
print("\nCleaned Dataset:")
print(df)
marks = np.array(df["Marks"])

print("\nAverage Marks:", np.mean(marks))
print("Highest Marks:", np.max(marks))
print("\nStudents with marks above 70:")
print(df[df["Marks"] > 70])
