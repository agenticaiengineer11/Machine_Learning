import pandas as pd

data = {
    "Study_Hours": [1, 2, 3, 4, 5],
    "Marks": [20, 40, 60, 80, 100]
}

df = pd.DataFrame(data)

correlation = df["Study_Hours"].corr(df["Marks"])

print("Correlation:", correlation)