import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = {
    "Hours": [1, 2, 3, 4, 5],
    "Marks": [20, 40, 60, 80, 100],
    "Attendance": [50, 60, 70, 80, 90]
}

df = pd.DataFrame(data)

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.show()