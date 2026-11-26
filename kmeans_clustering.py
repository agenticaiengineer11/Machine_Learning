
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Create dataset
data = {
    "Age": [20, 22, 25, 28, 30, 35, 40, 42, 45, 50],
    "Spending": [15, 18, 20, 25, 30, 70, 75, 80, 85, 90]
}

df = pd.DataFrame(data)

# 2. Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 3. Create and train KMeans
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)

# 4. Create Cluster column
df["Cluster"] = kmeans.labels_

print(df)

# 5. Visualize clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    df["Age"],
    df["Spending"],
    c=df["Cluster"],
    cmap="viridis",
    s=100
)

plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation Using K-Means")

plt.show()
