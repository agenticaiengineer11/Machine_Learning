import pandas as pd
from sklearn.linear_model import LinearRegression
data = {
    "Area": [800, 1000, 1200, 1500, 1800],
    "Price": [40, 50, 60, 75, 90]
}
df = pd.DataFrame(data)
x = df[["Area"]]
y = df["Price"]
model = LinearRegression()
model.fit(x, y)
prediction = model.predict([[1600]])

print("Predicted House Price:", prediction[0], "Lakhs")

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
