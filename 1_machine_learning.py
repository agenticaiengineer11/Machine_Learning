from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]]
y = [20, 40, 60, 80, 100]
model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[6]])

print("Predicted Marks:", round(prediction[0], 2))
