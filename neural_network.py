
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Create dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Attendance": [40, 45, 50, 55, 60, 65, 75, 80, 90, 95],
    "Pass":       [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# 2. Separate features and target
X = df[["Study_Hours", "Attendance"]]
y = df["Pass"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Scale the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Create neural network
model = MLPClassifier(
    hidden_layer_sizes=(8,),
    activation="relu",
    solver="adam",
    max_iter=2000,
    random_state=42
)

# 6. Train model
model.fit(X_train, y_train)

# 7. Make predictions
predictions = model.predict(X_test)

# 8. Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("Actual:", y_test.values)
print("Predicted:", predictions)
print("Accuracy:", accuracy)

# 9. Predict for a new student
new_student = pd.DataFrame({
    "Study_Hours": [7],
    "Attendance": [85]
})

new_student_scaled = scaler.transform(new_student)

result = model.predict(new_student_scaled)

if result[0] == 1:
    print("Student will PASS")
else:
    print("Student will FAIL")
