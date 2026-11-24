
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 6, 3, 9],
    "Attendance": [40, 45, 50, 60, 65, 70, 80, 90, 50, 85, 55, 95],
    "Pass":        [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

print(df)

X = df[["Study_Hours", "Attendance"]]

y = df["Pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=3
)

dt_model.fit(X_train, y_train)

dt_prediction = dt_model.predict(X_test)

print("Decision Tree Accuracy:",
      accuracy_score(y_test, dt_prediction))

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_prediction = rf_model.predict(X_test)

print("Random Forest Accuracy:",
      accuracy_score(y_test, rf_prediction))

print("Decision Tree:")
print(classification_report(y_test, dt_prediction, zero_division=0))

print("Random Forest:")
print(classification_report(y_test, rf_prediction, zero_division=0))
