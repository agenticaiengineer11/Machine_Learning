from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

depths = [1, 2, 3, 5, 10, None]

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_prediction = model.predict(X_train)

    test_prediction = model.predict(X_test)

    train_accuracy = accuracy_score(
        y_train, train_prediction
    )

    test_accuracy = accuracy_score(
        y_test, test_prediction
    )

    print("\nMax Depth:", depth)
    print("Training Accuracy:", train_accuracy)
    print("Testing Accuracy:", test_accuracy)



model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("Fold Scores:", scores)
print("Average Accuracy:", scores.mean())

