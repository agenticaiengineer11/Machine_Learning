import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
data = {
    "Message": [
        "Win a free prize now",
        "Claim your free gift today",
        "Congratulations you won money",
        "Click here to claim your reward",
        "Free entry in a cash competition",
        "You have won a free ticket",
        "Urgent claim your prize now",
        "Get free money today",
        "Win cash by clicking this link",
        "Congratulations claim your reward",
        "Let's meet for lunch tomorrow",
        "Can you call me later",
        "I am going to university",
        "Please send me the assignment",
        "Are you coming to class today",
        "I will meet you at home",
        "Please share the lecture notes",
        "What time is our meeting",
        "I am learning Python programming",
        "Let's study machine learning"
    ],

    "Label": [
        "spam", "spam", "spam", "spam", "spam",
        "spam", "spam", "spam", "spam", "spam",
        "ham", "ham", "ham", "ham", "ham",
        "ham", "ham", "ham", "ham", "ham"
    ]
}

df = pd.DataFrame(data)
X = df["Message"]
y = df["Label"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

new_message = ["Congratulations! You won a free prize"]

result = model.predict(new_message)

print("New Message:", new_message[0])
print("Prediction:", result[0])
