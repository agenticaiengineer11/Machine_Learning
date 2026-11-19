import pandas as pd
from sklearn.linear_model import LogisticRegression
data = {
    "Free_Word": [1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
    "Has_Link":  [1, 1, 0, 0, 1, 0, 0, 1, 1, 0],
    "Spam":      [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

X = df[["Free_Word", "Has_Link"]]
y = df["Spam"]
model = LogisticRegression()
model.fit(X, y)
new_email = [[1, 1]]
prediction = model.predict(new_email)
probability = model.predict_proba(new_email)
print("Prediction:", prediction[0])
print("Probability:", probability[0])

if prediction[0] == 1:
    print("Email is SPAM")
else:
    print("Email is NOT SPAM")
