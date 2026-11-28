import re

text = "I LOVE Machine Learning!!! It is AMAZING."

text = text.lower()

text = re.sub(r"[^a-zA-Z\s]", "", text)

tokens = text.split()
stopwords = {"i", "it", "is", "the", "a", "an"}

clean_tokens = [
    word for word in tokens
    if word not in stopwords
]

print("Original:", text)
print("Tokens:", tokens)
print("Clean tokens:", clean_tokens)