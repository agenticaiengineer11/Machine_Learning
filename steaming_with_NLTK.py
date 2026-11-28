from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

words = ["playing", "played", "plays", "studies"]

for word in words:
    print(word, "->", stemmer.stem(word))