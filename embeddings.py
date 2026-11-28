from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences =[
    "I love machine learning",
    "I enjoy studying artificial intelligence",
    "The weather is very cold today"
]
embeddings = model.encode(sentences)
print("Embedding shape:", embeddings.shape)

print("\nFirst sentence embedding:")
print(embeddings[0])

similarity = cosine_similarity(embeddings)

print("\nSimilarity Matrix:")
print(similarity)

score = cosine_similarity(
    [embeddings[0]],
    [embeddings[1]]
)

print("\nSentence Similarity:", score[0][0])
