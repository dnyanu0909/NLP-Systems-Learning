from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "I am stressed about money",
    "I am worried about finances",
    "I love playing football"
]

embeddings = model.encode(sentences)
# Compute cosine similarity between the first two sentences
similarity = cosine_similarity([embeddings[0]], embeddings) # Compare the first sentence with all sentences
print("Cosine Similarity Scores:")
for i,score in enumerate(similarity[0]):
    print(f"{sentences[0]} <-> {sentences[i]}= {score:.4f}")