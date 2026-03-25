from sentence_transformers import SentenceTransformer

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example sentences
sentences = [
    "I am stressed about money",
    "I am worried about finances",
    "I love playing football"
]

# Generate embeddings for the sentences
embeddings = model.encode(sentences)

for i,emb in enumerate(embeddings):
    print(f"Sentence: {sentences[i]}")
    print(f"Embedding: {emb[:5]}...")  # Print the first 5 dimensions of the embedding
    print()
