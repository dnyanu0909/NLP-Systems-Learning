from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

#load a stronger transformer model
model = SentenceTransformer('all-mpnet-base-v2')

sentences = [
    "I love machine learning and natural language processing.",
    "I enjoy deep learning and data science.",
    "I am passionate about artificial intelligence and big data."
]

query = input("Enter your query: ")

#Encode the sentences and the query
sentence_embeddings = model.encode(sentences)
query_embedding = model.encode([query])

#Calculate cosine similarity
similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]

#Sort results
result = sorted(zip(sentences, similarities),key=lambda x: x[1],reverse=True)

print("\nTop matches:")
for sentence, score in result:
    print(f"{sentence} (Similarity: {score:.4f})")