from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

#Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample documents
document = """
Natural language processing (NLP) is a subfield of artificial intelligence.
The goal of NLP is to enable machines to understand human language.
NLP involves tasks like translation, sentiment analysis, and chatbots.
It uses machine learning, deep learning, and statistical techniques.
"""

#Smart chunking
chunks = [chunk.strip() for chunk in document.split("\n") if chunk.strip()]

# Generate embeddings
chunk_embeddings = model.encode(chunks)

#Convert to numpy array
chunk_embeddings = np.array(chunk_embeddings).astype('float32')

# Build FAISS index
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

# Input query
query = input("Enter your query: ")

# Generate query embedding
query_embedding = model.encode([query]).astype('float32')

# Search for similar chunks
k = 2
distances, indices = index.search(query_embedding, k)
print("\nTop matching chunks:")

for i in indices[0]:
    print(f"Chunk {i+1}: {chunks[i]}")


#Before - cosine loop , small scale , manual chunking
#After - FAISS index , scalable , optimized chunking

#now system can handle larger documents with many chunks and provide faster retrieval of relevant information based on the query.