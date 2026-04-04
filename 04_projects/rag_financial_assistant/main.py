from sentence_transformers import SentenceTransformer
import os
from retriever import retrieve
from generator import generate_response

# load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# load data
BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, "knowledge_base.txt")

with open(file_path, "r") as f:
    documents = [doc.strip() for doc in f.readlines() if doc.strip()]

doc_embeddings = model.encode(documents)

# run
query = input("Ask something: ")

retrieved = retrieve(query, model, doc_embeddings, documents)
response = generate_response(query, retrieved)

print("\nResponse:\n", response)