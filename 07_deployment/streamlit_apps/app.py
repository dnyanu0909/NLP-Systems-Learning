import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

st.title("💰 AI Financial Stress Assistant")

# load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# load data
BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, "../../04_projects/rag_system/knowledge_base.txt")

with open(file_path, "r") as f:
    documents = [doc.strip() for doc in f.readlines() if doc.strip()]

doc_embeddings = model.encode(documents)

#load memory
if "history" not in st.session_state:
    st.session_state.history = []

def retrieve(query):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]
    results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return results[:2]

def generate_response(query, retrieved_docs):
    context = [doc for doc, _ in retrieved_docs]

    return f"""
I understand you're dealing with: "{query}"

Here are some practical steps:

- {context[0]}
- {context[1]}
"""

# input
user_input = st.text_input("Ask your question:")

if user_input:
    full_query = " ".join(st.session_state.history[-2:] + [user_input])

    retrieved = retrieve(full_query)
    response = generate_response(user_input, retrieved)

    for msg in st.session_state.history:
        st.write("🧑:", msg)
    st.write(response)

    st.session_state.history.append(user_input)