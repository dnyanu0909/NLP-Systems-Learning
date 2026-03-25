from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# load knowledge base
BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, "knowledge_base.txt")

with open(file_path, "r") as f:
    documents = [doc.strip() for doc in f.readlines() if doc.strip()]

doc_embeddings = model.encode(documents)

# memory
conversation_history = []

def retrieve(query, top_k=2):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]

    results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return results[:top_k]


def build_context(history, current_query):
    context = ""

    for turn in history[-2:]:
        context += turn["user"] + " "
    
    context += current_query

    return context

def is_followup(query):
    keywords = ["first", "next", "after", "then"]
    return any(word in query.lower() for word in keywords)

def generate_response(query, retrieved_docs, conversation_history):
    context = [doc for doc, _ in retrieved_docs]

    if is_followup(query) and conversation_history:
        return f"""
Since you asked about next steps:

Step 1:
- Manage and track your expenses

Step 2:
- Start saving regularly

Keep going step by step.
"""
    else:
        return f"""
I understand you're dealing with: "{query}"

Here are some practical steps:

- {context[0]}
- {context[1]}
"""
# chat loop
while True:
    query = input("\nYou: ")

    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # include history (simple version)
    full_query = build_context(conversation_history, query)
    retrieved = retrieve(full_query)

    response = generate_response(query, retrieved, conversation_history)

    print("\nAI:", response)

    # update memory
    conversation_history.append({
    "user": query,
    "assistant": response
})