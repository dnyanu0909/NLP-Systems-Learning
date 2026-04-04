from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os
BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, "knowledge_base.txt")
with open(file_path, "r") as f:
    documents = f.readlines()

#load model
model = SentenceTransformer('all-MiniLM-L6-v2')

#clean documents
documents = [doc.strip() for doc in documents if doc.strip()]

print("Reading from:", file_path)
print("Documents loaded:", documents)

#create embeddings
doc_embeddings = model.encode(documents)

#retrieval function
def retrieve(query, top_k=2):
    query_embedding = model.encode([query])  # keep 2D

    scores = cosine_similarity(query_embedding, doc_embeddings)[0]
    results = []

    for i in range(len(documents)):
        score = scores[i]
        score = boost_score(documents[i], score)
        category = categorize(documents[i])
        results.append((documents[i], score, category))

# sort by:
# 1. category (action first)
# 2. score

    results = sorted(results,key=lambda x: (x[2] != "action", -priority_score(x[0]), -x[1]))
    return results[:top_k]

#boosting function
def boost_score(sentence, score):
    keywords = ["reduce", "improve", "manage", "help", "avoid"]

    for word in keywords:
        if word in sentence.lower():
            score += 0.05  # small boost

    return score

#categorization function
def categorize(sentence):
    if any(word in sentence.lower() for word in ["reduce", "improve", "manage", "saving", "avoid"]):
        return "action"
    else:
        return "info"

#priority scoring function
def priority_score(sentence):
    if "tracking" in sentence.lower():
        return 2
    elif "saving" in sentence.lower():
        return 1
    return 0   
#generation function
def generate_response(query, retrieved_docs):
    context = [doc for doc, _, _ in retrieved_docs]

    return f"""
I understand you're dealing with: "{query}"

Here are some practical steps you can take:

- {context[1]}
- {context[0]}

Stay consistent with small improvements — they compound over time.
"""

if __name__ == "__main__":
    query = input("Ask something: ")

    retrieved = retrieve(query)

    print("\nRetrieved Context:")
    for doc, score, category in retrieved:
        print(f"- {doc} ({score:.4f})")

    answer = generate_response(query, retrieved)

    print("\nFinal Answer:")
    print(answer)