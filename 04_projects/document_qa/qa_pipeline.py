from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

document = """
Natural language processing (NLP) is a subfield of artificial intelligence (AI) that focuses on the interaction between computers and humans through natural language.
The ultimate goal of NLP is to enable computers to understand, interpret, and generate human language in a valuable way.
NLP combines computational linguistics with machine learning, deep learning, statistical modeling, and more.
It involves tasks such as language translation, sentiment analysis, chatbots, and information retrieval."""

def filter_chunks(query, chunks):
    query = query.lower()

    if "application" in query or "involve" in query:
        return [c for c in chunks if "involves" in c.lower() or "tasks" in c.lower()]

    if "goal" in query:
        return [c for c in chunks if "goal" in c.lower()]

    if "what is" in query:
        return [c for c in chunks if "subfield" in c.lower()]

    return chunks

def rewrite_query(query):
    query = query.lower()

    if "application" in query or "involve" in query:
        return "tasks and applications of NLP"

    if "goal" in query:
        return "goal of NLP"

    if "what is" in query:
        return "definition of NLP"

    return query

def generate_answer(query, results):
    if "what is" in query.lower():
        return f"Definition:\n{results[0]}"

    if "goal" in query.lower():
        return f"Goal:\n{results[0]}"

    if "application" in query.lower() or "involve" in query.lower():
        return f"Applications:\n{results[0]}"

    return f"Answer:\n{results[0]}"

#split into chunks
chunks = [chunk.strip() for chunk in document.split("\n") if chunk.strip()]
query = input("Enter your query: ")
filtered_chunks = filter_chunks(query, chunks)
query = rewrite_query(query)
chunk_embeddings = model.encode(filtered_chunks)

query_embedding = model.encode([query])

scores = cosine_similarity(query_embedding, chunk_embeddings)[0]

top_k = 2
top_indices = np.argsort(scores)[-top_k:][::-1]
# print("\nTop matching chunks:")
# for idx in top_indices:
#     print(f"- {filtered_chunks[idx]}")

#get best matching chunk
# best_chunk = filtered_chunks[scores.argmax()]

answer = generate_answer(query, [filtered_chunks[i] for i in top_indices])
print("\nFinal Answer:")
print(answer)

# print("Best matching chunk:")
# print(best_chunk)

#ask query : What is NLP? , ans should be : Natural language processing (NLP) is a subfield of artificial intelligence (AI) that focuses on the interaction between computers and humans through natural language.
#or What is the goal of NLP? ANS : The ultimate goal of NLP is to enable computers to understand, interpret, and generate human language in a valuable way.
#or What are the applications of NLP? ans should be : NLP involves tasks such as language translation, sentiment analysis, chatbots, and information retrieval.
#or What are the techniques used in NLP? ans should be : NLP combines computational linguistics with machine learning, deep learning, statistical modeling, and more.