from sklearn.metrics.pairwise import cosine_similarity

def retrieve(query, model, doc_embeddings, documents, top_k=2):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]

    results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return results[:top_k]