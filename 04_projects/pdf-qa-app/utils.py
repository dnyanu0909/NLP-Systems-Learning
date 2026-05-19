from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
import re
from openai import OpenAI
import os
print(os.getenv("OPENROUTER_API_KEY"))

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================
# CHANGE:
# Good lightweight embedding model for semantic search
# No issue here
model = SentenceTransformer('all-MiniLM-L6-v2')


# =========================================================
# EXTRACT TEXT FROM PDF
# =========================================================
def extract_text_from_pdf(pdf_file):

    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

    text = ""

    for page in doc:
        text += page.get_text()

    return text


# =========================================================
# SMART CHUNKING WITH OVERLAP
# =========================================================
def smart_chunk(text, chunk_size=3, overlap=1):

    # CHANGE:
    # Replace newlines with spaces for cleaner text
    text = text.replace("\n", " ")

    # CHANGE:
    # Better sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []

    # CHANGE:
    # Overlap improves retrieval quality
    # Example:
    # chunk1 = sentence1 sentence2 sentence3
    # chunk2 = sentence3 sentence4 sentence5
    step = chunk_size - overlap

    for i in range(0, len(sentences), step):

        chunk = " ".join(sentences[i:i + chunk_size]).strip()

        # CHANGE:
        # Ignore tiny useless chunks
        if len(chunk) > 50:
            chunks.append(chunk)

    return chunks


# =========================================================
# CREATE FAISS INDEX
# =========================================================
def create_faiss_index(clean_chunks):

    # Generate embeddings
    embeddings = model.encode(clean_chunks)

    # Convert to float32 for FAISS
    embeddings = np.array(embeddings).astype('float32')

    # CHANGE:
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    # CHANGE:
    # IndexFlatIP + normalized embeddings
    # = cosine similarity search
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# =========================================================
# QUERY REWRITING
# =========================================================
def rewrite_query(query):

    query = query.lower()

    # CHANGE:
    # Expands weak user queries
    # Helps semantic retrieval
    if "application" in query:
        return query + " tasks uses examples"

    if "goal" in query:
        return query + " purpose objective"

    if "definition" in query:
        return query + " meaning explanation"

    return query


# =========================================================
# KEYWORD BOOSTING / RERANKING
# =========================================================
def keyword_boost(query, chunks):

    boosted = []

    for chunk in chunks:

        score = 0

        # CHANGE:
        # Basic keyword scoring
        for word in query.lower().split():

            if word in chunk.lower():
                score += 1

        boosted.append((chunk, score))

    # Sort by keyword score descending
    boosted.sort(key=lambda x: x[1], reverse=True)

    return [x[0] for x in boosted]


# =========================================================
# RETRIEVE RELEVANT CHUNKS
# =========================================================
def retrieve_answer(query, clean_chunks, index, top_k=5):

    # CHANGE:
    # Rewrite query before embedding
    query = rewrite_query(query)

    # Encode query
    query_embedding = model.encode([query]).astype('float32')

    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)

    # Semantic search
    distances, indices = index.search(query_embedding, top_k)

    # Get retrieved chunks
    semantic_results = [clean_chunks[i] for i in indices[0]]

    # CHANGE:
    # Keyword reranking AFTER semantic search
    boosted_results = keyword_boost(query, semantic_results)

    # Return top reranked chunks
    return boosted_results[:3]


# =========================================================
# DETECT QUERY TYPE
# =========================================================
def detect_query_type(query):

    query = query.lower()

    summary_keywords = [
        "topic",
        "summary",
        "about",
        "overview",
        "main idea"
    ]

    for word in summary_keywords:

        if word in query:
            return "summary"

    return "retrieval"


# =========================================================
# SIMPLE DOCUMENT SUMMARIZATION
# =========================================================
def summarize_document(text):

    sentences = smart_chunk(text)

    # CHANGE:
    # Simple extractive summary
    return " ".join(sentences[:3])


# =========================================================
# GENERATE FINAL ANSWER
# =========================================================
def generate_answer(query, retrieved_chunks):

    context = "\n".join(retrieved_chunks)

    prompt = f"""
You are a strict AI assistant.

Answer ONLY using the provided context.

If answer is not found, say:
'I could not find the answer in the document.'

Context:
{context}

Question:
{query}

Answer:
"""

    completion = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    return completion.choices[0].message.content

if __name__ == "__main__":

    with open("sample.pdf", "rb") as f:

        text = extract_text_from_pdf(f)

    print(text[:500])