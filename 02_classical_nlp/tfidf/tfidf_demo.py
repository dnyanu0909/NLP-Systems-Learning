from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample documents
documents =  [
    "I love my country",
    "I love India",
    "I am an Indian",
    "I am a citizen of India",
    "I am a citizen of USA",
    "I am a citizen of China"
]

# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

#convert text -> vectors
tfidf_matrix = vectorizer.fit_transform(documents)

#input
query = input("Enter a sentence: ")

# Validate input
if not query.strip():
    print("Error: Please enter a valid sentence.")
else:
    # Convert the query to a TF-IDF vector
    query_vector = vectorizer.transform([query])

    #similarity
    similarity = cosine_similarity(query_vector, tfidf_matrix)

    # sort results
    results = sorted(zip(documents, similarity[0]),key = lambda x:x[1],reverse = True)

    # Print the results
    print("Top matches:")
    for i , (sent,score) in enumerate(results[:3]):
        print(f"{i+1}. {sent} -> {score:.4f}")

