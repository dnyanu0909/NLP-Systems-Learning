from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#load embedding model 
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the sentiment analyzer for better understanding of user queries
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    return analyzer.polarity_scores(text)['compound']

documents =  [
    "I love my country",
    "I love India",
    "I am an Indian",
    "I am a citizen of India",
    "I am a citizen of USA",
    "I am a citizen of China"
]
# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english') #stop_words = 'english' -> remove common words

#convert text -> vectors
tfidf_matrix = vectorizer.fit_transform(documents)

#embeddings
embeddings = model.encode(documents)

#input
query = input("Enter a sentence: ")

#compute scores 
query_embedding = model.encode([query])
semantic_scores = cosine_similarity(query_embedding, embeddings)[0]

query_tfidf = vectorizer.transform([query])
tfidf_scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]

#combine scores
final_scores = []

for i in range(len(documents)):
    input_sentiment = get_sentiment(query)
    candidate_sentiment = get_sentiment(documents[i])
    
    combined_score = (0.7 * semantic_scores[i]) + (0.3 * tfidf_scores[i])

    # sentiment penalty
    if input_sentiment * candidate_sentiment < 0:
        combined_score *= 0.5
    final_scores.append((documents[i], combined_score))

#sort results
results = sorted(final_scores ,key=lambda x : x[1],reverse = True)

# Print the results
print("Top matches:")
for i , (sent,score) in enumerate(results[:3]):
    print(f"{i+1}. {sent} -> {score:.4f}")