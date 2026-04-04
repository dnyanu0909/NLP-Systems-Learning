from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    return score['compound']

def get_label(score):
    if score > 0.7:
        return "High"
    elif score > 0.4:
        return "Moderate"
    else:
        return "Low"

# Load the pre-trained model    
model = SentenceTransformer('all-MiniLM-L6-v2')

my_sentence = input("Enter a sentence: ")

sentences = [
    "I love my country",
    "I love India",
    "I am an Indian",
    "I am a citizen of India",
    "I am a citizen of USA",
    "I am a citizen of China"
]

def get_label(score):
    if score > 0.7:
        return "High"
    elif score > 0.4:
        return "Moderate"
    else:
        return "Low"

embeddings = model.encode(sentences)

# Compute cosine similarity between the second sentence and all sentences
similarity = cosine_similarity([model.encode(my_sentence)], embeddings) # Compare the user's sentence with all sentences

print("Top matches")
# for i,score in enumerate(similarity[0]):
#     print(f"{i+1}.{sentences[i]} -> {score:.4f}")
#     if i==2:
#         break

similarity_scores = similarity[0]
sorted_results = sorted(zip(sentences, similarity_scores), key=lambda x: x[1], reverse=True)
for i, score in enumerate(sorted_results):
    input_sentiment = get_sentiment(my_sentence)
    candidate_sentiment = get_sentiment(score[0])
    
    # if opposite sentiment → penalize score
    if input_sentiment * candidate_sentiment < 0:
        adjusted_score = score[1] * 0.5  
    else:
        adjusted_score = score[1]

    label = get_label(adjusted_score)
    print(f"{i+1}. {score[0]} -> {adjusted_score:.4f} ({label})")