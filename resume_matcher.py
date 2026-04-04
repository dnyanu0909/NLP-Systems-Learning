from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

SKILLS = [
    "python",
    "machine learning",
    "deep learning",
    "nlp",
    "data analysis",
    "sql",
    "pandas",
    "tensorflow",
    "keras",
    "scikit-learn",
    "r",
    "java",
    "c++",
    "javascript"
]

def extract_skills(text):
    text = text.lower()
    found = []

    for skill in SKILLS:
        if skill in text:
            found.append(skill)

    return found

def compute_similarity(resume, job_desc):
    vectorizer = TfidfVectorizer(stop_words='english')

    vectors = vectorizer.fit_transform([resume, job_desc])

    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    return similarity

def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

def missing_skills(resume, job_desc):
    resume_skills = set(extract_skills(resume))
    job_skills = set(extract_skills(job_desc))

    missing = job_skills - resume_skills

    return list(missing)

def generate_suggestions(missing_skills):
    suggestions = []

    for skill in missing_skills:
        if skill == "deep learning":
            suggestions.append("Add Deep Learning projects (e.g., CNN, NLP models)")
            suggestions.append("Mention frameworks like TensorFlow or PyTorch")

        elif skill == "machine learning":
            suggestions.append("Include ML algorithms (Regression, SVM, etc.)")
            suggestions.append("Show real-world ML projects")

        elif skill == "nlp":
            suggestions.append("Add NLP projects like chatbot or text classifier")

        elif skill == "python":
            suggestions.append("Highlight Python projects and libraries (NumPy, Pandas)")

        elif skill == "data analysis":
            suggestions.append("Include data analysis projects with tools like Pandas or SQL")
        elif skill == "sql":
            suggestions.append("Add SQL projects or certifications")
        elif skill == "pandas":
            suggestions.append("Show projects using Pandas for data manipulation")
        elif skill == "tensorflow":
            suggestions.append("Include TensorFlow projects (e.g., image classification)")
        elif skill == "keras":
            suggestions.append("Mention Keras for deep learning projects")
        elif skill == "scikit-learn":
            suggestions.append("Highlight scikit-learn for ML algorithms")
        elif skill == "r":
            suggestions.append("Add R projects or certifications")
        elif skill == "java":
            suggestions.append("Include Java projects or certifications")
        elif skill == "c++":
            suggestions.append("Add C++ projects or certifications")
        elif skill == "javascript":
            suggestions.append("Include JavaScript projects or certifications")
        else:
            suggestions.append(f"Consider adding experience in {skill}")

    return list(set(suggestions))

# test
resume = """
Python, Machine Learning, NLP, Data Analysis
"""

job_description = """
Responsible for using Python, Machine Learning, and NLP for Data Analysis."""

resume = clean_text(resume)
job_description = clean_text(job_description)

score = compute_similarity(resume, job_description)
missing = missing_skills(resume, job_description)

print("Match Score:", round(score, 2))
print("Missing Keywords:", missing)
suggestions = generate_suggestions(missing)

print("\nSuggestions:")
for s in suggestions:
    print("-", s)