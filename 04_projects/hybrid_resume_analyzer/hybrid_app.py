import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import re

model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------------
# SKILL LIST
# ------------------------
SKILLS = [
    "python", "machine learning", "deep learning","nlp", "data analysis", "sql", "pandas","tensorflow", "scikit-learn","keras", "r", "java", "c++", "javascript","numpy", "data visualization", "cloud computing", "docker", "kubernetes", "git", "linux","big data", "hadoop", "spark", "aws", "azure", "gcp","time series analysis", "reinforcement learning", "computer vision", "data engineering", "etl", "airflow","neural networks", "natural language processing", "data mining", "statistical analysis", "data wrangling", "data storytelling", "data-driven decision making","data science", "artificial intelligence", "deep learning frameworks", "ml algorithms", "data visualization tools", "cloud platforms", "big data technologies", "version control", "containerization", "orchestration", "linux command line"
]

# ------------------------
# FUNCTIONS
# ------------------------

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

def extract_skills(text):
    text = text.lower()
    found = []
    for skill in SKILLS:
        if skill in text:
            found.append(skill)
    return found

def compute_similarity(resume, job_desc):
    embeddings = model.encode([resume, job_desc])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return score

def missing_skills(resume, job_desc):
    resume_skills = set(extract_skills(resume))
    job_skills = set(extract_skills(job_desc))
    return list(job_skills - resume_skills)

def generate_suggestions(missing):
    suggestions = []
    for skill in missing:
        if skill == "deep learning":
            suggestions.append("Add Deep Learning projects (CNN, NLP models)")
        elif skill == "machine learning":
            suggestions.append("Include ML algorithms (Regression, SVM, etc.)")
        elif skill == "nlp":
            suggestions.append("Add NLP projects like chatbot or text classifier")
        elif skill == "python":
            suggestions.append("Highlight Python projects and libraries")
        elif skill == "scikit-learn":
            suggestions.append("Mention scikit-learn for ML implementations")
        elif skill == "data visualization":
            suggestions.append("Include data viz tools like Matplotlib, Seaborn, or Tableau")
        elif skill == "data analysis":
            suggestions.append("Showcase data analysis projects with Pandas or R")
        elif skill == "cloud computing":
            suggestions.append("Add experience with AWS, Azure, or GCP")
        elif skill == "docker":
            suggestions.append("Mention containerization experience with Docker")
        elif skill == "kubernetes":
            suggestions.append("Mention experience with Kubernetes orchestration")
        elif skill == "git":
            suggestions.append("Highlight version control experience with Git") 
        elif skill == "linux":
            suggestions.append("Mention experience with Linux command line and tools")
        elif skill == "big data":
            suggestions.append("Add experience with big data tools like Hadoop or Spark")
        elif skill == "sql":
            suggestions.append("Highlight SQL skills and database experience")
        elif skill == "pandas":
            suggestions.append("Showcase data manipulation projects using Pandas")  
        elif skill == "numpy":
            suggestions.append("Highlight numerical computing projects using NumPy")
        elif skill == "data engineering":
            suggestions.append("Add experience with data pipelines and ETL processes")
        elif skill == "airflow":
            suggestions.append("Mention experience with Airflow for workflow management")
        else:
            suggestions.append(f"Add experience in {skill}")
    return list(set(suggestions))

# ------------------------
# UI
# ------------------------
st.set_page_config(page_title="Resume Analyzer")

st.title("📄 AI Resume Analyzer")
st.caption("Upload your resume and compare it with a job description to get insights.")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

resume = ""

if uploaded_file is not None:
    resume = extract_text_from_pdf(uploaded_file)

job_desc = st.text_area("Paste Job Description here:")

if st.button("Analyze"):
    if resume and job_desc:

        clean_resume = clean_text(resume)
        clean_job = clean_text(job_desc)

        score = compute_similarity(clean_resume, clean_job)
        missing = missing_skills(resume, job_desc)
        suggestions = generate_suggestions(missing)

        st.subheader("📊 Match Score")
        st.write(round(score, 2))

        st.subheader("🧠 Detected Resume Skills")
        st.write(extract_skills(resume))

        st.subheader("❌ Missing Skills")
        st.write(missing if missing else "None")

        st.subheader("💡 Suggestions")

        if score > 0.7:
            st.success("Strong Match 🚀")
        elif score > 0.4:
            st.warning("Moderate Match ⚠️")
        else:
            st.error("Low Match ❌")

        if score < 0.4 and not missing:
            st.warning("Your resume may not match the job description well. Try aligning your experience more closely.")

        if suggestions:
            for s in suggestions:
                st.write("-", s)
        else:
            st.write("Your resume looks strong!")

    else:
        st.warning("Please enter both Resume and Job Description")



