import re

def extract_skills(text):
    skills = ["python", "machine learning", "data analysis", "deep learning", "nlp","cloud computing", "sql", "java", "c++", "javascript", "react", "angular", "docker", "kubernetes","node.js", "aws", "azure", "gcp", "git", "linux", "tensorflow", "pytorch", "scikit-learn","pandas", "numpy", "matplotlib", "seaborn", "tableau", "power bi", "hadoop", "spark", "scala", "r", "sas", "excel","communication", "teamwork", "problem-solving", "leadership", "project management", "time management", "adaptability", "critical thinking", "creativity", "collaboration", "emotional intelligence","conflict resolution", "decision making", "negotiation", "presentation skills", "public speaking", "writing skills", "analytical skills", "research skills", "customer service", "sales skills", "marketing skills", "financial analysis", "budgeting", "strategic planning", "business development", "product management", "ux/ui design", "mobile app development", "web development","cybersecurity", "networking", "devops", "agile methodologies","scrum","kanban","lean","six sigma"]

    found_skills = []

    text = text.lower()

    for skill in skills:
        if skill in text:
            found_skills.append(skill)

    return found_skills


# test
resume = """
Results-oriented professional with over five years of experience in operational management, team leadership, and strategic planning. Adept at leveraging data analytics tools, including SQL and Tableau, to identify process improvements that enhance efficiency and reduce costs. Proven ability to foster collaborative, cross-functional team environments, strengthening communication between technical and non-technical stakeholders. Strong background in project management, consistently meeting tight deadlines and implementing innovative solutions to complex problems.
"""

skills = extract_skills(resume)

print("Detected Skills:", skills)