import os
from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

api_key = os.getenv("GEMINI_API_KEY")

gemini_model = genai.GenerativeModel("gemini-1.5-flash")

genai.configure(api_key=api_key)


app = Flask(__name__)

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

def get_gemini_score(resume_text: str, job_desc: str) -> float:
    prompt = f"""
    Compare the following resume to the job description and return ONLY a similarity score (0-100).

    Resume:
    {resume_text}

    Job Description:
    {job_desc}
    """
    response = gemini_model.generate_content(prompt)
    try:
        score = float("".join([c for c in response.text if c.isdigit() or c == "."]))
        return min(max(score, 0), 100)  # clamp between 0-100
    except:
        return None
    
def get_gemini_suggestions(resume_text: str, job_desc: str) -> str:
    """
    Ask Gemini to provide a text description with suggested improvements
    for the resume according to the job description.
    """
    prompt = f"""
    Read the resume and the job description below.
    Suggest specific improvements to the resume to better match the job description.
    Only provide actionable feedback in plain text.

    Resume:
    {resume_text}

    Job Description:
    {job_desc}
    """
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

@app.route("/", methods=["GET", "POST"])
def index():
    sim_score = None
    sim_score_gemini = None
    resume_text = ""
    resume_suggestions = ""

    if request.method == "POST":
        job_desc = request.form.get("job_desc", "")

        if "resume" in request.files:
            pdf_file = request.files["resume"]
            if pdf_file.filename.endswith(".pdf"):
                resume_text = extract_text_from_pdf(pdf_file)
                embeddings = model.encode([resume_text, job_desc], convert_to_tensor=True)
                sim_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
                sim_score_gemini = get_gemini_score(resume_text, job_desc)
                resume_suggestions = get_gemini_suggestions(resume_text, job_desc)

    return render_template(
        "index.html",
        sim_score=round(sim_score * 100, 2) if sim_score else None,
        sim_score_gemini=sim_score_gemini,
        resume_text=resume_text,
        resume_suggestions=resume_suggestions
    )

if __name__ == "__main__":
    app.run(debug=True)
