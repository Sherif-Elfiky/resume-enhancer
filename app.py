import os
from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

@app.route("/", methods=["GET", "POST"])
def index():
    sim_score = None
    resume_text = ""

    if request.method == "POST":
        job_desc = request.form.get("job_desc", "")

        if "resume" in request.files:
            pdf_file = request.files["resume"]
            if pdf_file.filename.endswith(".pdf"):
                resume_text = extract_text_from_pdf(pdf_file)
                embeddings = model.encode([resume_text, job_desc], convert_to_tensor=True)
                sim_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

    return render_template(
        "index.html",
        sim_score=round(sim_score * 100, 2) if sim_score else None,
        resume_text=resume_text,
    )

if __name__ == "__main__":
    app.run(debug=True)
