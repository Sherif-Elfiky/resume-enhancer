import os
from flask import Flask, request, render_template
import PyPDF2
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'


model = SentenceTransformer('all-MiniLM-L6-v2')  

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        job_desc = request.form['job_desc']
        pdf_file = request.files['resume']

        if pdf_file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
            pdf_file.save(file_path)

            resume_text = extract_text_from_pdf(file_path)

            
            embeddings = model.encode([resume_text, job_desc], convert_to_tensor=True)
            sim_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

           
            suggestions = []
            if sim_score < 0.5:
                suggestions.append("Add more job-related keywords.")
            if "experience" not in resume_text.lower():
                suggestions.append("Include an 'Experience' section.")
            if len(resume_text.split()) < 200:
                suggestions.append("Expand with more details and achievements.")

            return render_template("index.html",
                                   score=round(sim_score*100, 2),
                                   suggestions=suggestions,
                                   resume_text=resume_text)

    return render_template("index.html", score=None, suggestions=None, resume_text=None)

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
