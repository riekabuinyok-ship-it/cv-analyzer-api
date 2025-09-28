import os
import json
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from docx import Document
import fitz # PyMuPDF

# --- Configuration ---
# Your API Key is used securely here. 
# It is vital this code is NOT accessible to the public via WordPress.
API_KEY = "AIzaSyBookMCDmUdO_YHD6GJZUCPq-DYi25y0z4" 
os.environ['GEMINI_API_KEY'] = API_KEY # Set for the client
app = Flask(__name__)

# IMPORTANT: Set 'origins' to your WordPress domain (e.g., 'cvanalyzer.space')
# '*' is used here for maximum compatibility but is less secure.
CORS(app, resources={r"/analyze_cv": {"origins": "*"}}) 

client = genai.Client()
MODEL = 'gemini-2.5-flash'

# --- File Extraction Functions ---
def extract_text_from_pdf(file_stream):
    """Extracts text from a PDF file stream."""
    try:
        doc = fitz.open(stream=file_stream.read(), filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def extract_text_from_docx(file_stream):
    """Extracts text from a DOCX file stream."""
    try:
        document = Document(file_stream)
        return '\n'.join(p.text for p in document.paragraphs)
    except Exception as e:
        return f"Error reading DOCX: {e}"

# --- Main API Endpoint ---
@app.route('/analyze_cv', methods=['POST'])
def analyze_cv():
    if 'cv_file' not in request.files or 'job_description' not in request.form:
        return jsonify({"error": "Missing CV file or job description"}), 400

    cv_file = request.files['cv_file']
    job_description = request.form['job_description']
    file_extension = cv_file.filename.split('.')[-1].lower()
    
    # Read the file data into a stream for parsing
    file_stream = io.BytesIO(cv_file.read())

    # 1. CV Text Extraction
    if file_extension == 'pdf':
        cv_text = extract_text_from_pdf(file_stream)
    elif file_extension in ['doc', 'docx']:
        cv_text = extract_text_from_docx(file_stream)
    else:
        return jsonify({"error": "Unsupported file type. Please upload a PDF or DOCX."}), 415

    if cv_text.startswith("Error"):
        return jsonify({"error": f"Failed to parse CV: {cv_text}"}), 500

    # 2. Gemini Analysis Prompt (Tuned for JSON Output)
    prompt = f"""
    You are an expert HR CV analyst. Compare the Candidate CV with the Job Description (JD).
    
    **CANDIDATE CV TEXT:**
    ---
    {cv_text}
    ---

    **JOB DESCRIPTION (JD):**
    ---
    {job_description}
    ---

    Generate a JSON response with the following four keys ONLY:
    1.  **fit_score_percent**: (Integer 0-100) The numerical match percentage.
    2.  **summary**: (String) A 3-sentence summary of the CV's alignment.
    3.  **issues_to_update**: (List of Strings) 3-5 specific, actionable bullet points the user must update or add to their CV (e.g., "Add quantified results for your experience.").
    4.  **alternative_summary**: (String) A new, best-alternative professional summary (4-6 lines) written for the CV, optimized specifically for this JD.
    
    Output ONLY the valid JSON object.
    """
    
    try:
        # 3. Call the Gemini API with JSON response configuration
        response = client.models.generate_content(
            model=MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        # Load the JSON from the response text
        ai_result = json.loads(response.text)
        
        return jsonify(ai_result), 200

    except Exception as e:
        print(f"Gemini API Error: {e}")
        return jsonify({"error": f"AI analysis failed due to a server error. Details: {str(e)}"}), 500

if __name__ == '__main__':
    # Run the server. Use host='0.0.0.0' and port 5000 for deployment
    app.run(host='0.0.0.0', port=5000, debug=True)