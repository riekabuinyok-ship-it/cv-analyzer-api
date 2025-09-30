import os
import json
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
# We switch to the OpenAI library for Deepseek API access
from openai import OpenAI
from docx import Document
import fitz # PyMuPDF

# --- Configuration ---
# IMPORTANT: This key must be set as a secure environment variable on Render!
# The script will attempt to read the key from the environment variable named 'DEEPSEEK_API_KEY'.
# Deepseek API keys start with 'sk-'. The old Gemini key won't work here.
# For local testing, you can uncomment this line and replace the placeholder:
# os.environ['DEEPSEEK_API_KEY'] = 'sk-2c475c30e490435f8bf9516fd17a040d' 

# The key will be securely read from the environment
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY environment variable not set.")

app = Flask(__name__)

# IMPORTANT: Replace '*' with your WordPress domain for better security (e.g., 'https://mycvanalyzer.com')
CORS(app, resources={r"/analyze_cv": {"origins": "*", "methods": ["POST", "OPTIONS"]}}) 

# Initialize the OpenAI client pointing to the Deepseek API base URL
try:
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com/v1" 
    )
    # Deepseek's recommended model for general tasks
    MODEL = 'deepseek-coder' 
except Exception as e:
    print(f"Error initializing Deepseek client: {e}")
    # Handle the error appropriately in a production environment

# --- Utility Functions for CV Parsing (No Change Needed) ---
def extract_text_from_pdf(file_stream):
    """Extracts text from a PDF file stream using PyMuPDF."""
    # (Same code as before: use fitz/PyMuPDF to extract text)
    text = ""
    try:
        doc = fitz.open(stream=file_stream.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def extract_text_from_docx(file_stream):
    """Extracts text from a DOCX file stream."""
    # (Same code as before: use python-docx to extract text)
    try:
        document = Document(file_stream)
        return '\n'.join([p.text for p in document.paragraphs])
    except Exception as e:
        return f"Error reading DOCX: {e}"

# --- Main API Endpoint ---
@app.route('/analyze_cv', methods=['POST'])
def analyze_cv():
    if 'cv_file' not in request.files or 'job_description' not in request.form:
        return jsonify({"error": "Missing CV file or job description"}), 400

    cv_file = request.files['cv_file']
    job_description = request.form['job_description']

    # 1. CV Text Extraction
    file_extension = cv_file.filename.split('.')[-1].lower()
    file_stream = io.BytesIO(cv_file.read())

    if file_extension == 'pdf':
        cv_text = extract_text_from_pdf(file_stream)
    elif file_extension in ['doc', 'docx']:
        cv_text = extract_text_from_docx(file_stream)
    else:
        return jsonify({"error": "Unsupported file type. Please upload a PDF or DOCX."}), 415

    if cv_text.startswith("Error"):
        return jsonify({"error": f"Failed to parse CV: {cv_text}"}), 500

    # 2. Deepseek Analysis Prompt (asking for JSON output)
    prompt = f"""
    You are an expert CV analyst. Your task is to compare a Candidate's CV with a specific Job Description (JD).

    **CANDIDATE CV TEXT:**
    ---
    {cv_text}
    ---

    **JOB DESCRIPTION (JD):**
    ---
    {job_description}
    ---

    Based on the comparison, generate a JSON response with the following keys:
    1.  **fit_score_percent**: (Integer 0-100) A numerical match percentage.
    2.  **summary**: (String) A 3-sentence summary of the CV's alignment with the JD.
    3.  **issues_to_update**: (List of Strings) A list of 3-5 specific, actionable bullet points the user must update or add to their CV to better match the JD (e.g., "Quantify your experience with Python by adding project results.").
    4.  **alternative_summary**: (String) A new, best-alternative professional summary (4-6 lines) written for the CV, optimized specifically for this JD.
    
    Ensure the output is ONLY a valid JSON object.
    """
    
    try:
        # 3. Call the Deepseek API
        completion = client.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"}, # Requesting JSON output format
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in HR and CV analysis. Your output must be a valid JSON object."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Get the JSON text from the response and parse it
        response_text = completion.choices[0].message.content
        ai_result = json.loads(response_text)
        
        return jsonify(ai_result), 200

    except Exception as e:
        print(f"Deepseek API Error: {e}")
        return jsonify({"error": f"AI analysis failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)