import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
import pypdf

app = Flask(__name__)
# Enable CORS so your Vercel frontend can securely communicate with this server
CORS(app)

# Initialize the official Google GenAI client using the environment variable
# The client will automatically pick up GEMINI_API_KEY from environment variables
client = genai.Client()

# In-memory document storage for the session (can be upgraded to Supabase later)
SESSION_STATE = {
    "extracted_text": "",
    "file_name": None
}

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "LexIntellect AI Litigation Core Online", "active_file": SESSION_STATE["file_name"]})

@app.route('/api/analyze', methods=['POST'])
def analyze_case():
    if 'file' not in request.files:
        return jsonify({"error": "No legal document provided."}), 400
    
    file = request.files['file']
    try:
        pdf_reader = pypdf.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
            
        SESSION_STATE["extracted_text"] = text
        SESSION_STATE["file_name"] = file.filename
        
        prompt = f"""
        You are an elite legal AI strategist. Perform a deep structural analysis on this legal document and return a detailed markdown response containing:
        1. **Executive Case Summary**: Core claims, parties involved, and primary cause of action.
        2. **Evidentiary Risks & Vulnerabilities**: Weaknesses, factual gaps, or timeline discrepancies.
        3. **Winning Precedents & Legal Strategies**: Recommended motions or strategic counters.

        Document Text:
        {text[:7000]}
        """
        
        # Using gemini-3.6-flash for high efficiency, speed, and accuracy
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
        )
        
        return jsonify({
            "status": "success",
            "fileName": file.filename,
            "pageCount": len(pdf_reader.pages),
            "analysis": response.text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/risk-scanner', methods=['POST'])
def risk_scanner():
    if not SESSION_STATE["extracted_text"]:
        return jsonify({"error": "No active case dossier loaded. Please upload a PDF file first."}), 400
        
    prompt = f"""
    You are an expert contract risk auditor. Scan the following document text specifically for high-risk clauses, hidden liabilities, penalty traps, unreasonable indemnities, or compliance dangers. 
    Format your findings clearly with severity labels (CRITICAL, MODERATE, LOW) and recommend changes.

    Document Text:
    {SESSION_STATE["extracted_text"][:7000]}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
        )
        return jsonify({"status": "success", "risks": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def cross_examination_chat():
    data = request.json
    user_message = data.get('message', '')
    
    if not SESSION_STATE["extracted_text"]:
        return jsonify({"response": "Please upload a case file dossier in the Case Audit tab before initiating cross-examination chat."})
        
    chat_prompt = f"""
    You are an AI co-counsel assisting in cross-examination and document interrogation. Answer the user's specific query strictly using facts and timelines from the uploaded case document context.
    
    Case Context:
    {SESSION_STATE["extracted_text"][:6000]}

    Co-Counsel Query: {user_message}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=chat_prompt,
        )
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/draft', methods=['POST'])
def legal_draft():
    data = request.json
    doc_type = data.get('docType', 'Legal Notice')
    instructions = data.get('instructions', '')
    
    draft_prompt = f"""
    Draft a formal, court-ready, professional {doc_type} based on the following instructions and legal context. Include formal jurisdiction headers, factual recitals, and demand/prayer clauses.
    
    Instructions & Case Notes:
    {instructions}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=draft_prompt,
        )
        return jsonify({"status": "success", "draft": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
