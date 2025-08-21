# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from med_chatbot import process_chat, request_count, error_count, guardrail_block_count
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

def validate_request():
    """Validate the incoming request and identify input types"""
    input_types = {
        'has_prompt': False,
        'has_pdf': False,
        'has_excel': False,
        'has_image': False,
        'has_video': False,
        'has_rag': False
    }
    
    # Check for prompt
    if request.form.get('content'):
        input_types['has_prompt'] = True
    
    # Check for files
    files = request.files
    if 'pdf' in files:
        input_types['has_pdf'] = True
    if 'excel' in files:
        input_types['has_excel'] = True
    if 'image' in files:
        input_types['has_image'] = True
    if 'video' in files:
        input_types['has_video'] = True
    if 'rag' in files:
        input_types['has_rag'] = True
    
    return input_types

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint with metrics"""
    try:
        return jsonify({
            "status": "healthy",
            "service": "Medical Assistant API",
            "metrics": {
                "total_requests": request_count,
                "total_errors": error_count,
                "guardrail_blocks": guardrail_block_count,
                "success_rate": f"{((request_count - error_count) / max(request_count, 1) * 100):.1f}%"
            }
        })
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return jsonify({
            "status": "degraded",
            "error": str(e)
        }), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Dedicated metrics endpoint for monitoring"""
    try:
        return jsonify({
            "metrics": {
                "total_requests": request_count,
                "total_errors": error_count,
                "guardrail_blocks": guardrail_block_count,
                "success_rate": f"{((request_count - error_count) / max(request_count, 1) * 100):.1f}%",
                "error_rate": f"{(error_count / max(request_count, 1) * 100):.1f}%",
                "guardrail_block_rate": f"{(guardrail_block_count / max(request_count, 1) * 100):.1f}%"
            }
        })
    except Exception as e:
        logger.error(f"Error fetching metrics: {str(e)}")
        return jsonify({
            "error": "Error fetching metrics",
            "details": str(e)
        }), 500

@app.route('/chat', methods=['POST'])
def medical_assistant():
    """
    Medical Assistant API endpoint
    Accepts:
    - Text queries about medical conditions, symptoms, and health (required)
    - PDF documents (optional: medical reports, lab results)
    - Excel sheets (optional: medical data, test results)
    - Images (optional: X-rays, scans, medical documents)
    - Videos (optional: medical procedures, physical symptoms)
    - RAG documents (optional: for detailed medical document Q&A)
    
    Request format:
    - Form data:
        - content: String (required) - The medical query or message
        - model: String (optional) - The model to use (gemma3:4b, gemma3:12b, llama3.2-vision:latest, mistral:latest)
    - Files (optional):
        - pdf: PDF file (for medical document analysis)
        - excel: Excel file (medical data)
        - image: Image file (medical images)
        - video: Video file (MP4 format recommended)
        - rag: PDF file (for RAG-based medical question answering)
        
    File Processing Types:
    - pdf: Standard medical document analysis and extraction
    - rag: Retrieval-Augmented Generation for detailed medical document question answering
      * Best for: Specific questions about medical reports and documents
      * Example usage: Upload medical report with key 'rag' and ask "What are the test results?"
        
    Video Guidelines:
    - Supported format: MP4
    - Maximum duration: 5 minutes
    - Will extract key frames for medical analysis
    
    RAG Guidelines:
    - Upload a medical document using the 'rag' key
    - Ask specific questions about the document content
    - RAG will chunk the document and find relevant context to answer your questions
    - Examples: "What are the key findings?", "What are the recommended treatments?", "What are the test results?"
    
    Note: This assistant provides general medical information and should not replace professional medical advice.
    Always consult with healthcare professionals for specific medical concerns.
    """
    try:
        # Validate and identify input types
        input_types = validate_request()
        
        # Must have at least a prompt
        if not input_types['has_prompt']:
            logger.warning("Request received without prompt")
            return jsonify({
                "error": "No prompt provided. Please provide a medical query.",
                "message": "Your query should include text describing your medical question or what you want to analyze."
            }), 400

        # Log the type of request being processed
        logger.info(f"Processing medical request with inputs: {input_types}")
        
        # Add input types to request for processing
        request.input_types = input_types
        
        # Process the request
        response = process_chat(request)
        
        return response

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            "error": "An error occurred while processing your request",
            "details": str(e),
            "message": "Please ensure your medical query is clear and files (if any) are in the correct format."
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Flask server with OpenLIT tracing and metrics...")
    print("📊 Metrics will be displayed in the terminal for each request")
    print("🔍 Tracing information will show request flow and guardrail results")
    logger.info("Starting Medical Assistant API...")
    app.run(host='0.0.0.0', port=9090, debug=True)