# BRD Chatbot

A powerful Business Requirements Document (BRD) analysis and processing tool built with Python and Flask. This application helps users analyze, understand, and work with BRD documents through an intelligent chat interface.

## Features

- 📄 **Document Processing**
  - PDF document analysis and text extraction
  - Word document (.docx) support
  - High-fidelity PDF ↔ Word conversions
  - Smart content appending to existing documents

- 🤖 **Intelligent Analysis**
  - BRD content understanding and analysis
  - Requirements extraction
  - Suggestions for improvements
  - Long-term memory for context retention
  - MLflow integration for tracking and monitoring

- 📊 **Visualization**
  - Automatic diagram generation
  - Flowchart creation
  - ASCII art support
  - Architecture visualization

- 💾 **Multiple Output Formats**
  - PDF generation with formatting
  - Word document export
  - Markdown output
  - JSON responses
  - Image generation for diagrams

## Prerequisites

- Python 3.8+
- Ollama (for LLM support)
- Optional: LibreOffice (for enhanced PDF ↔ Word conversion)

## Required Python Packages

```bash
flask
flask-cors
ollama
PyPDF2
python-doctr
python-dotenv
torch
Pillow
graphviz
python-markdown
reportlab
python-docx
mlflow
pdf2docx  # Optional
docx2pdf  # Optional
langmem   # Optional
```

## Environment Variables

```env
OLLAMA_MODEL=gemma3:12b  # Default Ollama model
LANGMEM_ENABLED=false    # Enable/disable long-term memory
LANGMEM_BACKEND=ollama   # Memory backend (ollama/langmem)
MLFLOW_TRACKING_URI=     # Optional: MLflow tracking server URI
MLFLOW_EXPERIMENT_NAME=BRDChatbot  # MLflow experiment name
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd brd-chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Optional: Install system dependencies for enhanced features:
   ```bash
   # For Ubuntu/Debian
   sudo apt-get install libreoffice graphviz
   ```

4. Create and configure your `.env` file based on the example above.

## Usage

1. Start the server:
   ```bash
   python brd_chatbot.py
   ```

2. The server will run on `http://localhost:9091`

3. API Endpoints:

   - `POST /chat`: Main chat endpoint
     - Supports multipart/form-data for file uploads
     - Accepts JSON for text-only queries
     - Parameters:
       - `content`: User query text
       - `pdf`: PDF file upload
       - `docx`: Word document upload
       - `output_format`: Desired output format (json/pdf/docx/image)
       - `convert_pdf_to_word`: Boolean flag for PDF to Word conversion
       - `convert_word_to_pdf`: Boolean flag for Word to PDF conversion

   - `GET /health`: Health check endpoint
   - `GET /debug/memories`: Debug endpoint for memory inspection

## Features in Detail

### Document Processing
- Extracts text from PDFs using both native extraction and OCR
- Supports Word document processing
- Maintains document formatting during conversions
- Smart content appending with format preservation

### Intelligent Analysis
- Processes and understands BRD content
- Extracts key requirements and information
- Maintains conversation context
- Stores important information in long-term memory
- Tracks operations and artifacts with MLflow

### Output Generation
- Generates professionally formatted PDFs
- Creates Word documents with styling
- Produces diagrams and flowcharts
- Supports markdown output
- Handles JSON responses for API integration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Acknowledgments

- Built with Flask and Python
- Uses Ollama for LLM capabilities
- Integrates with MLflow for experiment tracking
- Leverages various open-source libraries for document processing 