import ollama
import base64
import os
import re
from flask import Flask, request, jsonify
import pandas as pd
import PyPDF2
import tempfile
import cv2
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import shutil
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import openlit
from dotenv import load_dotenv
import time
import logging
import subprocess
import gc
import uuid
import collections
import torch
from werkzeug.utils import secure_filename
from typing import Union, List, Dict, Tuple
import asyncio
import concurrent.futures

# RAG specific imports
from sentence_transformers import SentenceTransformer
import faiss
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

# Add Tavily import and initialization
import os
from tavily import TavilyClient
from datetime import datetime, timedelta

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
logger.info("Loading environment variables...")
load_dotenv()

# Initialize Tavily client
tavily_api_key = os.getenv("TAVILY_API_KEY")
try:
    tavily_client = TavilyClient(api_key=tavily_api_key)
    logger.info("✓ Tavily client initialized successfully")
except Exception as e:
    logger.error(f"✗ Failed to initialize Tavily client: {str(e)}")
    tavily_client = None

# Initialize Flask app
app = Flask(__name__)

# Initialize global variables
global chat_history
chat_history = []

# --- Global metrics counters ---
request_count = 0
error_count = 0
guardrail_block_count = 0

# Update valid models list
valid_models = ['gemma3:4b', 'gemma3:12b', 'llama3.2-vision:latest', 'mistral:latest', 'simplyfillm']

# --- OpenLIT LLM-based Guardrail ---
custom_rules = [
    {"pattern": r"(?i)suicide|self-harm|kill", "classification": "harmful_content"},
    {"pattern": r"(?i)illegal drugs|cocaine|heroin", "classification": "illegal_substances"},
    {"pattern": r"(?i)prescribe|prescription", "classification": "prescription_request"},
]
sensitive_guard = openlit.guard.All(custom_rules=custom_rules)

TEMP_DIR = "temp_files"
PDF_UPLOAD_FOLDER = 'pdf_uploads'
TEMP_IMAGES_FOLDER = 'temp_images'
CLASSIFIED_IMAGES_FOLDER = 'classified_images'

# Create necessary directories
for folder in [TEMP_DIR, PDF_UPLOAD_FOLDER, TEMP_IMAGES_FOLDER, CLASSIFIED_IMAGES_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Initialize doctr model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
ocr_model = ocr_predictor(pretrained=True).to(device)
# --- RAG Functions ---
def extract_text_from_pdf_rag(pdf_path):
    """Extract text from PDF using doctr for RAG processing"""
    try:
        model = ocr_predictor(pretrained=True)
        doc = DocumentFile.from_pdf(pdf_path)
        result = model(doc)
       
        # Combine all pages into one text string
        text = ""
        for page in result.pages:
            text += page.render() + "\n"
        return text.replace("\n", " ")
    except Exception as e:
        logger.error(f"Error extracting text from PDF for RAG: {str(e)}")
        return ""

def chunk_text(text, chunk_size=300):
    """Chunking (simple word chunking)"""
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def build_vector_store(chunks):
    """Embedding and FAISS indexing"""
    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedder.encode(chunks)
        dim = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))
        return index, chunks, embedder
    except Exception as e:
        logger.error(f"Error building vector store: {str(e)}")
        return None, None, None

def query_with_rag(query, index, chunks, embedder, selected_model='gemma3:12b'):
    """RAG querying using Ollama or OpenAI"""
    try:
        query_embedding = embedder.encode([query])
        D, I = index.search(np.array(query_embedding), k=3)
        context = "\n".join([chunks[i] for i in I[0]])

        prompt = PromptTemplate.from_template(
            "Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        )

        # Validate model selection
        valid_models = ['gemma3:4b', 'gemma3:12b', 'llama3.2-vision:latest', 'mistral:latest', 'gpt-4.1-mini']
        if selected_model not in valid_models:
            selected_model = 'gemma3:12b'

        if selected_model == 'gpt-4.1-mini':
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
            return ask_openai(messages, model=selected_model)
        else:
            llm = Ollama(model=selected_model, temperature=0.2)
            chain = prompt | llm
            return chain.invoke({"context": context, "question": query})
    except Exception as e:
        logger.error(f"Error in RAG query: {str(e)}")
        return "Error processing RAG query"

def process_rag(pdf_file, query, selected_model='gemma3:12b'):
    """Process PDF for RAG and answer query"""
    try:
        # Save PDF to temporary file
        temp_pdf_path = os.path.join(TEMP_DIR, f"rag_temp_{uuid.uuid4()}.pdf")
        pdf_file.save(temp_pdf_path)
        
        logger.info("Processing PDF for RAG...")
        
        # Extract text from PDF
        extracted_text = extract_text_from_pdf_rag(temp_pdf_path)
        
        if not extracted_text or extracted_text.strip() == "":
            return "No text could be extracted from the PDF for RAG processing."
        
        logger.info(f"Extracted text length: {len(extracted_text)} characters")
        
        # Create chunks and build vector store
        chunks = chunk_text(extracted_text)
        index, chunks, embedder = build_vector_store(chunks)
        
        if index is None:
            return "Error building vector store for RAG processing."
        
        logger.info(f"Created {len(chunks)} chunks for RAG")
        
        # Validate model selection
        valid_models = ['gemma3:4b', 'gemma3:12b', 'llama3.2-vision:latest', 'mistral:latest', 'gpt-4.1-mini']
        if selected_model not in valid_models:
            selected_model = 'gemma3:12b'
        
        # Process the query
        answer = query_with_rag(query, index, chunks, embedder, selected_model)
        
        # Cleanup
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        
        return f"RAG Analysis Result:\n\n{answer}"
        
    except Exception as e:
        logger.error(f"Error in RAG processing: {str(e)}")
        return f"Error in RAG processing: {str(e)}"
    finally:
        # Cleanup temp file if it exists
        if 'temp_pdf_path' in locals() and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)





# --- Document Classification Functions ---

def get_classifier_prompt() -> str:
    """Return the classifier system prompt."""
    return (
        "You are an expert medical document classifier specializing in healthcare documentation. "
        "Your task is to analyze documents and classify them into categories based on their key characteristics:\n\n"
        "1. Medical Report\n"
        "- Contains patient information\n"
        "- Lists medical findings and diagnoses\n"
        "- Shows test results and interpretations\n"
        "- Includes doctor's recommendations\n\n"
        "2. Lab Results\n"
        "- Shows test parameters\n"
        "- Contains reference ranges\n"
        "- Lists measured values\n"
        "- Includes testing facility information\n\n"
        "3. Radiology Report\n"
        "- Contains imaging details (X-ray, MRI, CT)\n"
        "- Describes anatomical findings\n"
        "- Includes radiologist's interpretation\n"
        "- Shows imaging parameters\n\n"
        "4. Prescription\n"
        "- Lists medications and dosages\n"
        "- Contains prescribing doctor's information\n"
        "- Shows patient details\n"
        "- Includes usage instructions\n\n"
        "5. Medical Certificate\n"
        "- Contains official medical declarations\n"
        "- Shows validity period\n"
        "- Includes doctor's certification\n"
        "- Lists medical conditions or fitness status\n\n"
        
        "Instructions:\n"
        "- Carefully analyze the provided document\n"
        "- Identify key characteristics that match the categories above\n"
        "- Classify as exactly one of: 'Medical Report', 'Lab Results', 'Radiology Report', 'Prescription', 'Medical Certificate'\n"
        "- Respond ONLY with the classification - no explanation or additional text\n"
        "- If the document is not one of the above, return 'Other Medical Document'"
    )

def convert_pdf_to_images_for_classification(pdf_path, output_folder=TEMP_IMAGES_FOLDER):
    """
    Convert all pages of a PDF to PNG images and store them in a subfolder.
    
    Args:
        pdf_path (str): Path to the PDF file.
        output_folder (str): Folder to save images in.
        
    Returns:
        List of saved image file paths and pdf name.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")
    
    # Create a folder name based on PDF file name
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_subfolder = os.path.join(output_folder, pdf_name)
    os.makedirs(output_subfolder, exist_ok=True)

    logger.info(f"Converting PDF to images: {pdf_path}")
    images = convert_from_path(pdf_path, dpi=300)
    image_paths = []

    for i, img in enumerate(images):
        image_path = os.path.join(output_subfolder, f"page_{i + 1}.png")
        img.save(image_path, "PNG")
        image_paths.append(image_path)
        logger.info(f"Saved: {image_path}")

    return image_paths, pdf_name

def extract_text_with_doctr_classification(image_path):
    """
    Extract text from image using doctr OCR for classification.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        str: Extracted text from the image.
    """
    try:
        logger.info(f"Extracting text from image with doctr: {image_path}")
        # Load document
        doc = DocumentFile.from_images(image_path)
        # Run inference
        result = ocr_model(doc)
        # Extract text
        extracted_text = result.render()
        return extracted_text
    except Exception as e:
        logger.error(f"Error extracting text with doctr: {str(e)}")
        return ""

def clear_ocr_resources():
    """Clear OCR resources and force garbage collection."""
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    logger.info("OCR resources cleared")

def restart_ollama():
    """Restart Ollama service if it crashes"""
    logger.warning("Restarting Ollama service...")
    subprocess.run(["pkill", "-f", "ollama"], capture_output=True)
    subprocess.run(["ollama", "serve", "--restart"], capture_output=True)

def classify_document_image(image_path: Union[str, List[str]], batch_mode: bool = False) -> Union[str, List[str]]:
    """
    Extract text from image(s) and classify the document(s).
    
    Args:
        image_path: Path to single image or list of image paths
        batch_mode: Whether to process multiple images
    
    Returns:
        Classification result(s) as string or list of strings
    """
    try:
        # Handle single or multiple images
        image_paths = [image_path] if not batch_mode else image_path
        results = []

        for img_path in image_paths:
            try:
                # Check if the file exists
                if not os.path.exists(img_path):
                    logger.error(f"File not found: {img_path}")
                    results.append("File Not Found")
                    continue

                logger.info(f"Processing image: {img_path}")

                # Perform OCR with doctr
                extracted_text = extract_text_with_doctr_classification(img_path)
                
                # Check if text was extracted
                if not extracted_text or extracted_text.strip() == "":
                    logger.warning(f"No text detected in image: {img_path}")
                    results.append("No Text Detected")
                    continue

                logger.info(f"Extracted text: {extracted_text[:200]}...")  # Log first 200 chars

                # Set of valid classifications to prevent hallucination
                valid_classifications = {
                    'Medical Report', 
                    'Lab Results', 
                    'Radiology Report', 
                    'Prescription',
                    'Medical Certificate',
                    'Other Medical Document'
                }

                # Classify the document using selected model
                try:
                    selected_model = request.form.get('model', 'gemma3:12b')
                    valid_models = ['gemma3:4b', 'gemma3:12b', 'llama3.2-vision:latest', 'mistral:latest', 'gpt-4.1-mini']
                    if selected_model not in valid_models:
                        selected_model = 'gemma3:12b'
                    
                    # Create a deterministic prompt
                    messages = [
                        {"role": "system", "content": get_classifier_prompt()},
                        {"role": "user", "content": extracted_text}
                    ]
                    
                    if selected_model == 'gpt-4.1-mini':
                        raw_classification = ask_openai(messages, model=selected_model)
                    else:
                        response = ollama.chat(
                            model=selected_model,
                            messages=messages
                        )
                        raw_classification = response["message"]["content"].strip()
                    
                    # Normalize the classification to prevent hallucination
                    if raw_classification in valid_classifications:
                        classification = raw_classification
                    else:
                        # If not an exact match, find closest match or default to 'Others'
                        for valid_class in valid_classifications:
                            if valid_class.lower() in raw_classification.lower():
                                classification = valid_class
                                break
                        else:
                            classification = "Other Medical Document"
                    
                    logger.info(f"Final classification result: {classification}")

                except Exception as e:
                    logger.error(f"Model error: {str(e)}. Restarting service...")
                    restart_ollama()
                    classification = "Classification Failed"

                results.append(classification)

            except Exception as e:
                logger.error(f"Error processing image {img_path}: {str(e)}")
                results.append("Classification Failed")
            
            # Clear OCR resources after each image processing
            clear_ocr_resources()

        return results[0] if not batch_mode else results
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        return "Classification Failed"
    finally:
        # Force garbage collection again
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

def organize_images_by_classification(images_paths, classifications, pdf_name):
    """
    Organize images into folders based on their classification.
    
    Args:
        images_paths: List of image paths
        classifications: List of classifications
        pdf_name: Name of the original PDF file
        
    Returns:
        Dictionary with classification counts and paths
    """
    # Create a dictionary to track classification counts
    classification_counts = collections.defaultdict(int)
    classification_paths = collections.defaultdict(list)
    
    logger.info(f"Organizing {len(images_paths)} images into classification folders")
    
    # Group images by classification
    for i, (img_path, classification) in enumerate(zip(images_paths, classifications)):
        if classification == "Classification Failed" or classification == "No Text Detected":
            classification = "Other Medical Document"
            
        # Create subfolder for this classification if it doesn't exist
        target_folder = os.path.join(CLASSIFIED_IMAGES_FOLDER, classification, pdf_name)
        os.makedirs(target_folder, exist_ok=True)
        
        # Copy the file to the appropriate classification folder
        img_filename = os.path.basename(img_path)
        target_path = os.path.join(target_folder, img_filename)
        
        try:
            # Ensure source file exists
            if not os.path.exists(img_path):
                logger.error(f"Source image does not exist: {img_path}")
                continue
                
            # Ensure target directory exists and is writable
            if not os.path.exists(target_folder):
                logger.error(f"Failed to create target folder: {target_folder}")
                continue
                
            # Copy with explicit error handling
            shutil.copy2(img_path, target_path)
            
            if os.path.exists(target_path):
                logger.info(f"Successfully copied {img_path} to {target_path}")
                # Update counts and paths
                classification_counts[classification] += 1
                classification_paths[classification].append(target_path)
            else:
                logger.error(f"Failed to copy {img_path} to {target_path}")
        except Exception as e:
            logger.error(f"Error copying image {img_path} to {target_path}: {str(e)}")
    
    # Prepare results for response
    classification_results = {
        "document_counts": dict(classification_counts),
        "document_paths": {k: v for k, v in classification_paths.items()}
    }
    
    logger.info(f"Classification summary: {classification_results['document_counts']}")
    
    return classification_results

def process_pdf_for_classification(pdf_path):
    """
    Process PDF for document classification:
    1. Convert PDF to images
    2. Classify each image
    3. Organize images by classification
    
    Returns:
        Dictionary with classification results
    """
    try:
        # Step 1: Convert PDF to images
        image_paths, pdf_name = convert_pdf_to_images_for_classification(pdf_path)
        logger.info(f"Converted PDF to {len(image_paths)} images")
        
        # Verify images were created
        existing_images = [p for p in image_paths if os.path.exists(p)]
        if len(existing_images) != len(image_paths):
            logger.warning(f"Some images were not created: {len(existing_images)}/{len(image_paths)}")
            if not existing_images:
                return {"error": "Failed to convert PDF to images"}
        
        # Step 2: Classify each image
        classifications = classify_document_image(existing_images, batch_mode=True)
        logger.info(f"Classifications: {classifications}")
        
        if len(classifications) != len(existing_images):
            logger.warning(f"Classification count mismatch: {len(classifications)} classifications for {len(existing_images)} images")
        
        # Step 3: Organize images by classification
        classification_results = organize_images_by_classification(existing_images, classifications, pdf_name)
        
        # Check if any images were successfully classified
        total_classified = sum(classification_results["document_counts"].values())
        if total_classified == 0:
            logger.error("No images were successfully classified")
            return {"error": "Failed to classify any images"}
        
        return {
            "status": "success",
            "pdf_name": pdf_name,
            "total_pages": len(image_paths),
            "classifications": classifications,
            "organization_results": classification_results
        }
        
    except Exception as e:
        logger.error(f"Error processing PDF for classification: {str(e)}")
        return {"error": str(e)}

def should_classify_documents(user_content):
    """
    Check if user is specifically asking about medical document classification.
    Only triggers for explicit document identification queries.
    """
    # Very specific keywords that indicate document classification intent
    classification_keywords = [
        "what type of medical document",
        "what kind of medical report",
        "identify the medical documents",
        "classify the medical reports",
        "what type of scan",
        "what kind of test results",
        "categorize the medical documents",
        "sort the medical reports",
        "what medical documents are in",
        "list the medical documents",
        "identify medical report types"
    ]
    
    # Convert to lowercase for comparison
    user_content_lower = user_content.lower()
    
    # Check for exact phrase matches (more restrictive)
    return any(keyword in user_content_lower for keyword in classification_keywords)

def clear_chat_history():
    """Clear the chat history"""
    global chat_history
    chat_history = []

def add_to_chat_history(message):
    """Add a message to chat history"""
    global chat_history
    chat_history.append(message)
    # Keep only last 10 messages
    chat_history = chat_history[-50:]

def get_chat_history():
    """Get current chat history"""
    global chat_history
    return chat_history

def analyze_user_request(content):
    """Analyze user request to understand what medical information they're seeking"""
    try:
        # Ask LLM to analyze the user's request
        system_prompt = {
            "role": "system",
            "content": (
                "You are a medical analysis expert. Analyze the user's request and identify:\n"
                "1. What type of medical information they're looking for\n"
                "2. What specific symptoms or conditions they're asking about\n"
                "3. What type of medical documents they want to analyze\n"
                "Return the analysis in a structured format."
            )
        }
        
        user_message = {
            "role": "user",
            "content": f"Analyze this medical request: {content}"
        }
        
        # Get model selection from request
        selected_model = request.form.get('model', 'gemma3:12b')  # Default to 12b if not specified
        
        # Validate model selection
        valid_models = ['gemma3:4b', 'gemma3:12b', 'llama3.2-vision:latest', 'mistral:latest', 'gpt-4.1-mini']
        if selected_model not in valid_models:
            selected_model = 'gemma3:12b'  # Fallback to default if invalid
            
        if selected_model == 'gpt-4.1-mini':
            return ask_openai([system_prompt, user_message], model=selected_model)
        else:
            response = ollama.chat(
                model=selected_model,
                messages=[system_prompt, user_message],
                options={"max_tokens": 1000}
            )
            return response.get("message", {}).get("content", "")
    except Exception as e:
        print(f"Error analyzing user request: {str(e)}")
        return None

def analyze_extracted_fields(extracted_info, user_request_analysis):
    """Use LLM to analyze extracted fields based on user's request"""
    try:
        # Prepare the context for LLM
        context = {
            "user_request": user_request_analysis,
            "extracted_fields": extracted_info
        }
        
        system_prompt = {
            "role": "system",
            "content": (
                "You are a medical expert. Analyze the extracted fields and the user's request.\n"
                "1. Verify if all requested fields are present\n"
                "2. Validate the format and content of fields\n"
                "3. Identify any missing or incorrect information\n"
                "4. Provide insights about the document\n"
                "Return the analysis in a clear, structured format."
            )
        }
        
        user_message = {
            "role": "user",
            "content": f"Analyze these extracted fields based on the user's request:\n{context}"
        }

        # Get model selection from request
        selected_model = request.form.get('model', 'gemma3:12b')  # Default to 12b if not specified
        
        # Validate model selection
        valid_models = ['gemma3:4b', 'gemma3:12b', 'llama3.2-vision:latest', 'mistral:latest', 'gpt-4.1-mini']
        if selected_model not in valid_models:
            selected_model = 'gemma3:12b'  # Fallback to default if invalid
        
        if selected_model == 'gpt-4.1-mini':
            return ask_openai([system_prompt, user_message], model=selected_model)
        else:
            response = ollama.chat(
                model=selected_model,
                messages=[system_prompt, user_message],
                options={"max_tokens": 2000}
            )
            return response.get("message", {}).get("content", "")
    except Exception as e:
        print(f"Error analyzing fields: {str(e)}")
        return None

def analyze_image_with_llm(image_content, user_request, image_base64=None):
    """Use LLM to analyze image content based on user's request"""
    try:
        system_prompt = {
            "role": "system",
            "content": (
                "You are a focused banking document and image analysis expert. Your task is to:\n"
                "1. Provide ONLY the specific information requested by the user\n"
                "2. Give direct, concise answers without additional analysis or commentary\n"
                "3. If asked for a specific value (like amount, weight, date), return ONLY that value\n"
                "4. If the requested information is not found, respond with 'Information not found'\n"
                "5. Do not provide explanations unless specifically asked\n\n"
                "Example responses:\n"
                "- For 'what is the total amount': '₹50,000'\n"
                "- For 'what is the date': '15/03/2024'\n"
                "- If not found: 'Information not found'"
            )
        }

        # Get model selection from request
        selected_model = request.form.get('model', 'gemma3:12b')  # Default to 12b if not specified
        
        # Validate model selection
        valid_models = ['gemma3:4b', 'gemma3:12b', 'llama3.2-vision:latest', 'mistral:latest', 'gpt-4.1-mini']
        if selected_model not in valid_models:
            selected_model = 'gemma3:12b'  # Fallback to default if invalid
        
        if selected_model == 'gpt-4.1-mini':
            # For OpenAI models, we need to handle the image differently
            if image_base64:
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt["content"]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Find exactly this in the image: {user_request}"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ]
                return ask_openai(messages, model=selected_model, max_tokens=500)
            else:
                # If no image is provided, analyze just the text content
                messages = [
                    system_prompt,
                    {
                        "role": "user",
                        "content": f"Find exactly this in the text: {user_request}\n\nText Content: {image_content}"
                    }
                ]
                return ask_openai(messages, model=selected_model, max_tokens=500)
        else:
            # For Ollama models, use the text-based approach
            user_message = {
                "role": "user",
                "content": f"Find exactly this in the image: {user_request}\n\nImage Content: {image_content}"
            }
            
            response = ollama.chat(
                model=selected_model,
                messages=[system_prompt, user_message],
                options={"max_tokens": 500}  # Reduced token limit for concise responses
            )
            return response.get("message", {}).get("content", "")
    except Exception as e:
        print(f"Error in LLM image analysis: {str(e)}")
        return "Error analyzing image"

def extract_fields_from_document(image_path, user_request=None):
    """Extract structured fields from document using doctr and LLM"""
    # Load and analyze the document
    doc = DocumentFile.from_images(image_path)
    result = model(doc)
    
    # Extract structured information
    extracted_info = {
        'text_blocks': [],
        'fields': {},
        'tables': [],
        'key_value_pairs': []
    }
    
    # Process each page
    for page in result.pages:
        page_data = {
            'blocks': [],
            'fields': {}
        }
        
        # Extract text blocks with positions
        for block in page.blocks:
            block_text = ' '.join([word.value for line in block.lines for word in line.words])
            confidence = np.mean([word.confidence for line in block.lines for word in line.words])
            
            # Get block position
            bbox = block.geometry
            position = {
                'top': bbox[0][1],
                'left': bbox[0][0],
                'bottom': bbox[1][1],
                'right': bbox[1][0]
            }
            
            block_info = {
                'text': block_text,
                'confidence': float(confidence),
                'position': position
            }
            
            page_data['blocks'].append(block_info)
            
            # Try to identify key-value pairs
            if ':' in block_text:
                key, value = block_text.split(':', 1)
                extracted_info['key_value_pairs'].append({
                    'key': key.strip(),
                    'value': value.strip(),
                    'confidence': float(confidence)
                })
        
        # Get all text content
        text_content = ' '.join([block['text'] for block in page_data['blocks']])
        
        # Use LLM to identify fields if user request is provided
        if user_request:
            system_prompt = {
                "role": "system",
                "content": (
                    "You are a banking document expert. Given the text content and user's request:\n"
                    "1. Identify and extract the specific fields requested\n"
                    "2. Validate the format of extracted information\n"
                    "3. Look for contextual information that might be relevant\n"
                    "Return the extracted information in a structured format."
                )
            }
            
            user_message = {
                "role": "user",
                "content": f"User Request: {user_request}\n\nDocument Content: {text_content}"
            }
            
            # Get model selection from request
            selected_model = request.form.get('model', 'gemma3:12b')  # Default to 12b if not specified
            
            # Validate model selection
            valid_models = ['gemma3:4b', 'gemma3:12b', 'llama3.2-vision:latest', 'mistral:latest', 'gpt-4.1-mini']
            if selected_model not in valid_models:
                selected_model = 'gemma3:12b'  # Fallback to default if invalid
            
            response = ollama.chat(
                model=selected_model,
                messages=[system_prompt, user_message],
                options={"max_tokens": 1000}
            )
            
            llm_extracted_fields = response.get("message", {}).get("content", "")
            page_data['llm_extracted_fields'] = llm_extracted_fields
        
        # Also use regex for common banking fields
        fields_to_check = {
            'account_number': r'(?i)acc(?:ount)?\s*(?:no|number|#)?\s*[:.]?\s*([A-Z0-9]{8,})',
            'ifsc_code': r'(?i)ifsc\s*(?:code)?\s*[:.]?\s*([A-Z0-9]{11})',
            'date': r'(?i)date\s*[:.]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            'amount': r'(?i)(?:amount|rs|inr)\s*[:.]?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            'name': r'(?i)name\s*[:.]?\s*([A-Za-z\s]{2,50})',
            'branch': r'(?i)branch\s*[:.]?\s*([A-Za-z\s]{2,50})',
            'transaction_id': r'(?i)(?:transaction|ref|reference)\s*(?:id|no|number)\s*[:.]?\s*([A-Z0-9]{6,})'
        }
        
        import re
        for field, pattern in fields_to_check.items():
            matches = re.findall(pattern, text_content)
            if matches:
                page_data['fields'][field] = matches[0]
        
        extracted_info['text_blocks'].extend(page_data['blocks'])
        extracted_info['fields'].update(page_data['fields'])
        if 'llm_extracted_fields' in page_data:
            extracted_info['llm_analysis'] = page_data['llm_extracted_fields']
    
    return extracted_info

def process_scanned_pdf(pdf_path, user_request=None):
    """Process scanned PDF using doctr and LLM"""
    try:
        # First analyze user request
        request_analysis = None
        if user_request:
            request_analysis = analyze_user_request(user_request)
        
        # Create temporary directory for images - use TEMP_DIR for regular processing
        temp_img_dir = os.path.join(TEMP_DIR, "pdf_images")
        os.makedirs(temp_img_dir, exist_ok=True)
        
        # Convert PDF to images
        images = convert_from_path(pdf_path)
        
        text_content = []
        images_base64 = []
        extracted_fields = []
        
        # Process each page
        for i, image in enumerate(images):
            # Save image temporarily
            image_path = os.path.join(temp_img_dir, f'page_{i+1}.jpg')
            image.save(image_path, 'JPEG')
            
            # Extract fields and text using doctr with user request context
            page_info = extract_fields_from_document(image_path, user_request)
            
            # Add page number
            text_content.append(f"\n--- Page {i+1} ---\n")
            
            # Add extracted fields
            if page_info['fields']:
                text_content.append("\nExtracted Fields:")
                for field, value in page_info['fields'].items():
                    text_content.append(f"{field}: {value}")
            
            # Add LLM analysis if available
            if 'llm_analysis' in page_info:
                text_content.append("\nDetailed Analysis:")
                text_content.append(page_info['llm_analysis'])
            
            # Add key-value pairs
            if page_info['key_value_pairs']:
                text_content.append("\nDetected Key-Value Pairs:")
                for pair in page_info['key_value_pairs']:
                    text_content.append(f"{pair['key']}: {pair['value']}")
            
            # Add text blocks
            text_content.append("\nDocument Content:")
            for block in page_info['text_blocks']:
                text_content.append(block['text'])
            
            # Store extracted fields for later use
            extracted_fields.append(page_info)
            
            # Convert image to base64 for the model
            with open(image_path, "rb") as img_file:
                img_data = img_file.read()
                images_base64.append(base64.b64encode(img_data).decode('utf-8'))
        
        # If we have user request analysis, analyze the extracted fields
        if request_analysis:
            field_analysis = analyze_extracted_fields(extracted_fields, request_analysis)
            if field_analysis:
                text_content.append("\nField Analysis Based on Request:")
                text_content.append(field_analysis)
        
        return "\n".join(text_content), images_base64, extracted_fields
    
    finally:
        # Cleanup temporary image directory
        if os.path.exists(temp_img_dir):
            shutil.rmtree(temp_img_dir)

def extract_text_from_pdf(pdf_file):
    """Extract text content from PDF file"""
    try:
        # Save PDF to temporary file
        temp_pdf_path = os.path.join(TEMP_DIR, "temp.pdf")
        pdf_file.save(temp_pdf_path)
        
        # Try normal text extraction first
        pdf_reader = PyPDF2.PdfReader(temp_pdf_path)
        text = []
        has_text = False
        
        # Check if PDF has extractable text
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text.strip():
                has_text = True
                text.append(f"\n--- Page {page_num + 1} ---\n")
                text.append(page_text)
        
        # If no text found, treat as scanned document
        if not has_text:
            print("No extractable text found, processing as scanned document...")
            text_content, images, extracted_fields = process_scanned_pdf(temp_pdf_path)
            return text_content, images, True, extracted_fields  # True indicates scanned PDF
        
        return "\n".join(text), None, False, None  # False indicates normal PDF
        
    except Exception as e:
        return f"Error processing PDF: {str(e)}", None, False, None
    
    finally:
        # Cleanup
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

def process_excel(excel_file):
    """Process Excel file and return formatted string of content"""
    df = pd.read_excel(excel_file)
    return df.to_string()

def process_video(video_file):
    """Process video file and extract key frames"""
    temp_video_path = os.path.join(TEMP_DIR, "temp_video.mp4")
    temp_frame_path = os.path.join(TEMP_DIR, "frame.jpg")
    frames_base64 = []
    
    try:
        video_file.save(temp_video_path)
        cap = cv2.VideoCapture(temp_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Extract frames at 1-second intervals, up to 5 key frames
        frame_interval = fps
        frame_count = 0
        frames_extracted = 0
        
        while cap.isOpened() and frames_extracted < 5:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                cv2.imwrite(temp_frame_path, frame)
                with open(temp_frame_path, "rb") as img:
                    image_data = img.read()
                    frames_base64.append(base64.b64encode(image_data).decode('utf-8'))
                frames_extracted += 1
                
            frame_count += 1
            
        cap.release()
        
        # Add video metadata
        video_info = f"Video Analysis:\nDuration: {total_frames/fps:.2f} seconds\nFrames Analyzed: {frames_extracted}"
        return frames_base64, video_info
        
    finally:
        # Cleanup
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)

def process_files(request):
    """Process different types of medical files and return their contents"""
    file_contents = []
    images_base64 = []
    extracted_fields = None
    
    # Get user's request for context
    user_request = request.form.get('content', '').strip()
    selected_model = request.form.get('model', 'gemma3:12b')
    
    # Process medical images first (highest priority for medical use case)
    if request.input_types['has_image']:
        image_file = request.files['image']
        temp_image_path = os.path.join(TEMP_DIR, "temp_medical_image.jpg")
        try:
            image_file.save(temp_image_path)
            # Process medical image
            image_results = process_medical_image(temp_image_path)
            
            if "error" not in image_results:
                # Get base64 of both original and enhanced images
                for img_path in [temp_image_path, image_results["enhanced_image_path"]]:
                    with open(img_path, "rb") as img:
                        image_data = img.read()
                        images_base64.append(base64.b64encode(image_data).decode('utf-8'))
                
                # Add medical analysis results
                if image_results["medical_fields"]:
                    extracted_fields = [image_results]
                    file_contents.append("Medical Image Analysis Results:\n" + 
                                      generate_medical_summary(image_results["medical_fields"], 
                                                            {"confidence": 0.0}))
            else:
                file_contents.append(f"Error processing medical image: {image_results['error']}")
                
        finally:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
    
    # Process medical documents (PDFs)
    if request.input_types['has_pdf']:
        try:
            pdf_file = request.files['pdf']
            if pdf_file:
                pdf_text, scanned_images, is_scanned, fields = extract_text_from_pdf(pdf_file)
                if pdf_text:
                    # Analyze medical content
                    medical_analysis = analyze_medical_report(pdf_text)
                    if "error" not in medical_analysis:
                        file_contents.append(medical_analysis["summary"])
                        if scanned_images:
                            images_base64.extend(scanned_images)
                        if fields:
                            extracted_fields = fields
                    else:
                        file_contents.append(f"Error analyzing medical document: {medical_analysis['error']}")
                else:
                    file_contents.append("Error: Could not extract text from medical document.")
        except Exception as e:
            file_contents.append(f"Error processing medical PDF: {str(e)}")
    
    # Process lab results (Excel)
    if request.input_types['has_excel']:
        try:
            excel_file = request.files['excel']
            df = pd.read_excel(excel_file)
            
            # Convert lab results to structured format
            lab_results = []
            for _, row in df.iterrows():
                for col in df.columns:
                    try:
                        value = float(row[col])
                        lab_results.append({
                            'test': col,
                            'value': value,
                            'unit': 'N/A'  # Add units if available in Excel
                        })
                    except:
                        continue
            
            # Check for critical values
            critical_values = []
            for result in lab_results:
                if result['test'] in normal_ranges:
                    min_val, max_val = normal_ranges[result['test']]
                    if result['value'] < min_val or result['value'] > max_val:
                        critical_values.append({
                            'type': result['test'],
                            'value': result['value'],
                            'concern': 'Out of normal range'
                        })
            
            # Generate summary
            if lab_results:
                summary = "Lab Results Analysis:\n\n"
                summary += "Tests Performed:\n"
                for result in lab_results:
                    summary += f"• {result['test']}: {result['value']} {result['unit']}\n"
                
                if critical_values:
                    summary += "\nCritical Values Found:\n"
                    for cv in critical_values:
                        summary += f"• {cv['type']}: {cv['value']} - {cv['concern']}\n"
                
                summary += "\nDISCLAIMER: Lab results should be interpreted by healthcare professionals."
                file_contents.append(summary)
            
        except Exception as e:
            file_contents.append(f"Error processing lab results: {str(e)}")
    
    # Process medical videos
    if request.input_types['has_video']:
        try:
            video_file = request.files['video']
            video_frames, video_info = process_video(video_file)
            if video_frames:
                images_base64.extend(video_frames)
                # Analyze each frame for medical content
                frame_analyses = []
                for i, frame in enumerate(video_frames):
                    frame_path = os.path.join(TEMP_DIR, f"frame_{i}.jpg")
                    try:
                        # Save frame temporarily
                        with open(frame_path, "wb") as f:
                            f.write(base64.b64decode(frame))
                        # Process frame as medical image
                        frame_result = process_medical_image(frame_path)
                        if "error" not in frame_result:
                            frame_analyses.append(frame_result)
                    finally:
                        if os.path.exists(frame_path):
                            os.remove(frame_path)
                
                # Combine analyses
                if frame_analyses:
                    combined_analysis = "Medical Video Analysis:\n\n"
                    for i, analysis in enumerate(frame_analyses):
                        combined_analysis += f"Frame {i+1}:\n"
                        if analysis.get("medical_fields"):
                            combined_analysis += generate_medical_summary(
                                analysis["medical_fields"], 
                                {"confidence": 0.0}
                            )
                        combined_analysis += "\n---\n"
                    file_contents.append(combined_analysis)
        except Exception as e:
            file_contents.append(f"Error processing medical video: {str(e)}")
    
    return file_contents, images_base64, extracted_fields, None  # Remove classification_results

def ask_openai(messages, model="gpt-4.1-mini", max_tokens=4096):
    """Use OpenAI models for chat completion"""
    try:
        # Call OpenAI API
        completion = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error using OpenAI model: {str(e)}")
        return "Error using OpenAI model."

def evaluate_response_quality(response: str) -> float:
    """
    Evaluate the quality of a model's response based on various metrics.
    Returns a score between 0 and 1.
    """
    # Initialize score
    score = 0.0
    
    # Check response length (not too short, not too long)
    length = len(response)
    if 100 <= length <= 2000:
        score += 0.2
    elif length > 2000:
        score += 0.1
    
    # Check for specific keywords indicating detailed analysis
    analysis_keywords = ['analysis', 'evaluation', 'details', 'specifically', 'furthermore', 'however']
    keyword_count = sum(1 for keyword in analysis_keywords if keyword.lower() in response.lower())
    score += min(0.2, keyword_count * 0.033)  # Max 0.2 for keywords
    
    # Check for structured response (bullet points, numbering)
    if any(marker in response for marker in ['•', '-', '1.', '2.', '*']):
        score += 0.2
    
    # Check for balanced response (not just lists or single sentences)
    sentences = response.split('.')
    if 3 <= len(sentences) <= 15:
        score += 0.2
    
    # Check for technical terms and banking-specific vocabulary
    banking_terms = ['transaction', 'account', 'balance', 'payment', 'transfer', 'banking', 'financial']
    term_count = sum(1 for term in banking_terms if term.lower() in response.lower())
    score += min(0.2, term_count * 0.028)  # Max 0.2 for banking terms
    
    return min(1.0, score)

async def execute_model(model: str, messages: List[Dict], max_tokens: int = 4096) -> Tuple[str, float, str]:
    """
    Execute a single model and return its response along with quality score.
    """
    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"max_tokens": max_tokens}
        ).get("message", {}).get("content", "")
        
        # Evaluate response quality
        quality_score = evaluate_response_quality(response)
        return response, quality_score, model
    except Exception as e:
        logger.error(f"Error executing model {model}: {str(e)}")
        return f"Error with {model}: {str(e)}", 0.0, model

async def execute_simplyfillm(messages: List[Dict], max_tokens: int = 4096) -> Tuple[str, str]:
    """
    Execute all models in parallel and use gemma3:12b to evaluate their responses.
    """
    models = ['gemma3:4b', 'gemma3:12b', 'llama3.2-vision:latest', 'mistral:latest']
    tasks = []
    
    # Create tasks for all models
    for model in models:
        task = execute_model(model, messages, max_tokens)
        tasks.append(task)
    
    # Execute all models in parallel
    try:
        responses = await asyncio.gather(*tasks)
        all_responses = [(response, model) for response, _, model in responses]
        
        # Create evaluation prompt for gemma3:12b
        eval_messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert evaluator of AI responses. Your task is to:\n"
                    "1. Analyze each response for accuracy, completeness, and relevance\n"
                    "2. Consider the context and specific requirements of the query\n"
                    "3. Select the single best response based on these criteria\n"
                    "4. Return ONLY the index number (0-4) of the best response, nothing else"
                )
            },
            {
                "role": "user",
                "content": (
                    "Original query and context:\n"
                    f"{messages[-1]['content']}\n\n"
                    "Responses to evaluate:\n" +
                    "\n---\n".join(f"[{i}] {resp[0]}" for i, resp in enumerate(all_responses)) +
                    "\n\nReturn only the index number of the best response (0-4):"
                )
            }
        ]
        
        # Get evaluation from gemma3:12b
        eval_response = ollama.chat(
            model='gemma3:12b',
            messages=eval_messages,
            options={"max_tokens": 100}
        ).get("message", {}).get("content", "0")
        
        # Extract the index (default to 0 if parsing fails)
        try:
            best_index = int(eval_response.strip())
            best_index = max(0, min(len(all_responses) - 1, best_index))  # Ensure valid index
        except ValueError:
            best_index = 0
            
        # Get the best response and its model
        best_response, best_model = all_responses[best_index]
        
        # Format the response
        final_response = f"Response from {best_model}:\n\n{best_response}"
        
        return final_response, best_model
    except Exception as e:
        logger.error(f"Error in simplyfillm execution: {str(e)}")
        return f"Error executing simplyfillm: {str(e)}", "error"

# Medical-specific fields to extract
MEDICAL_FIELDS = {
    'patient_info': r'(?i)name:\s*([^\n]+)|age:\s*(\d+)|gender:\s*(M|F|Male|Female)',
    'vital_signs': r'(?i)BP:\s*(\d+/\d+)|Pulse:\s*(\d+)|Temp:\s*([\d\.]+)',
    'lab_values': r'(?i)(WBC|RBC|HGB|HCT|PLT):\s*([\d\.]+)',
    'medications': r'(?i)(?:prescribed|taking|medication):\s*([^\n]+)',
    'diagnosis': r'(?i)diagnosis:\s*([^\n]+)|assessment:\s*([^\n]+)',
    'doctor_info': r'(?i)Dr\.\s*([^\n]+)|physician:\s*([^\n]+)',
    'test_results': r'(?i)(positive|negative|normal|abnormal)(?:\s+for\s+([^\n]+))?',
    'medical_history': r'(?i)history:\s*([^\n]+)|previous conditions:\s*([^\n]+)'
}

def extract_medical_fields(text_content: str) -> Dict[str, List[str]]:
    """Extract medical-specific fields from text content"""
    extracted_fields = collections.defaultdict(list)
    
    for field_name, pattern in MEDICAL_FIELDS.items():
        matches = re.finditer(pattern, text_content)
        for match in matches:
            # Get all capturing groups and filter out None values
            values = [g for g in match.groups() if g is not None]
            if values:
                extracted_fields[field_name].extend(values)
    
    return dict(extracted_fields)

def analyze_medical_content(text_content: str) -> Dict:
    """Analyze medical content using Ollama"""
    try:
        # Get base analysis from Ollama
        ollama_response = ollama.chat(
            model='gemma3:12b',
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical expert. Analyze the following medical content and provide key findings. "
                        "Include potential diagnoses, critical values, and any concerning patterns. "
                        "Structure your response with clear sections and highlight any urgent findings."
                    )
                },
                {
                    "role": "user",
                    "content": text_content
                }
            ]
        )

        base_analysis = ollama_response.get("message", {}).get("content", "")

        # Extract key medical terms for context
        medical_terms = extract_medical_terms(text_content)
        
        return {
            'analysis': base_analysis,
            'medical_terms': medical_terms,
            'confidence': 1.0 if base_analysis else 0.0
        }
    except Exception as e:
        logger.error(f"Error in medical analysis: {str(e)}")
        return {
            'analysis': "Unable to complete medical analysis at this time.",
            'medical_terms': [],
            'confidence': 0.0
        }

def extract_medical_terms(text: str) -> List[str]:
    """Extract key medical terms from text for research"""
    try:
        # Use Ollama to extract key medical terms
        response = ollama.chat(
            model='gemma3:12b',
            messages=[
                {
                    "role": "system",
                    "content": "Extract key medical terms from the text. Return only the terms, separated by commas."
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        )
        terms = response.get("message", {}).get("content", "").split(",")
        return [term.strip() for term in terms if term.strip()]
    except Exception as e:
        print(f"Error extracting medical terms: {e}")
        return []

def generate_medical_summary(fields: Dict, analysis: Dict) -> str:
    """Generate a summary of medical findings with latest research"""
    summary_parts = []
    
    # Add patient info
    if 'patient_info' in fields:
        summary_parts.append("Patient Information: " + ", ".join(fields['patient_info']))
    
    # Add diagnosis if present
    if 'diagnosis' in fields:
        summary_parts.append("Diagnosis: " + ", ".join(fields['diagnosis']))
        
        # Add latest research for diagnosis
        if 'latest_research' in analysis:
            for diagnosis in fields['diagnosis']:
                if diagnosis in analysis['latest_research']:
                    research = analysis['latest_research'][diagnosis]
                    summary_parts.append(f"\nLatest Research for {diagnosis}:")
                    summary_parts.append(research['answer'])
                    summary_parts.append("\nSources:")
                    for source in research['sources'][:2]:  # Show top 2 sources
                        summary_parts.append(f"- {source['title']} ({source['source']})")
    
    # Add critical values
    critical_values = check_critical_values(fields)
    if critical_values:
        summary_parts.append("Critical Findings: " + 
                           ", ".join([f"{cv['type']}: {cv['value']} ({cv['concern']})" 
                                    for cv in critical_values]))
    
    # Add test results
    if 'test_results' in fields:
        summary_parts.append("Test Results: " + ", ".join(fields['test_results']))
    
    # Add medications
    if 'medications' in fields:
        summary_parts.append("Medications: " + ", ".join(fields['medications']))
    
    # Add medical analysis
    if 'analysis' in analysis:
        summary_parts.append("\nMedical Analysis:")
        summary_parts.append(analysis['analysis'])
    
    # Add disclaimer
    summary_parts.append("\nDISCLAIMER: This is an automated analysis for informational purposes only. "
                        "Please consult with healthcare professionals for medical advice and treatment decisions.")
    
    return "\n".join(summary_parts)

def process_medical_image(image_path: str) -> Dict:
    """Process medical images (X-rays, MRIs, etc.)"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Failed to load image"}

        # Convert to grayscale for medical imaging
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Basic image enhancement for medical images
        enhanced = cv2.equalizeHist(gray)
        
        # Save enhanced image
        enhanced_path = image_path.replace('.', '_enhanced.')
        cv2.imwrite(enhanced_path, enhanced)
        
        # Extract text from image using OCR
        doc = DocumentFile.from_images(enhanced_path)
        result = ocr_model(doc)
        extracted_text = result.render()
        
        # Extract medical fields from text
        medical_fields = extract_medical_fields(extracted_text)
        
        return {
            "enhanced_image_path": enhanced_path,
            "extracted_text": extracted_text,
            "medical_fields": medical_fields
        }
    except Exception as e:
        return {"error": f"Error processing medical image: {str(e)}"}

def analyze_medical_report(text_content: str) -> Dict:
    """Analyze medical report content"""
    try:
        # Extract medical fields
        medical_fields = extract_medical_fields(text_content)
        
        # Analyze with Ollama
        medical_analysis = analyze_medical_content(text_content)
        
        # Check for critical values
        critical_values = check_critical_values(medical_fields)
        
        # Generate summary
        summary = generate_medical_summary(medical_fields, medical_analysis)
        
        return {
            "fields": medical_fields,
            "analysis": medical_analysis,
            "critical_values": critical_values,
            "summary": summary
        }
    except Exception as e:
        return {"error": f"Error analyzing medical report: {str(e)}"}

def check_critical_values(medical_fields: Dict) -> List[Dict]:
    """Check for critical or abnormal values in medical data"""
    critical_values = []
    
    # Define normal ranges for common tests
    normal_ranges = {
        'WBC': (4.0, 11.0),
        'RBC': (4.5, 5.9),
        'HGB': (13.5, 17.5),
        'HCT': (41.0, 53.0),
        'PLT': (150, 450),
        'temperature': (36.5, 37.5),
        'systolic_bp': (90, 140),
        'diastolic_bp': (60, 90),
        'pulse': (60, 100)
    }
    
    # Check vital signs
    if 'vital_signs' in medical_fields:
        for vital in medical_fields['vital_signs']:
            if 'BP' in vital:
                try:
                    sys, dia = map(int, vital.split('/'))
                    if sys > normal_ranges['systolic_bp'][1] or sys < normal_ranges['systolic_bp'][0]:
                        critical_values.append({
                            'type': 'Blood Pressure',
                            'value': vital,
                            'concern': 'Abnormal systolic pressure'
                        })
                except:
                    pass
    
    # Check lab values
    if 'lab_values' in medical_fields:
        for lab in medical_fields['lab_values']:
            for test, (min_val, max_val) in normal_ranges.items():
                if test in lab:
                    try:
                        value = float(lab.split(':')[1].strip())
                        if value < min_val or value > max_val:
                            critical_values.append({
                                'type': test,
                                'value': value,
                                'concern': 'Out of normal range'
                            })
                    except:
                        pass
    
    return critical_values

def generate_medical_summary(fields: Dict, analysis: Dict) -> str:
    """Generate a summary of medical findings with latest research"""
    summary_parts = []
    
    # Add patient info
    if 'patient_info' in fields:
        summary_parts.append("Patient Information: " + ", ".join(fields['patient_info']))
    
    # Add diagnosis if present
    if 'diagnosis' in fields:
        summary_parts.append("Diagnosis: " + ", ".join(fields['diagnosis']))
        
        # Add latest research for diagnosis
        if 'latest_research' in analysis:
            for diagnosis in fields['diagnosis']:
                if diagnosis in analysis['latest_research']:
                    research = analysis['latest_research'][diagnosis]
                    summary_parts.append(f"\nLatest Research for {diagnosis}:")
                    summary_parts.append(research['answer'])
                    summary_parts.append("\nSources:")
                    for source in research['sources'][:2]:  # Show top 2 sources
                        summary_parts.append(f"- {source['title']} ({source['source']})")
    
    # Add critical values
    critical_values = check_critical_values(fields)
    if critical_values:
        summary_parts.append("Critical Findings: " + 
                           ", ".join([f"{cv['type']}: {cv['value']} ({cv['concern']})" 
                                    for cv in critical_values]))
    
    # Add test results
    if 'test_results' in fields:
        summary_parts.append("Test Results: " + ", ".join(fields['test_results']))
    
    # Add medications
    if 'medications' in fields:
        summary_parts.append("Medications: " + ", ".join(fields['medications']))
    
    # Add medical analysis
    if 'analysis' in analysis:
        summary_parts.append("\nMedical Analysis:")
        summary_parts.append(analysis['analysis'])
    
    # Add disclaimer
    summary_parts.append("\nDISCLAIMER: This is an automated analysis for informational purposes only. "
                        "Please consult with healthcare professionals for medical advice and treatment decisions.")
    
    return "\n".join(summary_parts)

# Update the welcome message
def get_welcome_message() -> str:
    return (
        "Hello! Welcome to the Medical Assistant! 🏥\n\n"
        "I'm your intelligent medical assistant powered by advanced AI models. Here's what I can help you with:\n\n"
        "🤖 **Available AI Models:**\n"
        "• Gemma3 Models - Medical analysis and queries\n"
        "• Vision Models - Medical image analysis\n"
        "• RAG System - Detailed medical document analysis\n\n"
        "📄 **Medical Document Processing:**\n"
        "• Medical Reports and Records\n"
        "• Lab Test Results\n"
        "• Radiology Reports (X-rays, MRIs, CT scans)\n"
        "• Prescriptions and Medications\n"
        "• Patient History Documents\n\n"
        "🔍 **Analysis Capabilities:**\n"
        "• Extract and analyze medical data\n"
        "• Identify critical or abnormal values\n"
        "• Process medical imaging results\n"
        "• Analyze lab test results\n"
        "• Review medication information\n"
        "• Generate medical report summaries\n\n"
        "⚠️ **Important Disclaimer:**\n"
        "This assistant provides general medical information for educational purposes only. "
        "It does not provide medical advice, diagnosis, or treatment. Always consult with "
        "qualified healthcare professionals for medical decisions.\n\n"
        "How can I assist you with your medical questions today?"
    )

# Modify process_chat function to handle simplyfillm
def process_chat(request):
    global request_count, error_count, guardrail_block_count
    
    # --- Metrics: Increment request count ---
    request_count += 1
    print(f"\n[METRIC] Total chat requests: {request_count}")
    
    # --- Tracing: Start timing ---
    start_time = time.time()
    
    # Initialize audit trail
    audit_trail = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "request_id": str(uuid.uuid4()),
        "metrics": {
            "total_requests": request_count,
            "error_count": error_count,
            "guardrail_blocks": guardrail_block_count
        },
        "events": []
    }
    
    try:
        user_content = request.form.get('content', '').strip()
        selected_model = request.form.get('model', 'gemma3:12b')
        
        # --- Tracing: Log user input ---
        print(f"[TRACE] User input: {user_content}")
        print(f"[TRACE] Selected model: {selected_model}")
        audit_trail["events"].append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": "input_received",
            "details": {
                "model": selected_model,
                "content_length": len(user_content)
            }
        })

        if not user_content:
            print("[TRACE] Error: No input provided")
            error_count += 1
            print(f"[METRIC] Total errors: {error_count}")
            audit_trail["events"].append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "event": "error",
                "details": "No input provided"
            })
            return jsonify({
                "error": "No input provided.",
                "audit_trail": audit_trail
            }), 400

        # --- OpenLIT LLM-based Guardrail Check ---
        print("[TRACE] Running guardrail check...")
        result = sensitive_guard.detect(text=user_content)
        result_dict = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        print(f"[TRACE] Guardrail result: {result_dict}")
        
        # Add guardrail check to audit trail
        audit_trail["events"].append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": "guardrail_check",
            "details": result_dict
        })
        
        if result_dict["verdict"] == "yes":
            guardrail_block_count += 1
            print(f"[METRIC] Guardrail blocks: {guardrail_block_count}")
            print(f"[TRACE] Input blocked by guardrail: {result_dict['classification']}")
            audit_trail["events"].append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "event": "guardrail_block",
                "details": {
                    "classification": result_dict['classification'],
                    "explanation": result_dict['explanation']
                }
            })
            return jsonify({
                "error": f"Input flagged as: {result_dict['classification']} ({result_dict['explanation']})",
                "audit_trail": audit_trail,
                "guardrail_info": result_dict
            }), 403

        # --- Continue with your existing functionality ---
        valid_models = ['gemma3:4b', 'gemma3:12b', 'llama3.2-vision:latest', 'mistral:latest', 'simplyfillm']
        if selected_model not in valid_models:
            selected_model = 'gemma3:12b'  # Fallback to default if invalid
            audit_trail["events"].append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "event": "model_fallback",
                "details": f"Invalid model selected, falling back to gemma3:12b"
            })
        
        # Process files (now includes classification)
        print("[TRACE] Processing files...")
        file_contents, images_base64, extracted_fields, classification_results = process_files(request)
        
        audit_trail["events"].append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": "files_processed",
            "details": {
                "has_file_contents": bool(file_contents),
                "has_images": bool(images_base64),
                "has_extracted_fields": bool(extracted_fields),
                "has_classification": bool(classification_results)
            }
        })

        # Your existing code for combining content and processing
        combined_content = user_content
        if file_contents:
            combined_content += "\n\nAttached Documents Content:\n" + "\n\n".join(file_contents)
            
        if extracted_fields:
            combined_content += "\n\nExtracted Fields Summary:\n"
            for page_num, page_fields in enumerate(extracted_fields):
                if page_fields['fields']:
                    combined_content += f"\nPage {page_num + 1} Fields:\n"
                    for field, value in page_fields['fields'].items():
                        combined_content += f"{field}: {value}\n"

        # Add classification results to content if available
        if classification_results and "error" not in classification_results:
            combined_content += "\n\nDocument Classification Analysis:\n"
            combined_content += f"The uploaded PDF contains {classification_results['total_pages']} pages with the following document types:\n"
            for doc_type, count in classification_results['organization_results']['document_counts'].items():
                combined_content += f"- {doc_type}: {count} page(s)\n"

        # Check for greeting keywords and provide welcome message
        greeting_keywords = ["hi", "hello", "hey", "start over", "new topic", "reset"]
        is_greeting = user_content.lower().strip() in greeting_keywords
        
        if is_greeting:
            clear_chat_history()
            audit_trail["events"].append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "event": "chat_history_cleared",
                "details": "Greeting detected, history cleared"
            })
            # Return welcome message directly for greetings
            welcome_message = get_welcome_message()
            
            # Add welcome message to history
            add_to_chat_history({"role": "assistant", "content": welcome_message})
            
            # --- Metrics: Calculate and log response time ---
            duration = time.time() - start_time
            print(f"[METRIC] Response time (greeting): {duration:.3f} seconds")
            
            audit_trail["events"].append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "event": "greeting_response",
                "details": {
                    "response_time": duration,
                    "message_type": "welcome"
                }
            })
            
            return jsonify({
                'description': welcome_message,
                'audit_trail': audit_trail,
                'guardrail_info': result_dict
            })

        # Create user message with all content
        user_message = {"role": "user", "content": combined_content}
        if images_base64:
            user_message["images"] = images_base64

        # Add user message to history
        add_to_chat_history(user_message)
        audit_trail["events"].append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": "message_added_to_history",
            "details": {
                "role": "user",
                "has_images": bool(images_base64)
            }
        })

        # Check if this is an image-related query or classification query
        has_image_query = request.input_types.get('has_image', False) or 'image' in user_content.lower()
        is_classification_query = should_classify_documents(user_content)
        is_rag_query = request.input_types.get('has_rag', False)

        # Banking and Finance focused system prompt
        system_message = {
            "role": "system",
            "content": (
                "You are a medical assistant, specialized in healthcare and medical information. "
                "Your responsibilities include:\n"
                "1. Answering queries about medical conditions, symptoms, and treatments\n"
                "2. Providing general health information and explanations\n"
                "3. Analyzing medical reports, lab results, and healthcare documentation\n"
                "4. Interpreting medical images like X-rays, MRIs, and CT scans\n"
                "5. Analyzing medical procedures and physical symptoms from videos\n"
                "6. RAG-based medical document question answering\n"
                "7. Maintaining professional, accurate, and compliant medical responses\n\n"
                "When handling documents and media:\n"
                "- For PDFs: Analyze medical reports and extract relevant information\n"
                "- For Excel: Interpret medical data and test results\n"
                "- For Images: Process and explain medical imaging and documentation\n"
                "- For Videos: Analyze medical procedures and physical symptoms\n"
                "- For RAG Processing: Use vector search to find relevant medical information\n\n"
                "When extracted fields are provided:\n"
                "- Verify and validate medical information\n"
                "- Check for required medical fields\n"
                "- Highlight any concerning or abnormal values\n"
                "- Provide relevant medical context\n\n"
                + ("For RAG queries, provide detailed medical answers based on the document context.\n\n" if is_rag_query else "")
                + ("For medical images, provide specific analysis of the visible medical conditions or findings.\n\n" if has_image_query else "")
                + "Important: Always include a disclaimer that this is for informational purposes only and "
                "does not replace professional medical advice. Recommend consulting healthcare professionals "
                "for specific medical concerns."
            )
        }

        # Combine history and system prompt
        messages = [system_message] + get_chat_history()

        # --- Tracing: Log model call ---
        print("[TRACE] Calling selected model...")
        audit_trail["events"].append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": "model_call_start",
            "details": {
                "model": selected_model,
                "query_type": {
                    "is_image": has_image_query,
                    "is_classification": is_classification_query,
                    "is_rag": is_rag_query
                }
            }
        })
        
        if selected_model == 'simplyfillm':
            # Create event loop if it doesn't exist
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            assistant_response, used_model = loop.run_until_complete(
                execute_simplyfillm(messages, 4096 if not has_image_query else 1000)
            )
        else:
            response = ollama.chat(
                model=selected_model,
                messages=messages,
                options={"max_tokens": 4096 if not has_image_query else 1000}
            )
            assistant_response = response.get("message", {}).get("content", "No response from model")
        
        # --- Tracing: Log response ---
        print(f"[TRACE] Bot response: {assistant_response[:100]}...")  # First 100 chars
        
        add_to_chat_history({"role": "assistant", "content": assistant_response})
        audit_trail["events"].append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": "model_response_received",
            "details": {
                "response_length": len(assistant_response)
            }
        })

        # --- Metrics: Calculate and log response time ---
        duration = time.time() - start_time
        print(f"[METRIC] Response time: {duration:.3f} seconds")
        print(f"[METRIC] Success rate: {((request_count - error_count) / request_count * 100):.1f}%")

        # Add final metrics to audit trail
        audit_trail["metrics"].update({
            "response_time": duration,
            "success_rate": ((request_count - error_count) / request_count * 100)
        })

        return jsonify({
            'description': assistant_response,
            'audit_trail': audit_trail,
            'guardrail_info': result_dict
        })

    except Exception as e:
        # --- Error handling with metrics and tracing ---
        error_count += 1
        duration = time.time() - start_time
        print(f"[TRACE] Error occurred: {str(e)}")
        print(f"[METRIC] Total errors: {error_count}")
        print(f"[METRIC] Response time (error): {duration:.3f} seconds")
        print(f"[METRIC] Error rate: {(error_count / request_count * 100):.1f}%")
        
        audit_trail["events"].append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": "error",
            "details": {
                "error_message": str(e),
                "response_time": duration
            }
        })
        audit_trail["metrics"].update({
            "error_rate": (error_count / request_count * 100),
            "response_time": duration
        })
        
        return jsonify({
            'error': f"Error in process_chat: {str(e)}",
            'audit_trail': audit_trail,
            'guardrail_info': result_dict if 'result_dict' in locals() else None
        }), 500
    
    finally:
        # Cleanup any temporary files
        for file in os.listdir(TEMP_DIR):
            try:
                os.remove(os.path.join(TEMP_DIR, file))
            except:
                pass

