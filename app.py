"""
AI KYC Verification - FastAPI Web Application

A modern web interface for AI-powered document verification.
Supports Aadhaar, PAN Card, Passport, and Voter ID verification.
"""

import os
import cv2
import re
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import YOLO for classification
try:
    from ultralytics import YOLO
    from huggingface_hub import hf_hub_download
    import json
    YOLO_AVAILABLE = True
except ImportError as e:
    logger.warning(f"YOLO not available: {e}")
    YOLO_AVAILABLE = False

# Import Surya OCR (better quality than PaddleOCR)
try:
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    SURYA_AVAILABLE = True
    logger.info("Surya OCR modules imported")
except ImportError as e:
    logger.warning(f"Surya OCR not available: {e}")
    SURYA_AVAILABLE = False

# Import mock database for verification
try:
    from mock_database import verify_aadhaar, MOCK_AADHAAR_DATABASE
    DATABASE_AVAILABLE = True
    logger.info(f"Mock database loaded with {len(MOCK_AADHAAR_DATABASE)} records")
except ImportError as e:
    logger.warning(f"Mock database not available: {e}")
    DATABASE_AVAILABLE = False

# Import face matching module
try:
    from face_matching import match_faces, get_demo_match_result
    FACE_MATCHING_AVAILABLE = True
    logger.info("Face matching module loaded")
except ImportError as e:
    logger.warning(f"Face matching not available: {e}")
    FACE_MATCHING_AVAILABLE = False

# Import PDF generator
try:
    from pdf_generator import generate_verification_pdf, generate_simple_pdf, REPORTLAB_AVAILABLE
    PDF_AVAILABLE = True
    logger.info(f"PDF generator loaded (ReportLab: {REPORTLAB_AVAILABLE})")
except ImportError as e:
    logger.warning(f"PDF generator not available: {e}")
    PDF_AVAILABLE = False
    REPORTLAB_AVAILABLE = False

# Initialize models
classifier_model = None
recognition_predictor = None
detection_predictor = None
foundation_predictor = None

def init_models():
    """Initialize AI models."""
    global classifier_model, recognition_predictor, detection_predictor, foundation_predictor
    
    if YOLO_AVAILABLE and classifier_model is None:
        try:
            model_path = hf_hub_download(
                repo_id="logasanjeev/indian-id-validator",
                filename="models/Id_Classifier.pt"
            )
            classifier_model = YOLO(model_path)
            logger.info("YOLO classifier loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")
    
    if SURYA_AVAILABLE and recognition_predictor is None:
        try:
            logger.info("Loading Surya OCR models...")
            foundation_predictor = FoundationPredictor()
            detection_predictor = DetectionPredictor()
            recognition_predictor = RecognitionPredictor(foundation_predictor)
            logger.info("Surya OCR models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Surya: {e}")

# Initialize FastAPI
app = FastAPI(
    title="AI KYC Verification",
    description="AI-powered document verification system",
    version="1.0.0"
)

# Mount static files
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Templates
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Config
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Document type configurations
DOCUMENT_TYPES = {
    'aadhaar': {
        'name': 'Aadhaar Card',
        'description': 'Indian unique identification card',
        'icon': 'fingerprint',
        'color': '#10b981'
    },
    'pan': {
        'name': 'PAN Card',
        'description': 'Permanent Account Number card',
        'icon': 'credit-card',
        'color': '#f59e0b'
    },
    'passport': {
        'name': 'Passport',
        'description': 'International travel document',
        'icon': 'globe',
        'color': '#3b82f6'
    },
    'voter_id': {
        'name': 'Voter ID',
        'description': 'Electoral photo identity card',
        'icon': 'badge-check',
        'color': '#8b5cf6'
    }
}


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def classify_document(image_path: str) -> Dict[str, Any]:
    """Classify document type using YOLO."""
    if classifier_model is None:
        return {"doc_type": "Unknown", "confidence": 0.0}
    
    try:
        results = classifier_model(image_path)
        doc_type = results[0].names[results[0].probs.top1]
        confidence = float(results[0].probs.top1conf)
        return {"doc_type": doc_type, "confidence": confidence}
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return {"doc_type": "Error", "confidence": 0.0}


def extract_text_surya(image_path: str) -> str:
    """Extract text using Surya OCR."""
    if recognition_predictor is None or detection_predictor is None:
        return ""
    
    try:
        image = Image.open(image_path).convert("RGB")
        predictions = recognition_predictor([image], det_predictor=detection_predictor)
        text = " ".join([line.text for line in predictions[0].text_lines])
        return text
    except Exception as e:
        logger.error(f"Surya OCR error: {e}")
        return ""


def parse_aadhaar_fields(ocr_text: str) -> Dict[str, Any]:
    """Parse Aadhaar card fields from OCR text."""
    result = {
        "Name": None,
        "DOB": None,
        "Gender": None,
        "Aadhaar Number": None
    }
    
    if not ocr_text:
        return result
    
    logger.info(f"Parsing OCR text: {ocr_text}")
    
    # Extract Aadhaar number - look for 4-4-4 or 4-4-3+ digit patterns
    # Aadhaar is always at the bottom, so prefer matches later in text
    aadhaar_patterns = [
        r'(\d{4}\s+\d{4}\s+\d{4})',  # With spaces: 6106 2339 9374
        r'(\d{4}\s+\d{4}\s+\d{3,4})',  # Might be partial: 6106 2339 937
        r'(\d{12})',  # No spaces: 610623399374
    ]
    for pattern in aadhaar_patterns:
        matches = re.findall(pattern, ocr_text)
        if matches:
            # Take the LAST match (Aadhaar is at bottom of card)
            result["Aadhaar Number"] = matches[-1]
            break
    
    # Extract DOB - look for "DOB" with flexible spacing around colon
    # Pattern handles: "DOB: 20/10/2004", "DOB : 20/10/2004", "DOB :20/10/2004"
    dob_patterns = [
        r'DOB\s*[:\s]\s*(\d{2}[/-]\d{2}[/-]\d{4})',
        r'D\.?O\.?B\.?\s*[:\s]\s*(\d{2}[/-]\d{2}[/-]\d{4})',
        r'(?:Date of Birth|जन्म\s*तिथि|పుట్టిన\s*తేదీ|பிறந்த\s*நாள்)\s*[:/]?\s*(\d{2}[/-]\d{2}[/-]\d{4})',
    ]
    for pattern in dob_patterns:
        dob_match = re.search(pattern, ocr_text, re.IGNORECASE)
        if dob_match:
            result["DOB"] = dob_match.group(1)
            break
    
    # Extract Gender - support multiple languages
    gender_patterns = [
        (r'\b(Male)\b', 'Male'),
        (r'\b(Female)\b', 'Female'),
        (r'\b(MALE)\b', 'Male'),
        (r'\b(FEMALE)\b', 'Female'),
        (r'(पुरुष)', 'Male'),
        (r'(महिला)', 'Female'),
        (r'(పురుషుడు)', 'Male'),  # Telugu
        (r'(స్త్రీ)', 'Female'),  # Telugu
        (r'(ஆண்)', 'Male'),  # Tamil
        (r'(பெண்)', 'Female'),  # Tamil
    ]
    for pattern, gender_value in gender_patterns:
        if re.search(pattern, ocr_text, re.IGNORECASE):
            result["Gender"] = gender_value
            break
    
    # Extract name - look for English name (Capitalized words)
    # Usually appears after regional script name
    name_patterns = [
        # Pattern 1: Name like "Adithya Vardan M" or "Kanika Manocha"
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:\s+[A-Z])?)(?:\s+(?:పుట్టిన|பிறந்த|जन्म|DOB|Issue))',
        r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z])?)',  # Two/three word names
        r'([A-Z][a-z]+\s+[A-Z][a-z]+)',  # Two word names like "Kanika Manocha"
        r'([A-Z][a-z]{2,}(?:\s+[A-Z])?)',  # Single name with optional initial
    ]
    
    excluded = {'Government', 'India', 'Unique', 'Authority', 'Aadhaar', 'Aadnaar',
                'Male', 'Female', 'DOB', 'Address', 'Issue', 'Date', 'Issued',
                'Proof', 'Identity', 'The', 'This', 'Card'}
    
    for pattern in name_patterns:
        matches = re.findall(pattern, ocr_text)
        for match in matches:
            name = match.strip()
            # Filter out false positives
            first_word = name.split()[0] if name else ''
            if name and first_word not in excluded and len(name) >= 3:
                result["Name"] = name
                break
        if result["Name"]:
            break
    
    return result


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    logger.info("Initializing AI models...")
    init_models()
    logger.info("Models initialized")


# =====================================================
# Page Routes
# =====================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render the main dashboard."""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "document_types": DOCUMENT_TYPES
    })


@app.get("/liveness", response_class=HTMLResponse)
async def liveness_page(request: Request):
    """Render the face liveness detection page."""
    return templates.TemplateResponse("liveness.html", {
        "request": request,
        "document_types": DOCUMENT_TYPES
    })


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    """Render the verification history page."""
    return templates.TemplateResponse("history.html", {
        "request": request,
        "document_types": DOCUMENT_TYPES
    })


@app.get("/verify/{doc_type}", response_class=HTMLResponse)
async def verify_page(request: Request, doc_type: str):
    """Render the verification page for a specific document type."""
    if doc_type not in DOCUMENT_TYPES:
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "document_types": DOCUMENT_TYPES,
            "error": "Invalid document type"
        })
    
    doc_info = DOCUMENT_TYPES[doc_type]
    return templates.TemplateResponse("verify.html", {
        "request": request,
        "doc_type": doc_type,
        "doc_info": doc_info,
        "document_types": DOCUMENT_TYPES
    })


# =====================================================
# API Routes
# =====================================================

@app.post("/api/verify")
async def verify_document(file: UploadFile = File(...)):
    """API endpoint to verify an uploaded document."""
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type. Use JPG, PNG, WebP, or GIF")
    
    # Check file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size must be less than 16MB")
    
    # Check if models are available
    if not YOLO_AVAILABLE or classifier_model is None:
        # Demo mode with sample data
        return JSONResponse({
            "success": True,
            "demo_mode": True,
            "result": {
                "doc_type": "Aadhaar (Front)",
                "confidence": 0.98,
                "extracted_data": {
                    "Name": "Rajesh Kumar",
                    "DOB": "15/08/1990",
                    "Gender": "Male",
                    "Aadhaar Number": "1234 5678 9012"
                }
            }
        })
    
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(contents)
            filepath = tmp.name
        
        # Step 1: Classify document
        classify_result = classify_document(filepath)
        doc_type = classify_result.get('doc_type', 'Unknown')
        confidence = classify_result.get('confidence', 0)
        
        logger.info(f"Classification: {doc_type} ({confidence:.2%})")
        
        # Step 2: Extract text using Surya OCR
        extracted_data = {}
        if SURYA_AVAILABLE and confidence > 0.5:
            ocr_text = extract_text_surya(filepath)
            logger.info(f"OCR text: {ocr_text[:200]}..." if len(ocr_text) > 200 else f"OCR text: {ocr_text}")
            
            # Parse fields based on document type
            if "aadhaar" in doc_type.lower() or "aadhar" in doc_type.lower():
                extracted_data = parse_aadhaar_fields(ocr_text)
        
        # Step 3: Verify against database
        verification_result = None
        if DATABASE_AVAILABLE and extracted_data.get("Aadhaar Number"):
            verification_result = verify_aadhaar(extracted_data)
            logger.info(f"Database verification: {verification_result.get('status')}")
        
        # Clean up
        os.unlink(filepath)
        
        # Format response
        return JSONResponse({
            "success": True,
            "result": {
                "doc_type": doc_type,
                "confidence": confidence,
                "extracted_data": extracted_data if extracted_data else {
                    "Status": "Document detected",
                    "Note": "Text extraction requires clearer image"
                },
                "verification": verification_result
            }
        })
        
    except Exception as e:
        logger.error(f"Verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/face-match")
async def face_match_endpoint(
    file: UploadFile = File(...),
    selfie: str = Form(None)
):
    """
    API endpoint to match face from document with selfie.
    Selfie should be passed as base64 encoded string in form data.
    """
    logger.info(f"Face match request - selfie provided: {selfie is not None}")
    
    # For demo, return sample result if face matching not available
    if not FACE_MATCHING_AVAILABLE:
        logger.info("Face matching not available, using demo mode")
        return JSONResponse({
            "success": True,
            "demo_mode": True,
            "result": get_demo_match_result()
        })
    
    try:
        # Save document temporarily
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(contents)
            doc_path = tmp.name
        
        logger.info(f"Document saved to: {doc_path}")
        
        # Perform face matching if selfie is provided
        if selfie and len(selfie) > 100:  # Base64 image should be >100 chars
            logger.info(f"Selfie data length: {len(selfie)} chars, performing real match")
            result = match_faces(selfie, doc_path)
            result["demo_mode"] = False
        else:
            logger.info("No selfie provided, using demo result")
            result = get_demo_match_result()
            result["demo_mode"] = True
        
        # Clean up
        os.unlink(doc_path)
        
        logger.info(f"Face match result: match={result.get('match')}, score={result.get('score')}")
        
        return JSONResponse({
            "success": True,
            "result": result
        })
        
    except Exception as e:
        logger.error(f"Face matching error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-pdf")
async def generate_pdf_endpoint(request: Request):
    """Generate a PDF verification certificate."""
    from fastapi.responses import Response
    
    try:
        data = await request.json()
        
        if not PDF_AVAILABLE:
            raise HTTPException(status_code=503, detail="PDF generation not available")
        
        # Generate PDF
        if REPORTLAB_AVAILABLE:
            pdf_bytes = generate_verification_pdf(data)
        else:
            pdf_bytes = generate_simple_pdf(data)
        
        if pdf_bytes is None:
            raise HTTPException(status_code=500, detail="Failed to generate PDF")
        
        # Return PDF
        filename = f"kyc_certificate_{data.get('id', 'unknown')}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "yolo_available": YOLO_AVAILABLE and classifier_model is not None,
        "surya_available": SURYA_AVAILABLE and recognition_predictor is not None,
        "database_available": DATABASE_AVAILABLE,
        "face_matching_available": FACE_MATCHING_AVAILABLE,
        "pdf_available": PDF_AVAILABLE
    }


# =====================================================
# Entry Point
# =====================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("  AI KYC Verification System")
    print("="*60)
    print(f"\n  🌐 Dashboard: http://localhost:5001")
    print(f"  📚 API Docs:  http://localhost:5001/docs")
    print(f"  🔍 YOLO:      {'Available' if YOLO_AVAILABLE else 'Not Available'}")
    print(f"  📝 Surya OCR: {'Available' if SURYA_AVAILABLE else 'Not Available'}")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)
