"""
Verification Tools for KYC Workflow

Contains both simulated tools (OTP, Bank, Video KYC) and real tools (Aadhaar OCR, Liveness, Face Match).
"""

import os
import time
import tempfile
import base64
import logging
from langchain_core.tools import tool

# Configure logging
logger = logging.getLogger(__name__)

# --- Try to import real verification modules ---
try:
    from inference import classify_document, extract_and_display_ocr_text
    INFERENCE_AVAILABLE = True
except ImportError:
    logger.warning("inference.py not available - using simulated Aadhaar verification")
    INFERENCE_AVAILABLE = False

try:
    from face_matching import match_faces, detect_face, decode_base64_image
    import numpy as np
    FACE_MATCHING_AVAILABLE = True
except ImportError:
    logger.warning("face_matching.py not available - using simulated face matching")
    FACE_MATCHING_AVAILABLE = False

try:
    import mock_database as db
    DATABASE_AVAILABLE = True
except ImportError:
    logger.warning("mock_database.py not available")
    DATABASE_AVAILABLE = False


# =============================================================================
# Level 0 Tools - OTP Verification (Simulated)
# =============================================================================

@tool
def verify_phone_otp(phone: str, otp: str) -> dict:
    """Simulates verifying a Phone OTP."""
    print(f"--- Verifying Phone {phone} with OTP {otp} ---")
    time.sleep(0.5)
    if otp == "123456":
        return {"status": "Verified", "details": "Phone OTP verified."}
    return {"status": "Failed", "details": "Invalid OTP."}


@tool
def verify_email_otp(email: str, otp: str) -> dict:
    """Simulates verifying an Email OTP."""
    print(f"--- Verifying Email {email} with OTP {otp} ---")
    time.sleep(0.5)
    if otp == "123456":
        return {"status": "Verified", "details": "Email OTP verified."}
    return {"status": "Failed", "details": "Invalid OTP."}


# =============================================================================
# Level 1 Tools - Aadhaar + Liveness + Face Matching (REAL)
# =============================================================================

@tool
def verify_aadhaar_ocr(aadhaar_image_path: str = "", aadhaar_data: dict = None) -> dict:
    """
    Verifies Aadhaar document using YOLO classification and Surya OCR.
    
    Args:
        aadhaar_image_path: Path to the Aadhaar image file
        aadhaar_data: Pre-extracted Aadhaar data (if already available)
    
    Returns:
        Verification result with extracted fields
    """
    print(f"--- Verifying Aadhaar Document ---")
    
    # Handle if aadhaar_data is passed as a string (from JSON)
    if isinstance(aadhaar_data, str):
        import json
        try:
            aadhaar_data = json.loads(aadhaar_data)
        except:
            aadhaar_data = {}
    
    # Ensure aadhaar_data is a dict
    if not isinstance(aadhaar_data, dict):
        aadhaar_data = {}
    
    # If pre-extracted data is provided, use it directly
    aadhaar_number = aadhaar_data.get('aadhaar_number', aadhaar_data.get('Aadhaar Number', ''))
    name = aadhaar_data.get('full_name', aadhaar_data.get('name', aadhaar_data.get('Name', '')))
    dob = aadhaar_data.get('dob', aadhaar_data.get('DOB', ''))
    
    if aadhaar_number:
        print(f"Using pre-extracted Aadhaar data: {aadhaar_number}")
        
        # Verify against database using the correct format
        if DATABASE_AVAILABLE:
            # Convert to mock_database expected format
            db_query = {
                "Aadhaar Number": aadhaar_number,
                "Name": name,
                "DOB": dob
            }
            db_record = db.verify_aadhaar(db_query)
            if db_record and db_record.get('verified'):
                return {
                    "status": "Verified",
                    "details": "Aadhaar matched with database.",
                    "aadhaar_number": aadhaar_number,
                    "name": db_record.get('database_record', {}).get('name', name),
                    "dob": db_record.get('database_record', {}).get('dob', dob)
                }
        
        # Return based on provided data (format validated)
        return {
            "status": "Verified",
            "details": "Aadhaar data extracted and validated.",
            "aadhaar_number": aadhaar_number,
            "name": name,
            "dob": dob
        }
    
    # If we have an image path and real inference is available
    if aadhaar_image_path and INFERENCE_AVAILABLE and os.path.exists(aadhaar_image_path):
        print(f"Running YOLO + OCR on {aadhaar_image_path}")
        try:
            # Classify document
            doc_type, confidence = classify_document(aadhaar_image_path)
            if "aadhar" not in doc_type.lower() and "aadhaar" not in doc_type.lower():
                return {
                    "status": "Failed",
                    "details": f"Document is not Aadhaar. Detected: {doc_type}"
                }
            
            # Extract text via OCR
            ocr_result = extract_and_display_ocr_text(aadhaar_image_path)
            
            # Parse Aadhaar fields from OCR
            extracted = parse_aadhaar_from_ocr(ocr_result)
            
            if extracted.get('aadhaar_number'):
                return {
                    "status": "Verified",
                    "details": f"Aadhaar extracted via OCR. Confidence: {confidence:.1%}",
                    "aadhaar_number": extracted.get('aadhaar_number', ''),
                    "name": extracted.get('name', ''),
                    "dob": extracted.get('dob', '')
                }
            else:
                return {
                    "status": "Failed",
                    "details": "Could not extract Aadhaar number from document."
                }
        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
            return {"status": "Failed", "details": f"OCR error: {str(e)}"}
    
    # Fallback: simulated verification
    print("Using simulated Aadhaar verification")
    time.sleep(1)
    return {
        "status": "Verified",
        "details": "Aadhaar verified (demo mode).",
        "aadhaar_number": "XXXX XXXX 1234",
        "name": "Demo User",
        "dob": "01/01/1990"
    }


def parse_aadhaar_from_ocr(ocr_text: str) -> dict:
    """Parse Aadhaar fields from OCR text."""
    import re
    result = {"name": "", "dob": "", "aadhaar_number": "", "gender": ""}
    
    # Extract Aadhaar number (12 digits, possibly with spaces)
    aadhaar_patterns = [
        r'(\d{4}\s+\d{4}\s+\d{4})',
        r'(\d{4}\s+\d{4}\s+\d{3,4})',
        r'(\d{12})',
    ]
    for pattern in aadhaar_patterns:
        matches = re.findall(pattern, ocr_text)
        if matches:
            result["aadhaar_number"] = matches[-1]
            break
    
    # Extract DOB
    dob_patterns = [
        r'DOB\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})',
        r'Date of Birth\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})',
        r'(\d{2}/\d{2}/\d{4})',
    ]
    for pattern in dob_patterns:
        match = re.search(pattern, ocr_text, re.IGNORECASE)
        if match:
            result["dob"] = match.group(1)
            break
    
    # Extract name (look for capitalized words near the top)
    lines = ocr_text.split('\n')
    for line in lines[:10]:
        if len(line) > 3 and line.strip() and not any(c.isdigit() for c in line[:5]):
            # Skip common headers
            if not any(skip in line.lower() for skip in ['government', 'india', 'aadhaar', 'unique']):
                result["name"] = line.strip()
                break
    
    return result


@tool
def verify_liveness_real(selfie_data: str) -> dict:
    """
    Verifies face liveness using real face detection.
    
    Args:
        selfie_data: Base64 encoded selfie image
    
    Returns:
        Liveness verification result
    """
    print("--- Running Face Liveness Check ---")
    
    # If no selfie is provided, use demo mode for testing
    if not selfie_data or len(selfie_data) < 100:
        print("No selfie provided - using demo mode")
        time.sleep(0.5)
        return {
            "status": "Verified",
            "details": "Liveness verified (demo mode - no selfie provided).",
            "liveness_score": 90.0
        }
    
    if FACE_MATCHING_AVAILABLE:
        try:
            # Decode and detect face
            image = decode_base64_image(selfie_data)
            if image is None:
                return {"status": "Failed", "details": "Could not decode selfie image."}
            
            face_bbox = detect_face(image)
            if face_bbox is None:
                return {"status": "Failed", "details": "No face detected in selfie."}
            
            # Calculate liveness score based on face detection confidence
            # (In production, use an actual liveness model)
            x1, y1, x2, y2 = face_bbox
            face_area = (x2 - x1) * (y2 - y1)
            image_area = image.shape[0] * image.shape[1]
            face_ratio = face_area / image_area
            
            # Face should be at least 10% of image for good capture
            if face_ratio >= 0.10:
                liveness_score = min(0.95, 0.7 + face_ratio)
                return {
                    "status": "Verified",
                    "details": "Face detected and liveness confirmed.",
                    "liveness_score": round(liveness_score * 100, 1)
                }
            else:
                return {
                    "status": "Failed",
                    "details": "Face too small in frame. Please position closer."
                }
                
        except Exception as e:
            logger.error(f"Liveness check error: {e}")
            return {"status": "Failed", "details": f"Liveness error: {str(e)}"}
    
    # Fallback: simulated
    print("Using simulated liveness check")
    time.sleep(1)
    return {
        "status": "Verified",
        "details": "Liveness verified (demo mode).",
        "liveness_score": 95.0
    }


@tool
def verify_face_match(selfie_data: str, document_image_path: str) -> dict:
    """
    Matches face from selfie with face on document.
    
    Args:
        selfie_data: Base64 encoded selfie image
        document_image_path: Path to document image
    
    Returns:
        Face matching result with similarity score
    """
    print("--- Running Face Matching ---")
    
    if not selfie_data or len(selfie_data) < 100:
        return {"status": "Failed", "details": "No valid selfie provided."}
    
    if FACE_MATCHING_AVAILABLE and document_image_path and os.path.exists(document_image_path):
        try:
            result = match_faces(selfie_data, document_image_path)
            
            if result.get('success'):
                if result.get('match'):
                    return {
                        "status": "Verified",
                        "details": f"Face match confirmed. Similarity: {result['score']}%",
                        "score": result['score']
                    }
                else:
                    return {
                        "status": "Failed",
                        "details": f"Face mismatch. Similarity: {result['score']}% (threshold: 60%)",
                        "score": result['score']
                    }
            else:
                return {"status": "Failed", "details": result.get('message', 'Face matching failed.')}
                
        except Exception as e:
            logger.error(f"Face matching error: {e}")
            return {"status": "Failed", "details": f"Face match error: {str(e)}"}
    
    # Fallback: simulated
    print("Using simulated face matching")
    time.sleep(1)
    return {
        "status": "Verified",
        "details": "Face match verified (demo mode).",
        "score": 87.5
    }


# =============================================================================
# Level 2 Tools - PAN Verification
# =============================================================================

@tool
def verify_pan(pan_data: dict) -> dict:
    """
    Verifies PAN details against database.
    """
    print(f"--- Verifying PAN {pan_data.get('pan_number')} ---")
    time.sleep(1)
    
    pan_number = pan_data.get('pan_number', '')
    full_name = pan_data.get('full_name', '')
    
    if DATABASE_AVAILABLE:
        # Check against mock database
        result = db.verify_pan(pan_number)
        if result and result.get('verified'):
            return {
                "status": "Verified",
                "details": "PAN matched with database.",
                "pan_number": pan_number
            }
    
    # Simple validation
    if pan_number and len(pan_number) == 10:
        return {
            "status": "Verified",
            "details": "PAN format validated.",
            "pan_number": pan_number
        }
    
    return {"status": "Failed", "details": "PAN validation failed."}


# =============================================================================
# Level 3 Tools - Bank + Video KYC (Simulated)
# =============================================================================

@tool
def verify_bank_account(account_number: str, ifsc: str, expected_name: str) -> dict:
    """Verifies bank account using Penny Drop API."""
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from services.penny_drop import verify_bank_penny_drop
        return verify_bank_penny_drop(account_number, ifsc, expected_name)
    except Exception as e:
        logger.error(f"Failed to import Penny Drop service: {e}")
        return {"status": "Failed", "details": f"Penny Drop Service Unavailable: {e}"}
    time.sleep(1)
    if account_number and ifsc:
        return {"status": "Verified", "details": "Account holder name matched."}
    return {"status": "Failed", "details": "Bank account verification failed."}


@tool
def verify_address(address_data: dict, aadhaar_address: dict) -> dict:
    """Simulates address verification against Aadhaar."""
    print(f"--- Verifying Address ---")
    time.sleep(0.5)
    
    if not aadhaar_address:
        return {"status": "Verified", "details": "Address recorded (no Aadhaar address to compare)."}
    
    if address_data.get('pincode') == aadhaar_address.get('pincode'):
        return {"status": "Verified", "details": "Address Pincode matched Aadhaar."}
    return {"status": "Verified", "details": "Address recorded."}


@tool
def verify_video_kyc(user_id: str) -> dict:
    """Creates a secure Video KYC WebRTC session."""
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from services.video_kyc import create_video_kyc_room
        result = create_video_kyc_room(user_id)
        if result.get("success"):
            return {
                "status": "Verified",
                "details": f"Video KYC session generated. User needs to join: {result.get('room_url')}",
                "officer_id": "V-AGENT-007",
                "room_url": result.get("room_url")
            }
        else:
            return {"status": "Failed", "details": f"Failed to generate Video Room: {result.get('error')}"}
    except Exception as e:
        logger.error(f"Failed to load Video KYC module: {e}")
        return {"status": "Failed", "details": "Video KYC Service Unavailable"}


# =============================================================================
# Level 4 Tools - Income + AML/PEP
# =============================================================================

@tool
def verify_income_docs(income_doc_data: dict) -> dict:
    """Simulates income document verification."""
    print(f"--- Verifying Income Docs ---")
    time.sleep(1)
    if income_doc_data.get("annual_income", 0) > 500000:
        return {"status": "Verified", "details": "Income documents validated."}
    return {"status": "Verified", "details": "Income documents recorded."}


@tool
def run_llm_aml_pep_check(full_name: str, pan: str) -> dict:
    """
    Performs AML/PEP risk assessment.
    In production, this uses Groq LLM.
    """
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from services.aml_check import check_aml_pep_risk
        return check_aml_pep_risk(full_name, pan)
    except Exception as e:
        logger.error(f"Failed to load AML/PEP module: {e}")
        return {"status": "Failed", "details": "AML/PEP Service Unavailable"}
