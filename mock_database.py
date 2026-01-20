"""
Mock Database for KYC Verification

This simulates a government/UIDAI database for Aadhaar verification.
In production, this would connect to actual verification APIs.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re


class VerificationStatus(Enum):
    """Verification result status."""
    VERIFIED = "verified"           # All details match
    PARTIAL_MATCH = "partial_match" # Some details match
    NOT_FOUND = "not_found"         # Aadhaar number not in database
    MISMATCH = "mismatch"           # Details don't match
    INVALID = "invalid"             # Invalid Aadhaar format


@dataclass
class AadhaarRecord:
    """Aadhaar database record."""
    aadhaar_number: str
    name: str
    dob: str
    gender: str
    address: str = ""
    phone: str = ""
    is_active: bool = True


# ============================================
# MOCK DATABASE - Sample Aadhaar Records
# ============================================
# In production, this would be an API call to UIDAI
MOCK_AADHAAR_DATABASE: Dict[str, AadhaarRecord] = {
    # Sample records for testing
    "2114 5270 9955": AadhaarRecord(
        aadhaar_number="2114 5270 9955",
        name="Kanika Manocha",
        dob="11/09/1993",
        gender="Female",
        address="Delhi, India",
        phone="9876543210"
    ),
    "6106 2339 9374": AadhaarRecord(
        aadhaar_number="6106 2339 9374",
        name="Adithya Vardan M",
        dob="20/10/2004",
        gender="Male",
        address="Hyderabad, India",
        phone="9876543211"
    ),
    "8016 2213 9063": AadhaarRecord(
        aadhaar_number="8016 2213 9063",
        name="Sajan A",
        dob="03/03/2005",
        gender="Male",
        address="Chennai, India",
        phone="9876543212"
    ),
    # Add more sample records for demo
    "1234 5678 9012": AadhaarRecord(
        aadhaar_number="1234 5678 9012",
        name="Rajesh Kumar",
        dob="15/08/1990",
        gender="Male",
        address="Mumbai, India",
        phone="9876543213"
    ),
    "9876 5432 1098": AadhaarRecord(
        aadhaar_number="9876 5432 1098",
        name="Priya Sharma",
        dob="25/12/1995",
        gender="Female",
        address="Bangalore, India",
        phone="9876543214"
    ),
}


def normalize_aadhaar(aadhaar: str) -> str:
    """Normalize Aadhaar number format (remove extra spaces, standardize)."""
    if not aadhaar:
        return ""
    # Remove all non-digits
    digits = re.sub(r'\D', '', aadhaar)
    # Format as XXXX XXXX XXXX
    if len(digits) >= 12:
        return f"{digits[:4]} {digits[4:8]} {digits[8:12]}"
    return aadhaar


def normalize_name(name: str) -> str:
    """Normalize name for comparison."""
    if not name:
        return ""
    # Remove extra spaces, convert to title case
    return " ".join(name.strip().split()).title()


def verify_aadhaar(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify extracted Aadhaar data against the mock database.
    
    Returns verification result with status and details.
    """
    result = {
        "status": VerificationStatus.INVALID.value,
        "message": "Invalid data provided",
        "match_score": 0,
        "matched_fields": [],
        "mismatched_fields": [],
        "database_record": None
    }
    
    # Get and normalize Aadhaar number
    aadhaar_number = extracted_data.get("Aadhaar Number", "")
    if not aadhaar_number:
        result["message"] = "No Aadhaar number provided"
        return result
    
    normalized_aadhaar = normalize_aadhaar(aadhaar_number)
    
    # Check if Aadhaar exists in database
    if normalized_aadhaar not in MOCK_AADHAAR_DATABASE:
        # Try partial match (in case OCR missed a digit)
        partial_match = None
        for db_aadhaar in MOCK_AADHAAR_DATABASE.keys():
            if normalized_aadhaar[:11] in db_aadhaar or db_aadhaar[:11] in normalized_aadhaar:
                partial_match = db_aadhaar
                break
        
        if partial_match:
            normalized_aadhaar = partial_match
        else:
            result["status"] = VerificationStatus.NOT_FOUND.value
            result["message"] = f"Aadhaar number {normalized_aadhaar} not found in database"
            return result
    
    # Get database record
    db_record = MOCK_AADHAAR_DATABASE[normalized_aadhaar]
    
    # Check if Aadhaar is active
    if not db_record.is_active:
        result["status"] = VerificationStatus.INVALID.value
        result["message"] = "This Aadhaar card has been deactivated"
        return result
    
    # Compare fields
    matched_fields = []
    mismatched_fields = []
    
    # Compare Name
    extracted_name = normalize_name(extracted_data.get("Name", ""))
    db_name = normalize_name(db_record.name)
    if extracted_name and db_name:
        # Check if names are similar (partial match allowed)
        if extracted_name.lower() == db_name.lower():
            matched_fields.append("Name")
        elif extracted_name.lower() in db_name.lower() or db_name.lower() in extracted_name.lower():
            matched_fields.append("Name (partial)")
        else:
            mismatched_fields.append(("Name", extracted_name, db_name))
    
    # Compare DOB
    extracted_dob = extracted_data.get("DOB", "")
    if extracted_dob and db_record.dob:
        if extracted_dob == db_record.dob:
            matched_fields.append("DOB")
        else:
            mismatched_fields.append(("DOB", extracted_dob, db_record.dob))
    
    # Compare Gender
    extracted_gender = extracted_data.get("Gender", "")
    if extracted_gender and db_record.gender:
        if extracted_gender.lower() == db_record.gender.lower():
            matched_fields.append("Gender")
        else:
            mismatched_fields.append(("Gender", extracted_gender, db_record.gender))
    
    # Calculate match score
    total_fields = len(matched_fields) + len(mismatched_fields)
    match_score = (len(matched_fields) / total_fields * 100) if total_fields > 0 else 0
    
    # Determine final status
    if len(mismatched_fields) == 0 and len(matched_fields) >= 2:
        status = VerificationStatus.VERIFIED
        message = "✅ Identity verified successfully! All details match."
    elif len(matched_fields) > 0 and len(mismatched_fields) > 0:
        status = VerificationStatus.PARTIAL_MATCH
        message = f"⚠️ Partial match. {len(matched_fields)} fields matched, {len(mismatched_fields)} fields differ."
    elif len(mismatched_fields) > 0:
        status = VerificationStatus.MISMATCH
        message = "❌ Details do not match database records."
    else:
        status = VerificationStatus.VERIFIED
        message = "✅ Aadhaar number verified in database."
    
    result = {
        "status": status.value,
        "message": message,
        "match_score": round(match_score, 1),
        "matched_fields": matched_fields,
        "mismatched_fields": [{"field": f[0], "extracted": f[1], "expected": f[2]} for f in mismatched_fields],
        "database_record": {
            "name": db_record.name,
            "dob": db_record.dob,
            "gender": db_record.gender,
            "aadhaar_masked": f"XXXX XXXX {normalized_aadhaar[-4:]}"
        }
    }
    
    return result
