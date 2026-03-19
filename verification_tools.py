import os
import time
from langchain_core.tools import tool
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from datetime import datetime

# Import our MongoDB functions
import mongo_db as db

# --- Groq LLM Imports ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
# Load .env file for GROQ_API_KEY
load_dotenv()

class GroqSettings(BaseSettings):
    GROQ_API_KEY: str = "YOUR_FALLBACK_KEY"

try:
    groq_settings = GroqSettings()
except Exception as e:
    print(f"Error loading Groq settings: {e}. Please set GROQ_API_KEY in .env")
    
# --- Tool Definitions ---

# --- Level 0 Tools ---
@tool
def verify_phone_otp(phone: str, otp: str) -> dict:
    """Simulates verifying a Phone OTP."""
    print(f"--- Verifying Phone {phone} with OTP {otp} ---")
    time.sleep(0.5) # Simulate network delay
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

# --- Level 1 Tools ---
@tool
def verify_pan(pan_data: dict) -> dict:
    """
    Verifies the extracted PAN JSON data against the mock 'source_pan_holders' DB.
    """
    print(f"--- Verifying PAN {pan_data.get('pan_number')} ---")
    time.sleep(1) # Simulate API delay
    
    # --- Date Conversion ---
    # The JSON will have a string date, but MongoDB was populated with datetime objects.
    # We must convert the string from the JSON to a datetime object for the query.
    try:
        if 'dob' in pan_data:
             pan_data['dob_dt'] = datetime.strptime(pan_data['dob'], '%Y-%m-%d')
    except ValueError:
        return {"status": "Failed", "details": "Invalid DOB format. Expected YYYY-MM-DD."}
    
    record = db.find_pan_record(pan_data)
    
    if not record:
        return {"status": "Failed", "details": "PAN details did not match."}
    
    if record['pan_status'] != "Active":
        return {"status": "Failed", "details": f"PAN status is '{record['pan_status']}'."}
        
    return {"status": "Verified", "details": "PAN details matched and status is Active."}

# --- Level 2 Tools ---
@tool
def verify_aadhaar(aadhaar_data: dict) -> dict:
    """
    Verifies the extracted Aadhaar JSON data against the mock 'source_aadhaar_holders' DB.
    """
    print(f"--- Verifying Aadhaar {aadhaar_data.get('aadhaar_number')} ---")
    time.sleep(1.5) # Simulate API delay
    
    # --- Date Conversion ---
    try:
        if 'dob' in aadhaar_data:
             aadhaar_data['dob_dt'] = datetime.strptime(aadhaar_data['dob'], '%Y-%m-%d')
    except ValueError:
        return {"status": "Failed", "details": "Invalid DOB format. Expected YYYY-MM-DD."}
        
    record = db.find_aadhaar_record(aadhaar_data)
    
    if not record:
        return {"status": "Failed", "details": "Aadhaar details did not match."}
        
    if record['status'] != "Active":
        return {"status": "Failed", "details": f"Aadhaar status is '{record['status']}'."}
    
    # Return the verified address for L3 checks
    return {"status": "Verified", "details": "Aadhaar details matched.", "address": record.get('address')}

@tool
def verify_liveness(selfie_image_path: str) -> dict:
    """MOCK: Simulates a liveness/selfie check."""
    print(f"--- Performing liveness check on {selfie_image_path} ---")
    time.sleep(2) # Simulate ML model processing
    if "good_selfie.jpg" in selfie_image_path.lower():
        return {"status": "Verified", "liveness_score": 0.95, "face_match": True}
    return {"status": "Failed", "details": "Liveness check failed (e.g., spoof, blur)."}

# --- Level 3 Tools ---
@tool
def verify_bank_account(account_number: str, ifsc: str) -> dict:
    """MOCK: Simulates a 'penny drop' bank account verification."""
    print(f"--- Verifying Bank Account {account_number} ---")
    time.sleep(1)
    if account_number and ifsc:
        return {"status": "Verified", "details": "Account holder name matched."}
    return {"status": "Failed", "details": "Bank account verification failed."}

@tool
def verify_address(address_data: dict, aadhaar_address: dict) -> dict:
    """
    MOCK: Simulates matching address from an OCR'd doc (e.g., utility bill)
    against the verified Aadhaar address.
    """
    print(f"--- Verifying Address ---")
    time.sleep(0.5)
    
    if not aadhaar_address:
         return {"status": "Failed", "details": "Aadhaar address not found for comparison."}
         
    # Simple mock check
    if address_data.get('pincode') == aadhaar_address.get('pincode'):
         return {"status": "Verified", "details": "Address Pincode matched Aadhaar."}
    return {"status": "Failed", "details": "Address proof did not match Aadhaar address."}
    
@tool
def verify_video_kyc(video_call_id: str) -> dict:
    """MOCK: Simulates a Video KYC (V-CIP) call completion check."""
    print(f"--- Checking status of video call {video_call_id} ---")
    time.sleep(3) # Simulate agent review
    return {
        "status": "Verified", 
        "details": "Video call completed and approved by agent.",
        "officer_id": "V-AGENT-007",
    }

# --- Level 4 Tools ---
@tool
def verify_income_docs(income_doc_data: dict) -> dict:
    """MOCK: Simulates income verification (e.g., ITR, salary slip)."""
    print(f"--- Verifying Income Docs ---")
    time.sleep(1)
    if income_doc_data.get("annual_income", 0) > 500000:
        return {"status": "Verified", "details": "Income documents look valid."}
    return {"status": "Failed", "details": "Income docs not provided or invalid."}

@tool
def run_llm_aml_pep_check(full_name: str, pan: str) -> dict:
    """
    Performs an LLM-based risk assessment for AML/PEP screening using Groq.
    """
    print(f"--- Running Groq LLM AML/PEP Check for: {full_name} ---")
    
    try:
        llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.1,
            api_key=groq_settings.GROQ_API_KEY
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
You are a senior compliance officer at a financial institution. Your task is to conduct an Anti-Money Laundering (AML) and Politically Exposed Person (PEP) risk assessment.

Analyze the user's name.
- Simulate a search against known sanctions lists, PEP databases, or adverse media mentions.
- Provide a concise one-paragraph summary of your findings.
- Conclude with a final, clear risk rating on its own line, in the format: "RISK: [Clear | Review | High]"

**Simulated Watchlist Data (for this exercise only):**
- 'Adithya Vardan M': Clear.
- 'Jane Doe': Clear.
- 'Robert K Mueller': Potential PEP match (US Politician).
- 'Vikram Singh': Potential adverse media (fraud allegations).
- 'Chang Wei': Potential sanctions list match.
"""),
            ("user", "Please perform risk assessment for the following individual:\nName: {full_name}\nPAN: {pan}")
        ])

        chain = prompt_template | llm | StrOutputParser()
        
        response = chain.invoke({
            "full_name": full_name,
            "pan": pan
        })

        print(f"--- Groq LLM Response ---\n{response}\n-------------------------")
        
        # Parse the LLM's response
        if "RISK: CLEAR" in response.upper():
            return {"status": "Clear", "details": response}
        elif "RISK: REVIEW" in response.upper() or "RISK: MEDIUM" in response.upper():
            return {"status": "Review", "details": response}
        elif "RISK: HIGH" in response.upper():
            return {"status": "High", "details": response}
        else:
            return {"status": "Review", "details": f"LLM analysis complete, but risk rating was unclear. Review required.\n{response}"}

    except Exception as e:
        print(f"Error during Groq AML check: {e}")
        return {"status": "Failed", "details": f"AML check failed due to an API error: {e}"}

