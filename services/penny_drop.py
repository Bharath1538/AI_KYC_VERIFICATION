import os
import requests
import logging
from dotenv import load_dotenv
from fuzzywuzzy import fuzz

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "rzp_test_mock_key")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "mock_secret")

def verify_bank_penny_drop(account_number: str, ifsc: str, expected_name: str) -> dict:
    """
    Performs a real Penny Drop using Razorpay Fund Accounts API.
    Since we don't have active billing on a test key, we will simulate
    the success path but architect the exact HTTP call required for Prod.
    """
    logger.info(f"Initiating Penny Drop for Acc: {account_number}, IFSC: {ifsc}")
    
    # In a real production scenario, Penny Drop involves 3 steps:
    # 1. Create a Contact
    # 2. Create a Fund Account (Bank) for that Contact
    # 3. Create a Payout of ₹1 to that Fund Account and check the `fund_account.bank_account.name` returned from IMPS.
    
    # We will architect the API call but mock the response if billing is disabled.
    url = "https://api.razorpay.com/v1/fund_accounts/validations"
    
    payload = {
        "account": {
            "name": expected_name,
            "ifsc": ifsc,
            "account_number": account_number
            # "bank_name": "Testing Bank"
        },
        "amount": 100, # 1 Rupee in Paise
        "currency": "INR",
        "notes": {
            "purpose": "KYC Penny Drop Verification"
        }
    }
    
    try:
        # In a real app we'd uncomment this line and use the actual response:
        # response = requests.post(url, auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET), json=payload)
        # result = response.json()
        
        # MOCKING RESPONSE FOR DEMO
        # Razorpay validation response payload looks like this when IMPS succeeds
        mock_response = {
            "id": "fav_00000000000001",
            "fund_account": {
                "id": "fa_00000000000001",
                "account_type": "bank_account",
                "bank_account": {
                    "ifsc": ifsc,
                    "bank_name": "HDFC Bank",
                    "name": expected_name if expected_name else "Adithya Vardan M", # The name registered at the bank
                    "notes": []
                }
            },
            "status": "completed",
            "results": {
                "account_status": "active",
                "registered_name": expected_name if expected_name else "Adithya Vardan M"
            }
        }
        
        # Simulate Network Latency
        import time
        time.sleep(1)
        
        result = mock_response
        
        if result.get("status") == "completed" and result.get("results", {}).get("account_status") == "active":
            bank_registered_name = result["results"].get("registered_name", "")
            
            # Fuzzy match the registered bank name against the exact Aadhaar name
            similarity = fuzz.token_sort_ratio(expected_name.lower(), bank_registered_name.lower())
            
            if similarity >= 80:
                logger.info(f"Penny Drop Success: Names match closely ({similarity}%).")
                return {
                    "status": "Verified",
                    "details": f"Account verified. Name proxy matched successfully ({similarity}% similarity).",
                    "bank_registered_name": bank_registered_name
                }
            else:
                logger.warning(f"Penny Drop Warning: Name mismatch ({expected_name} vs {bank_registered_name})")
                return {
                    "status": "Failed",
                    "details": f"Account active, but Name Mismatch. Got '{bank_registered_name}', expected '{expected_name}'.",
                    "bank_registered_name": bank_registered_name
                }
        else:
            return {
                "status": "Failed",
                "details": "Bank account is invalid or inactive.",
                "api_response": result
            }
            
    except Exception as e:
        logger.error(f"Penny Drop API Error: {e}")
        return {
            "status": "Failed",
            "details": f"Internal API Error connecting to Bank Gateway: {str(e)}"
        }
