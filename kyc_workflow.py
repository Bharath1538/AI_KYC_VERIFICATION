"""
KYC Workflow - Staged Verification using LangGraph

Levels:
- L0: Lite (OTP verification)
- L1: Basic (Aadhaar OCR + Face Liveness + Face Matching)
- L2: Verified (PAN verification)
- L3: Full (Bank + Address + Video KYC)
- L4: EDD (Income + AML/PEP)
"""

import mock_database as db
from verification_tools import (
    verify_phone_otp, verify_email_otp, verify_pan,
    verify_aadhaar_ocr, verify_liveness_real, verify_face_match,
    verify_bank_account, verify_video_kyc,
    verify_address, verify_income_docs, run_llm_aml_pep_check
)
from typing import TypedDict, Annotated, List, Dict, Any
import operator
from langgraph.graph import StateGraph, END


class StagedVerificationState(TypedDict):
    """The state object for our graph."""
    request_data: Dict[str, Any]
    user_profile: Dict[str, Any]
    verification_log: Annotated[List[str], operator.add]
    error_message: str | None


def log(msg: str) -> dict:
    """Helper function to append to the log."""
    print(msg)
    return {"verification_log": [msg]}


def load_user_profile(state: StagedVerificationState) -> dict:
    """Loads the user's profile from the mock DB."""
    user_id = state['request_data']['user_id']
    try:
        profile = db.get_user_by_aadhaar(user_id)
        if not profile:
            # Try to find or create a new user
            profile = {
                "user_id": user_id,
                "kyc_level": -1,
                "verified_data": {},
                "full_name": None,
                "dob": None
            }
        
        log_msg = f"Loaded profile for {user_id}. Current Level: {profile.get('kyc_level', -1)}"
        print(log_msg)
        return {
            "user_profile": profile,
            "verification_log": [log_msg]
        }
    except Exception as e:
        error_msg = f"Failed to load profile for {user_id}: {e}"
        print(error_msg)
        return {"error_message": error_msg}


def run_level_0_otp(state: StagedVerificationState) -> dict:
    """Level 0: Phone + Email OTP."""
    try:
        data = state['request_data']['data']['level0']
        profile = state['user_profile']

        phone_res = verify_phone_otp.invoke({"phone": data['phone'], "otp": data.get('phone_otp', '')})
        email_res = verify_email_otp.invoke({"email": data['email'], "otp": data.get('email_otp', '')})

        if phone_res['status'] == "Verified" and email_res['status'] == "Verified":
            profile['kyc_level'] = 0
            profile['verified_data']['phone'] = data['phone']
            profile['verified_data']['email'] = data['email']
            msg = "✅ Level 0 (Lite) VERIFIED - Phone & Email confirmed."
            print(msg)
            return {"user_profile": profile, "verification_log": [msg]}
        else:
            msg = f"❌ Level 0 Failed. Phone: {phone_res['details']} Email: {email_res['details']}"
            print(msg)
            return {"error_message": msg, "verification_log": [msg]}
    except Exception as e:
        error_msg = f"Error in Level 0 processing: {e}"
        print(error_msg)
        return {"error_message": error_msg, "verification_log": [error_msg]}


def run_level_1_aadhaar_verify(state: StagedVerificationState) -> dict:
    """Level 1: Aadhaar OCR + Face Liveness + Face Matching.
    
    This is the REAL verification using:
    - YOLO for document classification
    - Surya OCR for text extraction
    - Face detection for liveness
    - Face matching for identity confirmation
    """
    try:
        data = state['request_data']['data']['level1']
        profile = state['user_profile']
        
        # Step 1: Verify Aadhaar document via OCR
        msg = "🔍 Running Aadhaar document verification..."
        print(msg)
        
        aadhaar_res = verify_aadhaar_ocr.invoke({
            "aadhaar_image_path": data.get('aadhaar_image_path', ''),
            "aadhaar_data": data.get('aadhaar_data', {})
        })
        
        if aadhaar_res['status'] != "Verified":
            msg = f"❌ Level 1 Failed. Aadhaar: {aadhaar_res['details']}"
            print(msg)
            return {"error_message": msg, "verification_log": [msg]}
        
        # Step 2: Verify liveness (face detection in selfie)
        msg = "📸 Running face liveness check..."
        print(msg)
        
        liveness_res = verify_liveness_real.invoke({
            "selfie_data": data.get('selfie_data', '')
        })
        
        if liveness_res['status'] != "Verified":
            msg = f"❌ Level 1 Failed. Liveness: {liveness_res['details']}"
            print(msg)
            return {"error_message": msg, "verification_log": [msg]}
        
        # Step 3: Match face from selfie with Aadhaar photo
        msg = "🔗 Running face matching..."
        print(msg)
        
        face_match_res = verify_face_match.invoke({
            "selfie_data": data.get('selfie_data', ''),
            "document_image_path": data.get('aadhaar_image_path', '')
        })
        
        if face_match_res['status'] != "Verified":
            msg = f"❌ Level 1 Failed. Face Match: {face_match_res['details']}"
            print(msg)
            return {"error_message": msg, "verification_log": [msg]}
        
        # All checks passed!
        profile['kyc_level'] = 1
        profile['verified_data']['aadhaar'] = aadhaar_res.get('aadhaar_number', '')
        profile['verified_data']['liveness_score'] = liveness_res.get('liveness_score', 0)
        profile['verified_data']['face_match_score'] = face_match_res.get('score', 0)
        profile['full_name'] = aadhaar_res.get('name', '')
        profile['dob'] = aadhaar_res.get('dob', '')
        
        msg = f"✅ Level 1 (Basic) VERIFIED - Aadhaar confirmed for {profile['full_name']}"
        print(msg)
        return {"user_profile": profile, "verification_log": [msg]}
        
    except Exception as e:
        error_msg = f"Error in Level 1 processing: {e}"
        print(error_msg)
        return {"error_message": error_msg, "verification_log": [error_msg]}


def run_level_2_pan_verify(state: StagedVerificationState) -> dict:
    """Level 2: PAN verification."""
    from fuzzywuzzy import fuzz
    
    try:
        data = state['request_data']['data']['level2']
        profile = state['user_profile']

        # Ensure names match between Aadhaar (L1) and PAN (L2)
        expected_name = profile.get('full_name', '').lower()
        provided_name = data.get('full_name', '').lower()
        
        if expected_name and provided_name:
            name_similarity = fuzz.token_sort_ratio(expected_name, provided_name)
            if name_similarity < 80:
                msg = f"❌ Level 2 Failed. PAN name '{provided_name}' does not match Aadhaar name '{expected_name}' (Similarity: {name_similarity}%)."
                print(msg)
                return {"error_message": msg, "verification_log": [msg]}

        pan_res = verify_pan.invoke({"pan_data": data})

        if pan_res['status'] == "Verified":
            profile['kyc_level'] = 2
            profile['verified_data']['pan'] = data.get('pan_number', '')
            msg = f"✅ Level 2 (Verified) VERIFIED - PAN confirmed for {provided_name.title()}."
            print(msg)
            return {"user_profile": profile, "verification_log": [msg]}
        else:
            msg = f"❌ Level 2 Failed. {pan_res['details']}"
            print(msg)
            return {"error_message": msg, "verification_log": [msg]}
    except Exception as e:
        error_msg = f"Error in Level 2 processing: {e}"
        print(error_msg)
        return {"error_message": error_msg, "verification_log": [error_msg]}


def run_level_3_video_kyc(state: StagedVerificationState) -> dict:
    """Level 3: Full (Video KYC)."""
    try:
        data = state['request_data']['data']['level3']
        profile = state['user_profile']

        bank_res = verify_bank_account.invoke({
            "account_number": data['bank_account'], 
            "ifsc": data['ifsc'],
            "expected_name": profile.get("full_name", "")
        })
        addr_res = verify_address.invoke({"address_data": data['address_data'], "aadhaar_address": profile['verified_data'].get('aadhaar_address', {})})
        video_res = verify_video_kyc.invoke({"user_id": profile.get("user_id", "Unknown User")})

        if bank_res['status'] == "Verified" and addr_res['status'] == "Verified" and video_res['status'] == "Verified":
            profile['kyc_level'] = 3
            profile['verified_data']['bank_account'] = data['bank_account']
            profile['verified_data']['address'] = data['address_data']
            msg = "✅ Level 3 (Full V-CIP) VERIFIED."
            print(msg)
            return {"user_profile": profile, "verification_log": [msg]}
        else:
            msg = f"❌ Level 3 Failed. Bank: {bank_res['details']} Address: {addr_res['details']} Video: {video_res['details']}"
            print(msg)
            return {"error_message": msg, "verification_log": [msg]}
    except Exception as e:
        error_msg = f"Error in Level 3 processing: {e}"
        print(error_msg)
        return {"error_message": error_msg, "verification_log": [error_msg]}


def run_level_4_edd(state: StagedVerificationState) -> dict:
    """Level 4: Enhanced Due Diligence (EDD)."""
    try:
        data = state['request_data']['data']['level4']
        profile = state['user_profile']
        
        income_res = verify_income_docs.invoke({"income_doc_data": data['income_data']})
        aml_res = run_llm_aml_pep_check.invoke({
            "full_name": profile.get('full_name', ''),
            "pan": profile['verified_data'].get('pan', '')
        })
        
        if income_res['status'] == "Verified" and aml_res['status'] == "Clear":
            profile['kyc_level'] = 4
            profile['verified_data']['income_verified'] = True
            profile['verified_data']['aml_check'] = aml_res
            msg = "✅ Level 4 (EDD) VERIFIED - Full due diligence complete."
            print(msg)
            return {"user_profile": profile, "verification_log": [msg]}
        else:
            msg = f"❌ Level 4 Failed. Income: {income_res['details']} AML/PEP: {aml_res['details']}"
            print(msg)
            return {"error_message": msg, "verification_log": [msg]}
    except Exception as e:
        error_msg = f"Error in Level 4 processing: {e}"
        print(error_msg)
        return {"error_message": error_msg, "verification_log": [error_msg]}


def save_profile_and_finish(state: StagedVerificationState) -> dict:
    """Final node: Saves the updated profile back to the 'DB'."""
    profile = state['user_profile']
    try:
        msg = f"🎉 Verification complete. User '{profile.get('user_id', 'unknown')}' is now at Level {profile.get('kyc_level', -1)}."
        print(msg)
        # Save to mock database
        db.save_verification_result(profile)
        return {"verification_log": [msg]}
    except Exception as e:
        error_msg = f"Failed to save profile: {e}"
        print(error_msg)
        return {"error_message": error_msg, "verification_log": [error_msg]}


def verification_failed(state: StagedVerificationState) -> dict:
    """Error node: Logs the failure and updates DB with failure status."""
    profile = state.get("user_profile", {})
    error_msg = state.get("error_message", "Unknown error")
    
    msg = f"🚫 Verification FAILED. Error: {error_msg}"
    print(msg)
    
    try:
        if profile and profile.get("user_id"):
            profile["verification_status"] = "failed"
            profile["failure_reason"] = error_msg
            db.save_verification_result(profile)
    except Exception as e:
        print(f"Failed to save failed profile to DB: {e}")
    
    return {"verification_log": [msg]}


# --- Conditional Router ---
def route_verification_step(state: StagedVerificationState) -> str:
    """Decision maker: returns the next node to execute."""
    
    if state.get("error_message"):
        return "verification_failed"
    
    current = state['user_profile'].get('kyc_level', -1)
    target = state['request_data']['target_level']
    
    if current >= target:
        return "save_profile_and_finish"
    
    # L0: OTP Verification
    if current == -1 and target >= 0:
        return "run_level_0_otp"
    # L1: Aadhaar + Liveness + Face Matching (SWAPPED)
    if current == 0 and target >= 1:
        return "run_level_1_aadhaar_verify"
    # L2: PAN Verification (SWAPPED)
    if current == 1 and target >= 2:
        return "run_level_2_pan_verify"
    # L3: Video KYC
    if current == 2 and target >= 3:
        return "run_level_3_video_kyc"
    # L4: EDD
    if current == 3 and target >= 4:
        return "run_level_4_edd"
    
    return "save_profile_and_finish"


def router_node(state: StagedVerificationState) -> dict:
    """Pass-through node for routing decisions."""
    print("--- Routing... ---")
    return {}


def build_staged_graph():
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(StagedVerificationState)
    
    # Add all nodes
    workflow.add_node("load_user_profile", load_user_profile)
    workflow.add_node("router", router_node)
    workflow.add_node("run_level_0_otp", run_level_0_otp)
    workflow.add_node("run_level_1_aadhaar_verify", run_level_1_aadhaar_verify)  # SWAPPED
    workflow.add_node("run_level_2_pan_verify", run_level_2_pan_verify)  # SWAPPED
    workflow.add_node("run_level_3_video_kyc", run_level_3_video_kyc)
    workflow.add_node("run_level_4_edd", run_level_4_edd)
    workflow.add_node("save_profile_and_finish", save_profile_and_finish)
    workflow.add_node("verification_failed", verification_failed)

    # Entry point
    workflow.set_entry_point("load_user_profile")
    
    # Load profile, then go to the router
    workflow.add_edge("load_user_profile", "router")
    
    # Conditional routing
    workflow.add_conditional_edges(
        "router",
        route_verification_step,
        {
            "run_level_0_otp": "run_level_0_otp",
            "run_level_1_aadhaar_verify": "run_level_1_aadhaar_verify",
            "run_level_2_pan_verify": "run_level_2_pan_verify",
            "run_level_3_video_kyc": "run_level_3_video_kyc",
            "run_level_4_edd": "run_level_4_edd",
            "verification_failed": "verification_failed",
            "save_profile_and_finish": "save_profile_and_finish"
        }
    )
    
    # Loop back to router after each level
    workflow.add_edge("run_level_0_otp", "router")
    workflow.add_edge("run_level_1_aadhaar_verify", "router")
    workflow.add_edge("run_level_2_pan_verify", "router")
    workflow.add_edge("run_level_3_video_kyc", "router")
    workflow.add_edge("run_level_4_edd", "router")
    
    # End points
    workflow.add_edge("save_profile_and_finish", END)
    workflow.add_edge("verification_failed", END)
    
    return workflow.compile()
