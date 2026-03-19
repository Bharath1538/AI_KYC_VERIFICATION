import mongo_db as db
from verification_tools import (
    verify_phone_otp, verify_email_otp, verify_pan, verify_aadhaar,
    verify_liveness, verify_bank_account, verify_video_kyc,
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
        profile = db.get_user_profile(user_id)
        if not profile:
            return {"error_message": "User not found."}

        log_msg = f"Loaded profile for {user_id}. Current Level: {profile['kyc_level']}"
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
        profile = state['user_profile'] # Get profile from state

        phone_res = verify_phone_otp.invoke({"phone": data['phone'], "otp": data.get('phone_otp', '')})
        email_res = verify_email_otp.invoke({"email": data['email'], "otp": data.get('email_otp', '')})

        if phone_res['status'] == "Verified" and email_res['status'] == "Verified":
            profile['kyc_level'] = 0 # Modify profile in state
            profile['verified_data']['phone'] = data['phone']
            profile['verified_data']['email'] = data['email']
            msg = "Level 0 (Lite) VERIFIED."
            print(msg)
            return {"user_profile": profile, "verification_log": [msg]} # Return updated profile
        else:
            msg = f"Level 0 Failed. Phone: {phone_res['details']} Email: {email_res['details']}"
            print(msg)
            return {"error_message": msg, "verification_log": [msg]}
    except Exception as e:
        error_msg = f"Error in Level 0 processing: {e}"
        print(error_msg)
        return {"error_message": error_msg, "verification_log": [error_msg]}

def run_level_1_pan_verify(state: StagedVerificationState) -> dict:
    """Level 1: PAN-only."""
    try:
        data = state['request_data']['data']['level1']
        profile = state['user_profile']

        pan_res = verify_pan.invoke({"pan_data": data})

        if pan_res['status'] == "Verified":
            profile['kyc_level'] = 1 # Modify profile in state
            profile['verified_data']['pan'] = data['pan_number']
            profile['full_name'] = data['full_name']
            profile['dob'] = data['dob']
            msg = "Level 1 (Basic) VERIFIED."
            print(msg)
            return {"user_profile": profile, "verification_log": [msg]} # Return updated profile
        else:
            msg = f"Level 1 Failed. {pan_res['details']}"
            print(msg)
            return {"error_message": msg, "verification_log": [msg]}
    except Exception as e:
        error_msg = f"Error in Level 1 processing: {e}"
        print(error_msg)
        return {"error_message": error_msg, "verification_log": [error_msg]}

def run_level_2_aadhaar_verify(state: StagedVerificationState) -> dict:
    """Level 2: Aadhaar + Selfie."""
    try:
        data = state['request_data']['data']['level2']
        profile = state['user_profile']

        aadhaar_res = verify_aadhaar.invoke({"aadhaar_data": data['aadhaar_data']})
        liveness_res = verify_liveness.invoke({"selfie_image_path": data['selfie_path']})

        if aadhaar_res['status'] == "Verified" and liveness_res['status'] == "Verified":
            profile['kyc_level'] = 2 # Modify profile in state
            profile['verified_data']['aadhaar'] = data['aadhaar_data']['aadhaar_number']
            profile['verified_data']['liveness_score'] = liveness_res.get('liveness_score')
            msg = "Level 2 (Verified) VERIFIED."
            print(msg)
            return {"user_profile": profile, "verification_log": [msg]} # Return updated profile
        else:
            msg = f"Level 2 Failed. Aadhaar: {aadhaar_res['details']} Liveness: {liveness_res['details']}"
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

        bank_res = verify_bank_account.invoke({"account_number": data['bank_account'], "ifsc": data['ifsc']})
        addr_res = verify_address.invoke({"address_data": data['address_data'], "aadhaar_address": profile['verified_data'].get('aadhaar_address', {})})
        video_res = verify_video_kyc.invoke({"video_call_id": data['video_call_id']})

        if bank_res['status'] == "Verified" and addr_res['status'] == "Verified" and video_res['status'] == "Verified":
            profile['kyc_level'] = 3
            profile['verified_data']['bank_account'] = data['bank_account']
            profile['verified_data']['address'] = data['address_data']
            msg = "Level 3 (Full V-CIP) VERIFIED."
            print(msg)
            return {"user_profile": profile, "verification_log": [msg]}
        else:
            msg = f"Level 3 Failed. Bank: {bank_res['details']} Address: {addr_res['details']} Video: {video_res['details']}"
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
        aml_res = run_llm_aml_pep_check.invoke({"full_name": profile['full_name'], "pan": profile['verified_data']['pan']})
        
        if income_res['status'] == "Verified" and aml_res['status'] == "Clear":
            profile['kyc_level'] = 4
            profile['verified_data']['income_verified'] = True
            profile['verified_data']['aml_check'] = aml_res
            msg = "Level 4 (EDD) VERIFIED."
            print(msg)
            return {"user_profile": profile, "verification_log": [msg]}
        else:
            msg = f"Level 4 Failed. Income: {income_res['details']} AML/PEP: {aml_res['details']}"
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
        msg = f"Verification complete. User '{profile['user_id']}' is now at Level {profile['kyc_level']}."
        print(msg)
        db.update_user_kyc_status(profile["user_id"], profile["kyc_level"], profile["verified_data"], msg)
        return {"verification_log": [msg]}
    except Exception as e:
        error_msg = f"Failed to save profile: {e}"
        print(error_msg)
        return {"error_message": error_msg, "verification_log": [error_msg]}

def verification_failed(state: StagedVerificationState) -> dict:
    """Error node: Logs the failure and updates DB with failure status."""
    profile = state.get("user_profile", {})
    error_msg = state.get("error_message", "Unknown error")
    
    msg = f"Verification FAILED. Error: {error_msg}"
    print(msg)
    
    if profile.get("user_id"):
        db.update_user_kyc_status(
            profile["user_id"], 
            profile.get("kyc_level", -1), 
            profile.get("verified_data", {}), 
            msg 
        )
    return {"verification_log": [msg]} 

# --- 3. Define the Conditional Router ---

def route_verification_step(state: StagedVerificationState) -> str:
    """This function is the *decision maker*. It returns a string label."""
    
    if state.get("error_message"):
        return "verification_failed"
    
    current = state['user_profile']['kyc_level']
    target = state['request_data']['target_level']
    
    if current >= target:
        return "save_profile_and_finish"
    
    if current == -1 and target >= 0:
        return "run_level_0_otp"
    if current == 0 and target >= 1:
        return "run_level_1_pan_verify"
    if current == 1 and target >= 2:
        return "run_level_2_aadhaar_verify"
    if current == 2 and target >= 3:
        return "run_level_3_video_kyc"
    if current == 3 and target >= 4:
        return "run_level_4_edd"
    
    return "save_profile_and_finish"

# --- THIS IS THE FIX ---
def router_node(state: StagedVerificationState) -> dict:
    """
    This is a pass-through node. 
    It must return a dictionary, so it returns an empty one.
    Its only purpose is to be the junction for the loop.
    """
    print("--- Routing... ---")
    return {} 

def build_staged_graph():
    workflow = StateGraph(StagedVerificationState)
    
    # Add all nodes
    workflow.add_node("load_user_profile", load_user_profile)
    workflow.add_node("router", router_node) # <-- Use the new pass-through node
    workflow.add_node("run_level_0_otp", run_level_0_otp)
    workflow.add_node("run_level_1_pan_verify", run_level_1_pan_verify)
    workflow.add_node("run_level_2_aadhaar_verify", run_level_2_aadhaar_verify)
    workflow.add_node("run_level_3_video_kyc", run_level_3_video_kyc)
    workflow.add_node("run_level_4_edd", run_level_4_edd)
    workflow.add_node("save_profile_and_finish", save_profile_and_finish)
    workflow.add_node("verification_failed", verification_failed)

    # Entry point
    workflow.set_entry_point("load_user_profile")
    
    # Load profile, then go to the router
    workflow.add_edge("load_user_profile", "router")
    
    # ATTACH conditional logic AFTER the router node
    workflow.add_conditional_edges(
        "router", # The node to branch *after*
        route_verification_step, # The function that *decides where to go*
        {
            # The possible string outputs from the decision function
            "run_level_0_otp": "run_level_0_otp",
            "run_level_1_pan_verify": "run_level_1_pan_verify",
            "run_level_2_aadhaar_verify": "run_level_2_aadhaar_verify",
            "run_level_3_video_kyc": "run_level_3_video_kyc",
            "run_level_4_edd": "run_level_4_edd",
            "verification_failed": "verification_failed",
            "save_profile_and_finish": "save_profile_and_finish"
        }
    )
    
    # THIS IS THE LOOP: After each level, go *back* to the router
    workflow.add_edge("run_level_0_otp", "router")
    workflow.add_edge("run_level_1_pan_verify", "router")
    workflow.add_edge("run_level_2_aadhaar_verify", "router")
    workflow.add_edge("run_level_3_video_kyc", "router")
    workflow.add_edge("run_level_4_edd", "router")
    
    # End points
    workflow.add_edge("save_profile_and_finish", END)
    workflow.add_edge("verification_failed", END)
    
    return workflow.compile()

