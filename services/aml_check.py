import os
import logging
from groq import Groq

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "mock_groq_key")

# For testing out the AML system, we create a few "Sanctioned/PEP" fake identities
MOCK_SANCTION_LIST = [
    "viktor bout",
    "dawood ibrahim",
    "john doe (sanctioned)",
    "robert mueller"
]

def check_aml_pep_risk(full_name: str, pan: str) -> dict:
    """
    Acts as an AI Compliance Officer using Groq's high-speed inference.
    It takes the name, analyzes it against internal sanctioned list heuristics,
    and returns a risk decision.
    """
    logger.info(f"Initiating AI AML/PEP Compliance Check for: {full_name}")
    
    # 1. First Pass: Fast heuristic check against known bad actors database
    if full_name.lower() in MOCK_SANCTION_LIST:
        return {
            "status": "Review",
            "details": f"Flagged: Candidate '{full_name}' matched exactly against active Sanctions/PEP database. Manual Compliance Officer review mandated.",
            "risk_score": 95
        }
    
    # 2. Second Pass: LLM Analysis for fuzzy/alias matching and contextual risk
    if GROQ_API_KEY and GROQ_API_KEY != "mock_groq_key":
        try:
            client = Groq(api_key=GROQ_API_KEY)
            
            prompt = f"""
            You are an expert KYC/AML compliance officer for a major financial institution.
            Analyze the following applicant details for potential Anti-Money Laundering (AML) or Politically Exposed Person (PEP) risks.
            
            Applicant Name: {full_name}
            Applicant PAN: {pan}
            
            Given that this is a standard retail onboarding request, and the name does not immediately appear on the tier 1 OFAC list,
            determine if this profile requires enhanced due diligence.
            
            Respond ONLY with a JSON object in exactly this format:
            {{"decision": "Clear" | "Review" | "Flagged", "reasoning": "Brief 1 sentence explanation", "risk_score_1_to_100": integer}}
            """
            
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            return {
                "status": result.get("decision", "Review"),
                "details": f"AI Assessment: {result.get('reasoning', 'No reasoning provided')}",
                "risk_score": result.get("risk_score_1_to_100", 50)
            }
            
        except Exception as e:
            logger.error(f"Groq API Error: {e}")
            # Fall back to heuristic if Groq fails
            pass
            
    # 3. Fallback (Demo environment or no API Key)
    logger.info("Using Demo Mode Heuristic for AML/PEP check.")
    # If the user name has "suspect" or "scam" in it we mock a flag
    if "suspect" in full_name.lower() or "scam" in full_name.lower():
        return {
            "status": "Flagged",
            "details": "Heuristic Flag: Suspicious pattern detected in applicant identity.",
            "risk_score": 85
        }
        
    return {
        "status": "Clear",
        "details": f"Applicant '{full_name}' cleared automated AML/PEP screening.",
        "risk_score": 5
    }
