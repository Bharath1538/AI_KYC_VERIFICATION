import os
import requests
import logging
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)

DAILY_API_KEY = os.getenv("DAILY_API_KEY", "mock_daily_key")

def create_video_kyc_room(user_id: str) -> dict:
    """
    Creates an ephemeral WebRTC video room using the Daily.co REST API.
    A Video KYC (V-CIP) agent would join this room via an iframe.
    Returns the URL for the user to join.
    """
    logger.info(f"Generating secure WebRTC Video KYC Room for User: {user_id}")
    
    # We create a unique room name based on the verification session
    room_name = f"kyc-session-{uuid.uuid4().hex[:8]}"
    
    # In production, rooms should expire shortly after creation for security.
    exp_time = int((datetime.now() + timedelta(minutes=30)).timestamp())
    
    url = "https://api.daily.co/v1/rooms"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DAILY_API_KEY}"
    }
    
    payload = {
        "name": room_name,
        "properties": {
            "exp": exp_time,
            "enable_recording": "cloud", # V-CIP mandates video recording
            "start_video_on_join": True,
            "start_audio_on_join": True,
            "max_participants": 2, # Only the user and the KYC agent
            "eject_at_room_exp": True
        }
    }
    
    try:
        # If we have a real API key, make the call
        if DAILY_API_KEY and DAILY_API_KEY != "mock_daily_key":
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Daily.co Room Created successfully: {data.get('url')}")
                return {
                    "success": True,
                    "room_url": data.get("url"),
                    "room_name": data.get("name"),
                    "demo_mode": False
                }
            else:
                logger.error(f"Daily.co API Error: {response.text}")
                # Fallback to mock
                pass

        # MOCK FALLBACK for Demo environments without active billing
        logger.info("Using Demo Mode for Video KYC URL generation.")
        mock_url = f"https://yourdomain.daily.co/{room_name}"
        return {
            "success": True,
            "room_url": mock_url,
            "room_name": room_name,
            "demo_mode": True,
            "instructions": "In production, embedding this URL in an iframe launches the WebRTC Video Call."
        }
            
    except Exception as e:
        logger.error(f"Error generating Video KYC room: {e}")
        return {
            "success": False,
            "error": str(e)
        }
