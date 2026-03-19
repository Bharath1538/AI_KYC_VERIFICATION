import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from datetime import datetime

# --- Settings ---
class MongoSettings(BaseSettings):
    """Loads environment variables from .env file."""
    MONGO_URI: str
    MONGO_DB_NAME: str = "kyc_db"

    class Config:
        env_file = ".env"
        extra = "ignore"  # <-- FIX: Tell Pydantic to ignore extra fields (like GROQ_API_KEY)

try:
    load_dotenv()
    settings = MongoSettings()
except Exception as e:
    print(f"Error loading settings from .env: {e}")
    print("Please ensure .env file exists with MONGO_URI.")
    settings = None

# --- Client & DB Setup ---
def get_mongo_client():
    """Initializes and returns a MongoDB client."""
    if not settings:
        raise ConfigurationError("Settings not loaded. Cannot connect to MongoDB.")
        
    try:
        # Extract the cluster host for a nice print message
        host = settings.MONGO_URI.split('@')[-1].split('/')[0]
        print(f"Connecting to MongoDB at {host}...")
        
        client = MongoClient(
            settings.MONGO_URI,
            serverSelectionTimeoutMS=5000
        )
        client.admin.command('ping')
        print("MongoDB connection successful.")
        return client
    except ConnectionFailure as e:
        print(f"MongoDB connection failed: {e}")
        raise
    except ConfigurationError as e:
        print(f"MongoDB configuration error: {e}")
        print("This often means your MONGO_URI is invalid.")
        raise

try:
    client = get_mongo_client()
    db = client[settings.MONGO_DB_NAME] # Explicitly get DB by name
except Exception as e:
    print(f"Failed to initialize MongoDB. Exiting. Error: {e}")
    exit(1)


# --- Mock Data ---
def populate_mock_data():
    """Idempotently populates the DB with mock data."""
    print("Populating mock data...")
    
    try:
        source_pan = db.source_pan_holders
        # Check if data exists to avoid re-populating
        if source_pan.count_documents({}) == 0:
            source_pan.insert_many([
                {
                    "pan_number": "ABCDE1234F",
                    "full_name": "Adithya Vardan M",
                    "date_of_birth": datetime(2004, 5, 10),
                    "aadhaar_number": "123456789012",
                    "pan_status": "Active"
                },
                {
                    "pan_number": "FGHIJ5678K",
                    "full_name": "Jane Doe",
                    "date_of_birth": datetime(1990, 1, 1),
                    "aadhaar_number": "987654321098",
                    "pan_status": "Active"
                },
                {
                    "pan_number": "GKPDX1234R",
                    "full_name": "Robert K Mueller",
                    "date_of_birth": datetime(1944, 8, 7),
                    "aadhaar_number": "567812349012",
                    "pan_status": "Active"
                }
            ])
            print(f"Populated {source_pan.count_documents({})} records into source_pan_holders.")
        else:
            print("source_pan_holders already populated.")

        source_aadhaar = db.source_aadhaar_holders
        if source_aadhaar.count_documents({}) == 0:
            source_aadhaar.insert_many([
                {
                    "aadhaar_number": "123456789012",
                    "full_name": "Adithya Vardan M",
                    "date_of_birth": datetime(2004, 5, 10),
                    "gender": "Male",
                    "address": {"street": "123 Anna Salai", "city": "Chennai", "pincode": "600001"},
                    "mobile_number": "9876543210",
                    "status": "Active"
                },
                {
                    "aadhaar_number": "987654321098",
                    "full_name": "Jane Doe",
                    "date_of_birth": datetime(1990, 1, 1),
                    "gender": "Female",
                    "address": {"street": "456 Main St", "city": "New York", "pincode": "10001"},
                    "mobile_number": "1234567890",
                    "status": "Active"
                },
                {
                    "aadhaar_number": "567812349012",
                    "full_name": "Robert K Mueller",
                    "date_of_birth": datetime(1944, 8, 7),
                    "gender": "Male",
                    "address": {"street": "789 Capitol Hill", "city": "Washington", "pincode": "10001"},
                    "mobile_number": "555123456",
                    "status": "Active"
                }
            ])
            print(f"Populated {source_aadhaar.count_documents({})} records into source_aadhaar_holders.")
        else:
            print("source_aadhaar_holders already populated.")

        app_users = db.application_users
        if app_users.count_documents({}) == 0:
            app_users.insert_many([
                {
                    "user_id": "u123_adithya",
                    "email": "adithya@example.com",
                    "phone_number": "+919876543210",
                    "full_name": None,
                    "dob": None,
                    "kyc_level": -1,
                    "last_verification_status": "New user",
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                    "verified_data": {}
                },
                {
                    "user_id": "u456_jane",
                    "email": "jane@example.com",
                    "phone_number": "+11234567890",
                    "full_name": "Jane Doe",
                    "dob": datetime(1990, 1, 1),
                    "kyc_level": 1, # Jane is already at Level 1
                    "last_verification_status": "PAN Verified",
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                    "verified_data": {"pan": "FGHIJ5678K", "full_name": "Jane Doe", "dob": "1990-01-01"}
                },
                {
                    "user_id": "u789_robert",
                    "email": "robert.mueller@gov.us",
                    "phone_number": "+1555123456",
                    "full_name": "Robert K Mueller",
                    "dob": datetime(1944, 8, 7),
                    "kyc_level": 1, # Robert is at Level 1, to test L4 PEP check
                    "last_verification_status": "PAN Verified",
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                    "verified_data": {"pan": "GKPDX1234R", "full_name": "Robert K Mueller", "dob": "1944-08-07"}
                }
            ])
            print(f"Populated {app_users.count_documents({})} records into application_users.")
        else:
            print("application_users already populated.")
        
        db.verification_logs.delete_many({})
        print("Cleared verification_logs.")
        print("Mock data population complete.")
    
    except Exception as e:
        print(f"Error populating mock data: {e}")

# --- Query Functions ---

def get_all_users():
    """Fetches all users from the application_users collection."""
    try:
        users = list(db.application_users.find({}, {"_id": 0})) # Project to remove ObjectId
        return users
    except Exception as e:
        print(f"Error fetching all users: {e}")
        return []

def get_user_profile(user_id: str):
    """Fetches a single user's profile."""
    try:
        profile = db.application_users.find_one({"user_id": user_id}, {"_id": 0})
        return profile
    except Exception as e:
        print(f"Error fetching profile for {user_id}: {e}")
        return None

def update_user_kyc_status(user_id: str, kyc_level: int, verified_data: dict, status_message: str, full_name: str = None, dob: datetime = None):
    """Updates a user's KYC level, data, and status."""
    try:
        update_doc = {
            "$set": {
                "kyc_level": kyc_level,
                "verified_data": verified_data,
                "last_verification_status": status_message,
                "updated_at": datetime.now()
            }
        }
        if full_name:
            update_doc["$set"]["full_name"] = full_name
        if dob:
            update_doc["$set"]["dob"] = dob
            
        result = db.application_users.update_one({"user_id": user_id}, update_doc)
        print(f"Updated profile for {user_id}. Matched: {result.matched_count}, Modified: {result.modified_count}")
    except Exception as e:
        print(f"Error updating profile for {user_id}: {e}")

def create_verification_log(log_data: dict):
    """Creates a new log entry."""
    try:
        log_data["timestamp"] = datetime.now()
        db.verification_logs.insert_one(log_data)
    except Exception as e:
        print(f"Error creating verification log: {e}")

# --- Source DB Query Functions ---

def find_pan_record(pan_data: dict):
    """Finds a matching PAN record in the source DB."""
    try:
        query = {
            "pan_number": pan_data.get('pan_number'),
            "full_name": pan_data.get('full_name'),
            "date_of_birth": pan_data.get('dob_dt')
        }
        return db.source_pan_holders.find_one(query)
    except Exception as e:
        print(f"Error finding PAN record: {e}")
        return None

def find_aadhaar_record(aadhaar_data: dict):
    """Finds a matching Aadhaar record in the source DB."""
    try:
        query = {
            "aadhaar_number": aadhaar_data.get('aadhaar_number'),
            "full_name": aadhaar_data.get('full_name'),
            "date_of_birth": aadhaar_data.get('dob_dt')
        }
        return db.source_aadhaar_holders.find_one(query)
    except Exception as e:
        print(f"Error finding Aadhaar record: {e}")
        return None

if __name__ == "__main__":
    print("Running mongo_db.py as script...")
    populate_mock_data()

