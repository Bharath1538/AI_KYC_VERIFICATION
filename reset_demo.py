import sys
# FIX: Import the whole module to avoid potential circular import issues
import mongo_db 
from datetime import datetime

# The user_ids of the demo users you want to reset
DEMO_USER_IDS = ["u123_adithya", "u456_jane", "u789_bob"]

# The fields to force-reset to their original "new user" values
RESET_PAYLOAD = {
    "$set": {
        "full_name": None,
        "dob": None,
        "kyc_level": -1,
        "last_verification_status": "New user, reset for demo.",
        "updated_at": datetime.now(),
        "verified_data": {}
    }
}

def run_demo_reset():
    """
    Connects to the MongoDB and resets the kyc_level and verified_data
    for the demo users in the 'application_users' collection.
    """
    print("--- Running KYC Demo Reset Script ---")
    
    try:
        # FIX: Access the 'db' variable directly from the imported module
        db = mongo_db.db
        if db is None:
            raise Exception("Database object 'db' is None. Check mongo_db.py connection.")
            
        users_collection = db.application_users
        print(f"Successfully connected to MongoDB database: {db.name}")
    except Exception as e:
        print(f"FATAL: Could not connect to MongoDB. Is it running? Is .env correct?")
        print(f"Error: {e}")
        sys.exit(1)

    count = 0
    for user_id in DEMO_USER_IDS:
        print(f"Resetting user: {user_id} ...")
        
        try:
            # This update will find the user by their ID and force-set
            # the fields back to their original state.
            result = users_collection.update_one(
                {"user_id": user_id},
                RESET_PAYLOAD
            )
            
            if result.matched_count > 0:
                print(f" -> Success: User {user_id} was reset.")
                count += 1
            else:
                print(f" -> Warning: User {user_id} was not found in the database.")

        except Exception as e:
            print(f" -> ERROR: Failed to reset user {user_id}: {e}")

    print("\n--- Demo Reset Complete ---")
    print(f"Successfully reset {count} users back to kyc_level: -1.")
    
    # FIX: Access client using the module name
    mongo_db.client.close()
    print("MongoDB connection closed.")

if __name__ == "__main__":
    run_demo_reset()

