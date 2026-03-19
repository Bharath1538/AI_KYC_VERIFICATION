import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import mongo_db as db
from staged_kyc_graph import build_staged_graph, StagedVerificationState

# --- Pydantic Models ---
class StagedVerificationInput(BaseModel):
    user_id: str
    target_level: int
    data: dict

# --- App State ---
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup
    print("Populating mock data (if needed)...")
    db.populate_mock_data()
    print("Compiling LangGraph...")
    app_state["kyc_graph"] = build_staged_graph()
    print("Application startup complete.")
    yield
    # On shutdown
    print("Closing MongoDB connection...")
    db.client.close()

app = FastAPI(
    title="Staged KYC Verification API",
    lifespan=lifespan
)

# --- Streaming Verification Endpoint ---
@app.get("/api/verify-staged-stream")
async def verify_user_staged_stream(payload: str):
    
    kyc_graph = app_state.get("kyc_graph")
    if not kyc_graph:
        async def error_stream_graph_not_ready():
            yield {"event": "error", "data": json.dumps({"error": "Graph is not compiled. Please wait."})}
        return EventSourceResponse(error_stream_graph_not_ready())

    try:
        payload_dict = json.loads(payload)
        request_data = StagedVerificationInput(**payload_dict)
    except Exception as e:
        async def error_stream_invalid_payload():
            yield {"event": "error", "data": json.dumps({"error": f"Invalid payload format: {e}"})}
        return EventSourceResponse(error_stream_invalid_payload())

    initial_state = {
        "request_data": request_data.dict(),
        "user_profile": {},
        "verification_log": [],
        "error_message": None
    }

    async def event_stream():
        """The generator function that streams graph events."""
        try:
            # FIX: Increase recursion limit
            async for chunk in kyc_graph.astream(initial_state, {"recursion_limit": 25}):
                if "verification_log" in chunk:
                    last_log = chunk["verification_log"][-1]
                    yield {"event": "log_update", "data": json.dumps({"log": last_log})}

                if "__end__" in chunk:
                    final_state = chunk["__end__"]
                    if final_state.get("error_message"):
                        yield {"event": "error", "data": json.dumps({"error": final_state["error_message"]})}
                    else:
                        yield {"event": "final_result", "data": json.dumps(final_state)}
                    break 
        except Exception as e:
            print(f"Error during graph execution: {e}")
            yield {"event": "error", "data": json.dumps({"error": f"An unexpected server error occurred: {e}"})}

    return EventSourceResponse(event_stream())

# --- Standard REST Endpoints ---
@app.get("/api/users")
async def get_all_users_api():
    """Fetches all users from the application_users collection."""
    users = db.get_all_users()
    return users

@app.get("/api/user/{user_id}")
async def get_user_profile_api(user_id: str):
    """Fetches a single user's profile."""
    profile = db.get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    return profile

# --- Serve Frontend ---
@app.get("/")
async def get_frontend(request: Request):
    """Serves the main index.html file."""
    return FileResponse("index.html", media_type="text/html")

