Staged KYC Verification Agent (LangGraph + MongoDB + Groq)

This is a full-stack, end-to-end prototype of a staged KYC (Know Your Customer) verification system. It uses a LangGraph agentic workflow on the backend, a MongoDB Atlas database, and a React/Tailwind frontend.

The system is designed as per your specifications:

Backend: FastAPI serving a LangGraph workflow.

Database: MongoDB (designed for MongoDB Atlas).

Frontend: A single-file React + Tailwind application (index.html).

Workflow: 5-level staged verification (L0 to L4).

Input: Expects JSON data (simulating output from a separate OCR app).

LLM: Uses Groq for the Level 4 AML/PEP check.

1. Setup Instructions

Step 1: Set Up MongoDB Atlas (Online DB)

Go to MongoDB Atlas and create a free (M0) cluster.

Get Connection String:

Click "Connect" for your cluster.

Select "Drivers".

Copy the Connection String (URI). It will look like: mongodb+srv://<username>:<password>@your-cluster.mongodb.net/...

Whitelist IP:

In the "Security" tab on the left, go to "Network Access".

Click "Add IP Address".

Select "Allow Access From Anywhere" (0.0.0.0/0) for this prototype. (For production, use your server's IP).

Create Database User:

In "Database Access", create a user. Remember the username and password.

Replace <username> and <password> in your connection string with these credentials.

Important: In the connection string, make sure to specify a database name before the ?, e.g., ...mongodb.net/kyc_db?retryWrites=true&w=majority.

Step 2: Set Up Groq API Key

Go to Groq Console.

Sign up and go to the "API Keys" section.

Create a new API key and copy it.

Step 3: Configure Environment

Save all 8 files (.env.example, requirements.txt, mongo_db.py, verification_tools.py, staged_kyc_graph.py, main.py, index.html, README.md) in a single project directory.

Rename .env.example to .env.

Open the .env file and paste your MongoDB URI and Groq API Key:

MONGO_URI="mongodb+srv://your_user:your_password@your-cluster.mongodb.net/kyc_db?retryWrites=true&w=majority"
GROQ_API_KEY="gsk_YourGroqApiKeyHere..."


Step 4: Set Up Python Environment

Open a terminal in the project directory.

Create a virtual environment:

python -m venv venv


Activate it:

Mac/Linux: source venv/bin/activate

Windows: venv\Scripts\activate

Install all required packages:

pip install -r requirements.txt


2. Running the Application

You need one terminal to run the backend. The frontend is served automatically.

Run the Backend (FastAPI + LangGraph)

Make sure your virtual environment is active (source venv/bin/activate).

Run the Uvicorn server:

uvicorn main:app --reload --port 8000


The server will start up. On its first run, it will automatically connect to MongoDB and populate the mock data (you'll see the log messages).

The server is now running at http://127.0.0.1:8000.

Use the Application

Open your browser and go to:
http://127.0.0.1:8000 (The FastAPI server itself serves the index.html file).

How to use:

The app will automatically load the list of mock users.

Select a user from the "Select User" dropdown (e.g., "u123_adithya").

Their current profile will load.

Select a "Target Level" (e.g., Level 1).

A form for Level 1 will appear, pre-filled with sample JSON data.

Click "Start Verification".

Watch the "Live Verification Log" as the LangGraph agent executes the steps.

Once complete, the "Final Result" will appear, and the user's profile card will update with their new level.

How to Test Level 4 (AML/PEP)

Select the user "u789_robert" (Robert K Mueller).

This user is already at Level 1.

Set the Target Level to 4.

Forms for Level 2, 3, and 4 will appear, pre-filled with the correct data for this user.

Click "Start Verification".

The graph will run L2 and L3. When it gets to L4, watch the "Live Verification Log". You will see the log from run_llm_aml_pep_check, and the final summary from Groq should mention a "Potential PEP match". The verification will fail (as expected) and the user's status will be updated.