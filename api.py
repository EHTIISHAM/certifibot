import os
import uvicorn
import chromadb
from chromadb.utils import embedding_functions
from typing import Annotated, TypedDict, List, Optional
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import secrets
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from langgraph.graph import StateGraph, END
from google import genai
from google.genai import types
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
import json
import dotenv
from pathlib import Path
import sys
                                                                                                                                                
def get_env_path():
    """ 
    Returns the path to .env file based on execution mode.
    """
    if getattr(sys, 'frozen', False):
        # Running as EXE: Look in the same folder as the .exe file
        # sys.executable is the full path to the .exe (e.g., D:\dist\app.exe)
        # .parent gives us the folder (e.g., D:\dist\)
        base_path = Path(sys.executable).parent
    else:
        # Running in VS Code: Look in the project root
        base_path = Path(__file__).resolve().parent.parent
    
    return base_path / '.env'

# Load the env
env_file = "D:\certifibot\.env"
print(f"DEBUG: Loading .env from: {env_file}")

# Override=True ensures that if you change the .env file, it updates immediately
dotenv.load_dotenv(dotenv_path=env_file, override=True)


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("SERVICE_ACCOUNT_JSON_FILE")
PINECONE_KEY = os.environ.get("PINECONEAPI")

client = genai.Client(
      vertexai=True,
      project=os.environ.get("GOOGLE_CLOUD_PROJECT_ID"),
      #api_key=os.environ.get("GOOGLE_CLOUD_API_KEY"),
      location="us-central1",
  )
"""client_db = chromadb.PersistentClient()
ef = embedding_functions.SentenceTransformerEmbeddingFunction()
collection = client_db.get_or_create_collection(
    name="cert",
    embedding_function=ef
)"""
embed_model = SentenceTransformer('all-MiniLM-L6-v2',token=1024)
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index("certificates")
# Initialize Models
chat_model = "gemini-2.5-flash"
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. Setup Jinja2 Templates
templates = Jinja2Templates(directory="templates")
# --- 2. DATA MODELS & STATE ---

# The Schema we want Gemini to extract
class CertificateSchema(BaseModel):
    candidate_name: str
    title_name: str
    issuer: str
    issue_date: str
    expiry_date: Optional[str] = None
    credential_id: Optional[str] = None
    review: str
    is_valid_format: bool = Field(description="True if it looks like a legitimate certificate")

# The Brain's Memory (State)
class AgentState(TypedDict):
    messages: List[str]          # Chat history
    file_bytes: Optional[bytes]  # The uploaded file content
    file_type: Optional[str]     # Mime type (image/png, application/pdf)
    has_file: bool               # Trigger for routing
    final_response: str          # What we send back to UI

# --- 3. LANGGRAPH NODES (The Logic) ---

def steering_node(state: AgentState):
    """
    Handles text-only interaction. 
    Strictly steers user back to uploading a certificate.
    """
    user_msg = state['messages'][-1]
    
    system_prompt = """
    You are a single-purpose AI Agent designed to evaluate certificates.
    Your GOAL: Get the user to upload a certificate file.
    
    Rules:
    1. If the user says "hi" or asks a question, acknowledge briefly but IMMEDIATELY ask for a certificate.
    2. Do NOT answer general questions (e.g., "What is the capital of France?").
    3. Be polite but firm. You cannot function without a file.
    
    User Input: {user_msg}
    """

    response = client.models.generate_content(
        model=chat_model,
        contents=system_prompt.format(user_msg=user_msg),
    )

    return {"final_response": response.text}

def evaluation_node(state: AgentState):
    """
    Handles file interaction.
    Uses Vision to extract data and validate the cert.
    """
    file_content = state['file_bytes']
    mime_type = state['file_type']
    
    prompt = """
    Analyze this image. It should be a professional certificate.
    Extract the following details in JSON format:
    - candidate_name
    - title_name
    - issuer
    - issue_date
    - expiry_date
    - credential_id
    - is_valid_format (boolean)
    - review
    
    If it is NOT a certificate, set is_valid_format to False.
    Write a samll Review ( if is_valid_format False then say to upload a Certificate)
    Return ONLY raw JSON.
    """
    
    # Create the image part for Gemini
    image_part = {
        "mime_type": mime_type,
        "data": file_content
    }
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    data=image_part["data"], 
                    mime_type=image_part["mime_type"]
                ),
                types.Part.from_text(
                    text=prompt
                ),
            ]
        )
    ]
    try:
        response = client.models.generate_content(
        model=chat_model,
        contents=contents,
        )

        # Clean up JSON formatting if Gemini adds markdown
        clean_json = response.text.replace("```json", "").replace("```", "")
        data = json.loads(clean_json)
        
        if data['is_valid_format']:
            text_to_embed = f"{data['candidate_name']} got {data['title_name']} from {data['issuer']}."
            vector = embed_model.encode(text_to_embed).tolist()
            
            # 2. Upload to Pinecone
            # Pinecone expects: (ID, Vector, Metadata)
            unique_id = f"{data['candidate_name']}_{data['credential_id']}".replace(" ", "_")
            
            index.upsert(vectors=[{
                "id": unique_id,
                "values": vector,
                "metadata": {
                    "name": data['candidate_name'],
                    "issuer": data['issuer'],
                    "date": data['issue_date'],
                    "review": data['review']
                }
            }])
            formatted_msg = (
                f"**--Certificate Analyzed**\n"
                f"**--Name:** {data.get('candidate_name')}\n"
                f"**--Certificate Title:** {data.get('title_name')}"
                f"**--Issuer:** {data.get('issuer')}\n"
                f"**--Date:** {data.get('issue_date')}\n"
                f"**--ID:** {data.get('credential_id')}\n"
                f"**--Review:** {data.get('review')}"
            )
        else:
            formatted_msg = "This doesn't look like a valid certificate. Please upload a clear image of a certificate."
            
    except Exception as e:
        formatted_msg = f"Error analyzing file: {str(e)}"

    return {"final_response": formatted_msg}

# --- 4. BUILD THE GRAPH ---

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("steer", steering_node)
workflow.add_node("evaluate", evaluation_node)

# Add Conditional Logic (The Router)
def route_input(state: AgentState):
    if state["has_file"]:
        return "evaluate"
    return "steer"

workflow.set_conditional_entry_point(
    route_input,
    {
        "evaluate": "evaluate",
        "steer": "steer"
    }
)

workflow.add_edge("steer", END)
workflow.add_edge("evaluate", END)

# Compile the brain
agent_app = workflow.compile()


security = HTTPBasic()

# --- SECURITY LOGIC ---
def get_current_admin(credentials: HTTPBasicCredentials = Depends(security)):
    # In production, use environment variables: os.environ.get("ADMIN_USER")
    correct_username = secrets.compare_digest(credentials.username, "admin")
    correct_password = secrets.compare_digest(credentials.password, "supersecret123")
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, username: str = Depends(get_current_admin)):
    """Serves the secure HTML dashboard"""
    return templates.TemplateResponse("admin.html", {"request": request, "username": username})

@app.get("/api/admin/certs")
async def get_certificates(query: Optional[str] = None, username: str = Depends(get_current_admin)):
    """Fetches certificates from Pinecone. Uses a dummy vector to 'browse' if no query exists."""
    try:
        # 1. Prepare the Vector
        if query:
            # Convert user text -> [0.1, -0.2, ...]
            search_vector = embed_model.encode(query).tolist()
        else:
            # Trick: Create a "Dummy Vector" of zeros to browse the DB
            # 384 is the dimension size for 'all-MiniLM-L6-v2'
            search_vector = [0.0] * 384 

        # 2. Query Pinecone
        results = index.query(
            vector=search_vector,
            top_k=100, # Adjust this number based on how many you want to show
            include_metadata=True
        )

        # 3. Format Pinecone's output for your Frontend
        items = []
        for match in results['matches']:
            # Pinecone stores the review/text inside 'metadata', not a separate 'document' field
            # We assume your metadata has 'review' or we combine fields to make a summary
            metadata = match['metadata']
            
            items.append({
                "id": match['id'],
                # Create a readable summary from metadata
                "document": f"{metadata.get('name', 'Unknown')} - {metadata.get('issuer', 'Unknown')}", 
                "metadata": metadata,
                "score": match['score']
            })
            
        return {"status": "success", "data": items}

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.delete("/api/admin/certs/{cert_id}")
async def delete_certificate(cert_id: str, username: str = Depends(get_current_admin)):
    """Deletes a specific certificate by its ID in Pinecone."""
    try:
        # Pinecone delete is simple
        index.delete(ids=[cert_id])
        return {"status": "success", "message": f"Deleted {cert_id}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/chat")
async def chat_endpoint(
    message: str = Form(""), 
    file: Optional[UploadFile] = File(None)
):
    # 1. Prepare State inputs
    inputs = {
        "messages": [message],
        "has_file": False,
        "file_bytes": None,
        "file_type": None
    }

    # 2. Handle File if present
    if file:
        inputs["has_file"] = True
        inputs["file_bytes"] = await file.read()
        inputs["file_type"] = file.content_type

    # 3. Invoke the LangGraph Brain
    result = agent_app.invoke(inputs)

    return {"response": result["final_response"]}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Renders the HTML template"""
    return templates.TemplateResponse("index.html", {"request": request})
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)