from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from .retrieve import query_document
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/hackrx/run")
async def run(request: Request):
    try:
        auth = request.headers.get("Authorization")
        expected_token = f"Bearer {os.getenv('HACKRX_TOKEN')}"
        
        if auth != expected_token:
            raise HTTPException(status_code=401, detail="Invalid token")

        body = await request.json()
        url = body.get("documents")
        questions = body.get("questions")

        if not url or not questions:
            raise HTTPException(status_code=400, detail="Missing fields")

        result = await query_document(url, questions)
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
