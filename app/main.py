import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from .ingest import ingest_local_docs
from .retrieve import query_document
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

class RunRequest(BaseModel):
    documents: str        # e.g. "data/doc1.pdf"
    questions: list[str]

class RunResponse(BaseModel):
    answers: list[str]

@app.on_event("startup")
def startup_event():
    # Build FAISS index from your sample PDFs
    ingest_local_docs()

@app.post("/api/v1/hackrx/run", response_model=RunResponse)
def run(req: RunRequest, authorization: str = Header(...)):
    if authorization != os.getenv("HACKRX_TOKEN"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    answers = [
        query_document(req.documents, question) 
        for question in req.questions
    ]
    return RunResponse(answers=answers)
