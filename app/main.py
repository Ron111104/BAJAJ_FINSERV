from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from .retrieve import query_document
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or use specific domains instead of "*" for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/hackrx/run")
async def run(request: Request):
    auth = request.headers.get("Authorization")
    if auth != os.getenv("HACKRX_TOKEN"):
        raise HTTPException(status_code=401, detail="Invalid token")

    body = await request.json()
    url = body.get("documents")
    questions = body.get("questions")

    if not url or not questions:
        raise HTTPException(status_code=400, detail="Missing fields")

    result = query_document(url, questions)
    return JSONResponse(content=result)