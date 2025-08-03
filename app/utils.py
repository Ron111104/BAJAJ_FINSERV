import fitz  # PyMuPDF
import numpy as np
import faiss
import os
import requests
import textwrap
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def download_pdf(url: str) -> str:
    response = requests.get(url)
    with open("temp.pdf", "wb") as f:
        f.write(response.content)
    doc = fitz.open("temp.pdf")
    return "\n\n".join([page.get_text() for page in doc])

def chunk_text(text: str, max_len: int = 500) -> list[str]:
    return textwrap.wrap(text, max_len)

def embed_batch(chunks: list[str]):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=chunks
    )
    embeddings = [np.array(d.embedding, dtype="float32") for d in response.data]
    chunk_lookup = {i: chunk for i, chunk in enumerate(chunks)}
    return embeddings, chunk_lookup

def save_faiss_index(embeddings: list[np.ndarray], chunk_lookup: dict, path: str):
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    faiss.write_index(index, path)
    with open(path + ".lookup", "w", encoding="utf-8") as f:
        for i in chunk_lookup:
            f.write(f"{i}|||{chunk_lookup[i]}\n")

def load_faiss_index(path: str):
    index = faiss.read_index(path)
    lookup = {}
    with open(path + ".lookup", "r", encoding="utf-8") as f:
        for line in f:
            i, chunk = line.strip().split("|||", 1)
            lookup[int(i)] = chunk
    return index, lookup

def search_chunks(index, chunk_lookup, query: str, top_k=5):
    query_vec, _ = embed_batch([query])
    D, I = index.search(np.array(query_vec).reshape(1, -1), top_k)
    return "\n\n".join([chunk_lookup[i] for i in I[0]])

def call_llm(context: str, questions: list[str]) -> dict:
    prompt = f"""Answer the following questions based on the context below:

Context:
{context}

Questions:
"""
    prompt += "\n".join([f"- {q}" for q in questions])
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return {"answers": [response.choices[0].message.content]}
