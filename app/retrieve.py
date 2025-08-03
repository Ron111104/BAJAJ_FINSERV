import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Globals for lazy loading
FAISS_INDEX = None
METADATA = None
DIM = 1536
INDEX_DIR = "faiss_index"
INDEX_PATH = os.path.join(INDEX_DIR, "policy.index")
META_PATH = os.path.join(INDEX_DIR, "metadata.npy")

def load_faiss_index():
    global FAISS_INDEX, METADATA
    if FAISS_INDEX is None or METADATA is None:
        # Automatically ingest if index doesn't exist
        if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
            from app.ingest import ingest_local_docs
            ingest_local_docs()
        import faiss
        FAISS_INDEX = faiss.read_index(INDEX_PATH)
        METADATA = np.load(META_PATH, allow_pickle=True)
    return FAISS_INDEX, METADATA

def query_document(doc_path: str, question: str, top_k: int = 5) -> str:
    # Ensure FAISS index + metadata
    index, meta = load_faiss_index()

    # 1) Embed the question
    resp = openai.embeddings.create(
        input=question,
        model="text-embedding-ada-002"
    )
    q_emb = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)

    # 2) FAISS search
    distances, indices = index.search(q_emb, top_k)

    # 3) Gather snippets for the requested document
    snippets = []
    for idx in indices[0]:
        _, path, chunk = meta[idx]
        if path == doc_path:
            snippets.append(chunk)
    context = "\n---\n".join(snippets) if snippets else "No relevant context found."

    # 4) Generate answer via GPT-4
    prompt = f"""You are a policy assistant.
Given these excerpts:
{context}

Answer the question precisely, and cite which excerpt you used.

Q: {question}
A:"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You answer using the policy context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()
