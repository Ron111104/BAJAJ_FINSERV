import os
import faiss
import numpy as np
import fitz   # PyMuPDF
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Where to store FAISS index + metadata
INDEX_DIR = "faiss_index"
os.makedirs(INDEX_DIR, exist_ok=True)
INDEX_PATH = os.path.join(INDEX_DIR, "policy.index")
META_PATH  = os.path.join(INDEX_DIR, "metadata.npy")

DIM = 1536  # embedding dimension

def extract_text(path: str) -> str:
    doc = fitz.open(path)
    return "\n\n".join(page.get_text() for page in doc)

def chunk_text(text: str, size: int = 3000) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]

def embed_batch(chunks: list[str]) -> np.ndarray:
    vectors = []
    for chunk in chunks:
        resp = openai.embeddings.create(
            input=chunk,
            model="text-embedding-ada-002"
        )
        # correct attribute access:
        emb = np.array(resp.data[0].embedding, dtype="float32")
        vectors.append(emb)
    return np.vstack(vectors)

def ingest_local_docs():
    all_vectors = []
    all_meta    = []  # list of (idx, doc_path, chunk_text)
    idx_counter = 0

    for i in range(1, 6):
        path = f"data/doc{i}.pdf"
        text = extract_text(path)
        chunks = chunk_text(text)
        embs = embed_batch(chunks)

        for chunk in chunks:
            all_meta.append((idx_counter, path, chunk))
            idx_counter += 1

        all_vectors.append(embs)

    vectors = np.vstack(all_vectors)
    meta_arr = np.array(all_meta, dtype=object)

    index = faiss.IndexFlatL2(DIM)
    index.add(vectors)
    faiss.write_index(index, INDEX_PATH)
    np.save(META_PATH, meta_arr)

    print(f"Ingested {vectors.shape[0]} total chunks into FAISS.")
