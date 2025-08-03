import os
import psycopg2
from psycopg2.extras import RealDictCursor

def get_connection():
    return psycopg2.connect(os.environ["DATABASE_URL"], cursor_factory=RealDictCursor)

def get_document_by_url(url: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT faiss_path FROM documents WHERE url = %s;", (url,))
            result = cur.fetchone()
            return result["faiss_path"] if result else None

def insert_document(url: str, faiss_path: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (url, faiss_path) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
                (url, faiss_path)
            )
            conn.commit()