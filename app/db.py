import os
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

logger = logging.getLogger(__name__)

def get_connection():
    try:
        return psycopg2.connect(os.environ["DATABASE_URL"], cursor_factory=RealDictCursor)
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

def get_document_by_url(url: str):
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT faiss_path FROM documents WHERE url = %s;", (url,))
                result = cur.fetchone()
                return result["faiss_path"] if result else None
    except Exception as e:
        logger.error(f"Error getting document: {str(e)}")
        return None

def insert_document(url: str, faiss_path: str):
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO documents (url, faiss_path) VALUES (%s, %s) ON CONFLICT (url) DO UPDATE SET faiss_path = EXCLUDED.faiss_path;",
                    (url, faiss_path)
                )
                conn.commit()
    except Exception as e:
        logger.error(f"Error inserting document: {str(e)}")