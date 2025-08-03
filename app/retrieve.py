import hashlib, os
from .utils import download_pdf, chunk_text, embed_batch, save_faiss_index, load_faiss_index, search_chunks, call_llm
from .db import get_document_by_url, insert_document

def query_document(url: str, questions: list[str]):
    faiss_path = get_document_by_url(url)

    if not faiss_path:
        content = download_pdf(url)
        chunks = chunk_text(content)
        embeddings, chunk_lookup = embed_batch(chunks)

        doc_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        faiss_path = f"faiss_index/{doc_hash}.index"
        os.makedirs("faiss_index", exist_ok=True)

        save_faiss_index(embeddings, chunk_lookup, faiss_path)
        insert_document(url, faiss_path)

    index, chunk_lookup = load_faiss_index(faiss_path)
    context = search_chunks(index, chunk_lookup, questions[0])
    return call_llm(context, questions)