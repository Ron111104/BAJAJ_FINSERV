import hashlib
import os
from .utils import download_pdf, smart_chunk_text, embed_batch, save_faiss_index, load_faiss_index, search_chunks, call_llm_with_structured_prompt
from .db import get_document_by_url, insert_document
import asyncio
from typing import List, Dict

async def query_document(url: str, questions: List[str]) -> Dict:
    """
    Enhanced document querying with better error handling and multi-question search
    """
    try:
        # Check if document already processed
        faiss_path = get_document_by_url(url)
        
        if not faiss_path or not os.path.exists(faiss_path):
            # Download and process document
            content = await download_pdf(url)
            
            if not content.strip():
                return {"answers": ["Document appears to be empty or unreadable" for _ in questions]}
            
            # Smart chunking
            chunks = smart_chunk_text(content, max_chars=1500, overlap=200)
            
            if not chunks:
                return {"answers": ["Failed to process document content" for _ in questions]}
            
            # Create embeddings
            embeddings, chunk_lookup = embed_batch(chunks)
            
            # Save to FAISS
            doc_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            faiss_path = f"faiss_index/{doc_hash}.index"
            os.makedirs("faiss_index", exist_ok=True)
            
            save_faiss_index(embeddings, chunk_lookup, faiss_path)
            insert_document(url, faiss_path)
        
        # Load index
        index, chunk_lookup = load_faiss_index(faiss_path)
        
        # Multi-query search: combine all questions for comprehensive context retrieval
        combined_query = " ".join(questions)
        relevant_chunks = search_chunks(index, chunk_lookup, combined_query, top_k=15)
        
        # Also search for each question individually to ensure coverage
        all_relevant_chunks = set(relevant_chunks)
        
        for question in questions:
            question_chunks = search_chunks(index, chunk_lookup, question, top_k=5)
            all_relevant_chunks.update(question_chunks)
        
        # Combine all relevant context
        context = "\n\n---\n\n".join(list(all_relevant_chunks)[:20])  # Limit context size
        
        if not context.strip():
            return {"answers": ["No relevant information found in the document" for _ in questions]}
        
        # Get structured answers
        answers = call_llm_with_structured_prompt(context, questions)
        
        return {"answers": answers}
        
    except Exception as e:
        error_msg = f"Error processing document: {str(e)}"
        return {"answers": [error_msg for _ in questions]}