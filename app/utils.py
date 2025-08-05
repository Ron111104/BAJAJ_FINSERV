import fitz  # PyMuPDF
import numpy as np
import faiss
import os
import requests
import re
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import asyncio
import aiohttp

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def download_pdf(url: str) -> str:
    """Download PDF and extract text with better error handling"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download PDF: {response.status}")
                
                content = await response.read()
                
        with open("temp.pdf", "wb") as f:
            f.write(content)
            
        doc = fitz.open("temp.pdf")
        text_blocks = []
        
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            if page_text.strip():
                text_blocks.append(f"[Page {page_num + 1}]\n{page_text}")
        
        doc.close()
        os.remove("temp.pdf")
        
        return "\n\n".join(text_blocks)
    
    except Exception as e:
        raise Exception(f"Error downloading/processing PDF: {str(e)}")

def smart_chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> List[str]:
    """
    Intelligent chunking that preserves sentence boundaries and context
    """
    # Clean and normalize text
    text = re.sub(r'\s+', ' ', text.strip())
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed max_chars, save current chunk
        if len(current_chunk) + len(sentence) > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap from previous chunk
            if overlap > 0 and len(current_chunk) > overlap:
                overlap_text = current_chunk[-overlap:].strip()
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk = current_chunk + " " + sentence if current_chunk else sentence
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def embed_batch(chunks: List[str]) -> Tuple[List[np.ndarray], Dict[int, str]]:
    """Create embeddings with better error handling"""
    try:
        # Process in batches to avoid rate limits
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=batch
            )
            batch_embeddings = [np.array(d.embedding, dtype="float32") for d in response.data]
            all_embeddings.extend(batch_embeddings)
        
        chunk_lookup = {i: chunk for i, chunk in enumerate(chunks)}
        return all_embeddings, chunk_lookup
        
    except Exception as e:
        raise Exception(f"Error creating embeddings: {str(e)}")

def save_faiss_index(embeddings: List[np.ndarray], chunk_lookup: Dict[int, str], path: str):
    """Save FAISS index with metadata"""
    try:
        if not embeddings:
            raise ValueError("No embeddings to save")
            
        index = faiss.IndexFlatIP(len(embeddings[0]))  # Use inner product for better similarity
        
        # Normalize embeddings for cosine similarity
        embeddings_array = np.array(embeddings)
        faiss.normalize_L2(embeddings_array)
        
        index.add(embeddings_array)
        faiss.write_index(index, path)
        
        # Save metadata with better encoding
        with open(path + ".lookup", "w", encoding="utf-8") as f:
            for i, chunk in chunk_lookup.items():
                # Escape special characters
                escaped_chunk = chunk.replace("|||", "｜｜｜")
                f.write(f"{i}|||{escaped_chunk}\n")
                
    except Exception as e:
        raise Exception(f"Error saving FAISS index: {str(e)}")

def load_faiss_index(path: str) -> Tuple[faiss.Index, Dict[int, str]]:
    """Load FAISS index with metadata"""
    try:
        index = faiss.read_index(path)
        lookup = {}
        
        with open(path + ".lookup", "r", encoding="utf-8") as f:
            for line in f:
                if "|||" in line:
                    i, chunk = line.strip().split("|||", 1)
                    # Unescape special characters
                    chunk = chunk.replace("｜｜｜", "|||")
                    lookup[int(i)] = chunk
        
        return index, lookup
        
    except Exception as e:
        raise Exception(f"Error loading FAISS index: {str(e)}")

def search_chunks(index: faiss.Index, chunk_lookup: Dict[int, str], query: str, top_k: int = 10) -> List[str]:
    """Enhanced chunk search with better similarity scoring"""
    try:
        query_embedding, _ = embed_batch([query])
        query_vec = np.array(query_embedding).reshape(1, -1)
        
        # Normalize query vector
        faiss.normalize_L2(query_vec)
        
        # Search for similar chunks
        similarities, indices = index.search(query_vec, min(top_k, index.ntotal))
        
        # Filter by similarity threshold
        threshold = 0.3  # Adjust based on testing
        relevant_chunks = []
        
        for sim, idx in zip(similarities[0], indices[0]):
            if sim > threshold and idx in chunk_lookup:
                relevant_chunks.append(chunk_lookup[idx])
        
        return relevant_chunks[:top_k] if relevant_chunks else [chunk_lookup[0]]  # Fallback to first chunk
        
    except Exception as e:
        raise Exception(f"Error searching chunks: {str(e)}")

def call_llm_with_structured_prompt(context: str, questions: List[str]) -> List[str]:
    """
    Enhanced LLM call with insurance-domain specific prompting
    """
    try:
        # Create a more structured prompt
        prompt = f"""You are an expert insurance policy analyst. Your task is to answer questions about insurance policies based on the provided document context.

INSTRUCTIONS:
1. Answer each question precisely based ONLY on the information provided in the context
2. If information is not found in the context, respond with "Information not available in the provided document"
3. Use exact quotes from the document when possible
4. For numerical values (periods, percentages, amounts), be precise
5. Structure your answers clearly and concisely

CONTEXT:
{context}

QUESTIONS TO ANSWER:
"""
        
        for i, question in enumerate(questions, 1):
            prompt += f"{i}. {question}\n"
        
        prompt += "\nPlease provide answers in the following format:\n1. [Answer to question 1]\n2. [Answer to question 2]\n...\n"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use GPT-4 for better accuracy
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert insurance policy analyst. Provide accurate, precise answers based strictly on the provided document context."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.1,  # Low temperature for consistency
            max_tokens=16384
        )
        
        answer_text = response.choices[0].message.content
        
        # Parse numbered answers
        answers = []
        lines = answer_text.split('\n')
        current_answer = ""
        
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+\.', line):  # Numbered answer
                if current_answer:
                    answers.append(current_answer.strip())
                current_answer = re.sub(r'^\d+\.\s*', '', line)
            elif current_answer:
                current_answer += " " + line
        
        if current_answer:
            answers.append(current_answer.strip())
        
        # Ensure we have answers for all questions
        while len(answers) < len(questions):
            answers.append("Information not available in the provided document")
        
        return answers[:len(questions)]
        
    except Exception as e:
        # Fallback to simple response
        return [f"Error processing question: {str(e)}" for _ in questions]