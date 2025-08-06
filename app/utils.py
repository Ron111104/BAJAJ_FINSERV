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
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
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

def smart_chunk_text(text: str, max_chars: int = 2000, overlap: int = 300) -> List[str]:
    """
    Enhanced chunking that preserves context and ensures better coverage
    """
    # Clean text but preserve structure
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    chunks = []
    
    # Strategy 1: Split by clear section boundaries first
    section_markers = [
        r'(?i)(?:section|clause|article|part|chapter)\s*\d+',
        r'(?i)(?:definitions?|coverage|exclusions?|conditions?|benefits?|claims?)',
        r'(?i)(?:waiting\s*period|grace\s*period|premium|maternity)',
        r'\[Page \d+\]'
    ]
    
    # Find all section boundaries
    boundaries = [0]
    for marker in section_markers:
        for match in re.finditer(marker, text):
            boundaries.append(match.start())
    boundaries.append(len(text))
    boundaries = sorted(set(boundaries))
    
    # Create chunks based on sections
    for i in range(len(boundaries) - 1):
        section_start = boundaries[i]
        section_end = boundaries[i + 1]
        section_text = text[section_start:section_end].strip()
        
        if len(section_text) <= max_chars:
            # Section fits in one chunk
            if section_text:
                chunks.append(section_text)
        else:
            # Split large sections by sentences
            sentences = re.split(r'(?<=[.!?])\s+', section_text)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > max_chars and current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Add overlap
                    if overlap > 0:
                        words = current_chunk.split()
                        overlap_words = words[-20:] if len(words) > 20 else words  # Last 20 words
                        current_chunk = " ".join(overlap_words) + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
    
    # Strategy 2: Also create sliding window chunks for better coverage
    sliding_chunks = []
    words = text.split()
    window_size = max_chars // 5  # Roughly 400 words
    step_size = window_size // 2   # 50% overlap
    
    for i in range(0, len(words), step_size):
        chunk_words = words[i:i + window_size]
        if len(chunk_words) >= 50:  # Minimum chunk size
            sliding_chunks.append(" ".join(chunk_words))
    
    # Combine both strategies
    all_chunks = chunks + sliding_chunks
    
    # Remove duplicates and very short chunks
    final_chunks = []
    seen = set()
    
    for chunk in all_chunks:
        # Use first 100 characters as signature for deduplication
        signature = chunk[:100].lower().strip()
        if len(chunk) >= 100 and signature not in seen:
            seen.add(signature)
            final_chunks.append(chunk)
    
    return final_chunks

def embed_batch(chunks: List[str]) -> Tuple[List[np.ndarray], Dict[int, str]]:
    """Create embeddings with better error handling"""
    try:
        # Process in batches to avoid rate limits
        batch_size = 50
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Use text-embedding-ada-002 (reliable and available)
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

def search_chunks(index: faiss.Index, chunk_lookup: Dict[int, str], query: str, top_k: int = 15) -> List[str]:
    """Enhanced chunk search with much more permissive matching"""
    try:
        query_embedding, _ = embed_batch([query])
        query_vec = np.array(query_embedding).reshape(1, -1)
        
        # Normalize query vector
        faiss.normalize_L2(query_vec)
        
        # Search for similar chunks - get more candidates
        similarities, indices = index.search(query_vec, min(top_k * 3, index.ntotal))
        
        relevant_chunks = []
        
        # Much lower threshold - be very permissive
        threshold = 0.1  
        
        # Insurance-specific keyword matching as fallback
        query_lower = query.lower()
        insurance_keywords = [
            'grace period', 'waiting period', 'premium', 'maternity', 'pre-existing',
            'cataract', 'organ donor', 'no claim discount', 'ncd', 'health check',
            'hospital', 'ayush', 'room rent', 'icu', 'plan a', 'plan b',
            'thirty days', '36 months', 'twenty-four months', '24 months'
        ]
        
        query_keywords = [kw for kw in insurance_keywords if kw in query_lower]
        
        for sim, idx in zip(similarities[0], indices[0]):
            if idx in chunk_lookup:
                chunk_text = chunk_lookup[idx]
                chunk_lower = chunk_text.lower()
                
                # Accept chunk if either:
                # 1. Similarity above threshold, OR
                # 2. Contains query keywords (keyword matching fallback)
                if sim > threshold:
                    relevant_chunks.append(chunk_text)
                elif query_keywords and any(keyword in chunk_lower for keyword in query_keywords):
                    relevant_chunks.append(chunk_text)
        
        # If still no results, take top similarity matches regardless of threshold
        if not relevant_chunks:
            for idx in indices[0][:top_k]:
                if idx in chunk_lookup:
                    relevant_chunks.append(chunk_lookup[idx])
        
        return relevant_chunks[:top_k]
        
    except Exception as e:
        raise Exception(f"Error searching chunks: {str(e)}")

def call_llm_with_structured_prompt(context: str, questions: List[str]) -> List[str]:
    """
    Enhanced LLM call with insurance-domain specific prompting using GPT-4o-mini
    """
    try:
        # Create a more structured prompt based on expected answer patterns
        prompt = f"""You are an expert insurance policy analyst. Your task is to answer questions about insurance policies based strictly on the provided document context.

CRITICAL INSTRUCTIONS:
1. Answer ONLY based on information explicitly stated in the provided context
2. For numerical values (periods, percentages, amounts), be precise and quote exact figures
3. Use direct quotes from the document when possible
4. If information is not found in the context, respond exactly: "Information not available in the provided document"
5. Be specific about conditions, limitations, and exceptions
6. Pay special attention to waiting periods, coverage limits, and eligibility criteria

CONTEXT:
{context}

QUESTIONS TO ANSWER:
"""
        
        for i, question in enumerate(questions, 1):
            prompt += f"{i}. {question}\n"
        
        prompt += """
Please provide answers in the following format:
1. [Answer to question 1]
2. [Answer to question 2]
...

Focus on accuracy and completeness. Include specific timeframes, percentages, conditions, and limitations mentioned in the document."""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use GPT-4o-mini as requested
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert insurance policy analyst. Provide accurate, precise answers based strictly on the provided document context. Focus on extracting specific details like timeframes, percentages, conditions, and eligibility requirements."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.1,  # Low temperature for consistency
            max_tokens=16384  # Keep the previous token limit
        )
        
        answer_text = response.choices[0].message.content
        
        # Parse numbered answers more reliably
        answers = []
        lines = answer_text.split('\n')
        current_answer = ""
        
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+\.', line):  # Numbered answer
                if current_answer:
                    answers.append(current_answer.strip())
                current_answer = re.sub(r'^\d+\.\s*', '', line)
            elif current_answer and line:
                current_answer += " " + line
        
        if current_answer:
            answers.append(current_answer.strip())
        
        # Ensure we have answers for all questions
        while len(answers) < len(questions):
            answers.append("Information not available in the provided document")
        
        # Clean up answers
        processed_answers = []
        for answer in answers[:len(questions)]:
            answer = answer.strip()
            
            # Remove redundant prefixes
            answer = re.sub(r'^(Answer:\s*|A:\s*)', '', answer, flags=re.IGNORECASE)
            
            # Ensure proper capitalization
            if answer and not answer[0].isupper() and not answer.startswith("Information not available"):
                answer = answer[0].upper() + answer[1:] if len(answer) > 1 else answer.upper()
            
            processed_answers.append(answer)
        
        return processed_answers
        
    except Exception as e:
        # Fallback to simple response
        return [f"Error processing question: {str(e)}" for _ in questions]