# improved_utils.py
import fitz  # PyMuPDF
import numpy as np
import faiss
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import aiohttp
import asyncio

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def download_pdf(url: str) -> str:
    """Enhanced PDF download and text extraction"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download PDF: {response.status}")
                
                content = await response.read()
                
        # Save temporarily
        temp_path = "temp_insurance_doc.pdf"
        with open(temp_path, "wb") as f:
            f.write(content)
            
        # Extract text with better formatting preservation
        doc = fitz.open(temp_path)
        text_blocks = []
        
        for page_num, page in enumerate(doc):
            # Get text with layout information
            page_dict = page.get_text("dict")
            page_text = extract_structured_text(page_dict)
            
            if page_text.strip():
                text_blocks.append(f"[Page {page_num + 1}]\n{page_text}")
        
        doc.close()
        os.remove(temp_path)
        
        full_text = "\n\n".join(text_blocks)
        
        # Post-process to fix common PDF extraction issues
        full_text = clean_extracted_text(full_text)
        
        return full_text
    
    except Exception as e:
        raise Exception(f"Error downloading/processing PDF: {str(e)}")


def extract_structured_text(page_dict: Dict) -> str:
    """Extract text while preserving structure from PyMuPDF dict"""
    text_blocks = []
    
    for block in page_dict.get("blocks", []):
        if "lines" in block:  # Text block
            block_text = []
            for line in block["lines"]:
                line_text = []
                for span in line.get("spans", []):
                    span_text = span.get("text", "").strip()
                    if span_text:
                        line_text.append(span_text)
                
                if line_text:
                    block_text.append(" ".join(line_text))
            
            if block_text:
                text_blocks.append("\n".join(block_text))
    
    return "\n\n".join(text_blocks)


def clean_extracted_text(text: str) -> str:
    """Clean and normalize extracted PDF text"""
    # Fix common PDF extraction issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between joined words
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Normalize paragraph breaks
    
    # Fix common insurance document patterns
    text = re.sub(r'(\d+)\s*%', r'\1%', text)  # Fix percentage formatting
    text = re.sub(r'(\d+)\s+(days?|months?|years?)\b', r'\1 \2', text)  # Fix time periods
    text = re.sub(r'\b(SI|PED|NCD|ICU|AYUSH)\b', lambda m: m.group(1).upper(), text)  # Standardize acronyms
    
    # Preserve important numerical patterns
    text = re.sub(r'\bthirty six\b', 'thirty-six', text, flags=re.IGNORECASE)
    text = re.sub(r'\btwenty four\b', 'twenty-four', text, flags=re.IGNORECASE)
    
    return text


def embed_batch(chunks: List[str]) -> Tuple[List[np.ndarray], Dict[int, str]]:
    """Create embeddings with improved error handling and batching"""
    try:
        if not chunks:
            raise ValueError("No chunks to embed")
        
        # Process in smaller batches to avoid rate limits
        batch_size = 20  # Smaller batches for reliability
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Retry logic for robustness
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.embeddings.create(
                        model="text-embedding-ada-002",
                        input=batch
                    )
                    batch_embeddings = [np.array(d.embedding, dtype="float32") for d in response.data]
                    all_embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    print(f"Embedding attempt {attempt + 1} failed, retrying...")
                    import time
                    time.sleep(1)
        
        chunk_lookup = {i: chunk for i, chunk in enumerate(chunks)}
        return all_embeddings, chunk_lookup
        
    except Exception as e:
        raise Exception(f"Error creating embeddings: {str(e)}")


def save_faiss_index(embeddings: List[np.ndarray], chunk_lookup: Dict[int, str], path: str):
    """Save FAISS index with improved metadata handling"""
    try:
        if not embeddings:
            raise ValueError("No embeddings to save")
            
        # Use IndexFlatIP for cosine similarity (better for semantic search)
        index = faiss.IndexFlatIP(len(embeddings[0]))
        
        # Normalize embeddings for cosine similarity
        embeddings_array = np.array(embeddings)
        faiss.normalize_L2(embeddings_array)
        
        index.add(embeddings_array)
        faiss.write_index(index, path)
        
        # Save metadata with better encoding and error handling
        metadata_path = path + ".lookup"
        with open(metadata_path, "w", encoding="utf-8") as f:
            for i, chunk in chunk_lookup.items():
                # Escape special characters more robustly
                escaped_chunk = chunk.replace("|||", "｜｜｜").replace("\n", "\\n").replace("\r", "\\r")
                f.write(f"{i}|||{escaped_chunk}\n")
                
    except Exception as e:
        raise Exception(f"Error saving FAISS index: {str(e)}")


def load_faiss_index(path: str) -> Tuple[faiss.Index, Dict[int, str]]:
    """Load FAISS index with improved error handling"""
    try:
        index = faiss.read_index(path)
        lookup = {}
        
        metadata_path = path + ".lookup"
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    if "|||" in line:
                        parts = line.strip().split("|||", 1)
                        if len(parts) == 2:
                            i, chunk = parts
                            # Unescape special characters
                            chunk = chunk.replace("｜｜｜", "|||").replace("\\n", "\n").replace("\\r", "\r")
                            lookup[int(i)] = chunk
                except ValueError as e:
                    print(f"Warning: Skipping malformed line {line_num} in metadata: {e}")
                    continue
        
        return index, lookup
        
    except Exception as e:
        raise Exception(f"Error loading FAISS index: {str(e)}")


def search_chunks(index: faiss.Index, chunk_lookup: Dict[int, str], query: str, top_k: int = 15) -> List[str]:
    """Enhanced chunk search with insurance-specific optimizations"""
    try:
        # Create query embedding
        query_embedding, _ = embed_batch([query])
        query_vec = np.array(query_embedding).reshape(1, -1)
        
        # Normalize query vector
        faiss.normalize_L2(query_vec)
        
        # Search with more candidates for better recall
        search_k = min(top_k * 3, index.ntotal)
        similarities, indices = index.search(query_vec, search_k)
        
        relevant_chunks = []
        
        # Dynamic threshold based on query type
        base_threshold = 0.15
        
        # Lower threshold for specific insurance terms
        insurance_terms = [
            'grace', 'waiting', 'premium', 'maternity', 'pre-existing', 'cataract',
            'organ', 'donor', 'claim', 'discount', 'hospital', 'ayush', 'room', 'icu'
        ]
        
        query_lower = query.lower()
        if any(term in query_lower for term in insurance_terms):
            threshold = base_threshold * 0.7  # Lower threshold for insurance terms
        else:
            threshold = base_threshold
        
        # Collect relevant chunks
        for sim, idx in zip(similarities[0], indices[0]):
            if idx in chunk_lookup and sim > threshold:
                chunk_text = chunk_lookup[idx]
                relevant_chunks.append(chunk_text)
        
        # If no results above threshold, use keyword matching fallback
        if not relevant_chunks:
            relevant_chunks = keyword_fallback_search(chunk_lookup, query, top_k)
        
        # If still no results, take top similarity matches
        if not relevant_chunks:
            for idx in indices[0][:top_k]:
                if idx in chunk_lookup:
                    relevant_chunks.append(chunk_lookup[idx])
        
        return relevant_chunks[:top_k]
        
    except Exception as e:
        print(f"Error in semantic search: {e}")
        # Fallback to keyword search
        return keyword_fallback_search(chunk_lookup, query, top_k)


def keyword_fallback_search(chunk_lookup: Dict[int, str], query: str, top_k: int) -> List[str]:
    """Keyword-based fallback search for insurance documents"""
    query_terms = set(re.findall(r'\b\w{3,}\b', query.lower()))
    
    # Insurance-specific keyword expansion
    keyword_expansions = {
        'grace': ['grace period', 'premium payment'],
        'waiting': ['waiting period', 'pre-existing'],
        'maternity': ['childbirth', 'pregnancy', 'delivery'],
        'cataract': ['eye', 'surgery'],
        'organ': ['donor', 'transplant', 'harvesting'],
        'claim': ['discount', 'no claim', 'NCD'],
        'hospital': ['institution', 'inpatient', 'beds'],
        'ayush': ['ayurveda', 'yoga', 'naturopathy', 'unani', 'siddha', 'homeopathy'],
        'room': ['rent', 'daily', 'charges'],
        'icu': ['intensive', 'care', 'unit']
    }
    
    # Expand query terms
    expanded_terms = set(query_terms)
    for term in query_terms:
        if term in keyword_expansions:
            expanded_terms.update(keyword_expansions[term])
    
    # Score chunks based on keyword matches
    scored_chunks = []
    for idx, chunk in chunk_lookup.items():
        chunk_lower = chunk.lower()
        score = 0
        
        # Count exact matches
        for term in expanded_terms:
            if len(term) >= 3:
                matches = len(re.findall(r'\b' + re.escape(term) + r'\b', chunk_lower))
                score += matches * (2 if len(term) > 5 else 1)
        
        # Bonus for numerical patterns important in insurance
        numerical_patterns = [
            r'\b(?:thirty|30)\s+days?\b',
            r'\b(?:36|thirty-?six)\s+months?\b',
            r'\b(?:24|twenty-?four)\s+months?\b',
            r'\btwo\s+years?\b',
            r'\b5%\b',
            r'\b[12]%\s*(?:of\s+)?(?:SI|sum\s+insured)\b'
        ]
        
        for pattern in numerical_patterns:
            if re.search(pattern, chunk_lower):
                score += 5
        
        if score > 0:
            scored_chunks.append((chunk, score))
    
    # Sort by score and return top results
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in scored_chunks[:top_k]]


def call_llm_with_structured_prompt(context: str, questions: List[str]) -> List[str]:
    """
    Enhanced LLM call with insurance-domain specific prompting
    """
    try:
        # Create enhanced prompt with specific instructions for insurance documents
        prompt = f"""You are an expert insurance policy analyst specializing in extracting precise information from policy documents. Your expertise includes understanding insurance terminology, policy conditions, waiting periods, and coverage limitations.

CRITICAL INSTRUCTIONS:
1. Extract information EXACTLY as stated in the document - preserve precise wording and numerical values
2. For time periods: Look for and quote exact phrases like "thirty days", "36 months", "two years"
3. For percentages: Quote exact figures like "5% discount on base premium", "1% of SI", "2% of SI"
4. For definitions: Include all key requirements, conditions, and specifications mentioned
5. For coverage questions: Specify ALL conditions, limitations, eligibility requirements, and exclusions
6. For waiting periods: Include both the duration AND what triggers the coverage
7. If information is not explicitly found, state: "Information not available in the provided document"
8. Pay attention to plan-specific variations (Plan A vs Plan B, etc.)

INSURANCE DOCUMENT CONTEXT:
{context}

QUESTIONS TO ANALYZE:
"""
        
        for i, question in enumerate(questions, 1):
            prompt += f"{i}. {question}\n"
        
        prompt += """
Provide comprehensive answers in this format:
1. [Complete answer with all specific details, conditions, and exact quotes from document]
2. [Complete answer with all specific details, conditions, and exact quotes from document]
...

Focus on these critical elements:
- Extract exact time periods (include both numbers and words: "thirty days", "36 months", etc.)
- Include specific percentages, amounts, and limits with precise formatting
- Mention ALL conditions, requirements, and eligibility criteria
- Quote exact definitions including technical requirements
- Include relevant legal references, act names, or regulatory mentions
- Specify any plan-specific differences or variations
- Include any exceptions, exclusions, or special circumstances mentioned"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a senior insurance policy analyst with 20+ years of experience in policy interpretation. Your specialty is extracting precise, complete information from complex insurance documents. 

Key principles:
- Accuracy over brevity - include all relevant details
- Quote exact language from documents when possible
- Identify and include all conditions and limitations
- Pay special attention to numerical values and time periods
- Understand that insurance language is precise and every word matters"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,  # Maximum consistency
            max_tokens=16384,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        answer_text = response.choices[0].message.content
        
        # Enhanced answer parsing with better error handling
        answers = parse_structured_answers(answer_text, len(questions))
        
        return answers
        
    except Exception as e:
        error_msg = f"Error processing with LLM: {str(e)}"
        return [error_msg for _ in questions]


def parse_structured_answers(text: str, expected_count: int) -> List[str]:
    """Enhanced parsing with better handling of complex insurance answers"""
    
    # Clean up the text first
    text = text.strip()
    
    # Try multiple parsing strategies
    answers = []
    
    # Strategy 1: Split by numbered list pattern
    numbered_pattern = r'\n\s*(\d+)\.\s*'
    parts = re.split(numbered_pattern, text)
    
    if len(parts) > 2:  # We have numbered sections
        # Skip the first part (before first number)
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                answer = parts[i + 1].strip()
                # Clean up the answer
                answer = clean_answer_text(answer)
                answers.append(answer)
    
    # Strategy 2: If numbered parsing failed, try line-by-line
    if len(answers) < expected_count:
        lines = text.split('\n')
        current_answer = ""
        answer_count = 0
        
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+\.', line):
                if current_answer and answer_count < expected_count:
                    answers.append(clean_answer_text(current_answer))
                    answer_count += 1
                current_answer = re.sub(r'^\d+\.\s*', '', line)
            elif current_answer:
                current_answer += " " + line
        
        # Add the last answer
        if current_answer and answer_count < expected_count:
            answers.append(clean_answer_text(current_answer))
    
    # Ensure we have enough answers
    while len(answers) < expected_count:
        answers.append("Information not available in the provided document")
    
    # Trim to expected count
    return answers[:expected_count]


def clean_answer_text(answer: str) -> str:
    """Clean and format individual answers"""
    # Remove any trailing question numbering
    answer = re.sub(r'\n\s*\d+\.\s*.*,', '', answer, flags=re.DOTALL)
    
    # Clean up whitespace
    answer = re.sub(r'\s+', ' ', answer).strip()
    
    # Ensure proper sentence ending
    if answer and not re.search(r'[.!?]', answer):
        if not answer.endswith('"') and not answer.lower().endswith('act'):
            # Only add period if it doesn't end with certain patterns
            if not re.search(r'(?:limited|applicable|specified|mentioned)', answer.lower()):

                answer += '.'
    
    # Capitalize first letter if needed
    if answer and answer[0].islower() and not answer.startswith('e.g.'):
        answer = answer[0].upper() + answer[1:]
    
    return answer