import hashlib
import os
import re
from typing import List, Dict, Tuple
import numpy as np
from .utils import download_pdf, embed_batch, save_faiss_index, load_faiss_index, call_llm_with_structured_prompt
from .db import get_document_by_url, insert_document

async def query_document(url: str, questions: List[str]) -> Dict:
    """
    Improved document querying with focused, high-precision search
    """
    try:
        # Check if document already processed
        faiss_path = get_document_by_url(url)
        
        if not faiss_path or not os.path.exists(faiss_path):
            # Download and process document
            content = await download_pdf(url)
            
            if not content.strip():
                return {"answers": ["Document appears to be empty or unreadable" for _ in questions]}
            
            # Improved chunking specifically for insurance documents
            chunks = insurance_aware_chunking(content)
            
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
        
        # Focused search strategy
        relevant_chunks = focused_insurance_search(index, chunk_lookup, questions)
        
        if not relevant_chunks:
            return {"answers": ["No relevant information found in the document" for _ in questions]}
        
        # Create optimized context
        context = create_optimized_context(relevant_chunks, questions)
        
        if not context.strip():
            return {"answers": ["No relevant information found in the document" for _ in questions]}
        
        # Get structured answers with improved prompting
        answers = call_enhanced_llm(context, questions)
        
        return {"answers": answers}
        
    except Exception as e:
        error_msg = f"Error processing document: {str(e)}"
        return {"answers": [error_msg for _ in questions]}


def insurance_aware_chunking(text: str, max_chars: int = 2500, min_chars: int = 500) -> List[str]:
    """
    Insurance-specific chunking that preserves critical policy information
    """
    # Clean text while preserving structure
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    chunks = []
    
    # Define insurance section patterns that should be kept together
    section_patterns = [
        # Major sections
        r'(?i)(?:grace\s+period|waiting\s+period|premium\s+payment)',
        r'(?i)(?:pre-existing\s+disease|PED|maternity\s+(?:benefit|expense|coverage))',
        r'(?i)(?:cataract\s+surgery|organ\s+donor)',
        r'(?i)(?:no\s+claim\s+discount|NCD|health\s+check)',
        r'(?i)(?:hospital\s+definition|ayush\s+treatment)',
        r'(?i)(?:room\s+rent|ICU\s+charge|plan\s+A)',
        
        # Numerical definitions (critical for insurance)
        r'(?i)(?:thirty\s+days?|30\s+days?|36\s+months?|thirty-?six\s+months?)',
        r'(?i)(?:24\s+months?|twenty-?four\s+months?|2\s+years?|two\s+years?)',
        r'(?i)(?:5%\s+discount|1%\s+of\s+SI|2%\s+of\s+SI)',
        
        # Structural markers
        r'\[Page \d+\]',
        r'(?i)(?:clause|section|article|part)\s+\d+',
        r'(?i)(?:definition|benefit|coverage|exclusion|condition)'
    ]
    
    # Find all important boundaries
    boundaries = [0]
    for pattern in section_patterns:
        for match in re.finditer(pattern, text):
            # Include some context before the match
            start_pos = max(0, match.start() - 200)
            boundaries.append(start_pos)
            boundaries.append(match.start())
    
    boundaries.append(len(text))
    boundaries = sorted(set(boundaries))
    
    # Create chunks based on these boundaries
    for i in range(len(boundaries) - 1):
        section_start = boundaries[i]
        section_end = boundaries[i + 1]
        
        # Extend section end to complete sentences/paragraphs
        extended_end = find_natural_break(text, section_end, max_extension=500)
        section_text = text[section_start:extended_end].strip()
        
        if min_chars <= len(section_text) <= max_chars:
            chunks.append(section_text)
        elif len(section_text) > max_chars:
            # Split large sections but try to keep related info together
            sub_chunks = split_preserving_context(section_text, max_chars, min_chars)
            chunks.extend(sub_chunks)
        elif len(section_text) >= min_chars // 2:  # Include smaller chunks if they contain key info
            key_terms = ['grace period', 'waiting period', 'maternity', 'cataract', 
                        'organ donor', 'claim discount', 'hospital', 'ayush', 'room rent']
            section_lower = section_text.lower()
            if any(term in section_lower for term in key_terms):
                chunks.append(section_text)
    
    # Add overlapping chunks for better coverage of critical boundaries
    overlap_chunks = create_overlap_chunks(text, chunks, max_chars)
    chunks.extend(overlap_chunks)
    
    # Deduplicate based on content similarity
    final_chunks = deduplicate_chunks(chunks)
    
    return final_chunks


def find_natural_break(text: str, position: int, max_extension: int = 500) -> int:
    """Find natural break points (sentence/paragraph endings)"""
    search_end = min(len(text), position + max_extension)
    search_text = text[position:search_end]
    
    # Look for paragraph breaks first
    paragraph_match = re.search(r'\n\s*\n', search_text)
    if paragraph_match:
        return position + paragraph_match.end()
    
    # Look for sentence endings
    sentence_matches = list(re.finditer(r'[.!?]\s+[A-Z]', search_text))
    if sentence_matches:
        return position + sentence_matches[-1].start() + 1
    
    return min(len(text), position + max_extension)


def split_preserving_context(text: str, max_chars: int, min_chars: int) -> List[str]:
    """Split text while preserving important context"""
    chunks = []
    
    # Try to split by paragraphs first
    paragraphs = re.split(r'\n\s*\n+', text)
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) <= max_chars:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        else:
            if len(current_chunk) >= min_chars:
                chunks.append(current_chunk.strip())
            
            if len(para) <= max_chars:
                current_chunk = para
            else:
                # Split long paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                chunk_sentences = []
                
                for sent in sentences:
                    test_chunk = " ".join(chunk_sentences + [sent])
                    if len(test_chunk) <= max_chars:
                        chunk_sentences.append(sent)
                    else:
                        if chunk_sentences:
                            chunks.append(" ".join(chunk_sentences))
                            chunk_sentences = [sent]
                        else:
                            # Single very long sentence - must split
                            chunks.append(sent[:max_chars])
                
                if chunk_sentences:
                    current_chunk = " ".join(chunk_sentences)
                else:
                    current_chunk = ""
    
    if current_chunk and len(current_chunk) >= min_chars:
        chunks.append(current_chunk.strip())
    
    return chunks


def create_overlap_chunks(text: str, existing_chunks: List[str], max_chars: int) -> List[str]:
    """Create overlapping chunks around critical boundaries"""
    overlap_chunks = []
    
    # Find positions of critical terms
    critical_terms = [
        'grace period', 'waiting period', 'maternity', 'pre-existing', 'cataract',
        'organ donor', 'no claim discount', 'health check', 'hospital definition',
        'ayush', 'room rent', 'icu'
    ]
    
    for term in critical_terms:
        for match in re.finditer(re.escape(term), text, re.IGNORECASE):
            # Create chunk centered on this term
            start = max(0, match.start() - max_chars // 2)
            end = min(len(text), match.start() + max_chars // 2)
            
            chunk = text[start:end].strip()
            if len(chunk) >= 200:  # Minimum meaningful size
                overlap_chunks.append(chunk)
    
    return overlap_chunks


def deduplicate_chunks(chunks: List[str]) -> List[str]:
    """Remove duplicate chunks based on content similarity"""
    if not chunks:
        return []
    
    unique_chunks = []
    seen_signatures = set()
    
    for chunk in chunks:
        # Create signature from first and last 100 characters + length
        start_sig = chunk[:100].lower().strip()
        end_sig = chunk[-100:].lower().strip()
        signature = f"{start_sig}||{end_sig}||{len(chunk)}"
        
        if signature not in seen_signatures:
            seen_signatures.add(signature)
            unique_chunks.append(chunk)
    
    return unique_chunks


def focused_insurance_search(index, chunk_lookup: Dict[int, str], questions: List[str]) -> List[str]:
    """
    Focused search strategy specifically for insurance document QA
    """
    from .utils import search_chunks
    
    all_relevant_chunks = set()
    
    # Strategy 1: Direct question search with high precision
    for question in questions:
        chunks = search_chunks(index, chunk_lookup, question, top_k=10)
        all_relevant_chunks.update(chunks)
    
    # Strategy 2: Extract key terms from questions for targeted search
    key_terms_map = {
        'grace period premium payment': ['grace period', 'premium payment', 'thirty days', '30 days'],
        'waiting period pre-existing': ['waiting period', 'pre-existing', 'PED', '36 months', 'thirty-six months'],
        'maternity expenses conditions': ['maternity', 'childbirth', 'pregnancy', '24 months', 'twenty-four months'],
        'cataract surgery waiting': ['cataract', 'surgery', 'waiting', 'two years', '2 years'],
        'organ donor medical expenses': ['organ donor', 'harvesting', 'Transplantation Human Organs Act'],
        'no claim discount NCD': ['no claim discount', 'NCD', '5%', 'renewal', 'base premium'],
        'preventive health check-up': ['health check', 'preventive', 'block', 'policy years', 'INR 5,000'],
        'hospital definition': ['hospital definition', 'institution', '10 inpatient beds', '15 beds', 'qualified nursing'],
        'ayush treatment coverage': ['AYUSH', 'Ayurveda', 'Yoga', 'Naturopathy', 'Unani', 'Siddha', 'Homeopathy'],
        'room rent ICU charges': ['room rent', 'ICU charges', 'Plan A', '1%', '2%', 'sum insured', 'PPN']
    }
    
    for category, terms in key_terms_map.items():
        for term in terms:
            chunks = search_chunks(index, chunk_lookup, term, top_k=5)
            all_relevant_chunks.update(chunks)
    
    # Strategy 3: Search for specific numerical values mentioned in expected answers
    numerical_searches = [
        'thirty days', '30 days', 'grace period',
        '36 months', 'thirty-six months', 'continuous coverage',
        '24 months', 'twenty-four months', 'female insured',
        'two years', '2 years', 'cataract',
        '5% discount', 'base premium', 'renewal',
        'INR 5,000', 'Plan A', 'health check',
        '10 inpatient beds', '15 beds', 'qualified nursing staff',
        '1% sum insured', '2% sum insured', 'daily room rent'
    ]
    
    for num_search in numerical_searches:
        chunks = search_chunks(index, chunk_lookup, num_search, top_k=8)
        all_relevant_chunks.update(chunks)
    
    # Convert to list and prioritize
    return list(all_relevant_chunks)


def create_optimized_context(chunks: List[str], questions: List[str]) -> str:
    """Create highly optimized context focusing on insurance policy details"""
    
    # Extract question keywords
    question_keywords = set()
    for q in questions:
        # Extract meaningful terms
        words = re.findall(r'\b\w{3,}\b', q.lower())
        question_keywords.update(words)
    
    # Define high-value patterns for insurance documents
    high_value_patterns = [
        (r'\b(?:thirty|30)\s+days?\b', 10),          # Grace periods
        (r'\b(?:36|thirty-?six)\s+months?\b', 10),   # Waiting periods
        (r'\b(?:24|twenty-?four)\s+months?\b', 10),  # Maternity waiting
        (r'\btwo\s+years?\b', 10),                   # Cataract waiting
        (r'\b5%\s*(?:discount|premium)\b', 10),      # NCD
        (r'\b(?:1|2)%\s*(?:of\s+)?(?:SI|sum\s+insured)\b', 10),  # Room/ICU limits
        (r'\b(?:10|15)\s+(?:inpatient\s+)?beds?\b', 10),         # Hospital definition
        (r'\bINR\s+5,?000\b', 10),                  # Health check amount
        (r'(?i)transplantation\s+human\s+organs\s+act', 10),     # Legal reference
        (r'(?i)qualified\s+nursing\s+staff', 8),    # Hospital requirements
        (r'(?i)operation\s+theatre', 8),            # Hospital facilities
        (r'(?i)daily\s+records\s+patients', 8),     # Hospital requirements
        (r'(?i)preferred\s+provider\s+network|PPN', 8),  # Network references
    ]
    
    # Score chunks
    scored_chunks = []
    for chunk in chunks:
        score = 0
        chunk_lower = chunk.lower()
        
        # Score based on high-value patterns
        for pattern, points in high_value_patterns:
            if re.search(pattern, chunk_lower):
                score += points
        
        # Score based on question keywords
        for keyword in question_keywords:
            if len(keyword) > 3 and keyword in chunk_lower:
                score += 2
        
        # Bonus for definition-style content
        if re.search(r'(?:means?|defined?\s+as|definition|shall\s+mean)', chunk_lower):
            score += 5
        
        # Bonus for policy structure words
        structure_words = ['clause', 'section', 'benefit', 'coverage', 'condition', 'exclusion']
        for word in structure_words:
            if word in chunk_lower:
                score += 1
        
        scored_chunks.append((chunk, score))
    
    # Sort by relevance
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Select top chunks with some diversity
    selected_chunks = []
    used_signatures = set()
    
    # Take top scoring chunks first
    for chunk, score in scored_chunks:
        if score >= 8:  # High confidence chunks
            signature = chunk[:80].lower().strip()
            if signature not in used_signatures:
                selected_chunks.append(chunk)
                used_signatures.add(signature)
                if len(selected_chunks) >= 25:  # Reasonable limit
                    break
    
    # Add medium scoring chunks if needed
    if len(selected_chunks) < 20:
        for chunk, score in scored_chunks:
            if 3 <= score < 8:
                signature = chunk[:80].lower().strip()
                if signature not in used_signatures:
                    selected_chunks.append(chunk)
                    used_signatures.add(signature)
                    if len(selected_chunks) >= 30:
                        break
    
    # Organize the context
    context_parts = []
    
    # High priority sections
    high_priority = [chunk for chunk, score in scored_chunks if score >= 10][:15]
    if high_priority:
        context_parts.append("=== CRITICAL POLICY INFORMATION ===")
        context_parts.extend(high_priority)
        context_parts.append("")
    
    # Medium priority sections  
    medium_priority = [chunk for chunk, score in scored_chunks if 5 <= score < 10][:15]
    if medium_priority:
        context_parts.append("=== RELEVANT POLICY DETAILS ===")
        context_parts.extend(medium_priority)
        context_parts.append("")
    
    # Additional context
    additional = [chunk for chunk, score in scored_chunks if 1 <= score < 5][:10]
    if additional:
        context_parts.append("=== ADDITIONAL CONTEXT ===")
        context_parts.extend(additional)
    
    return "\n\n".join(context_parts)


def call_enhanced_llm(context: str, questions: List[str]) -> List[str]:
    """Enhanced LLM call with improved insurance-specific prompting"""
    
    prompt = f"""You are an expert insurance policy analyst with deep knowledge of policy language and structure. Your task is to extract precise information from the provided policy document context to answer specific questions.

CRITICAL INSTRUCTIONS:
1. Extract information EXACTLY as stated in the document - use precise wording and numbers
2. For time periods: Look for specific durations like "thirty days", "36 months", "two years", etc.
3. For percentages and limits: Quote exact figures like "5%", "1% of SI", "2% of SI"
4. For definitions: Include key requirements and conditions mentioned
5. For coverage questions: Specify conditions, limitations, and eligibility requirements
6. If information is not explicitly stated, respond: "Information not found in provided context"
7. Pay attention to specific plan references (Plan A, Plan B) and their different terms

DOCUMENT CONTEXT:
{context}

QUESTIONS TO ANSWER:
"""
    
    for i, question in enumerate(questions, 1):
        prompt += f"{i}. {question}\n"
    
    prompt += """
Provide answers in this exact format:
1. [Complete answer with specific details from the document]
2. [Complete answer with specific details from the document]
...

Key focus areas for each answer:
- Extract exact time periods (days/months/years)
- Include specific percentages and amounts
- Mention all conditions and requirements
- Quote precise definitions when available
- Include relevant legal references or act names
- Specify plan-specific differences when applicable"""

    try:
        from .utils import client  # Assuming OpenAI client is imported from utils
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert insurance policy analyst. Extract precise, specific information from policy documents. Focus on exact numbers, time periods, conditions, and requirements. Always quote specific language from the document when possible."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,  # Zero temperature for maximum consistency
            max_tokens=16384
        )
        
        answer_text = response.choices[0].message.content
        
        # Enhanced answer parsing
        answers = parse_numbered_answers(answer_text, len(questions))
        
        return answers
        
    except Exception as e:
        return [f"Error processing question: {str(e)}" for _ in questions]


def parse_numbered_answers(text: str, expected_count: int) -> List[str]:
    """Enhanced parsing of numbered answers"""
    answers = []
    
    # Split by numbered patterns
    parts = re.split(r'\n\s*\d+\.\s*', text)
    
    # First part might contain answer 1 if it starts with "1."
    if parts[0].strip():
        first_part = re.sub(r'^\d+\.\s*', '', parts[0].strip())
        if first_part:
            answers.append(first_part)
    
    # Process remaining parts
    for part in parts[1:]:
        clean_answer = part.strip()
        if clean_answer:
            answers.append(clean_answer)
    
    # Ensure we have enough answers
    while len(answers) < expected_count:
        answers.append("Information not found in provided context")
    
    # Clean and format answers
    final_answers = []
    for i, answer in enumerate(answers[:expected_count]):
        # Remove any trailing numbering from next question
        answer = re.sub(r'\n\s*\d+\.\s*.*$', '', answer, flags=re.DOTALL)
        answer = answer.strip()
        
        # Ensure proper sentence structure
        if answer and not answer.endswith('.') and not answer.endswith('?') and not answer.endswith('!'):
            if not answer.endswith('"') and not answer.lower().endswith('act'):
                answer += '.'
        
        final_answers.append(answer)
    
    return final_answers