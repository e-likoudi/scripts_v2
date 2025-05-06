import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ollama
from langchain.schema import Document
from langchain_community.vectorstores.chroma import Chroma
from protocol_tools.classify_protocol import classify_protocol
from protocol_tools.factual_protocol import factual_protocol
from protocol_tools.analytical_protocol import analytical_protocol
from protocol_tools.contexual_protocol import contextual_protocol
from protocol_tools.group_similar import safe_grouping
from protocol_tools.format_report import format_report
from protocol_tools.critical_components import contains_critical_components, ensure_critical_components
from basic_tools.config import PROTOCOL_FILE, CHROMA_PATH, BOOK_FOR_QA, PROTOCOL_MODEL

vectorstore = Chroma(persist_directory=CHROMA_PATH)

def load_chroma():

    collection = vectorstore._client.get_collection(BOOK_FOR_QA)
    data = collection.get(include=["documents"])
    
    raw_documents = data.get("documents", [])
    ids = data.get("ids", [])  
    
    return [
        {"id": id, "text": text}
        for id, text in zip(ids, raw_documents)
        if isinstance(text, str)
    ]

# Modify extract_data in protocol.py
def extract_data(chunks_list):
    extracted = []
    
    for chunk in chunks_list:
        if not isinstance(chunk, dict) or 'text' not in chunk:
            continue
            
        text = chunk["text"]
        classification = classify_protocol(text)
        components = contains_critical_components(text)
        
        processed = {
            "id": chunk.get("id", "N/A"),
            "text": text,
            "classification": classification[0],
            "components": components  # Now a proper dict
        }


        if classification[0] == "factual":
            processed.update(factual_protocol(text))   
        elif classification[0] == "analytical":
            processed.update(analytical_protocol(text))
        elif classification[0] == "contextual":
            processed.update(contextual_protocol(text))
        else:
            processed.update(factual_protocol(text))
        
        extracted.append(processed)
    
    return safe_grouping(extracted)

def generate_final_report(structured_data):
    refinement_prompt = f"""
    You are a stem cell protocol expert. Refine this draft protocol report with special attention to differentiation steps:

    === DRAFT PROTOCOL ===
    {structured_data}

    === CRITICAL PRIORITIES ===
    1. Remove duplicates but preserve all unique information
    2. Ensure differentiation steps progress logically (no repetition)
    3. Verify temporal progression makes biological sense
    4. Verify all time durations make sense in sequence
    5. Ensure complete component documentation

    === COMPONENT REQUIREMENTS ===
    Serums/Supplements: Must specify:
    - Base medium (e.g., DMEM/F12)
    - Serum type and % (e.g., 10% FBS)
    - Key supplements (e.g., 1x B27)
    
    Growth Factors: Must specify:
    - Full name (e.g., BMP4 not just "BMP")
    - Concentration (e.g., 10ng/ml)
    - Temporal application (e.g., Days 1-3)
    
    Culture Matrices: Must specify:
    - Coating details (e.g., Matrigel 1:30)
    - Incubation parameters
    - Pre-treatment if any
    
    Gene Markers: Should link to:
    - Expected expression timing
    - Validation method (IF, qPCR etc.)
    - Positive/Negative expectations

    FINAL CHECK: 
    1. Have all components been explicitly named?
    2. Does each differentiation stage have clear:
       - Input cell state
       - Molecular triggers
       - Expected output
    3. Are all durations biologically plausible?
    """
    
    response = ollama.chat(
        model=PROTOCOL_MODEL,
        messages=[
            {
                "role": "system", 
                "content": """You are a QA specialist for stem cell protocols. Your task is to:
                - Preserve all factual information
                - Enhance clarity and completeness
                - Ensure technical accuracy
                - Maintain protocol logic"""
            },
            {
                "role": "user", 
                "content": refinement_prompt
            }
        ],
        options={
            "temperature": 0,  # Lower temperature for less creativity
            "repeat_penalty": 1.5  # Penalize repetitive output
        }
    )
    return response['message']['content']

def save_final_report(report: str, filename: str = None):
    """Saves with error handling and metadata"""
    filename = filename or PROTOCOL_FILE
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=== STEM CELL PROTOCOL ===\n\n")
            f.write(report)
            f.write(f"\n\n=== END ===")
        print(f"Report saved to {filename}")
    except Exception as e:
        print(f"Error saving report: {str(e)}")
        # Fallback to alternate location
        with open("protocol_fallback.txt", 'w') as f:
            f.write(report)


# Example usage
def protocol():
    chunks_list = load_chroma()
    extracted_data = extract_data(chunks_list)
    draft_report = format_report(extracted_data)
    draft_report = ensure_critical_components(draft_report, extracted_data)
    final_report = generate_final_report(draft_report)
    save_final_report(final_report)