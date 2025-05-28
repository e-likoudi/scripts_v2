import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ollama
from langchain.schema import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from new_protocol_tools.small_summaries import generate_summary
from new_protocol_tools.classify_summaries import classify_summary
from new_protocol_tools.phase_protocol import phase_protocol
from new_protocol_tools.detail_protocol import detail_protocol
from new_protocol_tools.focus_protocol import focus_protocol
from new_protocol_tools.organize_similar import group_by_category, organize_by_priority
from new_protocol_tools.report_format import format_protocol_report
from new_protocol_tools.critical_comps import detect_critical_components, generate_missing_components_report
from new_protocol_tools.rich_format_report import generate_data_rich_report
from basic_tools.config import NEW_PROTOCOL_FILE, CHROMA_PATH, BOOK_FOR_QA, PROTOCOL_MODEL, MODEL

def get_documents_from_chroma():
    embedding_function = OllamaEmbeddings(model=MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            
    collection = vectorstore._client.get_collection(BOOK_FOR_QA)
    data = collection.get(include=["documents"])
            
    raw_documents = data.get("documents", [])
    documents = [Document(page_content=doc) for doc in raw_documents if isinstance(doc, str)]

    return documents

def identify_cell_line(documents):  #not used for new protocol file
    cell_keywords = {
        'ESCs': ['embryonic stem', 'esc'],
        'iPSCs': ['induced pluripotent', 'ipsc', 'ips cell']
    }
    
    esc_count = 0  # Initialize with 0
    ipsc_count = 0  # Initialize with 0
    
    for doc in documents[:5]:  # Only check first 5 documents
        text = doc.page_content.lower()
        for keyword in cell_keywords['ESCs']:
            if keyword in text:
                esc_count += 1
        for keyword in cell_keywords['iPSCs']:
            if keyword in text:
                ipsc_count += 1
    
    if esc_count > ipsc_count:
        return 'ESCs'
    else:
        return 'iPSCs'

def extract_data(summaries):
    """Process summaries using new classification system"""
    extracted = []
    
    for summary in summaries:
        if not isinstance(summary, dict) or 'text' not in summary:
            continue
            
        text = summary["text"]
        
        # Classify using new system
        classification = classify_summary(text)
        components = detect_critical_components(text)
        
        processed = {
            "id": summary.get("id", "N/A"),
            "text": text,
            "classification": classification,
            "components": components,
            "original_chunks": summary.get("original_chunks", [])
        }

        # Extract data based on focus areas
        focus_data = {}
        if "timeline" in classification["focus"]:
            focus_data.update(phase_protocol(text, classification["phase"]))
        if any(f in classification["focus"] for f in ["efficiency", "gene_markers"]):
            focus_data.update(detail_protocol(text, classification["detail_level"]))
        if "critical_steps" in classification["focus"]:
            focus_data.update(focus_protocol(text, classification["focus"]))
        
        processed.update(focus_data)
        extracted.append(processed)
    
    return {
        "by_category": group_by_category(extracted),
        "by_priority": organize_by_priority(extracted),
        "raw_data": extracted
    }

def generate_final_report(structured_data):
    """Generate refined protocol report"""
    # Get the most common classification for header
    phases = structured_data["by_category"].get("phases", {})
    main_classification = max(
        phases.items(),
        key=lambda x: len(x[1]),
        default=("unclassified", [])
    )[0]

    # Safely get detail level with fallback
    detail_levels = structured_data["by_category"].get("detail_levels", {})
    detail_level = max(
        detail_levels.items(),
        key=lambda x: len(x[1]),
        default=("unclassified", [])
    )[0]

    # Safely get focus areas
    focus_areas = list(structured_data["by_category"].get("focus_areas", {}).keys())
    if not focus_areas:
        focus_areas = ["unclassified"]

    # Format report (now with safe defaults)
    draft = format_protocol_report(
        structured_data["by_priority"],
        phase=main_classification,
        detail_level=detail_level,
        focus_areas=focus_areas
    )
    
    # Add missing components
    draft = generate_missing_components_report(
        draft,
        structured_data["raw_data"]
    )
    
    # Refine with LLM
    refinement_prompt = f"""
    Refine this stem cell protocol report with focus on:
    1. Logical progression through {main_classification} phase
    2. Technical accuracy for {main_classification} stage
    3. Completeness of critical components
    
    === PROTOCOL ===
    {draft}
    
    === PHASE-SPECIFIC CHECKS ===
    {"- Verify initial seeding density" if main_classification == "early" else ""}
    {"- Check differentiation markers" if main_classification == "mid" else ""}
    {"- Validate maturation criteria" if main_classification == "late" else ""}
    """
    
    response = ollama.chat(
        model=PROTOCOL_MODEL,
        messages=[
            {"role": "system", "content": "You are a stem cell protocol specialist"},
            {"role": "user", "content": refinement_prompt}
        ],
        options={"temperature": 0}
    )
    return response['message']['content']


def save_final_report(protocol_info):
    with open(NEW_PROTOCOL_FILE, 'w', encoding='utf-8') as f:
        f.write("Results\n\n")
        f.write(protocol_info)
            
    print(f"Report saved to {NEW_PROTOCOL_FILE}")
    
def protocol():
    documents = get_documents_from_chroma()
    #cell_line = identify_cell_line(documents)
    summaries_list = generate_summary(documents)
    extracted = extract_data(summaries_list)

    final_report = generate_final_report(extracted)
    save_final_report(final_report)

    #protocol_summary = simple_protocol(cell_type, summaries_list) // txt in google drive


if __name__ == "__main__":
    protocol()
    