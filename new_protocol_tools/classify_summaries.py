import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ollama
from basic_tools.config import PROTOCOL_MODEL

classify_prompt = """
    You are an expert at classifying stem cell protocol queries. 
    First, classify the query into exactly one category:
    - Phase: which part of the protocol this refers to — early, mid, or late stage
    - Detail_level: how granular the information is — low, medium, or high
    - Focus: what type(s) of protocol information are discussed — can include multiple sub-tags
    
    Depending on your choice, use the corresponding sub-tags exactly as defined below:

    Phase (choose one):
    - early: setup or initial treatments (e.g., Day 0–3, induction)
    - mid: intermediate steps (e.g., transition phases, Day 4–10)
    - late: maturation, final stages, or functional assessment

    Detail_Level (choose one):
    - low: vague or high-level descriptions
    - medium: moderately detailed, includes some key terms
    - high: very specific, includes names, timings, concentrations, etc.

    FOCUS (choose one or more of the following):
    - timeline: contains timing info (e.g., "Day 0–2: CHIR99021")
    - reagents: lists of small molecules, inhibitors, etc.
    - media: mentions of basal media (e.g., RPMI, DMEM, B27, etc.)
    - gene_markers: specific gene expression info (e.g., "upregulation of TNNT2")
    - culture_matrix: refers to physical support (e.g., Matrigel, laminin)
    - outcomes: refers to observed results (e.g., morphology, beating cells)
    - efficiency: yield metrics, percentages of differentiation
    - critical_steps: warnings, essential conditions, or must-do actions

    Return ONLY in this format: "Category|Target1,Target2,..."
    Examples: 
    - Phase|early 
    - Focus|timeline,reagents
    """
    
def classify_summary(summary_text):
    response = ollama.chat(
        model=PROTOCOL_MODEL,
        messages=[
            {"role": "system", "content": classify_prompt},
            {"role": "user", "content": summary_text}
        ],
        options={"temperature": 0}
    )

    raw = response['message']['content'].strip()
    #print(f"Raw LLM response: {raw}")  # Debugging output

 # Parse all classifications
    classifications = {}
    for line in raw.split('\n'):
        line = line.strip()
        if '|' in line:
            try:
                category, value = line.split('|', 1)
                category = category.strip().lower()
                classifications[category] = [v.strip().lower() for v in value.split(',') if v.strip()]
            except Exception as e:
                print(f"Error processing line '{line}': {e}")

    # Return all three classifications
    return {
        'phase': classifications.get('phase', ['unclassified'])[0],
        'detail_level': classifications.get('detail_level', ['unclassified'])[0],
        'focus': classifications.get('focus', ['unclassified'])
    }
