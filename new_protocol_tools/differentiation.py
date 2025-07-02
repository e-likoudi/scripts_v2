import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.llms.ollama import Ollama 
from langchain.prompts import ChatPromptTemplate

def differentiation_stage(summary_data):  
    prompt_template = """
    You are an expert in stem cell biology analyzing protocol summaries.  
    For the given summary, determine if it describes a specific stage in cell differentiation. 
    Classify it into one of these stages:  
        1. Undifferentiated cells:
            Criteria: Cells are in a pluripotency/maintenance medium (e.g., mTeSR1, E8) or no differentiation signals are mentioned.
            Example: "Cells were passaged in mTeSR1." → Undifferentiated  
        2. Differentiation Process:  
            Criteria: Must mention:
                - A specific signal (e.g., BMP4, Wnt3a, retinoic acid).
                - A lineage-commitment step (e.g., "induced toward mesoderm").
                - A medium change to a differentiation cocktail (e.g., switching to RPMI+B27+Activin A).
            Example: "Day 1: Added BMP4 to induce trophoblast differentiation." → Differentiation Process
        3. Differentiated cells: 
            Criteria: The summary explicitly states a terminal cell type is achieved (e.g., "beating cardiomyocytes," "mature neurons").
            Example: "After 21 days, cells expressed cTNT and exhibited contractility." → Differentiated

    Rules:
    - If no differentiation-related action is described, return: "No differentiation step in this summary." 
    - Respond concisely using the following format:  

        "stage": "[Undifferentiated cells/Differentiation Process/Differentiated cells/No differentiation step]",  
        "reason": "One paragraph describing the procedure in that stage",  
        "specific_step": "[Optional: lineage specification if applicable]"  
    
    Input Summary to Analyze:
    "{summary}"

    """

    diff_list = []

    prompt_template = ChatPromptTemplate.from_template(prompt_template)
    model = Ollama(model="llama3.1:latest")

    for i, summary in enumerate(summary_data['summaries']):
        prompt = prompt_template.format(summary=summary)
        result = model.invoke(prompt)

        entry = {
            "stage": "No differentiation step in this summary",
            "reason": "",
            "specific_step": "",
            "source_documents": summary_data['source_documents'][i] if i < len(summary_data['source_documents']) else None
        }

        # Parse the result
        for line in result.splitlines():
            line = line.strip().strip('"').strip(',')
            if line.startswith('"stage":'):
                entry["stage"] = line.split(':', 1)[1].strip().strip('"')
            elif line.startswith('"reason":'):
                entry["reason"] = line.split(':', 1)[1].strip().strip('"')
            elif line.startswith('"specific_step":'):
                entry["specific_step"] = line.split(':', 1)[1].strip().strip('"')
        
        diff_list.append(entry)

    return diff_list