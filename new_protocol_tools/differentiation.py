import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.llms.ollama import Ollama 
from langchain.prompts import ChatPromptTemplate
from basic_tools.config import PROTOCOL_MODEL

def differentiation_stage(summary):  
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
        "reason": "Brief explanation",  
        "specific_step": "[Optional: lineage specification if applicable]"  
    
    Input Summary to Analyze:
    "{summary}"

    """

    prompt_template = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt_template.format(summary=summary)
    model = Ollama(model=PROTOCOL_MODEL)
    result = model.invoke(prompt)

    return result.strip()

#def differentiation_steps(stages):
#Step X: [Name of the differentiation stage]  
#Duration: [State the duration if available, else write 'Not specified']  
#[One paragraph describing the procedure in that stage]

#    - Step 0 must describe the condition of the undifferentiated cells before any differentiation starts.

