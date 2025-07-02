import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
from basic_tools.config import PROTOCOL_MODEL

def create_protocol(cell_line, gene_markers):
    refine_prompt = """
    You are an expert in stem cell differentiation protocols.    
    You are given a protocol about the differentiation process of {cell_line}.
    Your task is to refine and organize this information into a clear and structured differentiation protocol.
        
    Raw Protocol Data:
    {raw_data}

    The protocol should be structured as follows:
    - Step 0: All the information available about the undifferentiated cells.
    - Steps 1 through n-1: All the information available about the differentiation process.
    - The last step (n): All the information available about the final differentiated state of the cells.
    """
    #raw_data = []
    #for step in gene_markers:
    #    raw_data.append(
    #        f"Stage: {step.get('stage', 'N/A')}\n"
    #        f"Duration: {step.get('duration', 'N/A')}\n"
    #        f"Description: {step.get('reason', 'N/A')}\n"
    #        f"Specific Step: {step.get('specific_step', 'N/A')}\n"
    #        f"Media: {step.get('basic_media', 'N/A')}\n"
    #        f"Supplements: {step.get('serums_supplements', 'N/A')}\n"
    #        f"Growth Factors: {step.get('growth_factors', 'N/A')}\n"
    #        f"Cytokines: {step.get('cytokines_supplements', 'N/A')}\n"
    #       f"Passaging: {step.get('passaging', 'N/A')}\n"
    #       f"Markers: {step.get('gene_markers', 'N/A')}\n"
    #       "-----\n"
    #   )

    prompt_template = ChatPromptTemplate.from_template(refine_prompt)
    prompt = prompt_template.format(cell_line=cell_line, raw_data=gene_markers)
    model = Ollama(model=PROTOCOL_MODEL)
    response = model.invoke(prompt)

    return response