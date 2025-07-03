import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
from basic_tools.config import PROTOCOL_MODEL

def create_protocol(cell_line, gene_markers):
    refine_prompt = """
    You are an expert in stem cell differentiation protocols.    
    You are given a protocol about the differentiation process of the cell line: {cell_line}.
    Your task is to refine and organize this information into a complete guide for replicating the experiment.
        
    Raw Protocol Data:
    {raw_data}

    If any information is missing, should use the 'source_documents' field to find the missing information.

    The protocol should be structured as follows:
    - Step 0: All information available about the undifferentiated cells.
    - Steps 1 through n-1: All information available about the differentiation process.
    - The last step (n): All information available about the final differentiated state of the cells.
    - Steps should be organized in chronological order and contain timestamps.
    - Steps should not have overlapping information.
    - For each step, include the following details:
      - Stage
      - Duration
      - Description
      - Specific Step
      - Media
      - Serums
      - Supplements
      - Growth Factors
      - Cytokines
      - Passaging
      - Markers
    """

    prompt_template = ChatPromptTemplate.from_template(refine_prompt)
    prompt = prompt_template.format(cell_line=cell_line, raw_data=gene_markers)
    model = Ollama(model=PROTOCOL_MODEL)
    response = model.invoke(prompt)

    return response