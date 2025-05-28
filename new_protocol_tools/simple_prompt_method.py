import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain.chat_models import ChatOllama
from langchain.schema import HumanMessage
from basic_tools.config import PROTOCOL_MODEL
    
def simple_protocol(cell_type, summaries):

    context = " ".join([summary['text'] for summary in summaries])

    extraction_prompt = f"""
    
    You are an expert assistant. Your only task is to extract **structured data** from the following protocol summary.

    Your output MUST strictly follow the format below.
    You must NOT include evaluation, interpretation, or commentary.
    If any field is missing, leave it blank. Do not explain why.

    === BEGIN OUTPUT FORMAT ===

    Protocol Title: <...>
    Stem Cell Type: {cell_type}
    Target Cell Type: <...>
    Timeline:
      - Day X–Y: <Factor(s)>
    Basal Media: <...>
    Culture Matrix: <...>
    Growth Factors: <...>
    Serums: <...>
    Gene Markers:
      - <Gene>: <upregulated/downregulated>

    === END OUTPUT FORMAT ===

    Summary:
    {context}
    """

    llm = ChatOllama(model=PROTOCOL_MODEL, temperature=0)
    extract_prompt_text = extraction_prompt.format(context=context)

    messages = [
        HumanMessage(content=extract_prompt_text)
    ]

    extracted_info = llm(messages).content

    return extracted_info