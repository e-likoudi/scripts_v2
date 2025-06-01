import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.llms.ollama import Ollama 
from langchain.prompts import ChatPromptTemplate
from basic_tools.config import PROTOCOL_MODEL

def identify_cell_line(documents):  
    prompt_template = """
    You are an expert in cell line identification.
    Given the following documents, identify the cell line mentioned in each document.
    Provide the cell line name and any relevant details.
    Documents:
    {documents}
    """
    formatted_docs = "\n".join(doc.page_content for doc in documents)

    prompt_template = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt_template.format(documents=formatted_docs)

    model = Ollama(model=PROTOCOL_MODEL)
    response = model.invoke(prompt)

    return response