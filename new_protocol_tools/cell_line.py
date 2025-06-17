from langchain_community.llms.ollama import Ollama 
from langchain.prompts import ChatPromptTemplate

def identify_cell_line(documents):  
    prompt_template = """
    You are an expert in cell line identification.
    Given the following documents, identify the cell line and the differentiation target mentioned in each document.
    Provide the cell line name and the differentiation targetname without any more details.
    Documents:
    {documents}
    """
    formatted_docs = "\n".join(doc.page_content for doc in documents)

    prompt_template = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt_template.format(documents=formatted_docs)

    model = Ollama(model="gemma3:12b")
    response = model.invoke(prompt)

    return response