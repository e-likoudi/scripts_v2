import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import ollama
from langchain.schema import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from new_protocol_tools.cell_line import identify_cell_line
from new_protocol_tools.diff_steps import differentiation_steps
from new_protocol_tools.small_summaries import generate_summary
from new_protocol_tools.sort_steps import sorted_steps
from new_protocol_tools.merge_steps import merge_similar_steps
from new_protocol_tools.refine_desc import refine_with_prompt
from basic_tools.config import CHROMA_PATH, BOOK_FOR_QA, PROTOCOL_MODEL, MODEL, PROTOCOL_FILE

def get_documents_from_chroma():
    embedding_function = OllamaEmbeddings(model=MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            
    collection = vectorstore._client.get_collection(BOOK_FOR_QA)
    data = collection.get(include=["documents"])
            
    raw_documents = data.get("documents", [])
    documents = [Document(page_content=doc) for doc in raw_documents if isinstance(doc, str)]

    return documents

def summaries_for_steps(summaries_list):

    steps = []
    step_index = 0

    for summary in summaries_list:
        result = differentiation_steps([summary])
        if "No differentiation step" not in result:
            formatted_result = result.replace("Step X", f"Step {step_index}")
            steps.append(formatted_result)
            step_index += 1

    return steps

def save_final_report(cell_line, steps):
    with open(PROTOCOL_FILE, 'w', encoding='utf-8') as f:
        f.write("Identified Cell Line and Differentiation Target:\n\n")
        f.write(cell_line)
        f.write("\n\nDifferentiation Steps:\n\n")
        f.write(steps)
            
    print(f"Report saved to {PROTOCOL_FILE}")
    
def protocol():
    documents = get_documents_from_chroma()
    docs_for_cell_line = documents[:5]  # Use the first 5 documents for cell line identification
    cell_line = identify_cell_line(docs_for_cell_line)

    summaries_list = generate_summary(documents)
    print(f"Generated {len(summaries_list)} summaries")

    steps = summaries_for_steps(summaries_list)
    sort_steps = sorted_steps(steps)
    merge_steps = merge_similar_steps(sort_steps)
    refine_desc = refine_with_prompt(merge_steps)

    save_final_report(cell_line, refine_desc)

if __name__ == "__main__":
    protocol()
    