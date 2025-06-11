import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain.schema import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from new_protocol_tools.cell_line import identify_cell_line
from new_protocol_tools.differentiation import differentiation_stage
from new_protocol_tools.small_summaries import generate_summary
from new_protocol_tools.sort_stages import process_stages
from new_protocol_tools.merge_stages import merge_similar_steps
from new_protocol_tools.refine_desc import create_protocol, format_protocol_steps
from basic_tools.config import CHROMA_PATH, BOOK_FOR_QA, MODEL, PROTOCOL_FILE

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

    for summary in summaries_list:
        result = differentiation_stage([summary])
        stage_name, stage_data = next(iter(result.items()))
        steps.append({
            'stage': stage_name,
            'reason': stage_data.get('reason', ''),
            'specific_step': stage_data.get('specific_step', '')
        })
    return steps

def save_final_report(cell_line, protocol_result):
    with open(PROTOCOL_FILE, 'w', encoding='utf-8') as f:
        f.write("Identified Cell Line and Differentiation Target:\n\n")
        f.write(cell_line)
        f.write("\n\nDifferentiation Protocol:\n\n")
        f.write(protocol_result)
            
    print(f"Report saved to {PROTOCOL_FILE}")
    
def protocol():
    documents = get_documents_from_chroma()
    docs_for_cell_line = documents[:5]  # Use the first 5 documents for cell line identification
    cell_line = identify_cell_line(docs_for_cell_line)

    summaries_list = generate_summary(documents)
    print(f"Generated {len(summaries_list)} summaries")

    steps = summaries_for_steps(summaries_list)
    sort_steps = process_stages(steps)
    print(f"Processed {len(sort_steps)} stages")
    
    #merge_steps = merge_similar_steps(sort_steps)
    #print(f"Merged into {len(merge_steps)} steps") -> Merged into 1 steps
    
    protocol_steps = create_protocol(sort_steps)
    print(f"Created {len(protocol_steps)} protocol steps")
    
    #protocol_result = format_protocol_steps(protocol_steps)

    save_final_report(cell_line, protocol_steps) 

if __name__ == "__main__":
    protocol()
    