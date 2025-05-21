import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain.schema import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from new_protocol_tools.small_summaries import generate_summary
from basic_tools.config import MODEL, CHROMA_PATH, BOOK_FOR_QA, PROTOCOL_FILE

def get_documents_from_chroma():
   
    embedding_function = OllamaEmbeddings(model=MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            
    collection = vectorstore._client.get_collection(BOOK_FOR_QA)
    data = collection.get(include=["documents"])
            
    raw_documents = data.get("documents", [])
    documents = [Document(page_content=doc) for doc in raw_documents if isinstance(doc, str)]

    return documents

def identify_cell_line(documents):
    cell_keywords = {
        'ESCs': ['embryonic stem', 'esc'],
        'iPSCs': ['induced pluripotent', 'ipsc', 'ips cell']
    }
    
    esc_count = 0  # Initialize with 0
    ipsc_count = 0  # Initialize with 0
    
    for doc in documents[:5]:  # Only check first 5 documents
        text = doc.page_content.lower()
        for keyword in cell_keywords['ESCs']:
            if keyword in text:
                esc_count += 1
        for keyword in cell_keywords['iPSCs']:
            if keyword in text:
                ipsc_count += 1
    
    if esc_count > ipsc_count:
        return 'ESCs'
    else:
        return 'iPSCs'
    
def save_final_report(cell_type, summaries, filename):
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Results\n\n")
        f.write(cell_type, '\n\n')
        for summary in summaries:
            f.write(summary['text'])
            
    print(f"Report saved to {filename}")
    

def protocol():
    documents = get_documents_from_chroma()
    cell_type = identify_cell_line(documents)
    summaries_list = generate_summary(documents, cell_type)
    save_final_report(cell_type, summaries_list, filename=PROTOCOL_FILE)

if __name__ == "__main__":
    protocol()
    