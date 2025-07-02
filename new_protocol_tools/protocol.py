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
from new_protocol_tools import IdentifyDetails
from new_protocol_tools.refine_desc import create_protocol
from basic_tools.config import CHROMA_PATH, BOOK_FOR_QA, MODEL, PROTOCOL_FILE

def get_documents_from_chroma():
    embedding_function = OllamaEmbeddings(model=MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            
    collection = vectorstore._client.get_collection(BOOK_FOR_QA)
    data = collection.get(include=["documents"])
            
    raw_documents = data.get("documents", [])
    documents = [Document(page_content=doc) for doc in raw_documents if isinstance(doc, str)]

    return documents

def summaries_for_steps(summary_data):

    steps = []

    for summary in summary_data['summaries']:
        result = differentiation_stage([summary])
        stage_name, stage_data = next(iter(result.items()))
        steps.append({
            'stage': stage_name,
            'reason': stage_data.get('reason', ''),
            'specific_step': stage_data.get('specific_step', ''),
            'source_documents': summary['source_documents']
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

    summary_data = generate_summary(documents)    # generate summaries for all documents
    print(f"Generated {len(summary_data['summaries'])} summaries")

    steps = differentiation_stage(summary_data)
    sort_steps = process_stages(steps)          # Sort by stage
    print(f"Processed {len(sort_steps)} stages")

    durations = IdentifyDetails.calculate_durations(sort_steps)
    print(f"Calculated durations for {len(durations)} steps")

    basic_media = IdentifyDetails.basic_media(durations)
    print(f"Identified basic media for {len(basic_media)} steps")

    serums_supplements = IdentifyDetails.serums_supplements(basic_media)
    print(f"Identified serums and supplements for {len(serums_supplements)} steps")

    growth_factors = IdentifyDetails.growth_factors(serums_supplements)
    print(f"Identified growth factors for {len(growth_factors)} steps")

    cytokines_supplements = IdentifyDetails.cytokines_supplements(growth_factors)
    print(f"Identified cytokines and supplements for {len(cytokines_supplements)} steps")

    passaging = IdentifyDetails.passaging(cytokines_supplements)
    print(f"Identified passaging for {len(passaging)} steps")

    gene_markers = IdentifyDetails.gene_markers(passaging)
    print(f"Identified gene markers for {len(gene_markers)} steps")
    #print(f"Result sample: {gene_markers[:3]}")    #debugging
    
    protocol_steps = create_protocol(cell_line, gene_markers)
    print(f"Created protocol steps")
    
    save_final_report(cell_line, protocol_steps) 

if __name__ == "__main__":
    protocol()
    