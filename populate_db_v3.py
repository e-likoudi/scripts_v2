import os
import argparse
import shutil
import chromadb
from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma

BOOKS_PATH = "/Users/chara/Documents/thesis/scripts_v2/books/"
MODEL = "nomic-embed-text"
CHROMA_PATH = f"./{MODEL}_db"

client = chromadb.PersistentClient(CHROMA_PATH)


def load_pdf(file_name):
    loader = PyPDFDirectoryLoader(BOOKS_PATH, glob=file_name)   #glob=file_name: διαβαζει ενα ενα τα αρχεια οχι ολο το folder μαζι
    documents = loader.load()
    return documents


def split_pdfs(books: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, #try 10000
        chunk_overlap=80, #try 3000
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(books)


def calculate_chunk_embeddings(chunks: list[Document]):
    embedding_model = OllamaEmbeddings(model=MODEL)
    embeddings = []  

    embeddings = embedding_model.embed_documents([chunk.page_content for chunk in chunks])

    return embeddings  



def process_pdf(file_name, chunks):                   #Process a single PDF file and add its data to the corresponding Chroma collection.

    collection_name = os.path.splitext(file_name)[0]  # Get the name without .pdf extension

    # Check if the collection already exists
    existing_collections = client.list_collections()
    if collection_name in [col.name for col in existing_collections]:
        print(f"📚The book '{collection_name}' is already in the db.")
        return

    collection = client.get_or_create_collection(name=collection_name)

    # Check if the collection already contains data
    existing_data = collection.get(include=[])
    existing_ids = set(existing_data["ids"])
   

    # Calculate embeddings and assign IDs
    chunks_with_embeddings = calculate_chunk_embeddings(chunks) 
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Prepare data for adding to the collection
    documents_to_add = []
    for chunk in chunks_with_ids:           # Ensure the chunk ID is unique

        if chunk.metadata["id"] not in existing_ids:
            documents_to_add.append(chunk.page_content)  
              

    # Add documents to the collection
    if documents_to_add:
        for i, chunk in enumerate(chunks):
            embedding = chunks_with_embeddings[i]
            metadata = {"source": file_name, "chunk_index": chunks_with_ids[i]}
            doc_id = f"{collection_name}_{i}"

            try:
                collection.add(ids=[doc_id], documents=[chunk], embeddings=[embedding], metadatas=[metadata])  

                print(f"📥 Added {len(documents_to_add)} chunks to the collection '{collection_name}'.")
            except Exception as e:
                print(f"❌ Failed to add chunks to the collection '{collection_name}': {e}")    #error somewhere around here
    else:
        print(f"No new chunks to add for '{collection_name}'.")


def calculate_chunk_ids(chunks: list[Document]):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks



    

def main():
    parser = argparse.ArgumentParser(description="Process PDF books into Chroma collections.")
    parser.add_argument(
        "--books-path",
        type=str,
        default=BOOKS_PATH,
        help="Path to the directory containing PDF books."
    )
    args = parser.parse_args()

    for file_name in os.listdir(BOOKS_PATH):
        if file_name.endswith(".pdf"):
            books = load_pdf(file_name)
            chunks = split_pdfs(books)

            process_pdf(file_name, chunks)


if __name__ == "__main__":
    main()