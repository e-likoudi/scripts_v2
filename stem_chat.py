import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain.schema import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from basic_tools.config import MODEL, CHROMA_PATH, BOOK_FOR_QA, BOOKS_PATH, QUESTION
from more_tools.classify_query import classify_query
from more_tools.response_generation import generate_response
from more_tools.ar_core import adaptive_retrieval

embedding_function = OllamaEmbeddings(model=MODEL)
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
query = QUESTION


def rag_with_adaptive_retrieval(pdf_path, query, k=4, user_context=None):
    """
    Complete RAG pipeline with adaptive retrieval.
    
    Args:
        pdf_path (str): Path to PDF document
        query (str): User query
        k (int): Number of documents to retrieve
        user_context (str): Optional user context
        
    Returns:
        Dict: Results including query, retrieved documents, query type, and response
    """
    print("\n=== RAG WITH ADAPTIVE RETRIEVAL ===")
    print(f"Query: {query}")
    
    # Classify the query to determine its type
    query_type = classify_query(query)
    print(f"Query classified as: {query_type}")
    
    # Retrieve documents using the adaptive retrieval strategy based on the query type
    retrieved_docs = adaptive_retrieval(query, k, user_context)
    
    # Generate a response based on the query, retrieved documents, and query type
    response = generate_response(query, retrieved_docs, query_type)
    
    # Compile the results into a dictionary
    result = {
        "query": query,
        "query_type": query_type,
        "retrieved_documents": retrieved_docs,
        "response": response
    }
    
    print("\n=== RESPONSE ===")
    print(response)
    
    return result