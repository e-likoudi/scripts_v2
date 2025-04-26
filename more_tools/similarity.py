import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain.schema import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from basic_tools.config import MODEL, CHROMA_PATH, BOOK_FOR_QA

embedding_function = OllamaEmbeddings(model=MODEL)
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
collection = vectorstore._client.get_collection(BOOK_FOR_QA)
data = collection.get(include=["documents"])
        
raw_documents = data.get("documents", [])
documents = [Document(page_content=doc) for doc in raw_documents if isinstance(doc, str)]

vectordb = vectorstore.from_documents(collection_name=BOOK_FOR_QA, documents=documents, embedding=embedding_function)

class SimilarityMethods:
    @staticmethod
    def factual_similarity(query_embedding, k):
        initial_results = vectordb.similarity_search(query_embedding, k=k*2)
        return initial_results
    
    def analytical_sub_similarity(sub_query_embedding, k):
        results = vectordb.similarity_search(sub_query_embedding, k=k)
        return results
    
    def analytical_main_similarity(main_query_embedding, k):
        main_results = vectordb.similarity_search(main_query_embedding, k=k)
        return main_results
    
    def opinion_similarity(viewpoint_embedding, k):
        results = vectordb.similarity_search(viewpoint_embedding, k=k)
        return results
    
    def contexual_similarity(query_embedding, k):
        initial_results = vectordb.similarity_search(query_embedding, k=k*2)
        return initial_results
    
__all__ = ["factual_similarity", 
           "analytical_sub_similarity", 
           "analytical_main_similarity",
           "opinion_similarity",
           "contexual_similarity"]



 
