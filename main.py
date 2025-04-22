from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings

from basic_tools.config import MODEL, CHROMA_PATH, BOOK_FOR_QA, QUESTION
from basic_tools import CheckChromaIntegrity
from basic_tools.populate_db_v3 import populate_db
from basic_tools.query_data import query_rag
from basic_tools.summaries_v3 import generate_summary
from more_tools.classify_query import classify_query

embedding_function = OllamaEmbeddings(model=MODEL)
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

def list_books():
    
    # Remove comment if used for the first time
    #populate_db()
   
    collection_names = vectorstore._client.list_collections()
    book_titles = [c.name for c in collection_names]
        
    if not book_titles:
        print("No books found in the database.")
        return []

    if BOOK_FOR_QA not in book_titles:
        print(f"Book '{BOOK_FOR_QA}' not found in the database.")
        populate_db() 
    
    return book_titles

def interactive_chat():

    print(f"\nStarting interactive chat with '{BOOK_FOR_QA}'")
          
    try:
        query = QUESTION.strip()
        query_type = classify_query(query)
        print(f"Query: {query_type}")

        summary_triggers = [
            'summary', 'summarize', 'overview', 
            'tell me about', 'describe the book',
            'what is this book about', 'explain this book'
        ]

        if any(trigger in query for trigger in summary_triggers):
            generate_summary(BOOK_FOR_QA)
        else:
            query_rag(query, BOOK_FOR_QA)
        
    except KeyboardInterrupt:
        print("\nExiting chat...")
        
def main():
    CheckChromaIntegrity.check_chroma_integrity()
    book_titles = list_books()

    print(book_titles)
    interactive_chat()
    
if __name__ == "__main__":
    main()
