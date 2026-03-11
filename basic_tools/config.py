BOOKS_PATH = ""  #path for the folder with the papers
MODEL = "nomic-embed-text"  #embedding model
CHROMA_PATH = f"./{MODEL}_db"  #path to save chromadb
SUMMARIES_FILE = "/summaries.txt"  #path to save the summaries.txt

PROTOCOL_MODEL = "llama3.1:latest"  #base model for data extraction from the paper 
PROTOCOL_FILE = f"/protocol_{PROTOCOL_MODEL}.txt"  #path to save the differentiation steps and other details 

BOOK_FOR_QA = ""  #name of the paper we want to use for asking a question
QUESTION = "What is the main finding of the paper?"  #sample question

