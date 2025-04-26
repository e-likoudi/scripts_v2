from langchain.schema import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from basic_tools.config import MODEL, CHROMA_PATH

embedding_function = OllamaEmbeddings(model=MODEL)
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str, book_for_qa):

    collection = vectorstore._client.get_collection(book_for_qa)
    data = collection.get(include=["documents"])
    metadatas = collection.get(include=["metadatas"])

    raw_documents = data.get("documents", [])
    documents = [Document(page_content=doc) for doc in raw_documents if isinstance(doc, str)]

    if not documents:
        return "No valid documents found for this book."

    if not collection:
        return "Book not found in the database."

    vectordb = vectorstore.from_documents(collection_name=book_for_qa, documents=documents, embedding=embedding_function) 

    # Search the DB.
    results = vectordb.similarity_search_with_score(query_text, k=5) 

    if not results:
        return "No results found for the query"

    print(results)

    processed_docs = []
    for doc in results:
        if isinstance(doc, tuple):  
            doc = Document(page_content=" ".join(map(str, doc)))  
        processed_docs.append(doc)

    context_text = "\n\n---\n\n".join([doc.page_content for doc in processed_docs])


    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = Ollama(model="mistral")

    response_text = model.invoke(prompt)
    sources = [metadatas.get("id", None) for doc, _ in results]  
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

