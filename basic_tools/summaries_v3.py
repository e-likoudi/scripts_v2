import numpy as np
from langchain.schema import Document
from langchain.chat_models import ChatOllama
from langchain.chains.summarize import load_summarize_chain
from sklearn.cluster import KMeans
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores.chroma import Chroma
from basic_tools.config import CHROMA_PATH, SUMMARIES_FILE

vectorstore = Chroma(persist_directory=CHROMA_PATH)

llm3 = ChatOllama(
    temperature=0,              
    model="llama3.1",           
    max_tokens=1000             
)

map_prompt = """
You will be given a single passage of a book. This section will be enclosed in triple backticks (```)
Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
Your response should be at least three paragraphs and fully encompass what was said in the passage.

```{text}```
FULL SUMMARY:
"""
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

map_chain = load_summarize_chain(llm=llm3,
                             chain_type="stuff",
                             prompt=map_prompt_template)


combine_prompt = """
You will be given a series of summaries from a book. The summaries will be enclosed in triple backticks (```)
Your goal is to give a verbose summary of what happened in the story.
The reader should be able to grasp what happened in the book.

```{text}```
VERBOSE SUMMARY:
"""
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

reduce_chain = load_summarize_chain(llm=llm3,
                             chain_type="stuff",
                             prompt=combine_prompt_template,
                                   )


def get_clusters(vectors):
    num_clusters = 15
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)

    closest_indices = []

    for i in range(num_clusters):
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1) 
        closest_index = np.argmin(distances) 
        closest_indices.append(closest_index) 

    selected_indices = sorted(closest_indices)

    return selected_indices

def list_of_summaries(selected_indices, docs):
    selected_docs = [docs[doc] for doc in selected_indices]

    summary_list = []
    
    # Loop through a range of the lenght of your selected docs
    for doc in enumerate(selected_docs):
        
        if isinstance(doc, tuple):  
            doc = Document(page_content=" ".join(map(str, doc)))  

        chunk_summary = map_chain.run([doc])    
        summary_list.append(chunk_summary)

    return summary_list 

def generate_summary(book_title):                           
    print(f"\nGenerating summary for: {book_title}...")
    
    collection = vectorstore._client.get_collection(book_title)
    data = collection.get(include=["documents", "embeddings"])
    
    raw_documents = data.get("documents", [])
    documents = [Document(page_content=doc) for doc in raw_documents if isinstance(doc, str)]  
    embeddings = np.array(data.get("embeddings", []))  
    
    if not documents:
        print("No valid documents found for this book!")
        return
    
    selected_indices = get_clusters(embeddings)
    summary_list = list_of_summaries(selected_indices, documents)

    summaries = " ".join(summary_list)
    print(summaries)

    # Convert it back to a document
    summaries = Document(page_content=summaries)    
    output = reduce_chain.run([summaries])

    with open(SUMMARIES_FILE, "w", encoding="utf-8") as f:  
        f.write(f"Title: {book_title}\n\n")
        f.write("Summary:\n")
        f.write(output)

    return f"\nBook summary saved to: {SUMMARIES_FILE}"

