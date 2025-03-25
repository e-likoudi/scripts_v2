from langchain_community.vectorstores.chroma import Chroma
from basic_tools.config import CHROMA_PATH


vectorstore = Chroma(persist_directory=CHROMA_PATH)

class CheckChromaIntegrity:
    @staticmethod
    def check_chroma_integrity():
        collection_names = vectorstore._client.list_collections()
        book_titles = [c.name for c in collection_names]

        for collection_name in book_titles:
            print(f"Checking collection: {collection_name}")

            collection = vectorstore._client.get_collection(collection_name)
            data = collection.get(include=["documents", "embeddings"])  

            chunks = data.get("documents") 
            embeddings = data.get("embeddings")  

            null_chunks = []
            null_embeddings = []

            for chunk in chunks:
                if chunk is None:
                    null_chunks.append(chunk)

            for chunk, embedding in zip(chunks, embeddings):
                if embedding is None:
                    null_embeddings.append(chunk)

        if null_chunks:
            print(f"Collection '{collection_name}' has {len(null_chunks)} embeddings without chunks.\n")

        if null_embeddings:
            print(f"Collection '{collection_name}' has {len(null_embeddings)} chunks without embeddings.\n")

        if not null_embeddings and not null_chunks:
            print("\nNo issues found. The database is consistent.")

__all__ = ['CheckChromaIntegrity']