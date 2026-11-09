from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


class VectorStoreManager():

    def __init__(self,
                    persist_directory: str = "data/db",
                    collection_name: str = "local_rag"):

        self.model_name    = "all-MiniLM-L6-v2"
        self.model_kwargs  = {"device": "cpu"}
        self.encode_kwargs = {"normalize_embeddings": False}
        self.model = HuggingFaceEmbeddings(
                        model_name      = self.model_name,
                        model_kwargs    = self.model_kwargs,
                        encode_kwargs   = self.encode_kwargs,
                    )

        self.persist_directory = persist_directory
        self.collection_name   = collection_name

        self.vector_store = Chroma(
            collection_name    = self.collection_name,
            embedding_function = self.model,
            persist_directory  = self.persist_directory or ":memory:",
        )

    def add_documents(self, docs: list[Document]) -> list[str]:
        return self.vector_store.add_documents(docs) #ya devuelve una lista de UUIDs

    def search_similar(self, query: str, k: int = 3) -> list[Document]:
        return self.vector_store.similarity_search(query, k)
