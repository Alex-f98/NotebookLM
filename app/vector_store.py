from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


class VectorStoreManager():

    def __init__(self,
                    persist_directory: str = "data/db",
                    collection_name: str = "local_rag"):

        # Modelo multilingüe: entiende preguntas en español sobre textos en
        # inglés (o viceversa) mucho mejor que all-MiniLM-L6-v2.
        self.model_name    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.model_kwargs  = {"device": "cpu"}
        self.encode_kwargs = {"normalize_embeddings": True}
        self.model = HuggingFaceEmbeddings(
                        model_name      = self.model_name,
                        model_kwargs    = self.model_kwargs,
                        encode_kwargs   = self.encode_kwargs,
                    )

        self.persist_directory = persist_directory
        self.collection_name   = collection_name

        self.vector_store = self._open_collection()

    def _open_collection(self) -> Chroma:
        return Chroma(
            collection_name    = self.collection_name,
            embedding_function = self.model,
            persist_directory  = self.persist_directory or ":memory:",
            # Con embeddings normalizados, la similitud coseno es la métrica
            # correcta (la distancia L2 por defecto penaliza el largo).
            collection_metadata = {"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Borra la colección y la recrea (necesario al cambiar de embeddings)."""
        try:
            self.vector_store.delete_collection()
        except Exception:
            pass
        self.vector_store = self._open_collection()

    def add_documents(self, docs: list[Document]) -> list[str]:
        return self.vector_store.add_documents(docs) #ya devuelve una lista de UUIDs

    def search_similar(self, query: str, k: int = 3,
                       sources: list[str] | None = None) -> list[Document]:
        """
        Recupera los `k` fragmentos más similares.

        Si se pasa `sources`, la búsqueda se restringe a esos archivos
        (el valor de `metadata["source"]`).
        """
        where = {"source": {"$in": list(sources)}} if sources else None
        return self.vector_store.similarity_search(query, k, filter=where)

    def get_sources(self) -> dict[str, dict]:
        """
        Devuelve un resumen de las fuentes indexadas:

        {
            "data/uploads/manual.pdf": {
                "chunks": 14,
                "pages": [0, 1, 2],
            },
            ...
        }
        """
        data = self.vector_store.get(include=["metadatas"])
        sources: dict[str, dict] = {}

        for meta in data.get("metadatas") or []:
            source = (meta or {}).get("source")
            if not source:
                continue

            entry = sources.setdefault(source, {"chunks": 0, "pages": set()})
            entry["chunks"] += 1
            page = (meta or {}).get("page")
            if isinstance(page, int):
                entry["pages"].add(page)

        for entry in sources.values():
            entry["pages"] = sorted(entry["pages"])

        return sources

    def get_documents(self, sources: list[str] | None = None,
                      limit: int | None = None) -> list[Document]:
        """Devuelve los fragmentos almacenados (opcionalmente filtrados por fuente)."""
        where = {"source": {"$in": list(sources)}} if sources else None
        data = self.vector_store.get(
            where=where,
            limit=limit,
            include=["documents", "metadatas"],
        )

        documents: list[Document] = []
        for content, meta in zip(data.get("documents") or [], data.get("metadatas") or []):
            if content is not None:
                documents.append(Document(page_content=content, metadata=meta or {}))
        return documents

    def get_first_documents(self,
                            sources: list[str] | None = None) -> list[Document]:
        """
        Devuelve el primer fragmento (menor página) de cada fuente.

        Sirve como "portada/abstract" para preguntas globales del tipo
        «¿de qué trata el documento?», donde la búsqueda por similitud no
        tiene suficiente señal.
        """
        first: dict[str, Document] = {}
        for doc in self.get_documents(sources=sources):
            source = doc.metadata.get("source", "")
            page = doc.metadata.get("page")
            current = first.get(source)
            if current is None:
                first[source] = doc
                continue
            current_page = current.metadata.get("page")
            if isinstance(page, int) and (
                not isinstance(current_page, int) or page < current_page
            ):
                first[source] = doc
        return list(first.values())

    def delete_by_source(self, source: str) -> None:
        """Elimina todos los fragmentos asociados a una fuente."""
        self.vector_store.delete(where={"source": source})

    def count(self) -> int:
        """Cantidad total de fragmentos indexados."""
        return self.vector_store._collection.count()
