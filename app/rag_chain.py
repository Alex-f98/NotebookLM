from langchain_ollama import OllamaLLM
from app.vector_store import VectorStoreManager

PROMPT_TEMPLATE = """
Responde a la pregunta del usuario utilizando ÚNICAMENTE la información provista en el contexto. Si no puedes responder con este contexto, di "No encuentro información sobre eso en los documentos".

Contexto:
{context}

Pregunta: {query}
Respuesta:
"""


class RAGService():
    def __init__(self,
            vector_store: VectorStoreManager,
            model_name: str = "llama3:8b",
            base_url: str = "http://localhost:11434"):

        self.vector_store = vector_store
        #self.model = OllamaLLM(
        #    model=model_name, 
        #    base_url=base_url
        #    )

        self.model = OllamaLLM(
                model       = model_name,
                base_url    = base_url,
                temperature = 0.7,      # creatividad (0 = determinista, 1+ = más variado)
                num_predict = 500,      # máx tokens de salida → controla el LARGO
                num_ctx     = 4096,     # tamaño del contexto
            )   

    def prepare(self, query: str, k: int = 3,
                sources: list[str] | None = None) -> tuple[list, str]:
        """
        Recupera el contexto y arma el prompt final.

        Devuelve `(docs_recuperados, prompt)`.
        """
        docs = self.vector_store.search_similar(query, k=k, sources=sources)

        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = PROMPT_TEMPLATE.format(context=context, query=query)
        return docs, prompt

    def generate_answer(self, query: str, k: int = 3,
                        sources: list[str] | None = None) -> dict:
        docs, prompt = self.prepare(query, k=k, sources=sources)
        answer = self.model.invoke(prompt)

        return {
            "query": query,
            "answer": answer,
            "sources": self._format_sources(docs),
        }

    def stream_answer(self, query: str, k: int = 3,
                      sources: list[str] | None = None):
        """
        Igual que `generate_answer` pero devuelve `(docs, generador_de_tokens)`
        para poder mostrar la respuesta en streaming en la UI.
        """
        docs, prompt = self.prepare(query, k=k, sources=sources)
        return docs, self.model.stream(prompt)

    @staticmethod
    def _format_sources(docs: list) -> list[dict]:
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("page"),
            }
            for doc in docs
        ]




def main():
    # 1. Ingesta
    docs = load_document("reporte.pdf")
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
    print(f">> {len(docs)} docs cargados -> {len(chunks)} chunks\n")

    # 2. Vector store (persistente en data/db)
    vs = VectorStoreManager(persist_directory="data/db")
    ids = vs.add_documents(chunks)
    print(f">> {len(ids)} chunks indexados en Chroma\n")

    # 3. RAG
    rag = RAGService(vector_store=vs, model_name="llama3:8b")
    pregunta = input("Pregunta: ")

    result = rag.generate_answer(pregunta)

    print("\n=== RESPUESTA ===")
    print(result["answer"])
    print("\n=== FUENTES ===")
    for s in result["sources"]:
        print(f"- [{s['source']}] {s['content'][:120]}...")


if __name__ == "__main__":
    from app.ingester import load_document, split_documents
    from app.vector_store import VectorStoreManager
    from app.rag_chain import RAGService

    main()