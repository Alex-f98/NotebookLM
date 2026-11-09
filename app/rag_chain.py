from pathlib import Path

from langchain_ollama import OllamaLLM
from app.vector_store import VectorStoreManager

PROMPT_TEMPLATE = """
Sos un asistente que responde preguntas sobre documentos, siempre en español.

Reglas:
- Usá ÚNICAMENTE la información del CONTEXTO.
- Si el contexto no alcanza para responder, decí exactamente qué información falta. NO inventes.
- Respondé directo y conciso. Usá viñetas si ayuda a ordenar la respuesta.
- Citá el archivo y la página que figuran en la etiqueta del fragmento usado.
- Escribí las fórmulas en LaTeX: en línea entre $...$ y en bloque entre $$...$$.
  No uses delimitadores con paréntesis ni corchetes, ni bloques de código para fórmulas.
- Mostrá cada fórmula una sola vez; no la repitas también como texto.
- Explicá en palabras qué representa cada fórmula y qué significan sus símbolos.

HISTORIAL DE LA CONVERSACIÓN:
{history}

CONTEXTO:
{context}

PREGUNTA: {query}
RESPUESTA:
"""


OVERVIEW_HINTS = (
    "de que trata", "de qué trata", "sobre que trata", "sobre qué trata",
    "de que se trata", "de qué se trata", "resumen", "resumí", "resumi",
    "tema principal", "idea principal", "ideas principales", "overview",
    "summary", "abstract", "para que sirve el documento",
)


def is_overview_query(query: str) -> bool:
    """Detecta preguntas globales donde la búsqueda por similitud rinde poco."""
    text = query.lower()
    return any(hint in text for hint in OVERVIEW_HINTS)


class RAGService():
    def __init__(self,
            vector_store: VectorStoreManager,
            model_name: str = "llama3:8b",
            base_url: str = "http://localhost:11434"):

        self.vector_store = vector_store

        self.model = OllamaLLM(
                model       = model_name,
                base_url    = base_url,
                temperature = 0.15,     # bajo = respuestas fieles al contexto
                num_predict = 900,      # margen para respuestas largas
                num_ctx     = 8192,     # contexto amplio para k chunks + historial
            )

    @staticmethod
    def _format_context(docs: list) -> str:
        parts = []
        for doc in docs:
            source = Path(doc.metadata.get("source", "")).name or "documento"
            page = doc.metadata.get("page")
            label = f"{source}, pág. {page + 1}" if isinstance(page, int) else source
            parts.append(f"[{label}]\n{doc.page_content}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_history(history: list[dict] | None,
                        max_messages: int = 6,
                        max_chars: int = 2000) -> str:
        if not history:
            return "No hay historial."
        lines = []
        for message in history[-max_messages:]:
            role = "Usuario" if message.get("role") == "user" else "Asistente"
            content = str(message.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
        text = "\n".join(lines)
        return text[-max_chars:] if len(text) > max_chars else text

    def prepare(self, query: str, k: int = 3,
                sources: list[str] | None = None,
                history: list[dict] | None = None) -> tuple[list, str]:
        """
        Recupera el contexto y arma el prompt final.

        Devuelve `(docs_recuperados, prompt)`.
        """
        docs = self.vector_store.search_similar(query, k=k, sources=sources)

        if is_overview_query(query):
            overview = self.vector_store.get_first_documents(sources=sources)
            seen = set()
            merged = []
            for doc in list(overview) + list(docs):
                key = (
                    doc.metadata.get("source"),
                    doc.metadata.get("page"),
                    doc.page_content[:80],
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(doc)
            docs = merged

        context = self._format_context(docs)
        prompt = PROMPT_TEMPLATE.format(
            context=context,
            history=self._format_history(history),
            query=query,
        )
        return docs, prompt

    def generate_answer(self, query: str, k: int = 3,
                        sources: list[str] | None = None,
                        history: list[dict] | None = None) -> dict:
        docs, prompt = self.prepare(query, k=k, sources=sources, history=history)
        answer = self.model.invoke(prompt)

        return {
            "query": query,
            "answer": answer,
            "sources": self._format_sources(docs),
        }

    def stream_answer(self, query: str, k: int = 3,
                      sources: list[str] | None = None,
                      history: list[dict] | None = None):
        """
        Igual que `generate_answer` pero devuelve `(docs, generador_de_tokens)`
        para poder mostrar la respuesta en streaming en la UI.
        """
        docs, prompt = self.prepare(query, k=k, sources=sources, history=history)
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