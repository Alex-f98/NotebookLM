import re
from pathlib import Path

from app.rag_chain import is_overview_query


class GraphRAGService:
    def __init__(self, vector_store, graph_store, llm_client):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.llm = llm_client

    @staticmethod
    def _matches(node: str, text: str) -> bool:
        """Coincidencia por palabra/frase completa (evita 'agua' en 'aguas')."""
        node = node.strip().lower()
        if len(node) < 3:
            return False
        pattern = re.compile(r"(?<!\w)" + re.escape(node) + r"(?!\w)")
        return bool(pattern.search(text))

    def _collect_graph_facts(self, query: str, relevant_docs: list,
                             depth: int = 1, limit: int = 40) -> list[str]:
        """Busca entidades del grafo mencionadas en la consulta o en los chunks."""
        query_lower = query.lower()
        docs_lower = [doc.page_content.lower() for doc in relevant_docs]

        graph_facts = []
        for node in self.graph_store.graph.nodes():
            # En la pregunta aceptamos entidades cortas; en el texto exigimos
            # algo más de largo para no arrastrar términos genéricos.
            mentioned_in_query = self._matches(node, query_lower)
            mentioned_in_docs = len(node.strip()) >= 4 and any(
                self._matches(node, text) for text in docs_lower
            )
            if mentioned_in_query or mentioned_in_docs:
                graph_facts.extend(
                    self.graph_store.get_entity_neighborhood(node, depth=depth)
                )

        # Deduplicar preservando orden y acotar para no inundar el prompt.
        unique = list(dict.fromkeys(graph_facts))
        return unique[:limit]

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

    def prepare(self, query: str, k: int = 2,
                sources: list[str] | None = None, depth: int = 1,
                history: list[dict] | None = None):
        """
        Recupera contexto vectorial + grafo y arma el prompt.

        Devuelve `(docs, prompt, hechos_del_grafo)`.
        """
        # 1. Recuperar chunks vectoriales
        relevant_docs = self.vector_store.search_similar(
            query, k=k, sources=sources
        )

        # Para preguntas globales sumamos el primer fragmento de cada fuente.
        if is_overview_query(query):
            overview = self.vector_store.get_first_documents(sources=sources)
            seen = set()
            merged = []
            for doc in list(overview) + list(relevant_docs):
                key = (
                    doc.metadata.get("source"),
                    doc.metadata.get("page"),
                    doc.page_content[:80],
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(doc)
            relevant_docs = merged

        text_context = self._format_context(relevant_docs)

        # 2. Identificar entidades clave y traer su vecindario
        graph_facts = self._collect_graph_facts(query, relevant_docs, depth=depth)
        graph_context = "\n".join(graph_facts)

        # 3. Armar prompt enriquecido
        prompt = f"""
        Sos un asistente experto en documentación técnica, electrónica y científica.
        Respondé siempre en español.

        Reglas:
        - Usá el TEXTO DE CONTEXTO y las RELACIONES DEL GRAFO como evidencia.
        - El grafo puede tener relaciones ruidosas: si no coincide con el texto, priorizá el texto.
        - Si la información no alcanza, decí exactamente qué falta. NO inventes.
        - Respondé directo y conciso. Citá el archivo y la página tal como figuran en las etiquetas del contexto.
        - Escribí las fórmulas en LaTeX: en línea entre $...$ y en bloque entre $$...$$.
          No uses delimitadores con paréntesis ni corchetes, ni bloques de código para fórmulas.
        - Mostrá cada fórmula una sola vez; no la repitas también como texto.
        - Explicá en palabras qué representa cada fórmula y qué significan sus símbolos.

        HISTORIAL DE LA CONVERSACIÓN:
        {self._format_history(history)}

        RELACIONES DEL GRAFO:
        {graph_context if graph_context else "No hay relaciones directas en el grafo."}

        TEXTO DE CONTEXTO:
        {text_context}

        PREGUNTA: {query}
        RESPUESTA TÉCNICA:
        """
        return relevant_docs, prompt, graph_facts

    def generate_answer(self, query: str, k: int = 2,
                        sources: list[str] | None = None, depth: int = 1,
                        history: list[dict] | None = None) -> dict:
        relevant_docs, prompt, graph_facts = self.prepare(
            query, k=k, sources=sources, depth=depth, history=history
        )
        response = self.llm.invoke(prompt)

        return {
            "query": query,
            "answer": response,
            "text_sources": [doc.metadata.get("source") for doc in relevant_docs],
            "graph_relations_used": graph_facts,
        }

    def stream_answer(self, query: str, k: int = 2,
                      sources: list[str] | None = None, depth: int = 1,
                      history: list[dict] | None = None):
        """
        Igual que `generate_answer` pero devuelve
        `(docs, hechos_del_grafo, generador_de_tokens)` para streaming en la UI.
        """
        relevant_docs, prompt, graph_facts = self.prepare(
            query, k=k, sources=sources, depth=depth, history=history
        )
        return relevant_docs, graph_facts, self.llm.stream(prompt)
