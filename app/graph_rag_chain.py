class GraphRAGService:
    def __init__(self, vector_store, graph_store, llm_client):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.llm = llm_client

    def _collect_graph_facts(self, query: str, relevant_docs: list,
                             depth: int = 1) -> list[str]:
        """Busca entidades del grafo mencionadas en la consulta o en los chunks."""
        graph_facts = []
        for node in self.graph_store.graph.nodes():
            node_lower = node.lower()
            mentioned = (
                node_lower in query.lower()
                or any(node_lower in doc.page_content.lower() for doc in relevant_docs)
            )
            if mentioned:
                graph_facts.extend(
                    self.graph_store.get_entity_neighborhood(node, depth=depth)
                )

        # Eliminar duplicados preservando el orden de aparición
        return list(dict.fromkeys(graph_facts))

    def prepare(self, query: str, k: int = 2,
                sources: list[str] | None = None, depth: int = 1):
        """
        Recupera contexto vectorial + grafo y arma el prompt.

        Devuelve `(docs, prompt, hechos_del_grafo)`.
        """
        # 1. Recuperar chunks vectoriales
        relevant_docs = self.vector_store.search_similar(
            query, k=k, sources=sources
        )
        text_context = "\n".join([doc.page_content for doc in relevant_docs])

        # 2. Identificar entidades clave y traer su vecindario
        graph_facts = self._collect_graph_facts(query, relevant_docs, depth=depth)
        graph_context = "\n".join(graph_facts)

        # 3. Armar prompt enriquecido
        prompt = f"""
        Eres un experto en documentación técnica, electrónica y científica.
        Responde a la pregunta utilizando los hechos relacionales del grafo y el texto de contexto.

        REGLAS DE RELACIÓN (GRAFO):
        {graph_context if graph_context else "No hay relaciones directas en el grafo."}

        TEXTO DE CONTEXTO:
        {text_context}

        PREGUNTA: {query}
        RESPUESTA TÉCNICA:
        """
        return relevant_docs, prompt, graph_facts

    def generate_answer(self, query: str, k: int = 2,
                        sources: list[str] | None = None, depth: int = 1) -> dict:
        relevant_docs, prompt, graph_facts = self.prepare(
            query, k=k, sources=sources, depth=depth
        )
        response = self.llm.invoke(prompt)

        return {
            "query": query,
            "answer": response,
            "text_sources": [doc.metadata.get("source") for doc in relevant_docs],
            "graph_relations_used": graph_facts,
        }

    def stream_answer(self, query: str, k: int = 2,
                      sources: list[str] | None = None, depth: int = 1):
        """
        Igual que `generate_answer` pero devuelve
        `(docs, hechos_del_grafo, generador_de_tokens)` para streaming en la UI.
        """
        relevant_docs, prompt, graph_facts = self.prepare(
            query, k=k, sources=sources, depth=depth
        )
        return relevant_docs, graph_facts, self.llm.stream(prompt)
