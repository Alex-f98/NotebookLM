import networkx as nx
from app.graph_extractor import KnowledgeTriple

class GraphStoreManager:
    def __init__(self):
        self.graph = nx.DiGraph() # Grafo Dirigido

    def add_triples(self, triples: list[KnowledgeTriple]) -> None:
        """Agrega nodos y aristas con atributos al grafo."""
        for t in triples:
            self.graph.add_nodes_from([(t.source, {"type": "entity"}), (t.target, {"type": "entity"})])
            self.graph.add_edge(t.source, t.target, relation=t.relation)

    def get_entity_neighborhood(self, entity_name: str, depth: int = 1) -> list[str]:
        """
        Dada una entidad (ej. "ESP32"), recupera todas las entidades y 
        relaciones conectadas hasta 'depth' saltos de distancia.
        Retorna una lista de cadenas descriptivas.
        """
        if entity_name not in self.graph:
            return []
 
        # Distancia en saltos desde la entidad (BFS sobre grafo dirigido)
        distances = nx.single_source_shortest_path_length(self.graph, entity_name)
 
        # Nodos alcanzables dentro de 'depth' saltos
        reachable = {n for n, d in distances.items() if d <= depth}
 
        # Aristas del subgrafo inducido → "u RELACION v"
        facts = []
        for u, v, data in self.graph.edges(data=True):
            if u in reachable and v in reachable:
                facts.append(f"{u} {data['relation']} {v}")
        return facts

    def save_to_file(self, path: str) -> None:
        """Guarda el grafo en formato GraphML o JSON."""
        nx.write_graphml(self.graph, path)

    def load_from_file(self, path: str) -> None:
        """Carga el grafo desde disco."""
        self.graph = nx.read_graphml(path)
