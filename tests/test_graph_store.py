import pytest
from app.graph_store import GraphStoreManager
from app.graph_extractor import KnowledgeTriple

def test_graph_neighborhood_traversal():
    graph_store = GraphStoreManager()
    
    triples = [
        KnowledgeTriple(source="ESP32", relation="USA_BUS", target="SPI"),
        KnowledgeTriple(source="SPI", relation="CONECTA_CON", target="SENSOR_MAX31855")
    ]
    graph_store.add_triples(triples)
    
    # Búsqueda a 2 saltos desde "ESP32"
    neighborhood = graph_store.get_entity_neighborhood("ESP32", depth=2)
    
    assert len(neighborhood) == 2
    assert "ESP32 USA_BUS SPI" in neighborhood
    assert "SPI CONECTA_CON SENSOR_MAX31855" in neighborhood