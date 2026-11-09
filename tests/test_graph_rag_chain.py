import pytest
from unittest.mock import MagicMock
from app.graph_rag_chain import GraphRAGService

def test_graph_rag_pipeline(mocker):
    # Mock Vector Store
    mock_vector_store = MagicMock()
    mock_vector = MagicMock()
    mock_vector.page_content = "El circuito requiere un capacitor de desacople."
    mock_vector.metadata = {"source": "datasheet.pdf"}
    mock_vector_store.search_similar.return_value = [mock_vector]
    
    # Mock Graph Store
    mock_graph_store = MagicMock()
    mock_graph_store.graph.nodes.return_value = ["ESP32"]
    mock_graph_store.get_entity_neighborhood.return_value = ["ESP32 REQUIERE CAPACITOR"]

    # Mock LLM
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "Se debe colocar un capacitor en el pin de alimentación del ESP32."

    service = GraphRAGService(mock_vector_store, mock_graph_store, mock_llm)
    res = service.generate_answer("¿Qué capacitor lleva el ESP32?")

    assert "ESP32 REQUIERE CAPACITOR" in res["graph_relations_used"]
    assert res["answer"] == "Se debe colocar un capacitor en el pin de alimentación del ESP32."
