from fastapi.testclient import TestClient
from app.main import app, get_rag_service
from unittest.mock import MagicMock

client = TestClient(app)

def test_query_endpoint():
    # Mock de la capa de servicio RAG
    mock_rag_service = MagicMock()
    mock_rag_service.generate_answer.return_value = {
        "query": "hola",
        "answer": "Respuesta simulada",
        "sources": []
    }
    
    # Inyección de dependencia mockeada
    app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
    
    response = client.post("/query", json={"question": "hola"})
    
    assert response.status_code == 200
    assert response.json()["answer"] == "Respuesta simulada"
    app.dependency_overrides.clear()
