import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from app.rag_chain import RAGService

def test_generate_answer_success(mocker):
    # Mockear el VectorStore
    mock_vector_store = MagicMock()
    mock_vector_store.search_similar.return_value = [
        Document(page_content="Garantía de 1 año por fallas de fábrica.", metadata={"source": "garantia.txt"})
    ]
    
    # Mockear la llamada a Ollama para no depender de que esté corriendo en la máquina
    mock_ollama_response = "La garantía cubre 1 año por fallas de fábrica."
    mocker.patch("langchain_ollama.OllamaLLM.invoke", return_value=mock_ollama_response)
    
    rag_service = RAGService(vector_store=mock_vector_store)
    response = rag_service.generate_answer("¿Cuánto dura la garantía?")
    
    assert response["answer"] == mock_ollama_response
    assert len(response["sources"]) == 1
    assert response["sources"][0]["source"] == "garantia.txt"
