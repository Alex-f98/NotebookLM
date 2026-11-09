import pytest
from langchain_core.documents import Document
from app.vector_store import VectorStoreManager

@pytest.fixture
def in_memory_vector_store():
    # Instancia en memoria usando Chroma efímero
    return VectorStoreManager(persist_directory=None)

def test_add_and_search(in_memory_vector_store):
    doc1 = Document(page_content="El motor usa aceite 10W40.", metadata={"topic": "autos"})
    doc2 = Document(page_content="La receta lleva 200g de harina.", metadata={"topic": "cocina"})
    
    in_memory_vector_store.add_documents([doc1, doc2])
    results = in_memory_vector_store.search_similar("¿Qué lubricante necesita el auto?", k=1)
    
    assert len(results) == 1
    assert "10W40" in results[0].page_content
    assert results[0].metadata["topic"] == "autos"
