# scripts/verify_graphrag.py
from app.ingester import split_documents
from app.graph_extractor import extract_triples_from_chunk
from app.graph_store import GraphStoreManager
from app.vector_store import VectorStoreManager
from app.graph_rag_chain import GraphRAGService
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM

def main():
    print("--- 1. Texto de prueba técnico ---")
    texto_tecnico = """
    El microcontrolador ESP32 se comunica con el sensor BME280 mediante el bus I2C.
    El pin SDA del ESP32 está conectado al GPIO21 y el pin SCL al GPIO22.
    El sensor BME280 opera a un voltaje estricto de 3.3V. Si se aplican 5V se destruye el módulo.
    """
    doc = Document(page_content=texto_tecnico, metadata={"source": "datasheet_esp32.txt"})

    print("\n--- 2. Inicializando LLM Local (Ollama) ---")
    llm = OllamaLLM(model="llama3:8b", base_url="http://localhost:11434")

    print("\n--- 3. Extrayendo Tripletas para el Grafo ---")
    tripletas = extract_triples_from_chunk(texto_tecnico, llm)
    print("Tripletas detectadas:")
    for t in tripletas:
        print(f"  ({t.source}) --[{t.relation}]--> ({t.target})")

    print("\n--- 4. Cargando Grafo en NetworkX ---")
    graph_store = GraphStoreManager()
    graph_store.add_triples(tripletas)
    
    print("Nodos cargados en el grafo:", list(graph_store.graph.nodes()))

    print("\n--- 5. Cargando en Vector Store (ChromaDB) ---")
    vector_store = VectorStoreManager(persist_directory=None) # DB en RAM
    vector_store.add_documents([doc])

    print("\n--- 6. Ejecutando Consulta Híbrida (GraphRAG) ---")
    rag_service = GraphRAGService(vector_store, graph_store, llm)
    
    pregunta = "¿Qué voltaje requiere el BME280 y qué pasa si le pongo 5V?"
    resultado = rag_service.generate_answer(pregunta)

    print("\n>>> RESULTADO FINAL DEL LLM:")
    print(resultado["answer"])
    print("\n>>> RELACIONES DEL GRAFO USADAS COMO CONTEXTO:")
    for rel in resultado["graph_relations_used"]:
        print(f"  - {rel}")

if __name__ == "__main__":
    main()