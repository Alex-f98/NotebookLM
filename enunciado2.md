## Arquitectura GraphRAG para Documentos Científicos y Electrónicos

En *datasheets*, artículos científicos y manuales técnicos, la información clave rara vez vive en un solo párrafo. Está distribuida en redes de dependencias: componentes, pines, protocolos, voltajes, ecuaciones y metodologías.

Mientras que un RAG tradicional se pierde intentando relacionar términos dispersos en distintas páginas, **GraphRAG** construye un Grafo de Conocimiento (*Knowledge Graph*) donde:

* **Nodos (Entidades):** Componentes (`ESP32`, `BC547`), protocolos (`SPI`, `I2C`), métricas (`3.3V`, `80MHz`), ecuaciones o nombres de autores.
* **Aristas (Relaciones):** `SE_ALIMENTA_CON`, `COMUNICA_VIA`, `CONTROLA_A`, `DEPENDEN_DE`, `EQUIVALE_A`.

---

## Plan de Trabajo por Entregables

Para mantener el enfoque local y amigable sin lidiar de entrada con servidores pesados como Neo4j, utilizaremos **NetworkX** (librería de grafos en Python basada en memoria) junto con **ChromaDB** para el vector retrieval híbrido.

```text
local-graphrag-service/
├── app/
│   ├── ingester.py          # Reutilizado: Carga y chunking de PDFs
│   ├── graph_extractor.py   # Extracción de tripletas (Entidad -> Relación -> Entidad) vía LLM
│   ├── graph_store.py       # Creación, persistencia y recorrido de grafos con NetworkX
│   ├── vector_store.py      # Búsqueda semántica clásica
│   ├── graph_rag_chain.py   # Fusión de contexto (Vectores + Subgrafos) -> LLM
│   └── main.py              # Endpoints API REST
└── tests/
    ├── test_graph_extractor.py
    ├── test_graph_store.py
    └── test_graph_rag_chain.py

```

---

### Entregable 1: Extractor de Entidades y Relaciones (`app/graph_extractor.py`)

**Objetivo:** Recibir un fragmento de texto técnico y hacer que el LLM extraiga tripletas estructuradas en formato JSON.

#### Firma de Funciones y Contratos

```python
from dataclasses import dataclass

@dataclass
class KnowledgeTriple:
    source: str       # Ej: "ESP32"
    relation: str     # Ej: "REQUIERE_VOLTAJE"
    target: str       # Ej: "3.3V"

def extract_triples_from_chunk(chunk_text: str, llm_client) -> list[KnowledgeTriple]:
    """
    Analiza un texto técnico y retorna las relaciones detectadas.
    """
    pass

```

#### Estructura del Prompt de Extracción Interno

```text
Analiza el siguiente texto técnico/electrónico y extrae únicamente las entidades principales y sus relaciones directas.
Formato de salida estricto en JSON array:
[
  {"source": "EntidadA", "relation": "RELACION", "target": "EntidadB"}
]

Ejemplo: "El sensor BME280 utiliza el bus I2C para comunicarse a 3.3V."
Salida:
[
  {"source": "BME280", "relation": "USA_BUS", "target": "I2C"},
  {"source": "BME280", "relation": "VOLTAJE_OPERACION", "target": "3.3V"}
]

Texto a analizar:
{chunk_text}

```

#### Pruebas Unitarias (`tests/test_graph_extractor.py`)

```python
import pytest
from unittest.mock import MagicMock
from app.graph_extractor import extract_triples_from_chunk, KnowledgeTriple

def test_extract_triples_parsing(mocker):
    # Mock del LLM devuelviendo JSON válido
    mock_llm_output = '[{"source": "STM32", "relation": "POSEE_PIN", "target": "PA0"}]'
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_llm_output
    
    triples = extract_triples_from_chunk("El microcontrolador STM32 posee el pin PA0.", mock_llm)
    
    assert len(triples) == 1
    assert isinstance(triples[0], KnowledgeTriple)
    assert triples[0].source == "STM32"
    assert triples[0].relation == "POSEE_PIN"
    assert triples[0].target == "PA0"

```

---

### Entregable 2: Motor de Grafo de Conocimiento (`app/graph_store.py`)

**Objetivo:** Almacenar el grafo en memoria con `NetworkX`, permitir persistencia en JSON y realizar búsquedas por vecindario (recorrido de subgrafos a $N$ saltos).

#### Firma de Métodos de la Clase `GraphStoreManager`

```python
import networkx as nx

class GraphStoreManager:
    def __init__(self):
        self.graph = nx.DiGraph() # Grafo Dirigido

    def add_triples(self, triples: list[KnowledgeTriple]) -> None:
        """Agrega nodos y aristas con atributos al grafo."""
        pass

    def get_entity_neighborhood(self, entity_name: str, depth: int = 1) -> list[str]:
        """
        Dada una entidad (ej. "ESP32"), recupera todas las entidades y 
        relaciones conectadas hasta 'depth' saltos de distancia.
        Retorna una lista de cadenas descriptivas.
        """
        pass

    def save_to_file(self, path: str) -> None:
        """Guarda el grafo en formato GraphML o JSON."""
        pass

    def load_from_file(self, path: str) -> None:
        """Carga el grafo desde disco."""
        pass

```

#### Ejemplo de Entrada / Salida de `get_entity_neighborhood`

```python
# Si el grafo contiene: (ESP32) -> [USA_BUS] -> (SPI) -> [REQUIERE_PIN] -> (MOSI)

# Entrada
graph_manager.get_entity_neighborhood(entity_name="ESP32", depth=2)

# Salida esperada
[
  "ESP32 USA_BUS SPI",
  "SPI REQUIERE_PIN MOSI"
]

```

#### Pruebas Unitarias (`tests/test_graph_store.py`)

```python
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

```

---

### Entregable 3: Pipeline RAG Híbrido Vector + Grafo (`app/graph_rag_chain.py`)

**Objetivo:** Combinar lo mejor de ambos mundos. La búsqueda vectorial encuentra los bloques de texto semánticamente similares, y el grafo aporta las relaciones estructurales alrededor de las entidades detectadas.

```
 Petición del Usuario ("¿Cómo se conecta el sensor al microcontrolador?")
   ├── 1. Vector Search (ChromaDB) ──> Extrae Top-K Chunks de Texto
   ├── 2. Entity Match & Graph Traversal ──> Extrae Subgrafo de Relaciones
   └── 3. Prompt Augmentation ──> LLM (Ollama) ──> Respuesta Precisa

```

#### Código del Pipeline `GraphRAGService`

```python
class GraphRAGService:
    def __init__(self, vector_store, graph_store, llm_client):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.llm = llm_client

    def generate_answer(self, query: str) -> dict:
        # 1. Recuperar chunks vectoriales
        relevant_docs = self.vector_store.search_similar(query, k=2)
        text_context = "\n".join([doc.page_content for doc in relevant_docs])

        # 2. Identificar entidades clave en la consulta o chunks
        # Para cada entidad encontrada en el grafo, traer su vecindario
        graph_facts = []
        for node in self.graph_store.graph.nodes():
            if node.lower() in query.lower() or any(node.lower() in doc.page_content.lower() for doc in relevant_docs):
                graph_facts.extend(self.graph_store.get_entity_neighborhood(node, depth=1))
        
        # Eliminar duplicados en los hechos del grafo
        unique_graph_facts = list(set(graph_facts))
        graph_context = "\n".join(unique_graph_facts)

        # 3. Armar Prompt Enriquecido
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
        response = self.llm.invoke(prompt)
        
        return {
            "query": query,
            "answer": response,
            "text_sources": [doc.metadata.get("source") for doc in relevant_docs],
            "graph_relations_used": unique_graph_facts
        }

```

#### Pruebas Unitarias (`tests/test_graph_rag_chain.py`)

```python
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

```

---

## Dependencias Adicionales Requeridas

```bash
pip install networkx matplotlib

```

*(Nota: `matplotlib` es opcional, pero te permitirá renderizar y visualizar gráficamente el mapa de nodos y aristas de tus archivos PDF con `nx.draw(graph)`).*