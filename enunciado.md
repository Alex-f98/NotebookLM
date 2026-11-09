# Notebook LM local
``` 
local-rag-service/
├── app/
│   ├── __init__.py
│   ├── config.py             # Configuración central (rutas, modelos, chunk_size)
│   ├── ingester.py           # Carga y fragmentación de documentos
│   ├── vector_store.py       # Gestión de embeddings y base vectorial (Chroma)
│   ├── rag_chain.py          # Lógica del prompt y consulta al LLM local
│   └── main.py               # Endpoints FastAPI (/upload, /query)
├── tests/
│   ├── __init__.py
│   ├── test_ingester.py
│   ├── test_vector_store.py
│   ├── test_rag_chain.py
│   └── test_api.py
├── data/                     # Carpeta temporal para persistencia de Chroma y PDFs
│   └── db/
├── requirements.txt
└── README.md
``` 

## 1. Módulo de Ingesta (`app/ingester.py`)

### `load_document(file_path: str) -> list[Document]`

* **Propósito:** Lee un archivo del disco local y lo convierte en objetos `Document` estándar (con `page_content` y `metadata`).
* **Reglas de negocio:**
1. Si `file_path` termina en `.pdf`, usa `PyPDFLoader`.
2. Si `file_path` termina en `.txt`, usa `TextLoader` con codificación `utf-8`.
3. Si es otra extensión, lanza un `ValueError("Formato no soportado: <ext>")`.
4. Si el archivo no existe, lanza `FileNotFoundError`.



```python
# Ejemplo de entrada
file_path = "data/documento.txt"

# Ejemplo de salida esperada
[
    Document(
        page_content="El horario de atención es de 9 a 18 hs.",
        metadata={"source": "data/documento.txt"}
    )
]

```

---

### `split_documents(docs: list[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> list[Document]`

* **Propósito:** Fragmenta una lista de documentos en trozos más pequeños preservando los metadatos de origen.
* **Comportamiento:** Usa `RecursiveCharacterTextSplitter` con los separadores `["\n\n", "\n", " ", ""]`.

```python
# Ejemplo de entrada
docs = [Document(page_content="A" * 1200, metadata={"source": "doc.txt"})]
chunk_size = 500
chunk_overlap = 50

# Ejemplo de salida esperada (3 fragmentos generados por el solapamiento)
[
    Document(page_content="AAAA...", metadata={"source": "doc.txt"}),
    Document(page_content="AAAA...", metadata={"source": "doc.txt"}),
    Document(page_content="AAAA...", metadata={"source": "doc.txt"})
]

```

---

### Tests Unitarios (`tests/test_ingester.py`)

```python
import pytest
from langchain_core.documents import Document
from app.ingester import load_document, split_documents

def test_load_txt_file_success(tmp_path):
    # Setup: Crear archivo temporal
    file = tmp_path / "test.txt"
    file.write_text("Contenido de prueba", encoding="utf-8")
    
    # Exec
    result = load_document(str(file))
    
    # Assert
    assert len(result) == 1
    assert result[0].page_content == "Contenido de prueba"
    assert result[0].metadata["source"] == str(file)

def test_load_unsupported_extension():
    with pytest.raises(ValueError, match="Formato no soportado"):
        load_document("archivo.docx")

def test_split_documents_length():
    long_text = "Palabra " * 200 # ~1600 caracteres
    doc = Document(page_content=long_text, metadata={"source": "test.txt"})
    
    chunks = split_documents([doc], chunk_size=500, chunk_overlap=50)
    
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 550 # Tolerancia por palabras completas

```

---

## 2. Módulo de Almacenamiento Vectorial (`app/vector_store.py`)

### `class VectorStoreManager`

#### `__init__(self, persist_directory: str = "data/db", collection_name: str = "local_rag")`

* **Inicialización:** Instancia `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` y crea el cliente `Chroma(persist_directory=..., embedding_function=...)`.

#### `add_documents(self, docs: list[Document]) -> list[str]`

* **Propósito:** Guarda la lista de fragmentos y genera sus embeddings.
* **Retorno:** Lista de IDs generados (UUIDs en formato `string`).

#### `search_similar(self, query: str, k: int = 3) -> list[Document]`

* **Propósito:** Convierte `query` a vector y recupera los $k$ documentos más cercanos semánticamente.

```python
# Ejemplo de flujo
vs = VectorStoreManager(persist_directory=":memory:") # DB temporal en RAM para tests
vs.add_documents([
    Document(page_content="La política de devoluciones permite 30 días.", metadata={"id": 1}),
    Document(page_content="El envío a domicilio tarda 48 horas.", metadata={"id": 2})
])

# Entrada de búsqueda
res = vs.search_similar(query="¿Cuánto tiempo tengo para devolver un producto?", k=1)

# Salida esperada
[
    Document(page_content="La política de devoluciones permite 30 días.", metadata={"id": 1})
]

```

---

### Tests Unitarios (`tests/test_vector_store.py`)

```python
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

```

---

## 3. Módulo RAG Pipeline (`app/rag_chain.py`)

### `class RAGService`

#### `__init__(self, vector_store: VectorStoreManager, model_name: str = "llama3:8b", base_url: str = "http://localhost:11434")`

* **Inicialización:** Guarda la referencia a `vector_store` e instancia `OllamaLLM(model=model_name, base_url=base_url)`.

#### `generate_answer(self, query: str) -> dict`

* **Propósito:** Coordina la búsqueda, arma el prompt y llama a Ollama.
* **Estructura del Prompt interno:**
```text
Responde a la pregunta del usuario utilizando ÚNICAMENTE la información provista en el contexto. Si no puedes responder con este contexto, di "No encuentro información sobre eso en los documentos".

Contexto:
{context}

Pregunta: {query}
Respuesta:

```



```python
# Entrada
query = "¿Cuál es el plazo de devolución?"

# Salida esperada del método generate_answer(query)
{
    "query": "¿Cuál es el plazo de devolución?",
    "answer": "El plazo de devolución es de 30 días corridos.",
    "sources": [
        {
            "content": "La política de devoluciones permite 30 días.",
            "source": "manual_politicas.pdf"
        }
    ]
}

```

---

### Tests Unitarios (`tests/test_rag_chain.py`)

```python
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

```

---

## 4. API REST FastAPI (`app/main.py`)

### Endpoint 1: `POST /upload`

* **Input:** `multipart/form-data` con campo `file: UploadFile`.
* **Proceso:**
1. Guarda temporalmente el archivo recibido en `data/uploads/`.
2. Ejecuta `load_document()`.
3. Ejecuta `split_documents()`.
4. Ejecuta `vector_store.add_documents()`.
5. Elimina el archivo temporal.


* **Output (JSON - 200 OK):**
```json
{
  "filename": "manual.pdf",
  "status": "processed",
  "chunks_created": 14
}

```


* **Errores:** `400 Bad Request` si la extensión no es `.txt` o `.pdf`.

---

### Endpoint 2: `POST /query`

* **Input (JSON):**
```json
{
  "question": "¿Cuál es la política de garantía?"
}

```


* **Output (JSON - 200 OK):**
```json
{
  "query": "¿Cuál es la política de garantía?",
  "answer": "La garantía tiene una vigencia de 12 meses.",
  "sources": [
    {
      "content": "La garantía tiene una vigencia de 12 meses...",
      "source": "manual.pdf"
    }
  ]
}

```



---

### Tests Integración API (`tests/test_api.py`)

```python
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

```

---

## Guía de Inicio y Orden de Desarrollo

```bash
# 1. Instalar dependencias
pip install fastapi uvicorn langchain langchain-community langchain-ollama chromadb sentence-transformers pypdf pytest pytest-mock

# 2. Paso a paso sugerido:
# a) Crear app/ingester.py y correr pytest tests/test_ingester.py hasta que pase.
# b) Crear app/vector_store.py y correr pytest tests/test_vector_store.py.
# c) Crear app/rag_chain.py y verificar con mocks en test_rag_chain.py.
# d) Integrar todo en app/main.py y testear la API.

```

## Teoria:


La taxonomía moderna de RAG se divide en cinco grandes arquitecturas según su complejidad y capacidad de razonamiento:

| Tipo de RAG | Mecanismo Clave | Ventaja Principal | Problema que Resuelve |
| --- | --- | --- | --- |
| **1. Naive RAG** *(El que ya armaste)* | Búsqueda por similitud vectorial directa (Top-K) + LLM. | Fácil y rápido de implementar. | Útil solo para consultas simples y directas. |
| **2. Advanced RAG** | Procesamiento en 3 fases: Pre-Retrieval, Retrieval Híbrido y Post-Retrieval (Reranking). | Reduce drásticamente el ruido y las alucinaciones. | Resuelve fragmentos cortados sin contexto y fallos de similitud por vocabulario. |
| **3. Modular RAG** | Componentes intercambiables en flujo dinámico (routing, memoria, múltiples índices). | Altamente escalable y adaptable. | Permite combinar fuentes estructuradas, no estructuradas y memoria conversacional. |
| **4. Agentic RAG** | Un agente LLM decide cuándo buscar, cómo reformular y si la información alcanza (Loops/Multi-hop). | Maneja preguntas complejas multi-paso. | Resuelve consultas que requieren combinar datos dispersos en varios documentos. |
| **5. GraphRAG** | Utiliza Grafos de Conocimiento (Entidades + Relaciones) junto con embeddings vectoriales. | Excelente comprensión global y conceptual. | Resuelve preguntas analíticas abstractas (ej. *"¿Cuáles son los temas principales del corpus?"*). |
| **6. Corrective / Self-RAG** | Evaluación en tiempo real de los resultados recuperados con mecanismos de autocorrección. | Garantía de calidad en las respuestas. | Evita responder si los fragmentos recuperados son irrelevantes o contradictorios. |

---

### Detalle de las arquitecturas principales

#### 1. Advanced RAG (El paso natural tras Naive RAG)

No cambia el flujo base, sino que optimiza las tres etapas principales:

* **Pre-Retrieval (Antes de buscar):**
* *Query Rewriting / Expansion:* Reescribe la pregunta del usuario para hacerla más clara o genera 3 variaciones de la misma pregunta para buscar mejores coincidencias.
* *Parent-Document Retriever:* Busca fragmentos pequeños (100 tokens) para máxima precisión vectorial, pero le entrega al LLM el párrafo entero o documento padre (1000 tokens) para no perder contexto.


* **Retrieval Híbrido (Durante la búsqueda):**
* Combina búsqueda vectorial (semántica) con búsqueda léxica/BM25 (palabras clave exactas, códigos, nombres propios).


* **Post-Retrieval (Después de buscar):**
* *Reranking:* Reordena los fragmentos recuperados mediante un modelo especializado (Cross-Encoder) y descarta los que tengan bajo puntaje antes de armar el prompt final.



#### 2. Agentic RAG (RAG basado en Agentes)

En lugar de ejecutar un flujo rígido `Búsqueda -> Prompt -> Respuesta`, la consulta entra a un agente que razona sobre lo que necesita:

* **Ejemplo de flujo:** Si el usuario pregunta *"¿Qué diferencia hay entre el contrato de 2023 y el de 2024?"*, el agente:
1. Identifica que necesita hacer 2 búsquedas separadas.
2. Busca *"Contrato 2023"*, guarda los datos.
3. Busca *"Contrato 2024"*, guarda los datos.
4. Compara ambos resultados y sintetiza la respuesta.



#### 3. Self-RAG y Corrective RAG (CRAG)

Integran **nodos de evaluación (Graders)**:

* Un modelo clasificador verifica si los fragmentos recuperados por ChromaDB realmente responden la pregunta.
* **Si aprueban:** Se genera la respuesta.
* **Si reprueban:** El sistema activa un mecanismo alternativo (reescribe la consulta, busca en otra colección o consulta a internet).

#### 4. GraphRAG (Knowledge Graphs + Vectors)

Técnica popularizada por Microsoft Research. En lugar de dividir el texto únicamente en fragmentos planos:

1. Extrae nodos (ej: *Cliente*, *Producto*, *Garantía*) y bordes (*"Cliente COMPRA Producto"*, *"Producto TIENE Garantía"*).
2. Construye una red lógica.
3. Permite responder consultas sobre cómo se relacionan entidades a lo largo de cientos de archivos.
