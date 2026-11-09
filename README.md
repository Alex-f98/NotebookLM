# Notebook LM local

RAG + GraphRAG local sobre PDFs/TXT, con una interfaz tipo NotebookLM hecha en
Streamlit. Todo corre en tu máquina: **Ollama** para el LLM, **Chroma** para los
embeddings y **NetworkX** para el grafo de conocimiento.

```
Ollama (llama3:8b)
      │
      ├── Ingesta: PDF/TXT → chunks → embeddings → Chroma
      ├── RAG vectorial: pregunta → chunks → respuesta con citas
      └── GraphRAG: chunks → tripletas (Entidad→Relación→Entidad) → grafo
```

---

## Requisitos

- Python 3.10+
- [Ollama](https://ollama.com/) corriendo localmente
- Un modelo descargado, por ejemplo:

```bash
ollama pull llama3:8b
```

---

## 1. Levantar Ollama

Si no quedó como servicio del sistema:

```bash
ollama serve
```

Verificá que responda:

```bash
curl http://localhost:11434/api/tags
```

---

## 2. Instalar dependencias

Desde la raíz del proyecto:

```bash
python -m venv .env_lm          # si todavía no existe
. .env_lm/bin/activate
pip install -r requirements.txt
```

---

## 3. Interfaz tipo NotebookLM (Streamlit)

Es la forma recomendada de usar el proyecto.

```bash
# Con el venv activado, desde la raíz del proyecto
streamlit run frontend/streamlit_app.py
```

Se abre en **http://localhost:8501**.

### Uso paso a paso

#### 1) Agregar fuentes

1. En la barra lateral izquierda, tocá **＋ Agregar fuentes**.
2. Arrastrá uno o varios **PDFs** (o archivos `.txt`).
3. Tocá **Indexar fuentes**.

Cada archivo se guarda en `data/uploads/`, se fragmenta y se indexa en
`data/db/`. La app evita reindexar el mismo contenido comparando su hash
SHA-256 (`data/sources.json`). Si subís otro archivo con el mismo nombre pero
distinto contenido, reemplaza la versión anterior.

Los resultados de la indexación (indexado / duplicado / error) aparecen en la
barra lateral.

#### 2) Elegir qué fuentes usar

Cada fuente tiene un checkbox en la barra lateral:

- **Tildada** → entra en el contexto de las respuestas, el resumen y el grafo.
- **Destildada** → se ignora.
- El botón 🗑 elimina la fuente del índice y del registro.

Si ninguna fuente está tildada, el chat se deshabilita y te pide seleccionar al
menos una.

#### 3) Chatear

Escribí tu pregunta en el composer inferior y presioná Enter. La respuesta se
muestra en **streaming**, a medida que la genera Ollama.

Cada respuesta incluye:

- **Chips de citas** con el nombre del archivo y la página.
- Un expandible **Ver fragmentos consultados** con el texto exacto usado.
- El tiempo que tardó la respuesta.
- Un botón **📝 Guardar en notas**.

En la pantalla inicial aparecen preguntas sugeridas para arrancar.

#### 4) Studio (panel derecho)

El panel **Studio** tiene tres pestañas:

**Resumen**
- Genera una síntesis de las fuentes tildadas: contexto, ideas clave y
  preguntas sugeridas.
- Requiere al menos una fuente seleccionada y Ollama disponible.

**Mapa**
- **🕸️ Extraer relaciones** construye el grafo de conocimiento llamando al LLM
  fragmento por fragmento.
- Podés elegir cuántos fragmentos analizar (5 a 60). Cada fragmento es una
  llamada al LLM: menos fragmentos = más rápido.
- **Abrir mapa** muestra el grafo con Graphviz y un listado de relaciones.
- El grafo se guarda en `data/knowledge_graph.graphml` y persiste entre
  sesiones.
- La construcción es **transaccional**: si el LLM falla a mitad de camino, el
  grafo anterior queda intacto.
- Puede tardar varios minutos con modelos locales.

**Notas**
- Guardá respuestas del chat con **Guardar en notas**.
- Descargá todas las notas en Markdown.
- Borrá todas las notas con un botón.

#### 5) Ajustes de respuesta

Dentro de **⚙️ Ajustes de respuesta** en la barra lateral:

| Ajuste | Opciones | Qué hace |
| --- | --- | --- |
| **Modelo de Ollama** | modelos detectados | Modelo usado para responder y construir el grafo. |
| **Motor** | `Vectorial` / `GraphRAG` | `Vectorial` usa solo los chunks. `GraphRAG` suma las relaciones del grafo al contexto. |
| **Fragmentos recuperados (k)** | 1–8 | Cuántos chunks se envían al modelo. |

Si elegís **GraphRAG** y todavía no hay grafo, la app te avisa y te indica
construirlo desde Studio → Mapa.

#### 6) Acciones del cuaderno (menú ⋯)

En la cabecera podés:

- Editar el **título del cuaderno**.
- **Limpiar chat**.
- **Descargar chat (.md)**.
- **Vaciar notas**.
- **Vaciar cuaderno**: borra todas las fuentes, el grafo y el estado (pide
  confirmación).

---

## 4. API (opcional)

El backend FastAPI original sigue disponible si preferís usarlo por HTTP.

```bash
uvicorn app.main:app --reload
```

Deberías ver `Uvicorn running on http://127.0.0.1:8000`.

**Swagger UI**: abrí `http://localhost:8000/docs` para probar `/upload` y
`/query` sin tocar curl.

**Con curl:**

```bash
# Subir el PDF
curl -X POST http://localhost:8000/upload \
  -F "file=@reporte.pdf"

# Preguntar
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"¿De qué trata el documento?"}'
```

---

## Persistencia

Todo el estado queda en `data/`:

```text
data/
├── db/                     # Chroma (embeddings)
├── uploads/                # copia de los PDFs/TXT indexados
├── sources.json            # registro de hashes para evitar reindexar
└── knowledge_graph.graphml # grafo de conocimiento persistido
```

> Si borrás `data/db/`, se pierden los embeddings y hay que volver a indexar las
> fuentes. Si borrás `data/knowledge_graph.graphml`, se pierde el grafo y hay
> que reconstruirlo desde Studio → Mapa.

---

## Estructura del proyecto

```text
├── app/
│   ├── config.py            # Configuración central
│   ├── ingester.py          # Carga y chunking de documentos
│   ├── vector_store.py      # Embeddings + Chroma (filtros, listado, borrado)
│   ├── rag_chain.py         # RAG vectorial (prepare / stream / answer)
│   ├── graph_extractor.py   # Extracción de tripletas con el LLM
│   ├── graph_store.py       # Grafo NetworkX (vecindarios, persistencia)
│   ├── graph_rag_chain.py   # GraphRAG (vectorial + relaciones del grafo)
│   ├── verify_graph.py      # Utilidad de verificación del grafo
│   └── main.py              # API FastAPI (/upload, /query)
├── frontend/
│   ├── streamlit_app.py     # Interfaz NotebookLM
│   ├── rag_backend.py       # Ingesta, estado, grafo y helpers de la UI
│   └── styles.css           # Tema visual tipo NotebookLM
├── .streamlit/
│   └── config.toml          # Tema y configuración del servidor
├── data/                    # Estado persistente
├── tests/
└── requirements.txt
```

---

## Tests

```bash
# Desde la raíz, con el venv activado
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_mock -q
```

El prefijo `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` evita un conflicto con plugins de
pytest instalados a nivel de sistema. También podés correr simplemente:

```bash
pytest -q
```

---

## Solución de problemas

| Problema | Solución |
| --- | --- |
| La app dice **Ollama sin conexión** | Verificá `ollama serve` y el modelo (`ollama list`). Usá **Reintentar conexión** en la barra lateral. |
| El chat está deshabilitado | Tildá al menos una fuente y asegurate de que Ollama esté online. |
| **No hay grafo todavía** en GraphRAG | Construilo en Studio → Mapa → **Extraer relaciones**. |
| La construcción del grafo tarda mucho | Bajá «Fragmentos a analizar». Cada fragmento es una llamada al LLM. |
| Un PDF no devuelve texto | Puede ser un PDF escaneado (sin capa de texto). Probá con otro archivo o aplicá OCR antes. |
| Un archivo aparece duplicado / no se reindexa | La app deduplica por SHA-256. Si el contenido es idéntico, no lo vuelve a indexar. |
