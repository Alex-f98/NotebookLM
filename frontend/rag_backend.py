"""Utilidades compartidas por la interfaz Streamlit (ingesta, estado y grafo)."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
import networkx as nx
import streamlit as st

from app.ingester import load_document, split_documents
from app.vector_store import VectorStoreManager
from app.rag_chain import RAGService
from app.graph_rag_chain import GraphRAGService
from app.graph_extractor import extract_triples_from_chunk
from app.graph_store import GraphStoreManager


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_DIR = DATA_DIR / "db"
GRAPH_PATH = DATA_DIR / "knowledge_graph.graphml"
REGISTRY_PATH = DATA_DIR / "sources.json"

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

for directory in (UPLOAD_DIR, DB_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #

def ollama_models(base_url: str = DEFAULT_BASE_URL) -> list[str]:
    """Lista los modelos disponibles en Ollama (o [] si no está corriendo)."""
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=1.5)
        response.raise_for_status()
        return [model["name"] for model in response.json().get("models", [])]
    except Exception:
        return []


def ollama_online(base_url: str = DEFAULT_BASE_URL) -> bool:
    return bool(ollama_models(base_url))


# --------------------------------------------------------------------------- #
# Recursos cacheados
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def get_vector_store() -> VectorStoreManager:
    return VectorStoreManager(persist_directory=str(DB_DIR))


@st.cache_resource(show_spinner=False)
def get_rag(model_name: str = DEFAULT_MODEL,
            base_url: str = DEFAULT_BASE_URL) -> RAGService:
    return RAGService(
        vector_store=get_vector_store(),
        model_name=model_name,
        base_url=base_url,
    )


@st.cache_resource(show_spinner=False)
def get_graph_store() -> GraphStoreManager:
    store = GraphStoreManager()
    if GRAPH_PATH.exists():
        try:
            store.load_from_file(str(GRAPH_PATH))
        except Exception:
            # Un GraphML corrupto no debe romper la app.
            pass
    return store


def get_graph_rag(model_name: str, base_url: str = DEFAULT_BASE_URL) -> GraphRAGService:
    return GraphRAGService(
        vector_store=get_vector_store(),
        graph_store=get_graph_store(),
        llm_client=get_rag(model_name, base_url).model,
    )


# --------------------------------------------------------------------------- #
# Registro de fuentes (deduplicación por hash)
# --------------------------------------------------------------------------- #

def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_registry(registry: dict) -> None:
    REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_name(filename: str) -> str:
    name = os.path.basename(filename).strip().replace("\x00", "")
    return name or "documento.pdf"


def source_display_name(source: str) -> str:
    return Path(source).name or source


def list_sources() -> list[dict]:
    """Fuentes indexadas en Chroma, listas para mostrar en la barra lateral."""
    indexed = get_vector_store().get_sources()
    result = []
    for source, info in indexed.items():
        result.append(
            {
                "source": source,
                "name": source_display_name(source),
                "chunks": info.get("chunks", 0),
                "pages": info.get("pages", []),
            }
        )
    return sorted(result, key=lambda item: item["name"].lower())


def total_chunks() -> int:
    try:
        return get_vector_store().count()
    except Exception:
        return 0


# --------------------------------------------------------------------------- #
# Ingesta
# --------------------------------------------------------------------------- #

def ingest_uploads(files, progress_cb=None) -> list[dict]:
    """
    Indexa los PDFs/TXT subidos por el usuario.

    - Evita duplicados comparando el hash SHA-256 del contenido.
    - Si el mismo nombre se sube con otro contenido, reemplaza la fuente.
    - Guarda una copia en `data/uploads/` para que las fuentes persistan.
    """
    vector_store = get_vector_store()
    registry = _load_registry()
    indexed = set(vector_store.get_sources().keys())
    results: list[dict] = []

    for position, uploaded in enumerate(files):
        name = _safe_name(uploaded.name)
        if progress_cb:
            progress_cb(position, len(files), name)

        raw = uploaded.getvalue()
        digest = _sha256(raw)

        # ¿Ya está indexado el mismo contenido (con cualquier nombre)?
        duplicate = next(
            (
                source
                for source, meta in registry.items()
                if meta.get("sha256") == digest and source in indexed
            ),
            None,
        )
        if duplicate:
            results.append(
                {
                    "name": name,
                    "status": "duplicate",
                    "chunks": registry[duplicate].get("chunks", 0),
                    "message": f"Ya estaba indexado como {source_display_name(duplicate)}.",
                }
            )
            continue

        relative_source = f"data/uploads/{name}"
        target = UPLOAD_DIR / name

        try:
            # Si el nombre existe con otro contenido, eliminar la versión vieja.
            if relative_source in registry or relative_source in indexed:
                vector_store.delete_by_source(relative_source)

            target.write_bytes(raw)

            docs = load_document(str(target))
            for doc in docs:
                # Normalizamos el metadata para que no dependa del cwd.
                doc.metadata["source"] = relative_source

            chunks = split_documents(docs)
            if not chunks:
                raise ValueError("El documento no contiene texto extraíble.")

            ids = vector_store.add_documents(chunks)

            pages = sorted(
                {
                    doc.metadata["page"]
                    for doc in docs
                    if isinstance(doc.metadata.get("page"), int)
                }
            )
            registry[relative_source] = {
                "sha256": digest,
                "size": len(raw),
                "chunks": len(ids),
                "pages": pages,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }
            indexed.add(relative_source)
            results.append(
                {
                    "name": name,
                    "status": "indexed",
                    "chunks": len(ids),
                    "message": f"{len(ids)} fragmentos indexados.",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "name": name,
                    "status": "error",
                    "chunks": 0,
                    "message": str(exc),
                }
            )

    _save_registry(registry)
    if progress_cb:
        progress_cb(len(files), len(files), "")
    return results


def delete_source(source: str) -> None:
    get_vector_store().delete_by_source(source)
    registry = _load_registry()
    registry.pop(source, None)
    _save_registry(registry)


def clear_notebook() -> None:
    """Elimina todas las fuentes, el grafo y el historial del cuaderno."""
    store = get_vector_store()
    for source in store.get_sources():
        store.delete_by_source(source)

    _save_registry({})

    graph_store = get_graph_store()
    graph_store.graph.clear()
    if GRAPH_PATH.exists():
        GRAPH_PATH.unlink()


# --------------------------------------------------------------------------- #
# Grafo de conocimiento
# --------------------------------------------------------------------------- #

def build_graph(chunks: list, llm_client, progress_cb=None,
                reset: bool = True) -> GraphStoreManager:
    """
    Extrae tripletas de cada chunk y las acumula en el grafo persistente.

    La construcción es transaccional: si el LLM falla a mitad de camino, el
    grafo anterior queda intacto (no se vacía antes de terminar).
    """
    graph_store = get_graph_store()
    working = nx.DiGraph()
    if not reset:
        working = graph_store.graph.copy()

    total_triples = 0
    for position, chunk in enumerate(chunks, start=1):
        triples = extract_triples_from_chunk(chunk.page_content, llm_client)
        for t in triples:
            working.add_nodes_from(
                [(t.source, {"type": "entity"}), (t.target, {"type": "entity"})]
            )
            working.add_edge(t.source, t.target, relation=t.relation)
        total_triples += len(triples)
        if progress_cb:
            progress_cb(position, len(chunks), total_triples)

    # Sólo reemplazamos el grafo vivo cuando la extracción terminó bien.
    graph_store.graph = working
    if working.number_of_nodes():
        graph_store.save_to_file(str(GRAPH_PATH))

    return graph_store


def graph_to_dot(graph, max_edges: int = 120) -> str:
    """Convierte (una parte de) un MultiDiGraph/DiGraph a DOT para Graphviz."""
    def esc(value: str) -> str:
        text = str(value)
        # DOT no admite saltos de línea crudos dentro de las etiquetas.
        text = text.replace("\r", " ").replace("\n", " ")
        return text.replace("\\", "\\\\").replace('"', '\\"')

    lines = [
        "digraph G {",
        'rankdir=LR;',
        'bgcolor="transparent";',
        'graph [pad="0.2", nodesep="0.35", ranksep="0.7"];',
        'node [shape=box, style="rounded,filled", fillcolor="#eef0ff", '
        'color="#c7ccff", fontname="Inter", fontsize=10, fontcolor="#1f2328", '
        'margin="0.14,0.08"];',
        'edge [color="#9aa3b2", arrowsize=0.6, fontname="Inter", '
        'fontsize=8, fontcolor="#64748b"];',
    ]

    edges = list(graph.edges(data=True))[:max_edges]
    nodes = {node for edge in edges for node in edge[:2]}
    if not edges:
        for node in list(graph.nodes())[:max_edges]:
            nodes.add(node)

    for node in nodes:
        lines.append(f'"{esc(node)}";')

    for source, target, data in edges:
        relation = data.get("relation", "")
        lines.append(
            f'"{esc(source)}" -> "{esc(target)}" [label="{esc(relation)}"];'
        )

    lines.append("}")
    return "\n".join(lines)


def graph_relations(graph, limit: int = 60) -> list[str]:
    return [
        f"{source} —{data.get('relation', '')}→ {target}"
        for source, target, data in list(graph.edges(data=True))[:limit]
    ]


# --------------------------------------------------------------------------- #
# Auxiliares de UI
# --------------------------------------------------------------------------- #

def source_page_label(source: dict) -> str:
    page = source.get("page")
    return f"pág. {page + 1}" if isinstance(page, int) else ""


def format_sources(docs: list) -> list[dict]:
    """Convierte documentos de LangChain en referencias para la UI."""
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", ""),
            "page": doc.metadata.get("page"),
        }
        for doc in docs
    ]


def dedupe_sources(sources: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for item in sources:
        key = (item.get("source"), item.get("page"), item.get("content", "")[:80])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique
