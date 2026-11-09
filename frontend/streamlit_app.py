"""Notebook LM local — interfaz Streamlit.

Ejecutar desde la raíz del proyecto:

    streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import hashlib
import html
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = Path(__file__).resolve().parent

# La raíz va primero para que `import app...` resuelva el paquete real.
for path in (str(FRONTEND_DIR), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import rag_backend as backend  # noqa: E402


# --------------------------------------------------------------------------- #
# Configuración de página
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Notebook LM local",
    page_icon="📔",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS_FILE = FRONTEND_DIR / "styles.css"
if CSS_FILE.exists():
    st.markdown(f"<style>{CSS_FILE.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Estado
# --------------------------------------------------------------------------- #

if "messages" not in st.session_state:
    st.session_state.messages = []
if "notes" not in st.session_state:
    st.session_state.notes = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "summary_key" not in st.session_state:
    st.session_state.summary_key = None
if "upload_results" not in st.session_state:
    st.session_state.upload_results = []
if "dialog_round" not in st.session_state:
    st.session_state.dialog_round = 0
if "notebook_title" not in st.session_state:
    st.session_state.notebook_title = "Mi cuaderno"
if "engine" not in st.session_state:
    st.session_state.engine = "Vectorial"
if "k" not in st.session_state:
    st.session_state.k = 3

BASE_URL = backend.DEFAULT_BASE_URL


@st.cache_data(ttl=15, show_spinner=False)
def available_models(base_url: str) -> list[str]:
    return backend.ollama_models(base_url)


MODELS = available_models(BASE_URL)
OLLAMA_ONLINE = bool(MODELS)

if "model" not in st.session_state:
    if backend.DEFAULT_MODEL in MODELS:
        st.session_state.model = backend.DEFAULT_MODEL
    elif MODELS:
        st.session_state.model = MODELS[0]
    else:
        st.session_state.model = backend.DEFAULT_MODEL


# --------------------------------------------------------------------------- #
# Helpers de UI
# --------------------------------------------------------------------------- #

def esc(value) -> str:
    return html.escape(str(value or ""))


def source_key(source: str) -> str:
    return hashlib.md5(source.encode("utf-8")).hexdigest()[:12]


def queue_question(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.pending_prompt = question
    st.rerun()


def open_add_sources() -> None:
    st.session_state.dialog_round += 1
    add_sources_dialog(st.session_state.dialog_round)


def render_citations(sources: list[dict]) -> None:
    if not sources:
        return

    chips = []
    for index, source in enumerate(sources, start=1):
        page = source.get("page")
        page_text = f" · pág. {page + 1}" if isinstance(page, int) else ""
        chips.append(
            '<span class="nb-cite">'
            f'<b>{index}</b>'
            f'<span class="nb-cite-name">{esc(backend.source_display_name(source.get("source", "")))}</span>'
            f'<span class="nb-cite-page">{esc(page_text)}</span>'
            "</span>"
        )

    st.markdown(
        '<div class="nb-cites">' + "".join(chips) + "</div>",
        unsafe_allow_html=True,
    )


def render_source_details(sources: list[dict]) -> None:
    if not sources:
        return

    with st.expander(f"📄 Ver fragmentos consultados ({len(sources)})"):
        for index, source in enumerate(sources, start=1):
            name = backend.source_display_name(source.get("source", ""))
            page = source.get("page")
            page_text = f" · página {page + 1}" if isinstance(page, int) else ""
            st.markdown(
                f'<div class="nb-ref-head"><span class="nb-ref-num">{index}</span>'
                f'<span class="nb-ref-name">{esc(name)}</span>'
                f'<span class="nb-ref-page">{esc(page_text)}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="nb-snippet">{esc(source.get("content", "")[:900])}'
                f'{"…" if len(source.get("content", "")) > 900 else ""}</div>',
                unsafe_allow_html=True,
            )
            if index < len(sources):
                st.markdown('<div class="nb-ref-sep"></div>', unsafe_allow_html=True)


def render_notes_panel() -> None:
    notes = st.session_state.notes
    if not notes:
        st.caption("Todavía no guardaste notas. Usá «Guardar en notas» debajo de una respuesta.")
        return

    lines = ["# Notas — Notebook LM local", ""]
    for note in notes:
        with st.expander(note["question"][:90] or "Nota"):
            st.markdown(note["answer"])
            st.caption(f"Guardada el {note['created']}")
        lines.append(f"## {note['question']}\n\n{note['answer']}\n")

    st.download_button(
        "Descargar notas (.md)",
        data="\n".join(lines),
        file_name="notas-notebooklm.md",
        mime="text/markdown",
        use_container_width=True,
    )

    if st.button("Borrar todas las notas", use_container_width=True):
        st.session_state.notes = []
        st.rerun()


def chat_markdown() -> str:
    lines = ["# Chat — Notebook LM local", ""]
    for message in st.session_state.messages:
        role = "Vos" if message["role"] == "user" else "Asistente"
        lines.append(f"## {role}\n\n{message['content']}\n")
        for source in backend.dedupe_sources(message.get("sources", [])):
            name = backend.source_display_name(source.get("source", ""))
            lines.append(f"> Fuente: {name}\n")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Diálogo para agregar fuentes
# --------------------------------------------------------------------------- #

@st.dialog("Agregar fuentes", width="large")
def add_sources_dialog(round_id: int) -> None:
    st.caption(
        "Subí tus PDFs (o archivos .txt). Se indexan localmente con Chroma + "
        "embeddings y quedan disponibles para el chat."
    )

    files = st.file_uploader(
        "Archivos",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key=f"dialog_uploader_{round_id}",
        label_visibility="collapsed",
    )

    if files:
        st.caption(f"{len(files)} archivo(s) listo(s) para indexar.")

    if st.button(
        "Indexar fuentes",
        type="primary",
        use_container_width=True,
        disabled=not files,
    ):
        progress = st.progress(0.0, text="Preparando…")

        def on_progress(index: int, total: int, name: str) -> None:
            fraction = min(index / total, 1.0) if total else 1.0
            label = f"Procesando {name} ({min(index + 1, total)}/{total})…" if name else "Listo"
            progress.progress(fraction, text=label)

        with st.status("Indexando documentos…", expanded=True):
            results = backend.ingest_uploads(files, progress_cb=on_progress)

        st.session_state.upload_results = results
        st.rerun()


# --------------------------------------------------------------------------- #
# Diálogo con el mapa de conocimiento
# --------------------------------------------------------------------------- #

@st.dialog("Mapa de conocimiento", width="large")
def graph_dialog() -> None:
    graph_store = backend.get_graph_store()
    graph = graph_store.graph

    if not graph.number_of_nodes():
        st.info("Todavía no hay relaciones extraídas. Construí el grafo desde Studio → Mapa.")
        return

    st.caption(
        f"{graph.number_of_nodes()} entidades · {graph.number_of_edges()} relaciones. "
        "El grafo se arma con tripletas (Entidad → Relación → Entidad) extraídas por el LLM."
    )
    st.graphviz_chart(backend.graph_to_dot(graph), width="stretch")

    with st.expander("Ver listado de relaciones"):
        for relation in backend.graph_relations(graph, limit=300):
            st.markdown(f"- {relation}")


# --------------------------------------------------------------------------- #
# Barra lateral: fuentes y ajustes
# --------------------------------------------------------------------------- #

sources = backend.list_sources()
TOTAL_CHUNKS = sum(source["chunks"] for source in sources)

with st.sidebar:
    st.markdown(
        '<div class="nb-brand">'
        '<div class="nb-brand-mark">◈</div>'
        '<div><div class="nb-brand-name">Notebook LM</div>'
        '<div class="nb-brand-sub">local · LangChain + Chroma + Ollama</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    if st.button("＋  Agregar fuentes", type="primary", use_container_width=True):
        open_add_sources()

    for result in st.session_state.upload_results:
        kind = result.get("status")
        icon = {"indexed": "✓", "duplicate": "•", "error": "!"}.get(kind, "•")
        st.markdown(
            f'<div class="nb-upload-result {esc(kind)}">'
            f'<span class="nb-upload-icon">{icon}</span>'
            f'<span><b>{esc(result.get("name"))}</b><br>'
            f'<span class="nb-upload-msg">{esc(result.get("message"))}</span></span>'
            "</div>",
            unsafe_allow_html=True,
        )
    if st.session_state.upload_results:
        if st.button("Ocultar avisos", use_container_width=True):
            st.session_state.upload_results = []
            st.rerun()

    st.markdown(
        f'<div class="nb-side-title">FUENTES '
        f'<span class="nb-count">{len(sources)}</span></div>',
        unsafe_allow_html=True,
    )

    selected_sources: list[str] = []
    if not sources:
        st.markdown(
            '<div class="nb-empty-sources">Todavía no hay fuentes.<br>'
            "Subí un PDF para empezar a chatear.</div>",
            unsafe_allow_html=True,
        )
    else:
        for index, source in enumerate(sources):
            key = source_key(source["source"])
            with st.container(key=f"sourcebox_{key}"):
                col_check, col_delete = st.columns(
                    [0.86, 0.14], vertical_alignment="center"
                )
                checked = col_check.checkbox(
                    source["name"],
                    value=True,
                    key=f"sel_{key}",
                    help=source["source"],
                )
                if checked:
                    selected_sources.append(source["source"])

                if col_delete.button(
                    " ",
                    key=f"del_{key}",
                    icon="🗑",
                    help="Eliminar esta fuente del índice",
                ):
                    backend.delete_source(source["source"])
                    st.toast(f"Fuente eliminada: {source['name']}")
                    st.rerun()

                pages = len(source["pages"])
                st.markdown(
                    f'<div class="nb-source-meta">{source["chunks"]} fragmentos'
                    f'{" · " + str(pages) + " págs." if pages else ""}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="nb-side-sep"></div>', unsafe_allow_html=True)

    with st.expander("⚙️  Ajustes de respuesta", expanded=False):
        if MODELS:
            if st.session_state.model not in MODELS:
                st.session_state.model = MODELS[0]
            st.selectbox("Modelo de Ollama", MODELS, key="model")
        else:
            st.text_input(
                "Modelo de Ollama",
                key="model",
                help="Ollama no responde. Verificá que esté corriendo: ollama serve",
            )

        st.radio(
            "Motor",
            ["Vectorial", "GraphRAG"],
            key="engine",
            horizontal=True,
            help="GraphRAG suma las relaciones del grafo de conocimiento al contexto.",
        )
        st.slider(
            "Fragmentos recuperados (k)",
            min_value=1,
            max_value=8,
            key="k",
            help="Cuántos fragmentos de texto se envían al modelo.",
        )

        if st.session_state.engine == "GraphRAG":
            node_count = backend.get_graph_store().graph.number_of_nodes()
            if node_count:
                st.caption(f"Grafo disponible: {node_count} entidades.")
            else:
                st.warning("No hay grafo todavía. Construilo en Studio → Mapa.")

    status_class = "ok" if OLLAMA_ONLINE else "down"
    status_text = (
        f"Ollama conectado · {st.session_state.model}"
        if OLLAMA_ONLINE
        else "Ollama sin conexión"
    )
    st.markdown(
        f'<div class="nb-status"><span class="nb-dot {status_class}"></span>'
        f"{esc(status_text)}</div>",
        unsafe_allow_html=True,
    )
    if not OLLAMA_ONLINE:
        if st.button("Reintentar conexión", use_container_width=True):
            available_models.clear()
            st.rerun()


# --------------------------------------------------------------------------- #
# Cabecera
# --------------------------------------------------------------------------- #

with st.container(key="topbar"):
    logo_col, title_col, actions_col = st.columns(
        [0.055, 0.70, 0.245], vertical_alignment="center"
    )

    logo_col.markdown('<div class="nb-logo">◈</div>', unsafe_allow_html=True)

    with title_col:
        st.text_input(
            "Título del cuaderno",
            key="notebook_title",
            label_visibility="collapsed",
            placeholder="Título del cuaderno",
        )
        st.markdown(
            f'<div class="nb-meta">{len(sources)} fuentes · {TOTAL_CHUNKS} fragmentos'
            f' · motor {esc(st.session_state.engine)} · {esc(st.session_state.model)}</div>',
            unsafe_allow_html=True,
        )

    with actions_col:
        clear_col, menu_col = st.columns([0.62, 0.38], vertical_alignment="center")
        with clear_col:
            if st.button(
                "Limpiar chat",
                icon="🧹",
                use_container_width=True,
                disabled=not st.session_state.messages,
            ):
                st.session_state.messages = []
                st.session_state.pending_prompt = None
                st.rerun()
        with menu_col:
            with st.popover("⋯", use_container_width=True):
                st.markdown("**Cuaderno**")
                st.caption(f"{len(sources)} fuentes · {TOTAL_CHUNKS} fragmentos")

                if st.session_state.messages:
                    st.download_button(
                        "Descargar chat (.md)",
                        data=chat_markdown(),
                        file_name="chat-notebooklm.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )

                if st.button("Vaciar notas", use_container_width=True,
                             disabled=not st.session_state.notes):
                    st.session_state.notes = []
                    st.rerun()

                st.divider()
                confirm = st.checkbox(
                    "Confirmo borrar todas las fuentes",
                    key="confirm_clear_notebook",
                )
                if st.button(
                    "Vaciar cuaderno",
                    type="primary",
                    use_container_width=True,
                    disabled=not confirm,
                ):
                    backend.clear_notebook()
                    st.session_state.messages = []
                    st.session_state.notes = []
                    st.session_state.summary = None
                    st.session_state.summary_key = None
                    st.session_state.upload_results = []
                    st.toast("Cuaderno vaciado")
                    st.rerun()


st.markdown('<div class="nb-top-sep"></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Zona principal: chat + studio
# --------------------------------------------------------------------------- #

pending = st.session_state.pending_prompt
st.session_state.pending_prompt = None

chat_col, studio_col = st.columns([2.25, 1], gap="large")

with chat_col:
    with st.container(key="chat_area"):
        if not st.session_state.messages and not pending:
            hero_text = (
                "Tus fuentes están listas. Preguntá lo que quieras y las respuestas "
                "van a citar los fragmentos exactos de tus documentos."
                if sources
                else "Subí tus PDFs y convertí este cuaderno en un asistente que "
                "responde solo con la información de tus documentos."
            )
            st.markdown(
                '<div class="nb-hero">'
                '<div class="nb-hero-glow"></div>'
                '<div class="nb-hero-icon">✦</div>'
                "<h2>Explorá tus documentos</h2>"
                f"<p>{esc(hero_text)}</p>"
                "</div>",
                unsafe_allow_html=True,
            )

            if not sources:
                if st.button(
                    "＋  Agregar mis primeros PDFs",
                    type="primary",
                    use_container_width=True,
                ):
                    open_add_sources()
            else:
                suggestions = [
                    "Resumí las ideas principales de las fuentes.",
                    "¿Cuáles son los conceptos y entidades más importantes?",
                    f"¿De qué trata «{sources[0]['name']}»?",
                    "¿Qué relaciones o diferencias encontrás entre las fuentes?",
                ]
                st.markdown('<div class="nb-sugg-title">Probá con:</div>',
                            unsafe_allow_html=True)
                suggestion_cols = st.columns(2)
                for index, question in enumerate(suggestions):
                    if suggestion_cols[index % 2].button(
                        question,
                        key=f"suggestion_{index}",
                        use_container_width=True,
                        disabled=not selected_sources or not OLLAMA_ONLINE,
                    ):
                        queue_question(question)

        # Historial
        last_question = ""
        for index, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                last_question = message["content"]
                with st.container(key=f"msg_user_{index}"):
                    with st.chat_message("user", avatar="🧑"):
                        st.markdown(message["content"])
                continue

            refs = backend.dedupe_sources(message.get("sources", []))
            with st.container(key=f"msg_bot_{index}"):
                with st.chat_message("assistant", avatar="📔"):
                    st.markdown(message["content"])

                    if message.get("graph_facts"):
                        with st.expander(
                            f"🕸️ Relaciones del grafo usadas ({len(message['graph_facts'])})"
                        ):
                            for relation in message["graph_facts"]:
                                st.markdown(f"- {relation}")

                    render_citations(refs)
                    render_source_details(refs)

                    if message.get("elapsed"):
                        st.caption(f"⏱️ {message['elapsed']} s")

                    if message.get("error"):
                        continue

                    if message.get("saved"):
                        st.markdown(
                            '<div class="nb-saved">📝 Guardada en Studio → Notas</div>',
                            unsafe_allow_html=True,
                        )
                    elif st.button(
                        "Guardar en notas",
                        key=f"save_note_{index}",
                        icon="📝",
                    ):
                        st.session_state.notes.append(
                            {
                                "question": last_question,
                                "answer": message["content"],
                                "created": datetime.now().strftime("%d/%m/%Y %H:%M"),
                            }
                        )
                        message["saved"] = True
                        st.toast("Respuesta guardada en Studio → Notas")
                        st.rerun()

        # Respuesta en curso (después del historial, antes del composer)
        if pending:
            question = pending
            engine = st.session_state.engine
            model_name = st.session_state.model
            rag = backend.get_rag(model_name, BASE_URL)

            with st.container(key=f"msg_bot_{len(st.session_state.messages)}"):
                with st.chat_message("assistant", avatar="📔"):
                    started = datetime.now()
                    try:
                        if engine == "GraphRAG":
                            docs, graph_facts, tokens = backend.get_graph_rag(
                                model_name, BASE_URL
                            ).stream_answer(
                                question,
                                k=st.session_state.k,
                                sources=selected_sources or None,
                            )
                        else:
                            docs, tokens = rag.stream_answer(
                                question,
                                k=st.session_state.k,
                                sources=selected_sources or None,
                            )
                            graph_facts = []

                        answer = st.write_stream(tokens)
                        answer = answer if isinstance(answer, str) else "".join(map(str, answer))
                    except Exception as exc:  # noqa: BLE001
                        st.error(
                            "No se pudo generar la respuesta. Verificá que Ollama "
                            f"esté corriendo y que el modelo «{model_name}» exista.\n\n"
                            f"Detalle: {exc}"
                        )
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": "⚠️ No se pudo generar la respuesta. "
                                "Revisá la conexión con Ollama.",
                                "sources": [],
                                "graph_facts": [],
                                "error": True,
                            }
                        )
                    else:
                        refs = backend.dedupe_sources(
                            backend.format_sources(docs)
                        )
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer.strip(),
                                "sources": refs,
                                "graph_facts": graph_facts,
                                "elapsed": round(
                                    (datetime.now() - started).total_seconds(), 1
                                ),
                            }
                        )
                        render_citations(refs)
                        render_source_details(refs)
                        st.caption(
                            f"⏱️ {round((datetime.now() - started).total_seconds(), 1)} s"
                        )

                    # Re-render desde el historial para habilitar acciones
                    # (guardar nota) y dejar el mensaje en su estado final.
                    st.rerun()

with studio_col:
    with st.container(key="studio_panel"):
        st.markdown(
            '<div class="nb-studio-head"><span class="nb-studio-spark">✦</span>'
            "<span>Studio</span></div>",
            unsafe_allow_html=True,
        )

        tab_summary, tab_graph, tab_notes = st.tabs(
            ["Resumen", "Mapa", "Notas" + (f" ({len(st.session_state.notes)})"
                                          if st.session_state.notes else "")]
        )

        with tab_summary:
            summary_key = hashlib.md5(
                ("|".join(sorted(selected_sources)) + st.session_state.model).encode()
            ).hexdigest()

            if not selected_sources:
                st.caption("Seleccioná al menos una fuente para generar el resumen.")
            else:
                generate = st.button(
                    "✦  Generar resumen",
                    type="primary",
                    use_container_width=True,
                    disabled=not OLLAMA_ONLINE,
                )
                generated_now = False

                if generate:
                    docs = backend.get_vector_store().get_documents(
                        sources=selected_sources, limit=50
                    )
                    context = "\n\n".join(doc.page_content for doc in docs)[:9000]
                    prompt = (
                        "Sos un asistente experto en síntesis de documentos. "
                        "A partir de las fuentes, escribí un resumen en español con:\n"
                        "1) un párrafo breve de contexto,\n"
                        "2) 5 bullets con las ideas clave,\n"
                        "3) 3 preguntas sugeridas para seguir explorando.\n"
                        "Usá sólo la información de las fuentes.\n\n"
                        f"FUENTES:\n{context}\n\nRESUMEN:"
                    )
                    try:
                        with st.spinner("Sintetizando las fuentes…"):
                            summary = st.write_stream(
                                backend.get_rag(st.session_state.model, BASE_URL).model.stream(prompt)
                            )
                        st.session_state.summary = (
                            summary if isinstance(summary, str) else "".join(map(str, summary))
                        )
                        st.session_state.summary_key = summary_key
                        generated_now = True
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"No se pudo generar el resumen: {exc}")

                if (
                    not generated_now
                    and st.session_state.summary
                    and st.session_state.summary_key == summary_key
                ):
                    with st.container(key="summary_card"):
                        st.markdown(st.session_state.summary)

        with tab_graph:
            graph_store = backend.get_graph_store()
            node_count = graph_store.graph.number_of_nodes()
            edge_count = graph_store.graph.number_of_edges()

            if node_count:
                st.markdown(
                    f'<div class="nb-stats">'
                    f'<div><b>{node_count}</b><span>entidades</span></div>'
                    f'<div><b>{edge_count}</b><span>relaciones</span></div>'
                    "</div>",
                    unsafe_allow_html=True,
                )
                if st.button("🗺️  Abrir mapa", use_container_width=True):
                    graph_dialog()

                with st.expander("Relaciones principales"):
                    relations = backend.graph_relations(graph_store.graph, limit=25)
                    for relation in relations:
                        st.markdown(f"- {relation}")
                    if not relations:
                        st.caption("Sin relaciones todavía.")
            else:
                st.caption(
                    "GraphRAG extrae entidades y relaciones de tus documentos "
                    "para responder preguntas que cruzan varias fuentes."
                )

            with st.expander(
                "Construir grafo de conocimiento",
                expanded=not node_count,
            ):
                fragment_limit = st.slider(
                    "Fragmentos a analizar",
                    min_value=5,
                    max_value=60,
                    value=20,
                    step=5,
                    help="Cada fragmento es una llamada al LLM. Menos fragmentos = más rápido.",
                )
                st.caption(
                    "Puede tardar varios minutos con modelos locales. "
                    "Reemplaza el grafo anterior."
                )

                build_disabled = not selected_sources or not OLLAMA_ONLINE
                if not OLLAMA_ONLINE:
                    st.caption(
                        "⚠️ Ollama no está disponible: el grafo se construye "
                        "llamando al LLM, así que no se puede extraer ahora."
                    )
                elif not selected_sources:
                    st.caption("Seleccioná al menos una fuente para construir el grafo.")

                if st.button(
                    "🕸️  Extraer relaciones",
                    use_container_width=True,
                    disabled=build_disabled,
                ):
                    chunks = backend.get_vector_store().get_documents(
                        sources=selected_sources, limit=fragment_limit
                    )
                    if not chunks:
                        st.warning("No hay fragmentos para analizar en las fuentes seleccionadas.")
                    else:
                        progress = st.progress(0.0, text="Iniciando extracción…")

                        def on_graph_progress(index: int, total: int, triples: int) -> None:
                            progress.progress(
                                index / total,
                                text=f"Fragmento {index}/{total} · {triples} relaciones",
                            )

                        try:
                            with st.status("Construyendo el grafo…", expanded=True):
                                backend.build_graph(
                                    chunks,
                                    backend.get_rag(
                                        st.session_state.model, BASE_URL
                                    ).model,
                                    progress_cb=on_graph_progress,
                                    reset=True,
                                )
                            st.toast("Grafo actualizado")
                            st.rerun()
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"No se pudo construir el grafo: {exc}")

        with tab_notes:
            render_notes_panel()


# --------------------------------------------------------------------------- #
# Composer
# --------------------------------------------------------------------------- #

with chat_col:
    with st.container(key="composer"):
        if not sources:
            placeholder = "Agregá una fuente para empezar…"
        elif not selected_sources:
            placeholder = "Seleccioná al menos una fuente para preguntar…"
        else:
            placeholder = "Preguntá lo que quieras sobre tus fuentes…"

        prompt = st.chat_input(
            placeholder,
            disabled=not selected_sources or not OLLAMA_ONLINE,
        )

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.pending_prompt = prompt
    st.rerun()
