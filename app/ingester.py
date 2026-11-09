from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, PyMuPDFLoader
from pathlib import Path

import re
from collections import Counter

def load_document(file_path: str) -> list[Document]:
    """
    Lee un archivo del disco local y lo convierte en objetos 
    Document estándar (con page_content y metadata).


    ref:
    Estas librerias son siguen un loader base:
    https://reference.langchain.com/python/langchain-community/document_loaders/pdf/BasePDFLoader
    """


    if file_path.endswith(".pdf"):
        loader = PyMuPDFLoader(file_path)    #PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Formato no soportado: {file_path}")

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

  
    return loader.load() #->list[document]



def split_documents(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Document]:
    """
    Fragmenta una lista de documentos en trozos más pequeños preservando los metadatos de origen.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        separators    = ["\n\n", "\n", " ", ""],
        chunk_size    = chunk_size,
        chunk_overlap = chunk_overlap
    )
    
    return splitter.split_documents(docs)


def is_informative_text(text: str) -> bool:
    """
    Detecta fragmentos basura típicos de PDFs: páginas de gráficos que se
    extraen como "G G G G...", líneas de símbolos o texto muy repetitivo.

    Se aplica sólo en la ingesta; `split_documents` queda genérico.
    """
    all_tokens = re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)
    if len(all_tokens) < 20:
        return False

    # Los gráficos de puntos salen como cientos de letras sueltas ("G G G").
    # El texto real (aun con ecuaciones y variables) queda muy por debajo.
    single_ratio = sum(1 for token in all_tokens if len(token) == 1) / len(all_tokens)
    if single_ratio > 0.60:
        return False

    # Si una única palabra/letra domina el fragmento, es basura de figura
    # (p. ej. "G" repetida), no prosa ni ecuaciones.
    top_fraction = Counter(all_tokens).most_common(1)[0][1] / len(all_tokens)
    if top_fraction > 0.30:
        return False

    tokens = [token for token in all_tokens if len(token) >= 2]
    if len(tokens) < 8:
        return False

    counts = Counter(tokens)
    # Muy poca variedad de palabras → texto repetitivo/basura.
    if len(counts) / len(tokens) < 0.30:
        return False
    # Una sola palabra domina el fragmento.
    if max(counts.values()) / len(tokens) > 0.45:
        return False

    return True


def filter_informative_chunks(chunks: list[Document]) -> list[Document]:
    """Devuelve sólo los fragmentos con contenido textual aprovechable."""
    return [chunk for chunk in chunks if is_informative_text(chunk.page_content)]

    

if __name__ == "__main__":
    path = "reporte.pdf"
    doc = load_document(path)
    split = split_documents(doc)
    print(f"len: {len(doc)}: {doc[0].page_content}")
    print("######################")
    print(f"len = {len(split)}, content: \n {split[0].page_content}")

    print("######################")
    print(f" meta: \n {split[0].metadata}")
