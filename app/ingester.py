from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, PyMuPDFLoader
from pathlib import Path

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



def split_documents(docs: list[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> list[Document]:
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
    

if __name__ == "__main__":
    path = "reporte.pdf"
    doc = load_document(path)
    split = split_documents(doc)
    print(f"len: {len(doc)}: {doc[0].page_content}")
    print("######################")
    print(f"len = {len(split)}, content: \n {split[0].page_content}")

    print("######################")
    print(f" meta: \n {split[0].metadata}")
