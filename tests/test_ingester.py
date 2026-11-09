import pytest
from langchain_core.documents import Document
from app.ingester import (
    load_document,
    split_documents,
    is_informative_text,
    filter_informative_chunks,
)

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


def test_is_informative_text_rejects_plot_junk():
    # Página de gráfico extraída como letras sueltas.
    assert not is_informative_text("G G G G G G G G G G G G G G G G " * 30)


def test_is_informative_text_keeps_prose():
    texto = (
        "El tanque cónico piramidal tiene una bomba de agua que genera un flujo "
        "constante de entrada y una válvula que permite regularlo. El objetivo "
        "es controlar el nivel del tanque mediante un lazo de realimentación."
    )
    assert is_informative_text(texto)


def test_filter_informative_chunks_drops_junk():
    bueno = Document(
        page_content=(
            "El sensor de temperatura se comunica por el bus I2C y entrega "
            "lecturas cada segundo al microcontrolador principal del sistema."
        ),
        metadata={},
    )
    malo = Document(page_content="G " * 500, metadata={})
    result = filter_informative_chunks([bueno, malo])
    assert result == [bueno]

        