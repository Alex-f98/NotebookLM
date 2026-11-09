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


def test_extract_triples_ignores_prose_around_json(mocker):
    mock_llm_output = (
        "A continuación las relaciones:\n\n"
        '[{"source": "ESP32", "relation": "USA_BUS", "target": "SPI"}]\n\n'
        "Observaciones: ..."
    )
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_llm_output

    triples = extract_triples_from_chunk("El ESP32 usa SPI.", mock_llm)

    assert len(triples) == 1
    assert triples[0].source == "ESP32"


def test_extract_triples_handles_code_fence(mocker):
    mock_llm_output = '```json\n[{"source": "A", "relation": "R", "target": "B"}]\n```'
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_llm_output

    triples = extract_triples_from_chunk("texto", mock_llm)

    assert len(triples) == 1
    assert triples[0].relation == "R"
