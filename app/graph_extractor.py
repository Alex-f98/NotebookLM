import html
import json
import re
from dataclasses import dataclass

PROMPT_TEMPLATE = """
Analiza el siguiente texto técnico/electrónico y extrae únicamente las entidades principales y sus relaciones directas.
Formato de salida estricto en JSON array:
[
  {{"source": "EntidadA", "relation": "RELACION", "target": "EntidadB"}}
]
 
Ejemplo: "El sensor BME280 utiliza el bus I2C para comunicarse a 3.3V."
Salida:
[
  {{"source": "BME280", "relation": "USA_BUS", "target": "I2C"}},
  {{"source": "BME280", "relation": "VOLTAJE_OPERACION", "target": "3.3V"}}
]
 
Devolvé ÚNICAMENTE el JSON array, sin explicaciones, sin markdown y sin texto adicional.
Si no encontrás relaciones, devolvé [].

Texto a analizar:
{chunk_text}
"""


@dataclass
class KnowledgeTriple:
    source: str       # Ej: "ESP32"
    relation: str     # Ej: "REQUIERE_VOLTAJE"
    target: str       # Ej: "3.3V"


def _preproc(raw: str) -> str:
    """Recorta la respuesta del LLM y deja sólo el bloque JSON."""
    text = raw.strip()

    # Muchos modelos envuelven la respuesta en ```json ... ```
    if "```" in text:
        for block in text.split("```"):
            candidate = block.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("[") or candidate.startswith("{"):
                text = candidate
                break

    # El modelo a veces agrega explicaciones antes/después del JSON.
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        return html.unescape(text[start:end + 1])

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return html.unescape("[" + text[start:end + 1] + "]")

    return html.unescape(text)


def _parse_items(text: str) -> list[dict]:
    """Parsea el JSON esperado y, si falla, rescata objetos individuales."""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return [data]
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        pass

    # Fallback: el modelo devolvió varios objetos sueltos o con texto extra.
    items: list[dict] = []
    for match in re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL):
        try:
            obj = json.loads(match)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            items.append(obj)
    return items


def extract_triples_from_chunk(chunk_text: str, llm_client) -> list[KnowledgeTriple]:
    prompt = PROMPT_TEMPLATE.format(chunk_text=chunk_text)
    response = llm_client.invoke(prompt)

    data = _parse_items(_preproc(response))

    return [
        KnowledgeTriple(
            source=str(item["source"]).strip(),
            relation=str(item["relation"]).strip(),
            target=str(item["target"]).strip(),
        )
        for item in data
        if isinstance(item, dict)
        and all(k in item for k in ("source", "relation", "target"))
    ]
