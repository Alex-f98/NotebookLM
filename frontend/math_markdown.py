"""Normalización de LaTeX para que Streamlit lo renderice.

Streamlit usa KaTeX, que sólo reconoce los delimitadores `$...$` (en línea) y
`$$...$$` (bloque). Muchos modelos escriben `\\(...\\)` y `\\[...\\]`, o
directamente entornos como `\\begin{equation}...\\end{equation}`. En esos casos
Streamlit muestra el código crudo en vez de la fórmula.

Este módulo convierte esas variantes a los delimitadores que Streamlit entiende,
sin tocar el contenido que ya está dentro de `$...$`, `$$...$$` ni bloques de
código.
"""

from __future__ import annotations

import re

# Bloques de código: no se transforman (podrían contener LaTeX de ejemplo).
_FENCE_RE = re.compile(r"(```.*?```|~~~.*?~~~)", re.DOTALL)

# Contenido ya delimitado que hay que preservar tal cual.
# - $$...$$ puede ocupar varias líneas.
# - $...$ se limita a una línea para no confundir precios con fórmulas.
_MATH_RE = re.compile(
    r"(\$\$.*?\$\$|(?<!\\)\$(?!\$)[^$\n]*?(?<!\\)\$)",
    re.DOTALL,
)

# Delimitadores que KaTeX no entiende. El lookbehind evita romper `\\[2mm]`
# (salto de línea con espaciado dentro de un entorno align).
_DISPLAY_DELIM_RE = re.compile(r"(?<!\\)\\\[(.+?)(?<!\\)\\\]", re.DOTALL)
_INLINE_DELIM_RE = re.compile(r"(?<!\\)\\\((.+?)(?<!\\)\\\)", re.DOTALL)

_ENV_RE = re.compile(
    r"\\begin\{(equation\*?|displaymath|math|align\*?|aligned|gather\*?|gathered"
    r"|multline\*?|eqnarray\*?|split)\}(.*?)\\end\{\1\}",
    re.DOTALL,
)

# Entornos que KaTeX soporta tal cual dentro de modo display.
_KATEX_ENVS = {"align", "align*", "aligned", "gather", "gather*", "gathered", "split"}


def _environment_replacement(match: re.Match[str]) -> str:
    env = match.group(1)
    body = match.group(2).strip()

    if env in {"equation", "equation*", "displaymath", "math"}:
        return f"\n$$\n{body}\n$$\n"

    if env in _KATEX_ENVS:
        return f"\n$$\n\\begin{{{env}}}{body}\\end{{{env}}}\n$$\n"

    # eqnarray/multline no existen en KaTeX; `aligned` es el equivalente.
    return f"\n$$\n\\begin{{aligned}}{body}\\end{{aligned}}\n$$\n"


def _convert_delimiters(text: str) -> str:
    text = _DISPLAY_DELIM_RE.sub(
        lambda match: f"\n$$\n{match.group(1).strip()}\n$$\n", text
    )
    text = _INLINE_DELIM_RE.sub(
        lambda match: f"${match.group(1).strip()}$", text
    )
    return text


def _convert_segment(text: str) -> str:
    # Primero protegemos lo que ya está en formato válido.
    parts = _MATH_RE.split(text)

    converted: list[str] = []
    for index, part in enumerate(parts):
        if index % 2:
            converted.append(part)  # contenido matemático ya delimitado
            continue

        part = _ENV_RE.sub(_environment_replacement, part)
        converted.append(_convert_delimiters(part))

    return "".join(converted)


def normalize_math(text: str) -> str:
    """Convierte LaTeX con delimitadores no soportados al formato de Streamlit."""
    if not text:
        return text

    if not any(token in text for token in ("$", "\\(", "\\[", "\\begin{")):
        return text

    parts = _FENCE_RE.split(text)
    return "".join(
        part if index % 2 else _convert_segment(part)
        for index, part in enumerate(parts)
    )
