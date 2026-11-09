"""Reindexa las fuentes registradas con el modelo y chunking actuales.

Uso (desde la raíz del proyecto):

    python -m frontend.reindex

Es necesario después de cambiar el modelo de embeddings, el tamaño de chunk o
el filtro de calidad, porque los vectores viejos dejan de ser comparables.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend import rag_backend as backend  # noqa: E402


def main() -> None:
    def progress(index: int, total: int, name: str) -> None:
        if name:
            print(f"[{index + 1}/{total}] {name}", flush=True)

    results = backend.reindex_sources(progress_cb=progress)

    print("\nResultados:")
    for result in results:
        print(f"  {result['status']:>9}  {result['name']}: {result['message']}")

    print(f"\nTotal de fragmentos indexados: {backend.get_vector_store().count()}")


if __name__ == "__main__":
    main()
