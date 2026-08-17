"""Construye los .ipynb de notebooks/ a partir de notebook_cells.py.

Uso: python scripts/build_notebook.py [nombre.ipynb ...]   (sin argumentos: todos)
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).parent))
from notebook_cells import NOTEBOOKS

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"

KERNELSPEC = {
    "display_name": "Aurum Market (.venv)",
    "language": "python",
    "name": "aurum-market-catalog",
}


def build(name: str) -> Path:
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(source) if cell_type == "markdown" else nbf.v4.new_code_cell(source)
        for cell_type, source in NOTEBOOKS[name]
    ]
    notebook["metadata"]["kernelspec"] = KERNELSPEC
    output_path = NOTEBOOKS_DIR / name
    nbf.write(notebook, output_path)
    return output_path


def main(argv: list[str]) -> None:
    names = argv or list(NOTEBOOKS.keys())
    for name in names:
        path = build(name)
        print(f"Escrito {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
