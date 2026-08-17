"""Ejecuta un notebook con nbclient (allow_errors=False) y guarda las salidas.

Uso: python scripts/execute_notebook.py <nombre.ipynb> [...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"


def execute(name: str) -> Path:
    path = NOTEBOOKS_DIR / name
    notebook = nbf.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="aurum-market-catalog",
        allow_errors=False,
        resources={"metadata": {"path": str(NOTEBOOKS_DIR)}},
    )
    client.execute()
    nbf.write(notebook, path)
    return path


def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit("Uso: python scripts/execute_notebook.py <notebook.ipynb> [...]")
    for name in argv:
        path = execute(name)
        print(f"Ejecutado {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
