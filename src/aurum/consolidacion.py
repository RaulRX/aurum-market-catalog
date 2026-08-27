"""NB09: la tabla comparativa que junta lo que ya midieron NB01-NB06.

No recalcula nada -salvo el ranking del ANN elegido (R04), que ningun notebook
anterior dejo en un artefacto persistido-: lee lo que `baseline_lexico.json`
(NB01), `comparativa_representacion.json` (NB03) y `benchmark_ann.csv` (NB06)
ya midieron, y lo pone en una sola tabla con las mismas columnas para todas
las filas -para que a ninguna se le olvide una en silencio-.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

COLUMNAS_TABLA_COMPARATIVA = (
    "config", "modelo", "plantilla", "dim", "metrica", "ann",
    "ndcg_at_10", "recall_at_10", "mrr_at_10", "p50_ms", "p95_ms", "nota",
)


def fila_comparativa(config: str, **campos: Any) -> dict[str, Any]:
    """Una fila de la tabla comparativa de NB09, con sus columnas en un
    orden fijo. Los campos no dados quedan en `None` -y se ven como
    huecos en la tabla, no como un `0` que parecería una medición real-.

    Levanta si `campos` trae una clave que no es columna de la tabla: es
    mejor fallar aquí que dejar que una fila lleve una columna que las demas
    no tienen y que pandas rellene de NaN sin que nadie lo pida.
    """
    desconocidos = set(campos) - (set(COLUMNAS_TABLA_COMPARATIVA) - {"config"})
    if desconocidos:
        raise ValueError(
            f"Campos no reconocidos para la tabla comparativa: {sorted(desconocidos)}"
        )
    fila = {"config": config}
    fila.update(
        {columna: campos.get(columna) for columna in COLUMNAS_TABLA_COMPARATIVA if columna != "config"}
    )
    return fila


def tabla_comparativa(filas: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Junta las filas ya normalizadas por `fila_comparativa` en una tabla."""
    if not filas:
        raise ValueError("No hay filas con las que construir la tabla comparativa.")
    return pd.DataFrame(list(filas), columns=list(COLUMNAS_TABLA_COMPARATIVA))


def diagnosticar_consulta(
    query_id: str,
    *,
    ranking_oraculo: Sequence[str],
    ranking_ann: Sequence[str],
    qrels: Mapping[str, float],
    k: int = 10,
) -> dict[str, Any]:
    """Evidencia de las capas 1 y 2 de la atribución de errores (NB09 ·
    procedimiento): qué trajo el oráculo exacto, qué trajo el ANN, y
    cuáles de esos productos son relevantes de verdad (D01: `E`/`S`, score
    `>= RELEVANCE_THRESHOLD`).

    No concluye la capa -leer si el fallo es de representación o de índice
    es un juicio humano sobre la evidencia, no algo que esta función deba
    decidir-, solo reúne los hechos en un solo sitio:

    - `n_relevantes_en_oraculo` bajo → el oráculo exacto ya falla: es
      **representación** (modelo o plantilla), no tiene sentido mirar el ANN.
    - `perdidos_por_el_ann` no vacío (con `n_relevantes_en_oraculo` alto) →
      el oráculo encontraba relevantes que el ANN no trajo: es **índice**
      (candidato a subir `ef`, y se comprueba como prueba explícita).
    """
    from .evaluacion import RELEVANCE_THRESHOLD

    top_oraculo = list(ranking_oraculo[:k])
    top_ann = list(ranking_ann[:k])

    def _relevantes(ranking: Sequence[str]) -> list[str]:
        return [pid for pid in ranking if qrels.get(pid, 0.0) >= RELEVANCE_THRESHOLD]

    relevantes_en_oraculo = _relevantes(top_oraculo)
    return {
        "query_id": query_id,
        "top_oraculo": top_oraculo,
        "top_ann": top_ann,
        "relevantes_en_oraculo": relevantes_en_oraculo,
        "relevantes_en_ann": _relevantes(top_ann),
        "n_relevantes_en_oraculo": len(relevantes_en_oraculo),
        "n_relevantes_en_ann": len(_relevantes(top_ann)),
        "perdidos_por_el_ann": [pid for pid in relevantes_en_oraculo if pid not in top_ann],
    }
