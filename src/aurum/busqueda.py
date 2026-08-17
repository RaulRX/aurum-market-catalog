"""Interfaz común de recuperación y buscador denso exacto.

El enunciado (§3.3) pide *"una interfaz común -función, clase, API o comando-
que reciba una consulta y devuelva resultados normalizados"*. Ese contrato vive
aquí, no en cada implementación: `SearchResult` y `stable_top_k_indices` los
comparten el baseline léxico (`aurum.lexico`), este buscador denso y, más
adelante, el motor vectorial de NB04. Así una tabla comparativa puede mezclar
filas de sistemas distintos sin traducir formatos.

`DenseRetriever` es el **oráculo exacto**: recorre todos los vectores sin
índice aproximado. En NB02 es el buscador que compara modelos (la calidad de la
representación no debe medirse a través de la pérdida de un ANN); en NB06 es la
referencia contra la que se mide la fidelidad del ANN del motor.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

DEFAULT_TOP_K = 10

# Regla 5 de experimentación: `cosine` y `dot` son similitudes (mayor = mejor);
# `l2` es una distancia (menor = mejor). El contrato de `SearchResult` obliga a
# declarar cuál de las dos cosas es el score, para no mezclarlas en una tabla.
METRICS = ("cosine", "dot", "l2")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Un documento recuperado, con su posición y su puntuación."""

    rank: int
    document_id: str
    score: float
    score_es_similitud: bool = True


class Retriever(Protocol):
    """Lo mínimo que debe ofrecer cualquier buscador del proyecto."""

    name: str

    def search(
        self, query_text: str, *, k: int = DEFAULT_TOP_K
    ) -> tuple[SearchResult, ...]:
        """Devuelve los `k` documentos mejor puntuados para la consulta."""
        ...


def stable_top_k_indices(scores: NDArray[Any], *, k: int) -> NDArray[np.intp]:
    """Índices del top-k por puntuación, con el orden original como desempate.

    El desempate determinista es lo que hace que dos ejecuciones den métricas
    idénticas — un criterio de verificación explícito en NB09."""
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scores debe ser un vector unidimensional no vacío.")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k debe ser un entero positivo.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contiene NaN o infinito.")

    effective_k = min(k, scores.size)
    # np.lexsort ordena por la última clave: primero por -score, y a igualdad
    # de score, por el índice original ascendente.
    order = np.lexsort((np.arange(scores.size), -scores))
    return order[:effective_k]


def build_results(
    indices: NDArray[np.intp],
    scores: NDArray[Any],
    document_ids: Sequence[str],
    *,
    score_es_similitud: bool = True,
) -> tuple[SearchResult, ...]:
    """Empaqueta un top-k ya ordenado en el contrato común."""
    return tuple(
        SearchResult(
            rank=rank,
            document_id=document_ids[int(index)],
            score=float(scores[int(index)]),
            score_es_similitud=score_es_similitud,
        )
        for rank, index in enumerate(indices, start=1)
    )


def _validate_matrix(
    vectors: NDArray[Any], document_ids: Sequence[str]
) -> tuple[NDArray[np.float32], tuple[str, ...]]:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("vectors debe ser una matriz 2D no vacía.")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors contiene NaN o infinito.")
    ids = tuple(str(document_id) for document_id in document_ids)
    if len(ids) != matrix.shape[0]:
        raise ValueError(
            f"document_ids y vectores deben alinearse: {len(ids)} != {matrix.shape[0]}."
        )
    if len(set(ids)) != len(ids):
        raise ValueError("document_ids contiene identificadores duplicados.")
    return matrix, ids


def _l2_normalize(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Un vector nulo no se puede normalizar: se deja como está en vez de
    # producir NaN, que rompería el top-k aguas abajo.
    return np.divide(matrix, np.where(norms == 0, 1.0, norms)).astype(np.float32)


class DenseRetriever:
    """Búsqueda densa exacta por producto escalar o distancia euclídea.

    Recorre los `n` vectores en cada consulta. Con 15.000 × 768 en `float32`
    (~46 MB) eso son unos milisegundos, así que no compensa un índice: lo que
    aquí interesa es que el resultado sea **exacto**, para que la comparación
    entre modelos de NB02 no arrastre la pérdida de ningún ANN.

    `metric` cambia qué significa el score y, sin normalizar, también el orden:

    - `cosine` normaliza documentos y consulta antes del producto escalar.
    - `dot` no normaliza nada, así que premia los vectores de norma grande.
    - `l2` ordena por distancia euclídea ascendente.

    Con vectores ya L2-normalizados las tres dan el **mismo ranking**. Que se
    cumpla es la verificación de normalización que pide NB02; que deje de
    cumplirse significa que la normalización no se está aplicando."""

    def __init__(
        self,
        vectors: NDArray[Any],
        document_ids: Sequence[str],
        *,
        metric: str = "cosine",
        name: str = "dense",
    ) -> None:
        if metric not in METRICS:
            raise ValueError(f"metric debe ser uno de {METRICS}")
        matrix, self._document_ids = _validate_matrix(vectors, document_ids)
        self.name = name
        self.metric = metric
        self._matrix = _l2_normalize(matrix) if metric == "cosine" else matrix

    @property
    def document_ids(self) -> tuple[str, ...]:
        """IDs indexados, en el mismo orden que las filas de la matriz."""
        return self._document_ids

    @property
    def dim(self) -> int:
        """Dimensión de los vectores indexados."""
        return int(self._matrix.shape[1])

    @property
    def score_es_similitud(self) -> bool:
        """`False` para `l2`: su score es una distancia, no una similitud."""
        return self.metric != "l2"

    def _scores(self, query_vector: NDArray[Any]) -> NDArray[np.float32]:
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.size != self._matrix.shape[1]:
            raise ValueError(
                f"La consulta tiene dimensión {query.size} y el índice {self._matrix.shape[1]}."
            )
        if not np.isfinite(query).all():
            raise ValueError("El vector de consulta contiene NaN o infinito.")

        if self.metric == "l2":
            return np.linalg.norm(self._matrix - query, axis=1).astype(np.float32)
        if self.metric == "cosine":
            query = _l2_normalize(query.reshape(1, -1))[0]
        return (self._matrix @ query).astype(np.float32)

    def search_vector(
        self, query_vector: NDArray[Any], *, k: int = DEFAULT_TOP_K
    ) -> tuple[SearchResult, ...]:
        """Top-k para una consulta **ya codificada**.

        Es la entrada natural del buscador denso: codificar las consultas en
        un solo lote fuera de aquí evita una llamada al modelo (o a la API) por
        consulta."""
        scores = self._scores(query_vector)
        # Con `l2` se ordena por distancia ascendente, pero el score reportado
        # sigue siendo la distancia: negar solo sirve para ordenar.
        ranking_scores = -scores if self.metric == "l2" else scores
        indices = stable_top_k_indices(ranking_scores, k=k)
        return build_results(
            indices, scores, self._document_ids, score_es_similitud=self.score_es_similitud
        )


def rank_queries_dense(
    retriever: DenseRetriever,
    query_ids: Sequence[str],
    query_vectors: NDArray[Any],
    *,
    k: int = DEFAULT_TOP_K,
) -> dict[str, list[str]]:
    """Ejecuta un lote de consultas ya codificadas → `{query_id: [doc_id, ...]}`.

    Es la forma que espera `evaluacion.evaluate_rankings`. Guardar los IDs y no
    solo la métrica es lo que permite atribuir errores en NB09 (Regla 3)."""
    matrix = np.asarray(query_vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(query_ids):
        raise ValueError(
            f"Se esperaban {len(query_ids)} vectores de consulta, llegaron {matrix.shape}."
        )
    return {
        str(query_id): [
            result.document_id for result in retriever.search_vector(vector, k=k)
        ]
        for query_id, vector in zip(query_ids, matrix, strict=True)
    }


def results_frame(
    results: Mapping[str, Sequence[SearchResult]],
    *,
    metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> pd.DataFrame:
    """Aplana resultados al formato normalizado que exige el enunciado §3.3.

    Cada fila lleva `query_id`, posición, `product_id`, score y los metadatos
    que se le adjunten por documento (título y marca, típicamente). Es la
    estructura de la que salen los CSV de entrega de NB09."""
    filas = []
    for query_id, resultados in results.items():
        for result in resultados:
            fila: dict[str, object] = {
                "query_id": query_id,
                "rank": result.rank,
                "product_id": result.document_id,
                "score": round(result.score, 6),
                "score_es_similitud": result.score_es_similitud,
            }
            if metadata is not None:
                fila.update(metadata.get(result.document_id, {}))
            filas.append(fila)
    return pd.DataFrame(filas)
