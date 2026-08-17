"""Baselines léxicos de Aurum Market: TF-IDF y BM25 (D05).

Adaptado de `sesion_01/.../retrieval.py` (`TfidfRetriever`,
`stable_top_k_indices`). BM25 no existe en las sesiones: se añade aquí sobre
`rank_bm25` porque es el estándar en recuperación de información y porque es
el único baseline que trata la **longitud de documento** de forma distinta a
TF-IDF — que es justo el eje con recorrido en estos datos (dispersión de
3,29× en `text`, medida en NB01).

Ambos retrievers comparten tokenizador (`aurum.datos.tokenize`), de modo que
la comparación varía un único factor: la fórmula de puntuación, no los
términos que cada uno ve (Regla 2 de experimentación).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

# El contrato de resultado y el desempate viven en `busqueda`, que es la
# interfaz común del proyecto (enunciado §3.3): así el baseline léxico, el
# buscador denso y el motor vectorial devuelven exactamente la misma forma.
# Se reexportan para no romper los `from aurum.lexico import ...` existentes.
from .busqueda import DEFAULT_TOP_K, SearchResult, build_results, stable_top_k_indices
from .datos import tokenize

__all__ = [
    "DEFAULT_TOP_K",
    "Bm25Retriever",
    "LexicalRetriever",
    "SearchResult",
    "TfidfRetriever",
    "rank_queries",
    "stable_top_k_indices",
]


class LexicalRetriever(Protocol):
    """Interfaz común de los baselines léxicos.

    Los scores léxicos son siempre similitudes (mayor = mejor), así que sus
    `SearchResult` se construyen con `score_es_similitud=True` por defecto."""

    name: str

    def search(self, query_text: str, *, k: int = DEFAULT_TOP_K) -> tuple[SearchResult, ...]:
        """Devuelve los `k` documentos mejor puntuados para la consulta."""
        ...


def _validate_corpus(
    document_texts: Sequence[str], document_ids: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    texts = tuple("" if pd.isna(text) else str(text) for text in document_texts)
    ids = tuple(str(document_id) for document_id in document_ids)
    if not texts:
        raise ValueError("El corpus debe contener al menos un documento.")
    if len(ids) != len(texts):
        raise ValueError(
            f"document_ids y textos deben tener la misma longitud: {len(ids)} != {len(texts)}."
        )
    if len(set(ids)) != len(ids):
        raise ValueError("document_ids contiene identificadores duplicados.")
    return texts, ids


class TfidfRetriever:
    """Baseline disperso: TF-IDF con vectores normalizados L2 y coseno."""

    name = "tfidf"

    def __init__(
        self,
        document_texts: Sequence[str],
        document_ids: Sequence[str],
        *,
        strip_accents: bool = True,
        sublinear_tf: bool = False,
        min_df: int = 1,
        ngram_range: tuple[int, int] = (1, 1),
    ) -> None:
        texts, self._document_ids = _validate_corpus(document_texts, document_ids)
        self._strip_accents = strip_accents
        # `analyzer` propio en vez de la tokenización de sklearn: así TF-IDF y
        # BM25 ven exactamente los mismos términos, y los mismos que midió la
        # evidencia de cobertura de NB01.
        self._vectorizer = TfidfVectorizer(
            analyzer=lambda text: tokenize(text, strip_accents=strip_accents),
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            ngram_range=ngram_range,
            norm="l2",
            dtype=np.float32,
        )
        self._matrix = self._vectorizer.fit_transform(texts)

    @property
    def document_ids(self) -> tuple[str, ...]:
        """IDs indexados, en el mismo orden que la matriz."""
        return self._document_ids

    @property
    def vocabulary_size(self) -> int:
        """Número de términos distintos aprendidos del corpus."""
        return len(self._vectorizer.vocabulary_)

    def search(self, query_text: str, *, k: int = DEFAULT_TOP_K) -> tuple[SearchResult, ...]:
        """Ordena por similitud coseno dispersa."""
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text debe ser un string no vacío.")
        query_vector = self._vectorizer.transform([query_text])
        scores = np.asarray((query_vector @ self._matrix.T).todense()).ravel()
        return build_results(stable_top_k_indices(scores, k=k), scores, self._document_ids)


class Bm25Retriever:
    """Baseline disperso: BM25 Okapi.

    Frente a TF-IDF aporta dos cosas: satura la frecuencia de término (`k1`,
    una palabra repetida 20 veces no vale 20 veces más) y normaliza la
    longitud del documento de forma explícita y ajustable (`b`), comparando
    cada ficha con la longitud media del corpus.

    ⚠️ Detalle de `rank_bm25` que conviene conocer al leer los resultados: el
    IDF de Okapi se vuelve **negativo** para términos presentes en más de la
    mitad del corpus, y la librería los sustituye por `epsilon · IDF medio`.
    En el catálogo completo eso afecta a palabras como `en` (9.644 de 15.000
    fichas): no puntúan en negativo, pero su peso queda fijado a un suelo
    arbitrario en lugar de calculado."""

    name = "bm25"

    def __init__(
        self,
        document_texts: Sequence[str],
        document_ids: Sequence[str],
        *,
        strip_accents: bool = True,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        texts, self._document_ids = _validate_corpus(document_texts, document_ids)
        self._strip_accents = strip_accents
        self.k1 = k1
        self.b = b
        corpus = [tokenize(text, strip_accents=strip_accents) for text in texts]
        self._index = BM25Okapi(corpus, k1=k1, b=b)

    @property
    def document_ids(self) -> tuple[str, ...]:
        """IDs indexados, en el mismo orden que el corpus tokenizado."""
        return self._document_ids

    @property
    def vocabulary_size(self) -> int:
        """Número de términos distintos aprendidos del corpus."""
        return len(self._index.idf)

    def search(self, query_text: str, *, k: int = DEFAULT_TOP_K) -> tuple[SearchResult, ...]:
        """Ordena por puntuación BM25."""
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text debe ser un string no vacío.")
        query_tokens = tokenize(query_text, strip_accents=self._strip_accents)
        scores = np.asarray(self._index.get_scores(query_tokens), dtype=np.float64)
        return build_results(stable_top_k_indices(scores, k=k), scores, self._document_ids)


def rank_queries(
    retriever: LexicalRetriever,
    consultas: pd.DataFrame,
    *,
    id_col: str = "query_id",
    text_col: str = "query_text",
    k: int = DEFAULT_TOP_K,
) -> dict[str, list[str]]:
    """Ejecuta todas las consultas y devuelve `{id: [document_id, ...]}`.

    Es la forma que espera `evaluacion.evaluate_rankings`, y guardar los IDs
    (no solo la métrica) es lo que permite atribuir errores en NB09."""
    return {
        str(getattr(row, id_col)): [
            result.document_id
            for result in retriever.search(str(getattr(row, text_col)), k=k)
        ]
        for row in consultas.itertuples(index=False)
    }
