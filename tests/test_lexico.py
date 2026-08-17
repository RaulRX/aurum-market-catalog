"""Pruebas de los baselines léxicos (src/aurum/lexico.py)."""
import numpy as np
import pandas as pd
import pytest

from aurum.lexico import Bm25Retriever, TfidfRetriever, rank_queries, stable_top_k_indices

TEXTOS = [
    "taladro inalámbrico con batería de 24 voltios",
    "taladro con cable de 500 vatios",
    "pantalla táctil para portátil convertible",
    "lentejas sin gluten en conserva",
]
IDS = ["p1", "p2", "p3", "p4"]


@pytest.fixture(params=[TfidfRetriever, Bm25Retriever], ids=["tfidf", "bm25"])
def retriever_class(request):
    """Los dos baselines deben cumplir el mismo contrato de interfaz."""
    return request.param


def test_stable_top_k_indices_breaks_ties_by_original_order():
    scores = np.array([0.5, 0.9, 0.5, 0.9])

    assert stable_top_k_indices(scores, k=4).tolist() == [1, 3, 0, 2]


def test_stable_top_k_indices_returns_what_exists_when_k_exceeds_the_corpus():
    assert stable_top_k_indices(np.array([0.1, 0.2]), k=10).tolist() == [1, 0]


def test_stable_top_k_indices_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="NaN o infinito"):
        stable_top_k_indices(np.array([0.1, np.nan]), k=1)


def test_retriever_returns_k_results_ranked_from_one(retriever_class):
    retriever = retriever_class(TEXTOS, IDS)

    resultados = retriever.search("taladro batería", k=2)

    assert len(resultados) == 2
    assert [r.rank for r in resultados] == [1, 2]
    assert all(r.score_es_similitud for r in resultados)


def test_retriever_ranks_the_matching_document_first(retriever_class):
    retriever = retriever_class(TEXTOS, IDS)

    resultados = retriever.search("lentejas gluten", k=1)

    assert resultados[0].document_id == "p4"


def test_retriever_scores_are_monotonically_decreasing(retriever_class):
    retriever = retriever_class(TEXTOS, IDS)

    scores = [r.score for r in retriever.search("taladro", k=4)]

    assert scores == sorted(scores, reverse=True)


def test_retriever_never_returns_more_than_the_corpus(retriever_class):
    retriever = retriever_class(TEXTOS, IDS)

    assert len(retriever.search("taladro", k=100)) == len(TEXTOS)


def test_retriever_returns_unique_document_ids(retriever_class):
    retriever = retriever_class(TEXTOS, IDS)

    ids = [r.document_id for r in retriever.search("taladro", k=4)]

    assert len(set(ids)) == len(ids)


def test_retriever_rejects_an_empty_query(retriever_class):
    retriever = retriever_class(TEXTOS, IDS)

    with pytest.raises(ValueError, match="string no vacío"):
        retriever.search("   ", k=1)


def test_retriever_rejects_duplicated_document_ids(retriever_class):
    with pytest.raises(ValueError, match="duplicados"):
        retriever_class(TEXTOS, ["p1", "p1", "p3", "p4"])


def test_retriever_rejects_mismatched_ids_and_texts(retriever_class):
    with pytest.raises(ValueError, match="misma longitud"):
        retriever_class(TEXTOS, ["p1", "p2"])


def test_strip_accents_makes_the_unaccented_query_reach_the_accented_document(
    retriever_class,
):
    # D05.b: 'tactil' sin tilde debe alcanzar la ficha que escribe 'táctil'.
    con_normalizacion = retriever_class(TEXTOS, IDS, strip_accents=True)
    sin_normalizacion = retriever_class(TEXTOS, IDS, strip_accents=False)

    assert con_normalizacion.search("tactil", k=1)[0].score > 0
    assert sin_normalizacion.search("tactil", k=1)[0].score == 0


def test_bm25_length_normalisation_prefers_the_shorter_of_two_matching_documents():
    # Mismo término una vez en cada ficha: la corta gana porque 'b' penaliza
    # la larga. El corpus lleva relleno para que 'taladro' esté en minoría de
    # documentos: en BM25 Okapi, un término presente en más de la mitad del
    # corpus produce un IDF negativo que invierte el ranking.
    textos = [
        "taladro",
        "taladro " + " ".join(f"palabra{i}" for i in range(200)),
        "lentejas sin gluten",
        "pantalla táctil",
        "botines marrones",
    ]
    retriever = Bm25Retriever(textos, ["corta", "larga", "p3", "p4", "p5"])

    assert retriever.search("taladro", k=1)[0].document_id == "corta"


def test_rank_queries_returns_one_list_of_ids_per_query(retriever_class):
    retriever = retriever_class(TEXTOS, IDS)
    consultas = pd.DataFrame(
        {"query_id": [1, 2], "query_text": ["taladro batería", "lentejas gluten"]}
    )

    rankings = rank_queries(retriever, consultas, k=2)

    assert set(rankings) == {"1", "2"}
    assert rankings["2"][0] == "p4"
    assert all(len(ids) == 2 for ids in rankings.values())
