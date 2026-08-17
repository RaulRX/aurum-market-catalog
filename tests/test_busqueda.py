"""Pruebas del buscador denso exacto y del contrato común (src/aurum/busqueda.py)."""
import numpy as np
import pytest

from aurum.busqueda import (
    DenseRetriever,
    SearchResult,
    rank_queries_dense,
    results_frame,
    stable_top_k_indices,
)

# Tres documentos en un plano: A y B casi paralelos, C ortogonal a ambos. Las
# normas son deliberadamente distintas para que `dot` y `cosine` discrepen.
VECTORES = np.array(
    [
        [1.0, 0.0],  # doc-a: dirección de la consulta, norma 1
        [3.0, 0.3],  # doc-b: casi la misma dirección, norma ~3
        [0.0, 1.0],  # doc-c: ortogonal
    ],
    dtype=np.float32,
)
IDS = ["doc-a", "doc-b", "doc-c"]
CONSULTA = np.array([1.0, 0.0], dtype=np.float32)


def test_stable_top_k_desempata_por_orden_original():
    """Con scores idénticos gana el índice más bajo: dos ejecuciones coinciden."""
    assert stable_top_k_indices(np.array([0.5, 0.5, 0.5]), k=2).tolist() == [0, 1]


def test_stable_top_k_rechaza_scores_no_finitos():
    with pytest.raises(ValueError, match="NaN o infinito"):
        stable_top_k_indices(np.array([1.0, np.nan]), k=1)


def test_stable_top_k_devuelve_lo_que_hay_si_k_supera_el_corpus():
    assert stable_top_k_indices(np.array([0.1, 0.9]), k=10).tolist() == [1, 0]


def test_cosine_ignora_la_norma_y_dot_la_premia():
    """El caso que justifica declarar la métrica: sin normalizar cambian el orden."""
    coseno = DenseRetriever(VECTORES, IDS, metric="cosine").search_vector(CONSULTA, k=2)
    producto = DenseRetriever(VECTORES, IDS, metric="dot").search_vector(CONSULTA, k=2)

    # doc-a apunta exactamente a la consulta; doc-b se desvía pero es 3x más largo.
    assert [r.document_id for r in coseno] == ["doc-a", "doc-b"]
    assert [r.document_id for r in producto] == ["doc-b", "doc-a"]


def test_con_vectores_normalizados_las_tres_metricas_dan_el_mismo_ranking():
    """La verificación de normalización de NB02, en un test en vez de a ojo."""
    normalizados = VECTORES / np.linalg.norm(VECTORES, axis=1, keepdims=True)

    rankings = {
        metric: [
            r.document_id
            for r in DenseRetriever(normalizados, IDS, metric=metric).search_vector(
                CONSULTA, k=3
            )
        ]
        for metric in ("cosine", "dot", "l2")
    }

    assert rankings["cosine"] == rankings["dot"] == rankings["l2"]


def test_l2_declara_que_su_score_es_una_distancia():
    """Regla 5: una distancia y una similitud no van juntas en una tabla."""
    retriever = DenseRetriever(VECTORES, IDS, metric="l2")
    resultados = retriever.search_vector(CONSULTA, k=3)

    assert retriever.score_es_similitud is False
    assert all(not r.score_es_similitud for r in resultados)
    # Ordena de menor a mayor distancia y el primero es el vector idéntico.
    assert resultados[0].document_id == "doc-a"
    assert resultados[0].score == pytest.approx(0.0)
    assert [r.score for r in resultados] == sorted(r.score for r in resultados)


def test_rechaza_una_consulta_con_otra_dimension():
    retriever = DenseRetriever(VECTORES, IDS)

    with pytest.raises(ValueError, match="dimensión"):
        retriever.search_vector(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)


def test_rechaza_ids_duplicados_y_desalineados():
    with pytest.raises(ValueError, match="duplicados"):
        DenseRetriever(VECTORES, ["x", "x", "y"])
    with pytest.raises(ValueError, match="alinearse"):
        DenseRetriever(VECTORES, ["x", "y"])


def test_rechaza_vectores_no_finitos():
    rotos = np.array([[1.0, 0.0], [np.nan, 1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="NaN o infinito"):
        DenseRetriever(rotos, ["x", "y"])


def test_rank_queries_dense_devuelve_la_forma_que_espera_evaluate_rankings():
    retriever = DenseRetriever(VECTORES, IDS, metric="cosine")
    consultas = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    rankings = rank_queries_dense(retriever, ["q1", "q2"], consultas, k=2)

    assert rankings == {"q1": ["doc-a", "doc-b"], "q2": ["doc-c", "doc-b"]}


def test_rank_queries_dense_detecta_ids_y_vectores_descuadrados():
    retriever = DenseRetriever(VECTORES, IDS)

    with pytest.raises(ValueError, match="vectores de consulta"):
        rank_queries_dense(retriever, ["q1", "q2"], np.array([[1.0, 0.0]]), k=1)


def test_results_frame_adjunta_los_metadatos_por_documento():
    """Es la forma normalizada que exige el enunciado §3.3 y la base de los CSV."""
    resultados = {"q1": (SearchResult(rank=1, document_id="doc-a", score=0.9),)}

    frame = results_frame(resultados, metadata={"doc-a": {"title": "Taladro", "brand": "Einhell"}})

    assert frame.loc[0, "product_id"] == "doc-a"
    assert frame.loc[0, "rank"] == 1
    assert frame.loc[0, "brand"] == "Einhell"
