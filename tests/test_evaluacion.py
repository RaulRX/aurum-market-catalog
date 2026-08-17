"""Pruebas de las métricas de recuperación (src/aurum/evaluacion.py)."""
import pandas as pd
import pytest

from aurum.evaluacion import per_query_delta  # noqa: F401
from aurum.evaluacion import (
    ESCI_RELEVANCE,
    apply_tolerance_rule,
    evaluate_rankings,
    formulation_consistency,
    jaccard_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    qrels_from_judgements,
    recall_at_k,
)

# E=3 y S=2 son relevantes (D01); C=1 e I=0 no lo son.
QRELS = {"exact": 3.0, "substitute": 2.0, "complement": 1.0, "irrelevant": 0.0}


def test_esci_relevance_matches_the_manifest_contract():
    assert ESCI_RELEVANCE == {"E": 3.0, "S": 2.0, "C": 1.0, "I": 0.0}


def test_qrels_from_judgements_translates_labels_not_the_relevance_column():
    relevancias = pd.DataFrame(
        {
            "query_id": [1, 1, 2],
            "product_id": ["p1", "p2", "p3"],
            "esci_label": ["E", "I", "S"],
            "relevance": [99, 99, 99],  # columna incoherente a propósito
        }
    )

    qrels = qrels_from_judgements(relevancias)

    assert qrels["1"] == {"p1": 3.0, "p2": 0.0}
    assert qrels["2"] == {"p3": 2.0}


def test_qrels_from_judgements_rejects_unknown_labels():
    relevancias = pd.DataFrame(
        {"query_id": [1], "product_id": ["p1"], "esci_label": ["X"]}
    )

    with pytest.raises(ValueError, match="Etiquetas ESCI desconocidas"):
        qrels_from_judgements(relevancias)


def test_precision_divides_by_k_not_by_the_number_returned():
    # Un solo resultado, relevante: 1 acierto entre los 10 pedidos.
    assert precision_at_k(["exact"], QRELS, k=10) == pytest.approx(0.1)


def test_recall_counts_exact_and_substitute_but_not_complement():
    # 2 relevantes juzgados (exact, substitute); el ranking recupera uno.
    assert recall_at_k(["exact", "complement"], QRELS, k=10) == pytest.approx(0.5)


def test_unjudged_documents_score_zero_without_shifting_positions():
    # D04: 'desconocido' no está juzgado → cuenta como irrelevante en la 1ª
    # posición, así que el primer relevante sigue estando en la 2ª.
    assert mrr_at_k(["desconocido", "exact"], QRELS, k=10) == pytest.approx(0.5)


def test_mrr_is_zero_when_no_relevant_document_appears():
    assert mrr_at_k(["complement", "irrelevant"], QRELS, k=10) == 0.0


def test_ndcg_is_one_for_the_ideal_ranking_and_zero_for_the_worst():
    ideal = ["exact", "substitute", "complement", "irrelevant"]

    assert ndcg_at_k(ideal, QRELS, k=4) == pytest.approx(1.0)
    assert ndcg_at_k(["irrelevant"], QRELS, k=1) == pytest.approx(0.0)


def test_ndcg_penalises_putting_the_best_document_lower():
    ideal = ndcg_at_k(["exact", "substitute"], QRELS, k=2)
    invertido = ndcg_at_k(["substitute", "exact"], QRELS, k=2)

    assert invertido < ideal


def test_ndcg_gain_mode_changes_the_value_but_not_the_ordering():
    peor = ["substitute", "exact"]

    assert ndcg_at_k(peor, QRELS, k=2, gain="linear") != ndcg_at_k(peor, QRELS, k=2)
    assert ndcg_at_k(peor, QRELS, k=2, gain="linear") < 1.0


def test_ndcg_rejects_an_unknown_gain_mode():
    with pytest.raises(ValueError, match="gain debe ser"):
        ndcg_at_k(["exact"], QRELS, k=1, gain="cuadratico")


def test_duplicated_documents_in_the_top_k_are_rejected():
    with pytest.raises(ValueError, match="duplicados"):
        recall_at_k(["exact", "exact"], QRELS, k=10)


def test_evaluate_rankings_treats_a_missing_ranking_as_empty():
    qrels = {"q1": QRELS, "q2": QRELS}

    report = evaluate_rankings({"q1": ["exact"]}, qrels, k=10)
    por_consulta = {m.query_id: m for m in report.per_query}

    assert por_consulta["q2"].recall == 0.0
    assert por_consulta["q2"].ndcg == 0.0
    assert report.mean_recall == pytest.approx(por_consulta["q1"].recall / 2)


def test_evaluate_rankings_rejects_rankings_without_judgements():
    with pytest.raises(ValueError, match="rankings sin qrels"):
        evaluate_rankings({"fantasma": ["exact"]}, {"q1": QRELS}, k=10)


def test_evaluate_rankings_summary_uses_the_readme_datos_labels():
    report = evaluate_rankings({"q1": ["exact"]}, {"q1": QRELS}, k=10)

    assert set(report.summary) == {
        "precision_at_10",
        "recall_at_10",
        "mrr_at_10",
        "ndcg_at_10",
    }
    assert len(report.per_query_frame()) == 1


def test_jaccard_measures_overlap_between_two_top_k_lists():
    assert jaccard_at_k(["a", "b"], ["a", "b"], k=2) == pytest.approx(1.0)
    assert jaccard_at_k(["a", "b"], ["c", "d"], k=2) == pytest.approx(0.0)
    # 1 en común de 3 distintos en total.
    assert jaccard_at_k(["a", "b"], ["b", "c"], k=2) == pytest.approx(1 / 3)


def test_formulation_consistency_pairs_every_formulation_of_each_intent():
    rankings = {
        "EVAL-100455-direct": ["a", "b"],
        "EVAL-100455-context": ["a", "b"],
        "EVAL-100455-semantic": ["c", "d"],
    }

    consistencia = formulation_consistency(rankings, k=2).set_index("intencion")

    assert consistencia.loc["100455", "jaccard_context_direct"] == pytest.approx(1.0)
    assert consistencia.loc["100455", "jaccard_context_semantic"] == pytest.approx(0.0)


def test_formulation_consistency_rejects_ids_without_the_expected_shape():
    with pytest.raises(ValueError, match="EVAL-intencion-formulacion"):
        formulation_consistency({"consulta_suelta": ["a"]}, k=2)


# ─────────────────── D09b: la regla de desempate, ejecutada ──────────────────

# Tabla de juguete con la forma de la comparativa de NB02. La mejor métrica es
# 0.700 (dim 1024); con tolerancia 0.02 el suelo de admisión queda en 0.680.
CANDIDATOS = pd.DataFrame(
    [
        {"modelo": "grande", "dim": 1024, "ndcg_at_10": 0.700, "segundos": 900.0},
        {"modelo": "grande", "dim": 256, "ndcg_at_10": 0.685, "segundos": 900.0},
        {"modelo": "medio", "dim": 256, "ndcg_at_10": 0.690, "segundos": 300.0},
        {"modelo": "medio", "dim": 128, "ndcg_at_10": 0.640, "segundos": 300.0},
    ]
)


def test_apply_tolerance_rule_prefers_the_cheapest_admissible_configuration():
    """Dentro de la tolerancia gana la de menor dimensión, no la de mayor nDCG."""
    resultado = apply_tolerance_rule(CANDIDATOS, tolerancia=0.02)

    ganadora = resultado.iloc[0]
    assert (ganadora["modelo"], ganadora["dim"]) == ("medio", 256)
    assert ganadora["posicion_regla"] == 1


def test_apply_tolerance_rule_marks_what_falls_outside_the_tolerance():
    resultado = apply_tolerance_rule(CANDIDATOS, tolerancia=0.02).set_index(
        ["modelo", "dim"]
    )

    # 0.640 queda a más de 0.02 del mejor (0.700): fuera.
    assert bool(resultado.loc[("medio", 128), "admisible"]) is False
    assert bool(resultado.loc[("grande", 256), "admisible"]) is True


def test_apply_tolerance_rule_with_zero_tolerance_picks_the_best_metric():
    """Sin tolerancia la regla degenera en 'gana la mejor métrica'."""
    resultado = apply_tolerance_rule(CANDIDATOS, tolerancia=0.0)

    assert (resultado.iloc[0]["modelo"], resultado.iloc[0]["dim"]) == ("grande", 1024)


def test_apply_tolerance_rule_breaks_dimension_ties_by_metric_then_cost():
    """A igualdad de dimensión decide la métrica; solo después, el tiempo."""
    empate = pd.DataFrame(
        [
            {"modelo": "a", "dim": 256, "ndcg_at_10": 0.690, "segundos": 900.0},
            {"modelo": "b", "dim": 256, "ndcg_at_10": 0.690, "segundos": 100.0},
            {"modelo": "c", "dim": 256, "ndcg_at_10": 0.695, "segundos": 999.0},
        ]
    )

    resultado = apply_tolerance_rule(empate, tolerancia=0.02)

    assert list(resultado["modelo"]) == ["c", "b", "a"]


def test_apply_tolerance_rule_rejects_a_table_without_the_needed_columns():
    with pytest.raises(ValueError, match="Faltan columnas"):
        apply_tolerance_rule(pd.DataFrame([{"modelo": "a", "dim": 256}]))


def test_apply_tolerance_rule_rejects_an_empty_table():
    with pytest.raises(ValueError, match="vacía"):
        apply_tolerance_rule(
            pd.DataFrame(columns=["ndcg_at_10", "dim", "segundos"])
        )


# ───────────────────────────── per_query_delta ───────────────────────────────


def _tabla_larga():
    return pd.DataFrame([
        {"query_id": "1", "plantilla": "A3", "ndcg@10": 0.60},
        {"query_id": "1", "plantilla": "A3n", "ndcg@10": 0.90},
        {"query_id": "2", "plantilla": "A3", "ndcg@10": 0.70},
        {"query_id": "2", "plantilla": "A3n", "ndcg@10": 0.70},
    ])


def test_la_delta_por_consulta_revela_donde_esta_el_efecto():
    """El agregado diria +0,15 de media; la tabla dice que todo viene de una."""
    tabla = per_query_delta(_tabla_larga(), sistema_a="A3n", sistema_b="A3")

    assert list(tabla["query_id"]) == ["1", "2"]      # ordenada por delta desc
    assert list(tabla["delta"]) == [0.30, 0.00]


def test_avisa_si_el_sistema_no_esta_en_la_tabla():
    with pytest.raises(ValueError, match="no está en la columna"):
        per_query_delta(_tabla_larga(), sistema_a="A9", sistema_b="A3")


def test_avisa_de_las_columnas_que_faltan_en_la_tabla_larga():
    with pytest.raises(ValueError, match="faltan columnas"):
        per_query_delta(pd.DataFrame([{"query_id": "1"}]), sistema_a="A3n", sistema_b="A3")
