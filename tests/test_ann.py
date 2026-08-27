"""Pruebas de NB06: fidelidad ANN, latencia y selección bajo restricción (src/aurum/ann.py)."""
import time

import pandas as pd
import pytest

from aurum.ann import (
    aplicar_restriccion,
    ann_recall_at_k,
    ann_recall_per_query,
    barrido_ef,
    comparar_ndcg_con_oraculo,
    comparar_ndcg_por_consulta,
    measure_search_latency,
    resumen_recall,
    tabla_recall_por_consulta,
)
from aurum.busqueda import SearchResult

# ─────────────────────── fidelidad del índice (recall ANN) ───────────────────

ORACULO = {
    "q1": ["a", "b", "c", "d", "e"],
    "q2": ["f", "g", "h", "i", "j"],
}


def test_recall_perfecto_cuando_ann_devuelve_lo_mismo():
    ann = {"q1": ["a", "b", "c", "d", "e"], "q2": ["f", "g", "h", "i", "j"]}

    recalls = ann_recall_per_query(ORACULO, ann, k=5)

    assert recalls == {"q1": 1.0, "q2": 1.0}
    assert ann_recall_at_k(ORACULO, ann, k=5) == pytest.approx(1.0)


def test_recall_cuenta_la_interseccion_no_el_orden():
    """El recall ANN no juzga posiciones, solo pertenencia al top-k."""
    ann = {"q1": ["e", "d", "c", "b", "a"], "q2": ["x", "y", "h", "i", "j"]}

    recalls = ann_recall_per_query(ORACULO, ann, k=5)

    assert recalls["q1"] == pytest.approx(1.0)      # mismos 5, otro orden
    assert recalls["q2"] == pytest.approx(3 / 5)     # perdió f y g


def test_recall_respeta_k_aunque_el_ranking_traiga_mas():
    ann = {"q1": ["a", "b", "z", "z2", "z3"], "q2": ["f", "g", "h", "i", "j"]}

    recalls = ann_recall_per_query(ORACULO, ann, k=2)

    assert recalls["q1"] == pytest.approx(1.0)   # top-2 exacto: a, b -> los dos
    assert recalls["q2"] == pytest.approx(1.0)


def test_recall_exige_ranking_ann_para_toda_consulta_del_oraculo():
    with pytest.raises(ValueError, match="q2"):
        ann_recall_per_query(ORACULO, {"q1": ["a", "b", "c", "d", "e"]}, k=5)


def test_recall_rechaza_k_no_entero_positivo():
    with pytest.raises(ValueError, match="k debe ser"):
        ann_recall_per_query(ORACULO, {"q1": [], "q2": []}, k=0)


def test_resumen_recall_da_media_minimo_y_p5():
    resumen = resumen_recall({"q1": 1.0, "q2": 0.6, "q3": 0.8}, k=10)

    assert resumen["recall_ann_at_10"] == pytest.approx((1.0 + 0.6 + 0.8) / 3, abs=1e-4)
    assert resumen["recall_ann_at_10_min"] == pytest.approx(0.6)
    # p5 de 3 muestras cae muy cerca del mínimo con interpolación lineal.
    assert resumen["recall_ann_at_10_p5"] <= resumen["recall_ann_at_10_min"] + 1e-9 or (
        resumen["recall_ann_at_10_p5"] < resumen["recall_ann_at_10"]
    )


def test_resumen_recall_distingue_perdida_repartida_de_concentrada():
    """Misma media, historias distintas: es justo lo que el mínimo debe cazar."""
    repartida = resumen_recall({"q1": 0.9, "q2": 0.9, "q3": 0.9, "q4": 0.9}, k=10)
    concentrada = resumen_recall({"q1": 1.0, "q2": 1.0, "q3": 1.0, "q4": 0.6}, k=10)

    assert repartida["recall_ann_at_10"] == pytest.approx(concentrada["recall_ann_at_10"])
    assert repartida["recall_ann_at_10_min"] > concentrada["recall_ann_at_10_min"]


# ──────────────────────────── latencia de consulta ───────────────────────────


class BuscadorFalso:
    """Un buscador en memoria: cuenta llamadas y devuelve resultados fijos."""

    def __init__(self, ef=None, resultados=None):
        self.ef = ef
        self.llamadas = []
        self._resultados = resultados or (
            SearchResult(rank=1, document_id="a", score=0.9),
            SearchResult(rank=2, document_id="b", score=0.8),
        )

    def buscar(self, consulta, *, top_k=10, marca=None):
        self.llamadas.append(consulta)
        time.sleep(0.001)   # evita que ms_p50 redondee a 0.0 en una máquina rápida
        return self._resultados[:top_k]


def test_la_latencia_cuenta_calentamiento_y_repeticiones_por_separado():
    buscador = BuscadorFalso(ef=64)

    resultado = measure_search_latency(
        buscador, ["q1", "q2"], repeticiones=5, calentamiento=3
    )

    assert len(buscador.llamadas) == 5 + 3
    assert resultado["n_llamadas"] == 5
    assert resultado["ef"] == 64


def test_la_latencia_expone_percentiles_ordenados_y_qps():
    buscador = BuscadorFalso()

    resultado = measure_search_latency(buscador, ["q1"], repeticiones=10, calentamiento=0)

    assert resultado["ms_min"] <= resultado["ms_p50"] <= resultado["ms_p95"] <= resultado["ms_max"]
    # `rel=0.01`: `qps_estimado` se calcula del p50 sin redondear, `ms_p50` del
    # dict ya viene redondeado a 2 decimales -la comparación tolera esa vuelta.
    assert resultado["qps_estimado"] == pytest.approx(1000.0 / resultado["ms_p50"], rel=0.01)


def test_la_latencia_recorre_las_consultas_en_ciclo():
    buscador = BuscadorFalso()

    measure_search_latency(buscador, ["q1", "q2", "q3"], repeticiones=7, calentamiento=0)

    assert buscador.llamadas == ["q1", "q2", "q3", "q1", "q2", "q3", "q1"]


def test_la_latencia_rechaza_consultas_vacias():
    with pytest.raises(ValueError, match="consultas"):
        measure_search_latency(BuscadorFalso(), [])


# ──────────────────────── el barrido y la restricción D16 ────────────────────


def _buscador_con_recall(ef, recall_q1, recall_q2):
    """Un `BuscadorFalso` cuyo top-2 reproduce el recall pedido contra ORACULO."""
    top_q1 = ORACULO["q1"][:2] if recall_q1 == 1.0 else ["z1", "z2"]
    top_q2 = ORACULO["q2"][:2] if recall_q2 == 1.0 else ["z3", "z4"]
    resultados_por_consulta = {
        "q1": tuple(SearchResult(rank=i + 1, document_id=d, score=1.0) for i, d in enumerate(top_q1)),
        "q2": tuple(SearchResult(rank=i + 1, document_id=d, score=1.0) for i, d in enumerate(top_q2)),
    }

    class _Buscador(BuscadorFalso):
        def buscar(self, consulta, *, top_k=10, marca=None):
            self.llamadas.append(consulta)
            return resultados_por_consulta[consulta][:top_k]

    return _Buscador(ef=ef)


def test_barrido_ef_devuelve_una_fila_por_configuracion():
    consultas = {"q1": "q1", "q2": "q2"}

    tabla = barrido_ef(
        lambda ef: _buscador_con_recall(ef, 1.0, 1.0),
        consultas, ORACULO, valores_ef=[16, 32], top_k=2,
        repeticiones=3, calentamiento=0,
    )

    assert list(tabla["ef"]) == [16, 32]
    assert (tabla["recall_ann_at_2"] == 1.0).all()
    assert {"ms_p50", "ms_p95", "qps_estimado"}.issubset(tabla.columns)


def test_barrido_ef_rechaza_lista_vacia_de_valores():
    with pytest.raises(ValueError, match="valores_ef"):
        barrido_ef(lambda ef: BuscadorFalso(ef=ef), {"q1": "q1"}, {"q1": ["a"]}, valores_ef=[])


def test_aplicar_restriccion_descarta_lo_que_no_cumple_y_elige_lo_mas_barato():
    tabla = pd.DataFrame([
        {"ef": 16, "recall_ann_at_10": 0.85, "ms_p95": 5.0},   # no llega al recall
        {"ef": 32, "recall_ann_at_10": 0.95, "ms_p95": 8.0},   # cumple, más lento
        {"ef": 64, "recall_ann_at_10": 0.97, "ms_p95": 6.0},   # cumple, más barato
        {"ef": 128, "recall_ann_at_10": 0.99, "ms_p95": 25.0},  # recall sobra, p95 se pasa
    ])

    anotada = aplicar_restriccion(
        tabla, recall_minimo=0.90, p95_maximo_ms=20.0, columna_recall="recall_ann_at_10"
    )

    assert list(anotada["cumple_d16"]) == [False, True, True, False]
    assert anotada.loc[anotada["ef"] == 64, "elegido_r04"].item()
    assert anotada["elegido_r04"].sum() == 1


def test_aplicar_restriccion_sin_candidatas_no_elige_ninguna():
    tabla = pd.DataFrame([{"ef": 16, "recall_ann_at_10": 0.5, "ms_p95": 100.0}])

    anotada = aplicar_restriccion(
        tabla, recall_minimo=0.9, p95_maximo_ms=10.0, columna_recall="recall_ann_at_10"
    )

    assert not anotada["cumple_d16"].any()
    assert not anotada["elegido_r04"].any()


# ────────────── enseñar lo que sale: recall por consulta y nDCG ──────────────


def test_tabla_recall_por_consulta_lista_los_perdidos():
    ann = {"q1": ["a", "b", "z", "z2", "z3"], "q2": ["f", "g", "h", "i", "j"]}

    tabla = tabla_recall_por_consulta(
        ORACULO, ann, consultas={"q1": "taladro", "q2": "lentejas"}, k=5
    )

    fila_q1 = tabla[tabla["query_id"] == "q1"].iloc[0]
    assert fila_q1["consulta"] == "taladro"
    assert fila_q1["vecinos_recuperados"] == "2 de 5"
    assert fila_q1["perdidos"] == "c, d, e"

    fila_q2 = tabla[tabla["query_id"] == "q2"].iloc[0]
    assert fila_q2["perdidos"] == "—"


def test_comparar_ndcg_muestra_oraculo_y_ann_por_separado():
    qrels = {"q1": {"a": 3.0, "b": 2.0, "c": 0.0}}
    oraculo_ranking = {"q1": ["a", "b", "c"]}
    ann_peor = {"q1": ["c", "b", "a"]}   # mismo conjunto, peor orden -> nDCG baja

    tabla = comparar_ndcg_con_oraculo(ann_peor, oraculo_ranking, qrels, k=3)

    fila_oraculo = tabla[tabla["sistema"].str.contains("oráculo")].iloc[0]
    fila_ann = tabla[tabla["sistema"].str.contains("ANN")].iloc[0]
    assert fila_oraculo["ndcg_at_3"] > fila_ann["ndcg_at_3"]


def test_ndcg_por_consulta_aisla_donde_se_concentra_la_perdida():
    """El agregado puede esconder que la pérdida sea toda de una consulta."""
    qrels = {
        "q1": {"a": 3.0, "b": 2.0, "c": 0.0},
        "q2": {"f": 3.0, "g": 2.0},
    }
    oraculo_ranking = {"q1": ["a", "b", "c"], "q2": ["f", "g"]}
    ann_ranking = {"q1": ["c", "b", "a"], "q2": ["f", "g"]}   # q1 empeora, q2 intacta

    tabla = comparar_ndcg_por_consulta(
        ann_ranking, oraculo_ranking, qrels, consultas={"q1": "taladro", "q2": "lentejas"}, k=3
    )

    fila_q1 = tabla[tabla["query_id"] == "q1"].iloc[0]
    fila_q2 = tabla[tabla["query_id"] == "q2"].iloc[0]
    assert fila_q1["consulta"] == "taladro"
    assert fila_q1["delta"] < 0
    assert fila_q2["delta"] == pytest.approx(0.0)
    # Ordenada por delta ascendente: la que más pierde, arriba.
    assert tabla.iloc[0]["query_id"] == "q1"
