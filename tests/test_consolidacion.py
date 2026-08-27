"""Pruebas de NB09: la tabla comparativa (src/aurum/consolidacion.py)."""
import pandas as pd
import pytest

from aurum.consolidacion import (
    COLUMNAS_TABLA_COMPARATIVA,
    diagnosticar_consulta,
    fila_comparativa,
    tabla_comparativa,
)


def test_fila_comparativa_rellena_lo_no_dado_con_none():
    fila = fila_comparativa("C0 baseline", modelo="TF-IDF", ndcg_at_10=0.41)

    assert fila["config"] == "C0 baseline"
    assert fila["modelo"] == "TF-IDF"
    assert fila["ndcg_at_10"] == 0.41
    assert fila["p50_ms"] is None
    assert fila["ann"] is None


def test_fila_comparativa_rechaza_un_campo_desconocido():
    with pytest.raises(ValueError, match="latencia_media"):
        fila_comparativa("C0", latencia_media=12.0)


def test_tabla_comparativa_usa_siempre_las_mismas_columnas():
    filas = [
        fila_comparativa("C0", modelo="TF-IDF", ndcg_at_10=0.41),
        fila_comparativa("C1", modelo="gemini-2", dim=768, ndcg_at_10=0.55, p50_ms=8.5),
    ]

    tabla = tabla_comparativa(filas)

    assert list(tabla.columns) == list(COLUMNAS_TABLA_COMPARATIVA)
    assert len(tabla) == 2
    assert pd.isna(tabla.loc[0, "p50_ms"])
    assert tabla.loc[1, "p50_ms"] == 8.5


def test_tabla_comparativa_falla_sin_filas():
    with pytest.raises(ValueError, match="No hay filas"):
        tabla_comparativa([])


# ────────────────────────────── diagnosticar_consulta ─────────────────────────


def test_diagnostico_senala_representacion_cuando_el_oraculo_ya_falla():
    """Ni el oraculo exacto encuentra relevantes: no tiene sentido mirar el
    ANN, el problema esta antes -en el modelo o la plantilla-."""
    qrels = {"a": 0.0, "b": 0.0, "c": 1.0}  # ninguno E/S (>= 2.0, D01)

    diagnostico = diagnosticar_consulta(
        "q1", ranking_oraculo=["a", "b", "c"], ranking_ann=["a", "b", "c"], qrels=qrels,
    )

    assert diagnostico["n_relevantes_en_oraculo"] == 0
    assert diagnostico["perdidos_por_el_ann"] == []


def test_diagnostico_senala_indice_cuando_el_ann_pierde_relevantes_del_oraculo():
    """El oraculo si encuentra relevantes, pero el ANN no los trae: es la
    evidencia de un fallo de indice, no de representacion."""
    qrels = {"rel1": 3.0, "rel2": 2.0, "irrelevante": 0.0}

    diagnostico = diagnosticar_consulta(
        "q1",
        ranking_oraculo=["rel1", "rel2", "irrelevante"],
        ranking_ann=["irrelevante", "otro", "otro2"],
        qrels=qrels,
    )

    assert diagnostico["n_relevantes_en_oraculo"] == 2
    assert diagnostico["perdidos_por_el_ann"] == ["rel1", "rel2"]
    assert diagnostico["n_relevantes_en_ann"] == 0


def test_diagnostico_no_cuenta_complement_como_relevante():
    """D01: relevante para recall/MRR es E o S (score >= 2.0); C (1.0) no
    cuenta, aunque tenga ganancia en nDCG."""
    qrels = {"solo_complement": 1.0}

    diagnostico = diagnosticar_consulta(
        "q1", ranking_oraculo=["solo_complement"], ranking_ann=["solo_complement"], qrels=qrels,
    )

    assert diagnostico["n_relevantes_en_oraculo"] == 0


def test_diagnostico_respeta_k():
    qrels = {"rel": 3.0}

    diagnostico = diagnosticar_consulta(
        "q1", ranking_oraculo=["otro1", "otro2", "rel"], ranking_ann=[], qrels=qrels, k=2,
    )

    assert "rel" not in diagnostico["top_oraculo"]
    assert diagnostico["n_relevantes_en_oraculo"] == 0
