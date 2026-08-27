"""Tests de `aurum.graficas`.

Una figura no se puede "mirar" desde un test, pero sí se puede comprobar lo que
la haría mentir: que el número de series coincida con el de sistemas, que la
banda de tolerancia esté donde dice la regla, que el eje sea logarítmico y que
comparar métricas distintas falle en vez de dibujar un gráfico sin sentido.
"""
from __future__ import annotations

import pandas as pd
import pytest

from aurum.graficas import (
    apply_project_layout,
    plot_ann_pareto,
    plot_contract_delta,
    plot_duplicate_threshold_sweep,
    plot_effect_vs_exposure,
    plot_dimension_curve,
    plot_metric_comparison,
)

import plotly.graph_objects as go


METRICAS = {"precision_at_10": 0.4, "recall_at_10": 0.5, "mrr_at_10": 0.6, "ndcg_at_10": 0.7}


class InformeFalso:
    """Cualquier objeto con `.summary` sirve: es el contrato de duck typing."""

    def __init__(self, summary: dict[str, float]) -> None:
        self.summary = summary


@pytest.fixture
def barrido() -> pd.DataFrame:
    return pd.DataFrame([
        {"modelo": "jina-v3", "dim": 1024, "ndcg_at_10": 0.70, "bytes_por_vector": 4096},
        {"modelo": "jina-v3", "dim": 256, "ndcg_at_10": 0.69, "bytes_por_vector": 1024},
        {"modelo": "granite", "dim": 768, "ndcg_at_10": 0.61, "bytes_por_vector": 3072},
        {"modelo": "granite", "dim": 128, "ndcg_at_10": 0.55, "bytes_por_vector": 512},
    ])


# ─────────────────────────── apply_project_layout ────────────────────────────


def test_el_layout_exige_titulo_no_vacio():
    with pytest.raises(ValueError, match="title debe ser"):
        apply_project_layout(go.Figure(), title="   ")


def test_el_layout_rechaza_alturas_ilegibles():
    with pytest.raises(ValueError, match="height debe ser"):
        apply_project_layout(go.Figure(), title="Algo", height=120)


def test_el_subtitulo_se_incrusta_en_el_titulo():
    figura = apply_project_layout(go.Figure(), title="Calidad", subtitle="8 consultas")

    assert "<b>Calidad</b>" in figura.layout.title.text
    assert "8 consultas" in figura.layout.title.text


# ────────────────────────── plot_metric_comparison ───────────────────────────


def test_acepta_tanto_informes_como_diccionarios():
    """El motivo de existir de este módulo: la versión de sesion_01 solo
    aceptaba su propio EvaluationReport y trataba el resto como Mapping."""
    figura = plot_metric_comparison(
        {"bm25": METRICAS, "jina-v3": InformeFalso(METRICAS)}
    )

    assert len(figura.data) == 2
    assert {serie.name for serie in figura.data} == {"bm25", "jina-v3"}


def test_una_serie_por_sistema_con_sus_valores(barrido):
    figura = plot_metric_comparison({"bm25": METRICAS})

    assert list(figura.data[0].x) == list(METRICAS)
    assert list(figura.data[0].y) == list(METRICAS.values())


def test_no_compara_sistemas_medidos_con_metricas_distintas():
    with pytest.raises(ValueError, match="no trae las mismas métricas"):
        plot_metric_comparison({"bm25": METRICAS, "jina-v3": {"ndcg_at_10": 0.7}})


def test_rechaza_un_sistema_que_no_es_ni_informe_ni_mapping():
    with pytest.raises(TypeError, match="informe con `.summary`"):
        plot_metric_comparison({"bm25": [0.4, 0.5]})


def test_exige_al_menos_un_sistema():
    with pytest.raises(ValueError, match="no puede estar vacío"):
        plot_metric_comparison({})


# ─────────────────────────── plot_dimension_curve ────────────────────────────


def test_una_linea_por_modelo(barrido):
    figura = plot_dimension_curve(barrido)

    assert {serie.name for serie in figura.data} == {"jina-v3", "granite"}


def test_los_puntos_se_ordenan_por_dimension(barrido):
    """Sin ordenar, `px.line` une los puntos en el orden del DataFrame y dibuja
    un zigzag que sugiere una curva inexistente."""
    figura = plot_dimension_curve(barrido)
    jina = next(serie for serie in figura.data if serie.name == "jina-v3")

    assert list(jina.x) == [256, 1024]


def test_la_banda_cubre_la_tolerancia_bajo_el_mejor_valor(barrido):
    figura = plot_dimension_curve(barrido, tolerance=0.02)
    banda = next(forma for forma in figura.layout.shapes if forma.type == "rect")

    assert banda.y1 == pytest.approx(0.70)   # B = mejor nDCG de toda la tabla
    assert banda.y0 == pytest.approx(0.68)   # B - tau


def test_sin_tolerancia_no_se_dibuja_banda(barrido):
    figura = plot_dimension_curve(barrido, tolerance=0.0)

    assert not [forma for forma in figura.layout.shapes if forma.type == "rect"]


def test_el_eje_de_dimension_es_logaritmico_y_solo_marca_lo_medido(barrido):
    figura = plot_dimension_curve(barrido)

    assert figura.layout.xaxis.type == "log"
    assert list(figura.layout.xaxis.tickvals) == [128, 256, 768, 1024]


def test_el_trazo_separa_la_segunda_variable_sin_gastar_color(barrido):
    """Color por modelo y trazo por contrato: dos ejes cruzados en un gráfico
    sin convertirlos en cinco colores sueltos."""
    con_contrato = barrido.assign(contrato="nativo")
    sin_contrato = barrido.assign(contrato="sin_contrato", ndcg_at_10=0.66)
    ambos = pd.concat([con_contrato, sin_contrato])

    figura = plot_dimension_curve(ambos, dash_column="contrato")

    assert len(figura.data) == 4   # 2 modelos x 2 contratos
    colores = {serie.line.color for serie in figura.data}
    trazos = {serie.line.dash for serie in figura.data}
    assert len(colores) == 2       # un color por modelo, no por serie
    assert len(trazos) == 2        # el contrato viaja en el trazo


def test_el_trazo_tambien_exige_que_la_columna_exista(barrido):
    with pytest.raises(ValueError, match="faltan columnas"):
        plot_dimension_curve(barrido, dash_column="contrato")


def test_avisa_de_las_columnas_que_faltan():
    with pytest.raises(ValueError, match="faltan columnas"):
        plot_dimension_curve(pd.DataFrame([{"modelo": "jina-v3", "dim": 256}]))


def test_no_dibuja_un_barrido_vacio():
    vacio = pd.DataFrame(columns=["modelo", "dim", "ndcg_at_10"])

    with pytest.raises(ValueError, match="no tiene filas"):
        plot_dimension_curve(vacio)


# ──────────────────────────── plot_contract_delta ────────────────────────────


@pytest.fixture
def barrido_dos_ramas() -> pd.DataFrame:
    """Reproduce el patrón real de gemini: la Δ cambia de signo al truncar."""
    return pd.DataFrame([
        {"modelo": "gemini-2", "contrato": "nativo", "dim": 768, "ndcg_at_10": 0.7478},
        {"modelo": "gemini-2", "contrato": "sin_contrato", "dim": 768, "ndcg_at_10": 0.7718},
        {"modelo": "gemini-2", "contrato": "nativo", "dim": 128, "ndcg_at_10": 0.6648},
        {"modelo": "gemini-2", "contrato": "sin_contrato", "dim": 128, "ndcg_at_10": 0.5575},
    ])


def test_la_delta_cambia_de_signo_con_la_dimension(barrido_dos_ramas):
    figura = plot_contract_delta(barrido_dos_ramas)
    serie = figura.data[0]

    assert list(serie.x) == [128, 768]
    assert serie.y[0] == pytest.approx(0.1073, abs=1e-4)    # a 128 el contrato ayuda
    assert serie.y[1] == pytest.approx(-0.0240, abs=1e-4)   # a 768 estorba


def test_la_banda_rodea_el_cero_y_no_el_mejor_valor(barrido_dos_ramas):
    figura = plot_contract_delta(barrido_dos_ramas, tolerance=0.02)
    banda = next(f for f in figura.layout.shapes if f.type == "rect")

    assert banda.y0 == pytest.approx(-0.02)
    assert banda.y1 == pytest.approx(0.02)


def test_descarta_los_modelos_con_una_sola_rama(barrido_dos_ramas):
    """granite solo tiene `nativo`: sin las dos no hay diferencia que calcular."""
    con_granite = pd.concat([
        barrido_dos_ramas,
        pd.DataFrame([
            {"modelo": "granite", "contrato": "nativo", "dim": 768, "ndcg_at_10": 0.61}
        ]),
    ])

    figura = plot_contract_delta(con_granite)

    assert {serie.name for serie in figura.data} == {"gemini-2"}


def test_exige_que_existan_las_dos_ramas(barrido_dos_ramas):
    solo_nativo = barrido_dos_ramas.query("contrato == 'nativo'")

    with pytest.raises(ValueError, match="no contiene las dos ramas"):
        plot_contract_delta(solo_nativo)


# ─────────────────────────── plot_effect_vs_exposure ─────────────────────────


@pytest.fixture
def exposicion() -> pd.DataFrame:
    """El patrón real del control de D02: el efecto no sigue a la exposición."""
    return pd.DataFrame([
        {"query_id": "61533", "pct_sin_color": 100.0, "delta": 0.0055},
        {"query_id": "28703", "pct_sin_color": 26.0, "delta": 0.0662},
        {"query_id": "31224", "pct_sin_color": 0.0, "delta": 0.0000},
    ])


def test_un_punto_por_consulta_con_su_etiqueta(exposicion):
    figura = plot_effect_vs_exposure(exposicion, exposure="pct_sin_color")

    assert len(figura.data) == 1
    assert list(figura.data[0].text) == ["61533", "28703", "31224"]
    assert len(figura.data[0].x) == 3


def test_la_banda_marca_la_zona_indistinguible(exposicion):
    figura = plot_effect_vs_exposure(exposicion, exposure="pct_sin_color", tolerance=0.01)
    banda = next(f for f in figura.layout.shapes if f.type == "rect")

    assert banda.y0 == pytest.approx(-0.01)
    assert banda.y1 == pytest.approx(0.01)


def test_avisa_si_falta_la_columna_de_exposicion(exposicion):
    with pytest.raises(ValueError, match="faltan columnas"):
        plot_effect_vs_exposure(exposicion, exposure="no_existe")


# ─────────────────────────────── plot_ann_pareto ──────────────────────────────


@pytest.fixture
def barrido_ann() -> pd.DataFrame:
    return pd.DataFrame([
        {"ef": 16, "recall_ann_at_10": 0.88, "ms_p95": 5.0, "elegido_r04": False},
        {"ef": 32, "recall_ann_at_10": 0.95, "ms_p95": 7.0, "elegido_r04": True},
        {"ef": 64, "recall_ann_at_10": 0.97, "ms_p95": 9.0, "elegido_r04": False},
        {"ef": 128, "recall_ann_at_10": 0.99, "ms_p95": 25.0, "elegido_r04": False},
    ])


def test_un_punto_por_ef_unido_en_orden(barrido_ann):
    figura = plot_ann_pareto(
        barrido_ann, recall_minimo=0.90, p95_maximo_ms=20.0,
        recall_column="recall_ann_at_10",
    )

    assert list(figura.data[0].x) == [5.0, 7.0, 9.0, 25.0]
    assert list(figura.data[0].text) == ["16", "32", "64", "128"]


def test_el_elegido_se_distingue_por_color(barrido_ann):
    figura = plot_ann_pareto(
        barrido_ann, recall_minimo=0.90, p95_maximo_ms=20.0,
        recall_column="recall_ann_at_10",
    )

    colores = list(figura.data[0].marker.color)
    assert colores[1] != colores[0]            # ef=32, el elegido
    assert colores[0] == colores[2] == colores[3]


def test_la_region_sombreada_es_la_interseccion_de_d16(barrido_ann):
    figura = plot_ann_pareto(
        barrido_ann, recall_minimo=0.90, p95_maximo_ms=20.0,
        recall_column="recall_ann_at_10",
    )
    region = next(f for f in figura.layout.shapes if f.type == "rect")

    assert region.x0 == 0 and region.x1 == pytest.approx(20.0)
    assert region.y0 == pytest.approx(0.90)


def test_sin_columna_elegida_no_falla_y_no_distingue_ninguna(barrido_ann):
    sin_eleccion = barrido_ann.drop(columns=["elegido_r04"])

    figura = plot_ann_pareto(
        sin_eleccion, recall_minimo=0.90, p95_maximo_ms=20.0,
        recall_column="recall_ann_at_10",
    )

    assert len(set(figura.data[0].marker.color)) == 1


def test_avisa_si_falta_una_columna(barrido_ann):
    with pytest.raises(ValueError, match="faltan columnas"):
        plot_ann_pareto(
            barrido_ann.drop(columns=["ms_p95"]), recall_minimo=0.9,
            p95_maximo_ms=20.0, recall_column="recall_ann_at_10",
        )


def test_avisa_si_el_barrido_esta_vacio():
    vacio = pd.DataFrame(columns=["ef", "recall_ann_at_10", "ms_p95"])

    with pytest.raises(ValueError, match="no tiene filas"):
        plot_ann_pareto(
            vacio, recall_minimo=0.9, p95_maximo_ms=20.0,
            recall_column="recall_ann_at_10",
        )


# ───────────────────────── plot_duplicate_threshold_sweep ─────────────────────


@pytest.fixture
def barrido_duplicados() -> pd.DataFrame:
    return pd.DataFrame([
        {"umbral_texto_solo": 0.9, "margen_minimo": 0.1, "umbral_texto_corroborado": 0.5,
         "fp": 1, "precision": 0.88, "recall": 0.8, "cumple_d22": True, "elegido_r05": False},
        {"umbral_texto_solo": 0.8, "margen_minimo": 0.1, "umbral_texto_corroborado": 0.5,
         "fp": 2, "precision": 0.78, "recall": 1.0, "cumple_d22": True, "elegido_r05": True},
        {"umbral_texto_solo": 0.7, "margen_minimo": 0.1, "umbral_texto_corroborado": 0.5,
         "fp": 5, "precision": 0.58, "recall": 1.0, "cumple_d22": False, "elegido_r05": False},
    ])


def test_un_punto_por_combinacion(barrido_duplicados):
    figura = plot_duplicate_threshold_sweep(barrido_duplicados, max_fp=2)

    assert list(figura.data[0].x) == [0.88, 0.78, 0.58]
    assert list(figura.data[0].y) == [0.8, 1.0, 1.0]


def test_el_elegido_y_los_que_no_cumplen_se_distinguen_por_color(barrido_duplicados):
    figura = plot_duplicate_threshold_sweep(barrido_duplicados, max_fp=2)

    colores = list(figura.data[0].marker.color)
    assert colores[1] != colores[0]            # elegido, distinto del que solo cumple
    assert colores[2] != colores[0]            # no cumple el suelo, distinto de ambos
    assert len(set(colores)) == 3


def test_sin_columna_elegida_no_falla_y_no_distingue_el_ganador(barrido_duplicados):
    sin_eleccion = barrido_duplicados.drop(columns=["elegido_r05"])

    figura = plot_duplicate_threshold_sweep(sin_eleccion, max_fp=2)

    colores = list(figura.data[0].marker.color)
    assert colores[0] == colores[1]            # ambos cumplen, ninguno marcado elegido
    assert colores[2] != colores[0]            # el que no cumple sigue distinguiéndose


def test_avisa_si_falta_una_columna_del_barrido_de_duplicados(barrido_duplicados):
    with pytest.raises(ValueError, match="faltan columnas"):
        plot_duplicate_threshold_sweep(
            barrido_duplicados.drop(columns=["recall"]), max_fp=2
        )


def test_avisa_si_el_barrido_de_duplicados_esta_vacio():
    vacio = pd.DataFrame(columns=["precision", "recall", "cumple_d22"])

    with pytest.raises(ValueError, match="no tiene filas"):
        plot_duplicate_threshold_sweep(vacio, max_fp=2)
