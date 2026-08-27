"""Pruebas de NB07: senales, regla y barrido de duplicados (src/aurum/duplicados.py)."""
import pandas as pd
import pytest

from aurum.busqueda import Resultado
from aurum.duplicados import (
    SenalesDuplicado,
    barrido_umbrales,
    calcular_senales,
    colores_coinciden,
    elegir_punto_operacion,
    marcas_coinciden,
    normalizar_color,
    regla_duplicado,
    resultados_duplicados,
)

# ────────────────────────── D20 · igualdad de color ──────────────────────────


def test_color_coincide_con_mismas_palabras_en_otro_orden():
    assert colores_coinciden("Negro, Acero Inoxidable", "Acero Inoxidable, Negro") is True


def test_color_no_coincide_si_uno_declara_un_color_de_mas():
    """"Negro y Blanco" no es la misma informacion que "Negro" dicha de otra
    forma: es un color declarado de mas. D20 exige igualdad de conjunto, no
    solapamiento parcial."""
    assert colores_coinciden("Negro y Blanco", "Negro") is False


def test_color_indefinido_si_alguno_esta_vacio():
    assert colores_coinciden("", "Negro") is None
    assert colores_coinciden(None, "Negro") is None
    assert colores_coinciden(float("nan"), "Negro") is None


def test_color_coincide_identico():
    assert colores_coinciden("Negro (Black/White 011)", "Negro (Black/White 011)") is True


def test_normalizar_color_ignora_acentos_y_mayusculas():
    assert normalizar_color("Gris") == normalizar_color("GRIS")


# ────────────────────────── D20 · igualdad de marca ───────────────────────────


def test_marca_coincide_con_distinta_caja():
    assert marcas_coinciden("NIKE", "nike") is True


def test_marca_no_coincide_entre_marcas_distintas():
    assert marcas_coinciden("NIKE", "Adidas") is False


def test_marca_indefinida_si_alguna_esta_vacia():
    assert marcas_coinciden("", "NIKE") is None
    assert marcas_coinciden(None, "NIKE") is None


# ─────────────────────────── SenalesDuplicado.margen ──────────────────────────


def test_margen_es_la_diferencia_entre_top1_y_top2():
    s = SenalesDuplicado(
        incoming_id="x", score_top1=0.9, score_top2=0.7,
        matched_product_id="p1", marca_coincide=None, color_coincide=None,
    )
    assert s.margen == pytest.approx(0.2)


def test_margen_infinito_sin_segundo_candidato():
    """Sin rival, el top1 no tiene con que empatar: maxima confianza, no
    minima -por eso el fallback de `calcular_senales` no debe leerse como
    margen cero."""
    s = SenalesDuplicado(
        incoming_id="x", score_top1=0.9, score_top2=float("-inf"),
        matched_product_id="p1", marca_coincide=None, color_coincide=None,
    )
    assert s.margen == float("inf")


# ────────────────────────────── calcular_senales ──────────────────────────────


class BuscadorFalso:
    def __init__(self, resultados):
        self._resultados = resultados
        self.consultas = []

    def buscar(self, consulta, *, top_k=2, marca=None):
        self.consultas.append(consulta)
        return self._resultados[:top_k]


def _alta(incoming_id, brand, color, is_duplicate=None):
    fila = {"incoming_id": incoming_id, "text": f"texto de {incoming_id}", "brand": brand, "color": color}
    if is_duplicate is not None:
        fila["is_duplicate"] = is_duplicate
    return fila


def test_calcular_senales_lee_marca_color_y_matched_product_id_del_top1():
    top1 = Resultado(
        rank=1, document_id="B000G3T55M", score=0.95,
        metadatos={"brand": "NIKE", "color": "Negro"},
    )
    top2 = Resultado(rank=2, document_id="OTRO", score=0.80, metadatos={})
    buscador = BuscadorFalso([top1, top2])
    altas = pd.DataFrame([_alta("DEV-DUP-001", "NIKE", "Negro", is_duplicate=True)])

    senales = calcular_senales(buscador, altas)

    assert len(senales) == 1
    s = senales[0]
    assert s.matched_product_id == "B000G3T55M"
    assert s.score_top1 == pytest.approx(0.95)
    assert s.score_top2 == pytest.approx(0.80)
    assert s.marca_coincide is True
    assert s.color_coincide is True
    assert s.is_duplicate is True


def test_calcular_senales_is_duplicate_none_sin_columna_de_etiqueta():
    """El caso real de altas_evaluacion.csv: no trae `is_duplicate`."""
    top1 = Resultado(rank=1, document_id="X", score=0.5, metadatos={})
    buscador = BuscadorFalso([top1])
    altas = pd.DataFrame([_alta("EVAL-001", "Marca X", "Azul")])

    senales = calcular_senales(buscador, altas, etiquetas_col=None)

    assert senales[0].is_duplicate is None


def test_calcular_senales_falla_sin_ningun_candidato():
    buscador = BuscadorFalso([])
    altas = pd.DataFrame([_alta("EVAL-001", "Marca X", "Azul")])

    with pytest.raises(ValueError, match="EVAL-001"):
        calcular_senales(buscador, altas, etiquetas_col=None)


# ──────────────────────────────── D21 · la regla ──────────────────────────────


def _senal(score_top1, score_top2, marca=None, color=None, is_duplicate=None):
    return SenalesDuplicado(
        incoming_id="x", score_top1=score_top1, score_top2=score_top2,
        matched_product_id="p", marca_coincide=marca, color_coincide=color,
        is_duplicate=is_duplicate,
    )


def test_camino_1_no_necesita_marca_ni_color():
    s = _senal(score_top1=0.95, score_top2=0.70)  # margen 0.25
    assert regla_duplicado(
        s, umbral_texto_solo=0.9, margen_minimo=0.2, umbral_texto_corroborado=0.5
    ) is True


def test_camino_1_falla_si_el_margen_no_llega_aunque_el_score_sea_alto():
    s = _senal(score_top1=0.95, score_top2=0.94)  # margen 0.01
    assert regla_duplicado(
        s, umbral_texto_solo=0.9, margen_minimo=0.2, umbral_texto_corroborado=0.5
    ) is False


def test_camino_2_se_activa_solo_con_marca():
    s = _senal(score_top1=0.6, score_top2=0.1, marca=True, color=False)
    assert regla_duplicado(
        s, umbral_texto_solo=0.95, margen_minimo=0.2, umbral_texto_corroborado=0.5
    ) is True


def test_camino_2_se_activa_solo_con_color():
    s = _senal(score_top1=0.6, score_top2=0.1, marca=False, color=True)
    assert regla_duplicado(
        s, umbral_texto_solo=0.95, margen_minimo=0.2, umbral_texto_corroborado=0.5
    ) is True


def test_camino_2_no_se_activa_sin_ninguna_corroboracion():
    """marca/color en `None` (dato ausente) cuenta como no-corroborado, no
    como error: D20 documenta ese hueco, no lo hace fallar."""
    s = _senal(score_top1=0.6, score_top2=0.1, marca=None, color=None)
    assert regla_duplicado(
        s, umbral_texto_solo=0.95, margen_minimo=0.2, umbral_texto_corroborado=0.5
    ) is False


def test_ningun_camino_por_debajo_del_umbral_corroborado():
    s = _senal(score_top1=0.4, score_top2=0.1, marca=True, color=True)
    assert regla_duplicado(
        s, umbral_texto_solo=0.95, margen_minimo=0.2, umbral_texto_corroborado=0.5
    ) is False


# ──────────────────────────────── D22 · el barrido ────────────────────────────


def test_barrido_umbrales_exige_etiquetas_en_todas_las_filas():
    senales = [_senal(0.9, 0.5, is_duplicate=True), _senal(0.9, 0.5, is_duplicate=None)]
    with pytest.raises(ValueError, match="is_duplicate"):
        barrido_umbrales(
            senales,
            valores_umbral_texto_solo=[0.9],
            valores_margen_minimo=[0.1],
            valores_umbral_texto_corroborado=[0.5],
        )


def test_barrido_umbrales_falla_si_ningun_umbral_corroborado_es_menor_que_el_solo():
    senales = [_senal(0.9, 0.5, is_duplicate=True)]
    with pytest.raises(ValueError, match="umbral_texto_corroborado"):
        barrido_umbrales(
            senales,
            valores_umbral_texto_solo=[0.9],
            valores_margen_minimo=[0.1],
            valores_umbral_texto_corroborado=[0.9, 0.95],
        )


def test_barrido_umbrales_cuenta_tp_fp_fn_tn_correctamente():
    positivo_claro = _senal(0.95, 0.5, is_duplicate=True)          # camino 1 con margen 0.2
    negativo_claro = _senal(0.3, 0.1, is_duplicate=False)           # ningun camino
    senales = [positivo_claro, negativo_claro]

    tabla = barrido_umbrales(
        senales,
        valores_umbral_texto_solo=[0.9],
        valores_margen_minimo=[0.2],
        valores_umbral_texto_corroborado=[0.5],
    )

    fila = tabla.iloc[0]
    assert (fila["tp"], fila["fp"], fila["fn"], fila["tn"]) == (1, 0, 0, 1)
    assert fila["precision"] == pytest.approx(1.0)
    assert fila["recall"] == pytest.approx(1.0)
    assert fila["f1"] == pytest.approx(1.0)


# ───────────────────────── D22 · el punto de operacion ────────────────────────


def test_elegir_punto_operacion_maximiza_recall_dentro_del_suelo_de_fp():
    barrido = pd.DataFrame(
        [
            {"umbral_texto_solo": 0.9, "margen_minimo": 0.1, "umbral_texto_corroborado": 0.5,
             "fp": 1, "recall": 0.8, "f1": 0.85},
            {"umbral_texto_solo": 0.8, "margen_minimo": 0.1, "umbral_texto_corroborado": 0.5,
             "fp": 2, "recall": 1.0, "f1": 0.90},
            {"umbral_texto_solo": 0.7, "margen_minimo": 0.1, "umbral_texto_corroborado": 0.5,
             "fp": 5, "recall": 1.0, "f1": 0.60},
        ]
    )

    tabla = elegir_punto_operacion(barrido, max_fp=2)

    # La tercera fila queda fuera por el suelo (fp=5 > 2), aunque su recall
    # empate con la segunda.
    assert list(tabla["cumple_d22"]) == [True, True, False]
    assert list(tabla["elegido_r05"]) == [False, True, False]


def test_elegir_punto_operacion_sin_ninguna_combinacion_dentro_del_suelo():
    barrido = pd.DataFrame(
        [{"umbral_texto_solo": 0.9, "margen_minimo": 0.1, "umbral_texto_corroborado": 0.5,
          "fp": 5, "recall": 0.5, "f1": 0.5}]
    )

    tabla = elegir_punto_operacion(barrido, max_fp=2)

    assert list(tabla["elegido_r05"]) == [False]


# ─────────────────────────── el artefacto de entrega ──────────────────────────


def test_resultados_duplicados_deja_matched_product_id_vacio_si_es_negativo():
    positivo = SenalesDuplicado(
        incoming_id="EVAL-001", score_top1=0.95, score_top2=0.5,
        matched_product_id="P1", marca_coincide=None, color_coincide=None,
    )
    negativo = SenalesDuplicado(
        incoming_id="EVAL-002", score_top1=0.1, score_top2=0.05,
        matched_product_id="P2", marca_coincide=None, color_coincide=None,
    )

    tabla = resultados_duplicados(
        [positivo, negativo],
        umbral_texto_solo=0.9, margen_minimo=0.2, umbral_texto_corroborado=0.5,
    )

    assert list(tabla.columns) == [
        "incoming_id", "predicted_duplicate", "matched_product_id", "score",
    ]
    fila_positiva = tabla[tabla["incoming_id"] == "EVAL-001"].iloc[0]
    fila_negativa = tabla[tabla["incoming_id"] == "EVAL-002"].iloc[0]
    assert bool(fila_positiva["predicted_duplicate"]) is True
    assert fila_positiva["matched_product_id"] == "P1"
    assert bool(fila_negativa["predicted_duplicate"]) is False
    assert fila_negativa["matched_product_id"] == ""
