"""Tests de `aurum.plantillas`.

Lo que estas pruebas protegen es la comparabilidad del experimento: si dos
plantillas se diferencian en algo que no era el factor bajo estudio, la Δ de
nDCG que salga en NB03 no significa nada. Por eso se fija con tests que `A3` y
`A3n` difieren **solo** en la política de nulos, y que `A5` es `A3` sin `color`.
"""
from __future__ import annotations

import pandas as pd
import pytest

from aurum.plantillas import (
    CONTROLES,
    NULL_PLACEHOLDER,
    TEMPLATES,
    candidatas,
    CorpusContext,
    compose_product_text,
    corpus_context,
    render_template,
    template_stats,
    truncate_on_word_boundary,
)


FILA = {
    "title": "Vestido Largo De Navidad",
    "brand": "KanLin1986-Ropa",
    "color": "Negro",
    "text": ("Vestido Largo De Navidad. Marca: KanLin1986-Ropa. Color: Negro. "
             + "relleno " * 500).strip(),
}
SIN_COLOR = {**FILA, "color": None}


CTX = CorpusContext(a4_chars=64)


@pytest.fixture
def catalogo() -> pd.DataFrame:
    return pd.DataFrame([FILA, SIN_COLOR])


# ───────────────────────────── política de nulos ─────────────────────────────


def test_d02_omite_la_seccion_del_campo_vacio():
    texto = compose_product_text(SIN_COLOR)

    assert "Color" not in texto
    assert NULL_PLACEHOLDER not in texto
    assert "Marca: KanLin1986-Ropa" in texto


def test_a3n_rellena_donde_a3_omite():
    """El control de D02: la única diferencia entre las dos plantillas."""
    a3 = TEMPLATES["A3"](SIN_COLOR, CTX)
    a3n = TEMPLATES["A3n"](SIN_COLOR, CTX)

    assert a3 != a3n
    assert f"Color: {NULL_PLACEHOLDER}" in a3n
    assert "Color" not in a3


def test_a3_y_a3n_coinciden_cuando_no_hay_nulos():
    """Sin campos vacíos no hay nada que rellenar: si difirieran aquí, la Δ de
    NB03 mediría algo más que la política de nulos."""
    assert TEMPLATES["A3"](FILA, CTX) == TEMPLATES["A3n"](FILA, CTX)


def test_los_espacios_en_blanco_cuentan_como_vacio():
    texto = compose_product_text({**FILA, "color": "   "})

    assert "Color" not in texto


# ──────────────────────────── forma de cada receta ───────────────────────────


def test_a0_y_a1_son_las_columnas_tal_cual():
    assert TEMPLATES["A0"](FILA, CTX) == FILA["text"]
    assert TEMPLATES["A1"](FILA, CTX) == FILA["title"]


def test_a2_lleva_los_valores_pero_no_las_etiquetas():
    texto = TEMPLATES["A2"](FILA, CTX)

    assert "KanLin1986-Ropa" in texto and "Negro" in texto
    assert "Marca:" not in texto and "Color:" not in texto


def test_a5_es_a3_sin_color():
    a5 = TEMPLATES["A5"](FILA, CTX)

    assert "Marca: KanLin1986-Ropa" in a5
    assert "Color" not in a5


def test_a4_recorta_por_el_corte_del_contexto():
    a4 = TEMPLATES["A4"](FILA, CTX)

    assert len(a4) <= CTX.a4_chars
    assert FILA["text"].startswith(a4[:40])


def test_el_corte_de_a4_es_la_mediana_del_corpus():
    """El punto de corte sale de los datos, no de una constante: es lo que hace
    la plantilla defendible y la mantiene válida si cambia el catálogo."""
    corpus = pd.DataFrame([{**FILA, "text": "x" * n} for n in (100, 200, 900)])

    assert corpus_context(corpus).a4_chars == 200


def test_la_mediana_no_es_la_media():
    """Con la distribución sesgada del catálogo real las dos difieren, y elegir
    una u otra cambia a cuántos productos afecta el recorte."""
    corpus = pd.DataFrame([{**FILA, "text": "x" * n} for n in (100, 200, 3000)])

    assert corpus_context(corpus).a4_chars == 200      # mediana
    assert corpus_context(corpus).a4_chars != 1100     # media


def test_el_contexto_exige_un_corpus_con_texto():
    with pytest.raises(ValueError, match="columna 'text'"):
        corpus_context(pd.DataFrame([{"title": "algo"}]))


# ───────────────────────────────── recorte ───────────────────────────────────


def test_el_recorte_no_parte_palabras():
    """Cortar a media palabra fabrica subpalabras inexistentes y mediría 'texto
    roto', no 'menos texto'."""
    assert truncate_on_word_boundary("resistente al agua", 13) == "resistente"


def test_una_palabra_mas_larga_que_el_limite_se_corta_igual():
    """No hay frontera donde cortar: el corte duro es el único comportamiento
    posible, y vale la pena fijarlo para que no sorprenda."""
    assert truncate_on_word_boundary("resistente al agua", 8) == "resisten"


def test_el_recorte_deja_intacto_lo_que_ya_cabe():
    assert truncate_on_word_boundary("corto", 100) == "corto"


def test_el_recorte_exige_una_longitud_positiva():
    with pytest.raises(ValueError, match="max_chars"):
        truncate_on_word_boundary("texto", 0)


# ──────────────────────────── aplicación al corpus ───────────────────────────


def test_render_devuelve_un_texto_por_fila(catalogo):
    assert len(render_template(catalogo, "A3")) == len(catalogo)


def test_render_avisa_de_una_plantilla_inexistente(catalogo):
    with pytest.raises(ValueError, match="Plantilla desconocida"):
        render_template(catalogo, "A9")


def test_render_avisa_de_las_columnas_que_faltan():
    with pytest.raises(ValueError, match="faltan columnas"):
        render_template(pd.DataFrame([{"title": "algo"}]), "A0")


def test_las_estadisticas_comparan_cada_plantilla_contra_a0(catalogo):
    tabla = template_stats(catalogo, ["A0", "A1"])

    assert list(tabla["plantilla"]) == ["A0", "A1"]
    assert float(tabla.loc[tabla["plantilla"] == "A0", "pct_vs_A0"].iloc[0]) == 100.0
    assert float(tabla.loc[tabla["plantilla"] == "A1", "pct_vs_A0"].iloc[0]) < 100.0


def test_ninguna_plantilla_deja_el_texto_vacio(catalogo):
    """Un texto vacío produce un vector degenerado, y varios producen filas
    idénticas que `vector_health` denunciaría como duplicados."""
    for nombre in TEMPLATES:
        textos = render_template(catalogo, nombre)
        assert all(texto.strip() for texto in textos), nombre


# ────────────────────────── controles vs candidatas ──────────────────────────


def test_a3n_es_un_control_y_no_compite():
    """Su papel está declarado junto a la definición de la plantilla, no en el
    notebook: es lo que hace auditable que se excluya de la elección."""
    assert "A3n" in CONTROLES
    assert "A3n" not in candidatas()


def test_candidatas_y_controles_cubren_todas_las_plantillas():
    """Ninguna plantilla puede quedar en tierra de nadie: o compite o controla."""
    assert set(candidatas()) | set(CONTROLES) == set(TEMPLATES)
    assert not set(candidatas()) & set(CONTROLES)


def test_los_controles_se_siguen_renderizando(catalogo):
    """Excluirlo de la elección no es dejar de medirlo: sin su medición no hay
    con qué contrastar la decisión que controla."""
    assert len(render_template(catalogo, "A3n")) == len(catalogo)
