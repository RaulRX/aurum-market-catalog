import pandas as pd
import pytest

from aurum.datos import (
    relevant_field_nullity,
    brand_normalization_collisions,
    corpus_document_frequency,
    document_length_stats,
    esci_label_counts_per_query,
    literal_match_ceiling,
    normalize_brand,
    null_field_rates,
    qrels_coverage_in_catalog,
    qrels_pool_sizes,
    query_term_coverage,
    query_token_frequencies,
    text_field_label_summary,
    tokenize,
    value_frequency,
)


def test_esci_label_counts_per_query_counts_each_label_and_relevance_definitions():
    relevancias = pd.DataFrame(
        {
            "query_id": [1, 1, 1, 2, 2],
            "product_id": ["p1", "p2", "p3", "p4", "p5"],
            "esci_label": ["E", "S", "I", "E", "E"],
        }
    )

    counts = esci_label_counts_per_query(relevancias).set_index("query_id")

    assert counts.loc[1, ["E", "S", "C", "I"]].tolist() == [1, 1, 0, 1]
    assert counts.loc[1, "relevantes_solo_E"] == 1
    assert counts.loc[1, "relevantes_E_mas_S"] == 2
    assert counts.loc[2, "relevantes_solo_E"] == 2
    assert counts.loc[2, "relevantes_E_mas_S"] == 2


def test_null_field_rates_computes_percentage_over_all_rows():
    df = pd.DataFrame({"color": ["Negro", None, None, "Azul"], "brand": ["A", "B", "C", "D"]})

    rates = null_field_rates(df, ["color", "brand"]).set_index("campo")

    assert rates.loc["color", "pct_nulos"] == 50.0
    assert rates.loc["brand", "pct_nulos"] == 0.0


def test_normalize_brand_raw_mode_preserves_case_and_spacing():
    assert normalize_brand("  Nike ", "raw") == "Nike"


def test_normalize_brand_casefold_mode_ignores_case_and_surrounding_spaces():
    assert normalize_brand("  Nike ", "casefold") == normalize_brand("NIKE", "casefold") == "nike"


def test_normalize_brand_casefold_mode_keeps_accents():
    assert normalize_brand("Núñez Hogar", "casefold") != normalize_brand("Nunez Hogar", "casefold")


def test_normalize_brand_unaccent_mode_merges_accented_and_plain_variants():
    assert normalize_brand("Núñez Hogar", "unaccent") == normalize_brand("Nunez Hogar", "unaccent")


def test_normalize_brand_returns_none_for_missing_values():
    assert normalize_brand(float("nan"), "casefold") is None


def test_brand_normalization_collisions_raw_mode_finds_no_collisions():
    df = pd.DataFrame({"brand": ["Nike", "NIKE", "  nike ", "Adidas", "Núñez Hogar", "Nunez Hogar"]})

    collisions = brand_normalization_collisions(df, "raw")

    assert collisions.empty


def test_brand_normalization_collisions_casefold_mode_merges_case_and_spacing_variants():
    df = pd.DataFrame({"brand": ["Nike", "NIKE", "  nike ", "Adidas", "Núñez Hogar", "Nunez Hogar"]})

    collisions = brand_normalization_collisions(df, "casefold").set_index("normalizada")

    assert collisions.loc["nike", "n_marcas_crudas"] == 3
    assert "núñez hogar" not in collisions.index
    assert "nunez hogar" not in collisions.index


def test_brand_normalization_collisions_unaccent_mode_also_merges_accented_variants():
    df = pd.DataFrame({"brand": ["Nike", "NIKE", "  nike ", "Adidas", "Núñez Hogar", "Nunez Hogar"]})

    collisions = brand_normalization_collisions(df, "unaccent").set_index("normalizada")

    assert collisions.loc["nike", "n_marcas_crudas"] == 3
    assert collisions.loc["nunez hogar", "n_marcas_crudas"] == 2


def test_text_field_label_summary_detects_spanish_and_english_labels_case_insensitively():
    df = pd.DataFrame(
        {
            "text": [
                "Vestido largo. Marca: Acme. Color: Negro.",
                "Backpack. BRAND: Acme. Colour: Blue.",
                "Producto sin ninguna etiqueta de campo.",
            ],
            "brand": ["Acme", "Acme", "Acme"],
            "color": ["Negro", "Blue", None],
        }
    )

    summary = text_field_label_summary(df).set_index("campo")

    assert summary.loc["marca", "n_con_etiqueta_en_text"] == 2
    assert summary.loc["color", "n_con_etiqueta_en_text"] == 2


def test_text_field_label_summary_flags_label_present_with_structured_field_empty():
    df = pd.DataFrame(
        {
            "text": ["Prenda. Marca: . Color: Rojo.", "Prenda. Marca: Acme. Color: Rojo."],
            "brand": [None, "Acme"],
            "color": ["Rojo", "Rojo"],
        }
    )

    summary = text_field_label_summary(df).set_index("campo")

    assert summary.loc["marca", "n_etiqueta_con_campo_vacio"] == 1
    assert summary.loc["color", "n_etiqueta_con_campo_vacio"] == 0


def test_value_frequency_reports_percentage_per_distinct_value_including_empty():
    df = pd.DataFrame({"brand": ["Nike", "Nike", "Adidas", None]})

    freq = value_frequency(df, "brand").set_index("brand")

    assert freq.loc["Nike", "n_filas"] == 2
    assert freq.loc["Nike", "pct_filas"] == 50.0
    assert freq.loc["Adidas", "pct_filas"] == 25.0
    assert freq.loc["(vacío)", "pct_filas"] == 25.0


def test_qrels_pool_sizes_counts_distinct_products_per_query():
    relevancias = pd.DataFrame(
        {
            "query_id": [1, 1, 1, 2, 2],
            "product_id": ["p1", "p2", "p2", "p4", "p5"],
            "esci_label": ["E", "S", "I", "E", "E"],
        }
    )

    pool = qrels_pool_sizes(relevancias).set_index("query_id")

    assert pool.loc[1, "pool_size"] == 2
    assert pool.loc[2, "pool_size"] == 2


def test_tokenize_lowercases_and_keeps_alphanumeric_sequences():
    assert tokenize("Base Tapizada 160x200, sin patas") == [
        "base",
        "tapizada",
        "160x200",
        "sin",
        "patas",
    ]


def test_tokenize_keeps_accents_by_default_and_removes_them_on_demand():
    assert tokenize("Portátil táctil") == ["portátil", "táctil"]
    assert tokenize("Portátil táctil", strip_accents=True) == ["portatil", "tactil"]


def test_tokenize_returns_empty_list_for_missing_values():
    assert tokenize(float("nan")) == []


def test_document_length_stats_measures_dispersion_per_text_field():
    df = pd.DataFrame(
        {
            "title": ["uno dos", "uno dos tres cuatro"],
            "text": ["uno", "uno dos tres cuatro cinco seis"],
        }
    )

    stats = document_length_stats(df, ["title", "text"]).set_index("campo")

    assert stats.loc["title", "tokens_max"] == 4
    assert stats.loc["text", "tokens_max"] == 6
    # El campo con más dispersión relativa tiene mayor coeficiente de variación.
    assert stats.loc["text", "cv"] > stats.loc["title", "cv"]


def test_corpus_document_frequency_counts_documents_not_occurrences():
    df = pd.DataFrame({"text": ["taladro taladro batería", "taladro sin cable"]})

    frequency = corpus_document_frequency(df, "text")

    assert frequency["taladro"] == 2
    assert frequency["batería"] == 1


def test_query_term_coverage_flags_terms_absent_from_the_corpus():
    catalogo = pd.DataFrame({"text": ["taladro con batería", "taladro sin cable"]})
    consultas = pd.DataFrame(
        {"query_id": [1], "query_text": ["taladro inalámbrico batería"]}
    )

    coverage = query_term_coverage(consultas, catalogo).set_index("query_id")

    assert coverage.loc[1, "n_tokens"] == 3
    assert coverage.loc[1, "n_oov"] == 1
    assert coverage.loc[1, "tokens_oov"] == "inalámbrico"
    assert coverage.loc[1, "df_min"] == 1  # batería
    assert coverage.loc[1, "df_max"] == 2  # taladro


def test_query_term_coverage_recovers_accented_terms_when_stripping_accents():
    catalogo = pd.DataFrame({"text": ["portátil táctil convertible"]})
    consultas = pd.DataFrame({"query_id": [1], "query_text": ["portatil tactil"]})

    sin_strip = query_term_coverage(consultas, catalogo).set_index("query_id")
    con_strip = query_term_coverage(consultas, catalogo, strip_accents=True).set_index(
        "query_id"
    )

    assert sin_strip.loc[1, "n_oov"] == 2
    assert con_strip.loc[1, "n_oov"] == 0


def test_query_token_frequencies_gives_one_row_per_term_with_both_accent_modes():
    catalogo = pd.DataFrame({"text": ["portátil táctil", "portátil con cable", "otro"]})
    consultas = pd.DataFrame({"query_id": [1], "query_text": ["portatil tactil inventado"]})

    frecuencias = query_token_frequencies(consultas, catalogo).set_index("token")

    assert list(frecuencias["posicion"]) == [1, 2, 3]
    # Sin quitar acentos, la consulta apunta a listas vacías: son otras palabras.
    assert frecuencias.loc["portatil", "df_con_tildes"] == 0
    assert frecuencias.loc["portatil", "df_sin_tildes"] == 2
    assert frecuencias.loc["tactil", "df_sin_tildes"] == 1
    # Un término que no existe en el corpus tiene lista vacía en ambos modos.
    assert frecuencias.loc["inventado", "df_con_tildes"] == 0
    assert frecuencias.loc["inventado", "df_sin_tildes"] == 0


def test_literal_match_ceiling_counts_relevants_containing_every_query_term():
    consultas = pd.DataFrame({"query_id": [1], "query_text": ["taladro batería"]})
    relevancias = pd.DataFrame(
        {
            "query_id": [1, 1, 1],
            "product_id": ["p1", "p2", "p3"],
            "esci_label": ["E", "S", "I"],
        }
    )
    catalogo = pd.DataFrame(
        {
            "product_id": ["p1", "p2", "p3"],
            "title": ["taladro con batería", "taladro sin cable", "taladro batería"],
            "text": ["taladro con batería", "taladro con batería", "otro"],
        }
    )

    ceiling = literal_match_ceiling(consultas, relevancias, catalogo).set_index("query_id")

    assert ceiling.loc[1, "n_relevantes"] == 2  # p3 es I, no cuenta
    assert ceiling.loc[1, "n_todos_en_title"] == 1  # solo p1
    assert ceiling.loc[1, "n_todos_en_text"] == 2  # p1 y p2
    assert ceiling.loc[1, "pct_todos_en_title"] == 50.0


def test_literal_match_ceiling_ignores_relevants_absent_from_the_catalog():
    consultas = pd.DataFrame({"query_id": [1], "query_text": ["taladro"]})
    relevancias = pd.DataFrame(
        {"query_id": [1, 1], "product_id": ["p1", "ausente"], "esci_label": ["E", "E"]}
    )
    catalogo = pd.DataFrame(
        {"product_id": ["p1"], "title": ["taladro"], "text": ["taladro"]}
    )

    ceiling = literal_match_ceiling(consultas, relevancias, catalogo).set_index("query_id")

    assert ceiling.loc[1, "n_relevantes"] == 2
    assert ceiling.loc[1, "n_en_catalogo"] == 1
    assert ceiling.loc[1, "pct_todos_en_title"] == 100.0


def test_qrels_coverage_in_catalog_counts_judged_products_present_in_catalog():
    relevancias = pd.DataFrame(
        {
            "query_id": [1, 1, 1, 2, 2],
            "product_id": ["p1", "p2", "p3", "p4", "p5"],
            "esci_label": ["E", "S", "I", "E", "E"],
        }
    )
    catalog = pd.DataFrame({"product_id": ["p1", "p3", "p4"]})

    coverage = qrels_coverage_in_catalog(relevancias, catalog).set_index("query_id")

    assert coverage.loc[1, "n_juzgados"] == 3
    assert coverage.loc[1, "n_presentes"] == 2
    assert coverage.loc[2, "n_juzgados"] == 2
    assert coverage.loc[2, "n_presentes"] == 1
    assert coverage.loc[2, "pct_presentes"] == 50.0


# ─────────────────────── relevant_field_nullity (control D02) ────────────────


def _catalogo_con_nulos():
    return pd.DataFrame([
        {"product_id": "P1", "color": "Negro"},
        {"product_id": "P2", "color": None},
        {"product_id": "P3", "color": "   "},   # espacios: cuenta como vacío
        {"product_id": "P4", "color": "Rojo"},
    ])


def _relevancias():
    return pd.DataFrame([
        {"query_id": 1, "product_id": "P1", "esci_label": "E"},
        {"query_id": 1, "product_id": "P2", "esci_label": "S"},
        {"query_id": 1, "product_id": "P4", "esci_label": "I"},   # irrelevante: no cuenta
        {"query_id": 2, "product_id": "P2", "esci_label": "E"},
        {"query_id": 2, "product_id": "P3", "esci_label": "E"},
        {"query_id": 3, "product_id": "PX", "esci_label": "E"},   # fuera del catálogo
    ])


def test_la_exposicion_cuenta_solo_los_relevantes_presentes():
    tabla = relevant_field_nullity(_relevancias(), _catalogo_con_nulos(), field="color")

    fila1 = tabla[tabla["query_id"] == "1"].iloc[0]
    assert fila1["n_relevantes"] == 2          # P1 y P2; el I no entra
    assert fila1["pct_sin_color"] == 50.0


def test_una_consulta_con_todos_los_relevantes_vacios_da_exposicion_total():
    """Es el caso que decide el control de D02: exposición máxima al relleno."""
    tabla = relevant_field_nullity(_relevancias(), _catalogo_con_nulos(), field="color")

    assert tabla[tabla["query_id"] == "2"].iloc[0]["pct_sin_color"] == 100.0


def test_las_consultas_sin_relevantes_en_el_catalogo_se_omiten():
    tabla = relevant_field_nullity(_relevancias(), _catalogo_con_nulos(), field="color")

    assert "3" not in set(tabla["query_id"])


def test_avisa_si_el_campo_no_existe():
    with pytest.raises(ValueError, match="columna"):
        relevant_field_nullity(_relevancias(), _catalogo_con_nulos(), field="talla")
