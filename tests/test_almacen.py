"""Tests de `aurum.almacen`.

Lo que protegen estas pruebas es que la evidencia de D13–D15 mida lo que dice
medir: que la política de nulos cambie de verdad el punto que se guarda (y no
solo el nombre de la opción), que el presupuesto cuente los bytes que se pagan
15.000 veces, y que el alcance del filtro de marca se calcule con igualdad
exacta —como lo ejecuta el motor— y no con una comparación laxa de Python.
"""
from __future__ import annotations

import pandas as pd
import pytest

from aurum.almacen import (
    FILTER_KEYS,
    NULL_POLICIES,
    PAYLOAD_SCHEMAS,
    add_normalized_key,
    batch_footprint,
    build_payload,
    combined_filter_selectivity,
    field_byte_profile,
    filter_field_profile,
    filter_reach,
    filter_writing_robustness,
    index_footprint,
    payload_budget,
    payload_bytes,
    robustness_summary,
    writing_variants,
)

FILA = {
    "record_id": "000bd6e8-a995-56d0-ba03-559885ccef39",
    "product_id": "B0818K237B",
    "title": "Vestido Largo De Navidad",
    "brand": "KanLin1986-Ropa",
    "color": "Negro",
    "text": "Vestido Largo De Navidad. Marca: KanLin1986-Ropa. Color: Negro.",
    "catalog_version": 1,
    "active": True,
}
SIN_COLOR = {**FILA, "product_id": "B0000000X1", "color": None}


@pytest.fixture
def catalogo() -> pd.DataFrame:
    return pd.DataFrame([FILA, SIN_COLOR])


# ───────────────────────── política de nulos (D14) ──────────────────────────


def test_omitir_campo_deja_el_punto_sin_la_clave():
    payload = build_payload(SIN_COLOR, fields=("brand", "color"), null_policy="omitir_campo")

    assert "color" not in payload
    assert payload["brand"] == "KanLin1986-Ropa"


def test_cadena_vacia_y_centinela_si_escriben_la_clave():
    """La diferencia con `omitir_campo` no es cosmética: un filtro sobre una
    clave ausente y otro sobre una clave vacía no devuelven lo mismo."""
    vacia = build_payload(SIN_COLOR, fields=("color",), null_policy="cadena_vacia")
    centinela = build_payload(SIN_COLOR, fields=("color",), null_policy="centinela")

    assert vacia == {"color": ""}
    assert centinela["color"] and centinela["color"] != ""


def test_las_tres_politicas_coinciden_cuando_no_hay_vacios():
    """Sin campos vacíos no hay nada que decidir: si difirieran aquí, D14
    estaría cambiando algo más que el tratamiento del nulo."""
    payloads = [
        build_payload(FILA, fields=("brand", "color"), null_policy=policy)
        for policy in NULL_POLICIES
    ]

    assert all(payload == payloads[0] for payload in payloads)


def test_los_espacios_en_blanco_cuentan_como_vacio():
    payload = build_payload(
        {**FILA, "color": "   "}, fields=("color",), null_policy="omitir_campo"
    )

    assert payload == {}


def test_la_politica_de_nulos_es_obligatoria_y_validada():
    with pytest.raises(ValueError, match="null_policy"):
        build_payload(FILA, fields=("brand",), null_policy="lo_que_sea")


def test_el_payload_conserva_los_tipos_nativos():
    """`catalog_version` y `active` se comparan en NB08 (`== 2`, `is True`): si
    viajaran como texto, la verificación de las mutaciones sería otra cosa."""
    payload = build_payload(
        FILA, fields=("catalog_version", "active"), null_policy="omitir_campo"
    )

    assert payload == {"catalog_version": 1, "active": True}


# ──────────────────────────── coste del payload ─────────────────────────────


def test_los_bytes_cuentan_la_clave_ademas_del_valor():
    """El payload se guarda por punto: el nombre del campo se paga una vez por
    cada uno de los 15.000."""
    assert payload_bytes({"ab": "cd"}) == 4


def test_los_bytes_son_utf8_no_caracteres():
    assert payload_bytes({"c": "ñ"}) == 3  # 1 de la clave + 2 del carácter


def test_omitir_el_campo_vacio_sale_mas_barato_que_rellenarlo(catalogo):
    esquema = {"solo_color": ("color",)}
    omitido = payload_budget(catalogo, null_policy="omitir_campo", schemas=esquema)
    relleno = payload_budget(catalogo, null_policy="centinela", schemas=esquema)

    assert float(omitido["bytes_medios"].iloc[0]) < float(relleno["bytes_medios"].iloc[0])


def test_el_perfil_por_campo_localiza_los_vacios(catalogo):
    perfil = field_byte_profile(catalogo, ["brand", "color"]).set_index("campo")

    assert int(perfil.loc["brand", "n_vacios"]) == 0
    assert int(perfil.loc["color", "n_vacios"]) == 1
    assert float(perfil.loc["color", "pct_vacios"]) == 50.0


def test_el_perfil_avisa_de_un_campo_que_no_existe(catalogo):
    with pytest.raises(ValueError, match="columnas"):
        field_byte_profile(catalogo, ["precio"])


def test_los_esquemas_de_d13_estan_anidados():
    """`completo` tiene que contener a `minimo` para que la comparación mida
    'qué añade llevar más campos' y no dos esquemas distintos."""
    assert set(PAYLOAD_SCHEMAS["minimo"]) < set(PAYLOAD_SCHEMAS["completo"])
    assert set(PAYLOAD_SCHEMAS["completo"]) < set(PAYLOAD_SCHEMAS["completo_con_text"])


def test_ningun_esquema_guarda_el_record_id():
    """Es el ID del punto, no payload: duplicarlo se pagaría 15.000 veces sin
    aportar nada que el motor no tenga ya."""
    assert all("record_id" not in fields for fields in PAYLOAD_SCHEMAS.values())


def test_todos_los_esquemas_llevan_las_claves_de_filtro():
    """Regresión. Los esquemas se escribieron antes de que la sección B midiera
    que sin las claves derivadas el filtro pierde el 91 % del alcance, y se
    quedaron sin ellas: D13 habría presupuestado un payload que no era el que se
    iba a escribir. Las claves no son una opción de D13 -entran en los tres-, y
    este test impide que vuelvan a caerse de uno."""
    for nombre, campos in PAYLOAD_SCHEMAS.items():
        assert set(FILTER_KEYS) <= set(campos), f"{nombre} no puede filtrar"


def test_el_esquema_minimo_sigue_siendo_el_mas_pequeno():
    """Añadir las claves a los tres no puede haber roto el orden: si `minimo`
    dejara de serlo, la tabla de D13 compararía otra cosa."""
    tamanos = {n: len(c) for n, c in PAYLOAD_SCHEMAS.items()}

    assert tamanos["minimo"] < tamanos["completo"] < tamanos["completo_con_text"]


# ─────────────────────────── presupuesto de memoria ─────────────────────────


def test_el_vector_ocupa_dim_por_cuatro_bytes():
    huella = index_footprint(15_000, 768)

    assert huella["bytes_por_vector"] == 3072
    assert huella["mb_vectores"] == pytest.approx(15_000 * 3072 / 1024 ** 2, abs=0.01)


def test_el_payload_suma_a_la_huella_del_indice():
    solo_vectores = index_footprint(1000, 128)
    con_payload = index_footprint(1000, 128, payload_bytes_medios=200)

    assert con_payload["mb_total"] > solo_vectores["mb_total"]
    assert con_payload["mb_payload"] == pytest.approx(1000 * 200 / 1024 ** 2, abs=0.01)


def test_el_numero_de_lotes_redondea_hacia_arriba():
    """Con 15.000 puntos y lotes de 256 el último va incompleto, y aun así hay
    que enviarlo."""
    tabla = batch_footprint(768, n_points=15_000, batch_sizes=(256,))

    assert int(tabla["n_lotes"].iloc[0]) == 59


def test_el_lote_mas_grande_pide_mas_memoria():
    tabla = batch_footprint(768, n_points=15_000, payload_bytes_medios=300)

    assert tabla["mb_por_lote"].is_monotonic_increasing


def test_el_lote_grande_ahorra_viajes_y_paga_mas_al_fallar():
    """Los dos platos de la balanza de D15, que la RAM no arbitra: menos viajes
    de red a cambio de perder más trabajo cuando un lote se cae."""
    tabla = batch_footprint(
        768, n_points=15_000, payload_bytes_medios=300, batch_sizes=(64, 256)
    ).set_index("lote")

    assert int(tabla.loc[64, "n_lotes"]) > int(tabla.loc[256, "n_lotes"])
    assert (
        int(tabla.loc[64, "puntos_reintentados_si_falla"])
        < int(tabla.loc[256, "puntos_reintentados_si_falla"])
    )


def test_los_tres_lotes_candidatos_caben_en_un_mensaje_grpc():
    """El único techo duro de D15: pasarse del máximo de mensaje no ralentiza,
    corta la petición. Con 768 dimensiones ninguno se acerca, y por eso la
    decisión se resuelve por viajes y granularidad del fallo, no por tamaño."""
    tabla = batch_footprint(768, n_points=15_000, payload_bytes_medios=300)

    assert tabla["cabe_en_un_mensaje"].all()
    assert (tabla["pct_del_limite"] < 50).all()


def test_un_lote_desmesurado_se_senala_como_imposible():
    """La comprobación tiene que servir de algo: con un payload grande y un lote
    grande, la tabla avisa en vez de dejar que el fallo salga del SDK."""
    tabla = batch_footprint(
        3072, n_points=15_000, payload_bytes_medios=20_000, batch_sizes=(4096,)
    )

    assert not bool(tabla["cabe_en_un_mensaje"].iloc[0])
    assert float(tabla["pct_del_limite"].iloc[0]) > 100


# ────────────────────── alcance del filtro de marca (D03) ───────────────────


def test_el_filtro_crudo_pierde_las_variantes_de_escritura():
    """El número que decide si el payload necesita la clave normalizada: con
    `raw`, `NIKE` no alcanza a `Nike`."""
    catalogo = pd.DataFrame([
        {"brand": "NIKE"}, {"brand": "Nike"}, {"brand": "nike"}, {"brand": "Adidas"},
    ])

    alcance = filter_reach(catalogo, ["NIKE"]).set_index("modo")

    assert int(alcance.loc["raw", "n_productos"]) == 1
    assert int(alcance.loc["casefold", "n_productos"]) == 3


def test_la_normalizacion_sin_acentos_fusiona_lo_que_casefold_no():
    catalogo = pd.DataFrame([{"brand": "Müller"}, {"brand": "Muller"}])

    alcance = filter_reach(catalogo, ["muller"]).set_index("modo")

    assert int(alcance.loc["casefold", "n_productos"]) == 1
    assert int(alcance.loc["unaccent", "n_productos"]) == 2


def test_el_alcance_reporta_que_valores_crudos_ha_fusionado():
    """Sin la lista, un salto de 30 a 45 productos no se puede auditar: podría
    ser una variante de escritura o dos marcas distintas fusionadas."""
    catalogo = pd.DataFrame([{"brand": "NIKE"}, {"brand": "Nike"}])

    fila = filter_reach(catalogo, ["nike"], modes=("casefold",)).iloc[0]

    assert fila["n_valores_distintos"] == 2
    assert "NIKE" in fila["valores"] and "Nike" in fila["valores"]


# ─────────── cómo escribe el usuario: robustez del filtro ───────────────────


def test_las_variantes_cubren_las_tres_cajas():
    variantes = writing_variants("negro")

    assert set(variantes.values()) == {"negro", "Negro", "NEGRO"}


def test_solo_se_generan_variantes_con_tilde_si_el_valor_la_lleva():
    """No se puede adivinar dónde pondría una tilde quien escribe `azul`, pero
    sí quitársela a quien escribe `marrón`."""
    con_tilde = writing_variants("Marrón")
    sin_tilde = writing_variants("Azul")

    assert "marron" in con_tilde.values() and "marrón" in con_tilde.values()
    assert len(sin_tilde) == 3
    assert all("ó" not in v for v in sin_tilde.values())


def test_las_variantes_exigen_un_valor():
    with pytest.raises(ValueError, match="variantes"):
        writing_variants("   ")


def test_el_filtro_crudo_convierte_el_resultado_en_una_loteria():
    """El número que justifica la clave derivada: sin normalizar, lo que
    encuentra el usuario depende de la tecla de mayúsculas."""
    catalogo = pd.DataFrame([{"color": "Negro"}] * 3 + [{"color": "negro"}])

    tabla = filter_writing_robustness(catalogo, "negro", field="color")
    resumen = robustness_summary(tabla).set_index("modo")

    assert not bool(resumen.loc["raw", "consistente"])
    assert bool(resumen.loc["casefold", "consistente"])
    assert bool(resumen.loc["unaccent", "consistente"])


def test_consistente_no_significa_completo():
    """El caso que separa los dos ejes: con `casefold`, quien escribe la tilde y
    quien no encuentran cada uno **la mitad** del catálogo. Todos obtienen lo
    mismo, así que es consistente — y aun así el filtro está roto."""
    catalogo = pd.DataFrame([{"color": "Marrón"}, {"color": "marron"}])

    resumen = robustness_summary(
        filter_writing_robustness(catalogo, "Marrón", field="color")
    ).set_index("modo")

    assert bool(resumen.loc["casefold", "consistente"])
    assert not bool(resumen.loc["casefold", "alcanza_el_maximo"])
    assert float(resumen.loc["casefold", "pct_del_maximo"]) == 50.0

    assert bool(resumen.loc["unaccent", "consistente"])
    assert bool(resumen.loc["unaccent", "alcanza_el_maximo"])
    assert int(resumen.loc["unaccent", "minimo"]) == 2


def test_el_resumen_cuenta_las_variantes_medidas():
    catalogo = pd.DataFrame([{"color": "Negro"}])

    resumen = robustness_summary(
        filter_writing_robustness(catalogo, "negro", field="color", modes=("raw",))
    )

    assert int(resumen["n_variantes"].iloc[0]) == 3


# ──────────── el campo extra: un valor que lleva varios dentro ──────────────


def test_contains_alcanza_los_valores_compuestos_que_equals_deja_fuera():
    """El caso que motiva la política: quien pide negro no ve `Negro/Rojo` si
    el filtro compara el valor entero."""
    catalogo = pd.DataFrame([
        {"color": "Negro"}, {"color": "Negro/Rojo"},
        {"color": "Negro (Black)"}, {"color": "azul marino"},
    ])

    equals = filter_reach(catalogo, ["negro"], field="color", modes=("unaccent",))
    contains = filter_reach(
        catalogo, ["negro"], field="color", match="contains", modes=("unaccent",)
    )

    assert int(equals["n_productos"].iloc[0]) == 1
    assert int(contains["n_productos"].iloc[0]) == 3


def test_contains_cuenta_lo_que_entra_solo_como_subcadena():
    """La superficie de falsos positivos de la política, medida en vez de
    supuesta: `oro` dentro de `incoloro` no es el color oro."""
    catalogo = pd.DataFrame([{"color": "Oro"}, {"color": "Incoloro"}])

    fila = filter_reach(
        catalogo, ["oro"], field="color", match="contains", modes=("unaccent",)
    ).iloc[0]

    assert int(fila["n_productos"]) == 2
    assert int(fila["n_dentro_de_otra_palabra"]) == 1


def test_un_filtro_sin_resultados_no_rompe_el_recuento():
    """Regresión. Con cero coincidencias, contar los falsos positivos sumando
    una serie vacía de texto concatena en vez de sumar y devuelve la cadena
    vacía. Solo se ve cuando el filtro no encuentra nada, que es justo el caso
    que nadie prueba."""
    catalogo = pd.DataFrame([{"color": "Negro"}])

    fila = filter_reach(
        catalogo, ["turquesa"], field="color", match="contains", modes=("raw",)
    ).iloc[0]

    assert int(fila["n_productos"]) == 0
    assert int(fila["n_dentro_de_otra_palabra"]) == 0


def test_la_robustez_aguanta_variantes_que_no_encuentran_nada():
    """El caso real que lo destapó: una palabra con tilde escrita en mayúsculas
    no coincide con nada bajo el modo crudo."""
    catalogo = pd.DataFrame([{"color": "Marrón"}])

    resumen = robustness_summary(
        filter_writing_robustness(catalogo, "Marrón", field="color", match="contains")
    ).set_index("modo")

    assert int(resumen.loc["raw", "minimo"]) == 0
    assert bool(resumen.loc["unaccent", "consistente"])


def test_equals_no_reporta_subcadenas():
    """La columna solo tiene sentido en `contains`: en `equals` no hay
    coincidencias parciales que contar."""
    catalogo = pd.DataFrame([{"color": "Oro"}])

    fila = filter_reach(catalogo, ["oro"], field="color", modes=("unaccent",)).iloc[0]

    assert "n_dentro_de_otra_palabra" not in fila


def test_el_alcance_rechaza_un_modo_de_comparacion_inventado(catalogo):
    with pytest.raises(ValueError, match="match debe ser"):
        filter_reach(catalogo, ["x"], field="color", match="parecido")


def test_el_perfil_distingue_una_taxonomia_de_un_texto_libre():
    """Un campo donde casi todos los valores aparecen una vez y muchos llevan
    separador no es una categoría: es texto que alguien rellenó a mano."""
    taxonomia = pd.DataFrame([{"c": v} for v in ["Negro"] * 5 + ["Blanco"] * 5])
    libre = pd.DataFrame([
        {"c": "Negro/Rojo"}, {"c": "Azul marino"}, {"c": "Verde (Green)"},
        {"c": "como se muestra"},
    ])

    p_tax = filter_field_profile(taxonomia, ["c"]).iloc[0]
    p_libre = filter_field_profile(libre, ["c"]).iloc[0]

    assert p_tax["pct_valores_unicos"] == 0.0
    assert p_tax["pct_compuestos"] == 0.0
    assert p_libre["pct_valores_unicos"] == 100.0
    assert p_libre["pct_compuestos"] == 50.0      # "Negro/Rojo" y "Verde (Green)"
    assert p_libre["pct_multipalabra"] == 50.0    # "Azul marino" y "como se muestra"


def test_el_perfil_no_cuenta_los_vacios_como_valores():
    perfil = filter_field_profile(
        pd.DataFrame([{"c": "Negro"}, {"c": None}, {"c": "   "}]), ["c"]
    ).iloc[0]

    assert int(perfil["n_con_valor"]) == 1
    assert float(perfil["pct_vacios"]) == pytest.approx(66.67, abs=0.01)


def test_la_clave_normalizada_se_deriva_sin_tocar_la_cruda(catalogo):
    """D03 guarda el valor crudo; la clave de filtro se añade, no sustituye."""
    con_clave = add_normalized_key(catalogo, field="brand", mode="unaccent")

    assert list(con_clave["brand"]) == list(catalogo["brand"])
    assert list(con_clave["brand_normalized"]) == ["kanlin1986-ropa"] * 2


def test_la_misma_politica_sirve_para_cualquier_campo_filtrable(catalogo):
    """Que la normalización viva en una sola función es lo que impide que dos
    campos filtrables acaben normalizándose distinto."""
    con_clave = add_normalized_key(catalogo, field="color", mode="unaccent")

    assert con_clave["color_normalized"].iloc[0] == "negro"


def test_el_valor_vacio_no_produce_clave_normalizada():
    """Un producto sin marca no puede quedar bajo una clave de filtro
    inventada: sería alcanzable por un filtro que no le corresponde."""
    con_clave = add_normalized_key(
        pd.DataFrame([{"brand": None}]), field="brand", mode="unaccent"
    )

    assert con_clave["brand_normalized"].iloc[0] is None


# ─────────────── dos filtros a la vez: cuándo el cero no dice nada ────────────


def test_el_cero_por_falta_de_dato_se_distingue_de_la_ausencia_real():
    """Las dos marcas dan cero productos blancos, pero por razones opuestas: una
    tiene los colores anotados y ninguno es blanco, la otra no los tiene. Sin
    esta distinción, el cero de B.5 se leería como un hecho del catálogo."""
    catalogo = pd.DataFrame([
        {"brand": "Einhell", "color": None},
        {"brand": "Einhell", "color": "   "},
        {"brand": "NIKE", "color": "Negro"},
        {"brand": "NIKE", "color": "Rojo"},
    ])

    tabla = combined_filter_selectivity(
        catalogo, primary_values=["Einhell", "NIKE"], secondary_values=["blanco"]
    ).set_index("brand")

    assert int(tabla.loc["Einhell", "n_brand"]) == 2
    assert int(tabla.loc["Einhell", "n_brand_con_color"]) == 0
    assert int(tabla.loc["Einhell", "n_brand_y_color"]) == 0
    assert bool(tabla.loc["Einhell", "cero_por_falta_de_dato"])

    # Mismo cero, pero aquí sí es una afirmación sobre el catálogo.
    assert int(tabla.loc["NIKE", "n_brand_con_color"]) == 2
    assert int(tabla.loc["NIKE", "n_brand_y_color"]) == 0
    assert not bool(tabla.loc["NIKE", "cero_por_falta_de_dato"])


def test_la_marca_se_filtra_por_la_clave_normalizada_no_por_la_cruda():
    """Regresión. La celda original comparaba `brand` cruda con igualdad, que es
    justo lo que B.4 demuestra que no se puede hacer: una variante de escritura
    en el CSV recortaba el denominador en silencio."""
    catalogo = pd.DataFrame([
        {"brand": "Einhell", "color": "Blanco"},
        {"brand": "EINHELL", "color": "Blanco"},
        {"brand": " einhell ", "color": "Negro"},
    ])

    fila = combined_filter_selectivity(
        catalogo, primary_values=["einhell"], secondary_values=["blanco"]
    ).iloc[0]

    assert int(fila["n_brand"]) == 3
    assert int(fila["n_brand_y_color"]) == 2


def test_el_color_se_busca_como_texto_y_no_como_expresion_regular():
    """Los colores salen del propio catálogo, así que pueden traer paréntesis.
    Sin escapar, `negro (black)` se interpretaría como grupo y no encontraría el
    valor que lo originó."""
    catalogo = pd.DataFrame([{"brand": "NIKE", "color": "Negro (Black)"}])

    fila = combined_filter_selectivity(
        catalogo, primary_values=["NIKE"], secondary_values=["negro (black)"]
    ).iloc[0]

    assert int(fila["n_brand_y_color"]) == 1


def test_el_cruce_combina_las_politicas_de_cada_campo():
    """La marca con igualdad (vocabulario cerrado) y el color con *contiene*
    (texto libre): el enunciado exige que la marca siga siendo exacta."""
    catalogo = pd.DataFrame([
        {"brand": "NIKE", "color": "Negro/Blanco"},
        {"brand": "NIKE Kids", "color": "Blanco"},
    ])

    fila = combined_filter_selectivity(
        catalogo, primary_values=["nike"], secondary_values=["blanco"]
    ).iloc[0]

    assert int(fila["n_brand"]) == 1          # "NIKE Kids" no es "NIKE"
    assert int(fila["n_brand_y_color"]) == 1  # pero "Negro/Blanco" sí es blanco


def test_el_cruce_avisa_de_un_campo_que_no_existe():
    catalogo = pd.DataFrame([{"brand": "NIKE"}])

    with pytest.raises(ValueError, match="no tiene la columna"):
        combined_filter_selectivity(
            catalogo, primary_values=["NIKE"], secondary_values=["blanco"]
        )
