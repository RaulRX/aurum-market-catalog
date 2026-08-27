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


# ══════════ NB05 · la puerta de entrada al sistema, contra el motor ══════════
# Lo que se protege aquí: que "no hay resultados" y "no he podido buscar" salgan
# por canales distintos, que el filtro no deje de filtrar en silencio, y que el
# resultado llegue a NB09 sin necesidad de traducirlo.

from aurum.busqueda import (  # noqa: E402
    BuscadorVectorial,
    BusquedaAgotada,
    MotorNoDisponible,
    resultado_desde_hit,
)
from aurum.motores import SearchHit  # noqa: E402

PAYLOAD = {
    "product_id": "B0818K237B",
    "title": "Taladro percutor 24V",
    "brand": "Einhell",
    "brand_normalized": "einhell",
    "color": "",
}


def _hit(record_id="r-1", score=0.83, rank=1, payload=None, **kwargs):
    return SearchHit(
        record_id=record_id, score=score, rank=rank,
        payload=dict(PAYLOAD if payload is None else payload), **kwargs,
    )


class MotorFalso:
    """Un motor en memoria que devuelve lo que se le diga, o lanza lo que se le diga."""

    collection = "aurum_catalogo__test"

    def __init__(self, hits=(), error=None):
        self.hits, self.error = list(hits), error
        self.ultima_llamada = None

    def search(self, vector, *, top_k, filters=()):
        self.ultima_llamada = {"vector": vector, "top_k": top_k, "filters": tuple(filters)}
        if self.error is not None:
            raise self.error
        return self.hits[:top_k]


def _buscador(motor, **kwargs):
    return BuscadorVectorial(motor, lambda texto: [1.0, 0.0], **kwargs)


# ─────────────────── el tipo: hereda, luego no hay que traducir ──────────────


def test_un_resultado_es_un_searchresult():
    """Es el motivo de heredar: `evaluacion.py`, las gráficas y la tabla de NB09
    consumen `SearchResult` y aceptan un `Resultado` sin convertir nada."""
    assert isinstance(resultado_desde_hit(_hit()), SearchResult)


def test_el_resultado_es_inmutable_y_no_comparte_el_payload_del_motor():
    """`frozen` protege la asignación, no el contenido. Sin copiar el payload,
    recorrer los resultados podría modificar el diccionario del motor."""
    payload = dict(PAYLOAD)
    resultado = resultado_desde_hit(_hit(payload=payload))

    with pytest.raises(Exception):
        resultado.score = 0.99                       # frozen
    with pytest.raises(TypeError):
        resultado.metadatos["brand"] = "otra"        # copiado y cerrado
    payload["brand"] = "cambiada"
    assert resultado.metadatos["brand"] == "Einhell"


def test_el_mapeo_saca_los_dos_identificadores_de_donde_toca():
    """`document_id` es el `product_id` -lo que juzgan los qrels y piden los
    CSV- y `record_id` es el id del punto. Son cosas distintas y las dos hacen
    falta."""
    resultado = resultado_desde_hit(_hit(record_id="uuid-42"))

    assert resultado.document_id == "B0818K237B"
    assert resultado.record_id == "uuid-42"
    assert resultado.titulo == "Taladro percutor 24V"
    assert resultado.metadatos["brand"] == "Einhell"


def test_sin_product_id_el_mapeo_falla_en_vez_de_dejarlo_vacio():
    """Sin él no hay entrega: los dos CSV de salida se identifican con ese
    campo. Un hueco aquí se descubriría en NB09, con el índice ya construido."""
    with pytest.raises(ValueError, match="product_id"):
        resultado_desde_hit(_hit(payload={"title": "Sin id"}))


def test_un_titulo_ausente_si_puede_quedarse_vacio():
    """D14 dejó los huecos como cadena vacía; el título no rompe nada."""
    assert resultado_desde_hit(_hit(payload={"product_id": "B1"})).titulo == ""


def test_la_semantica_del_score_viene_del_motor_y_no_se_supone():
    """§3.2 prohíbe aplanar distancia y similitud. Weaviate devuelve distancia."""
    hit = _hit(score_kind="distance", higher_is_better=False)

    assert resultado_desde_hit(hit).score_es_similitud is False


def test_las_posiciones_van_de_uno_a_k_sin_huecos():
    motor = MotorFalso([_hit("r-1", 0.9, 1), _hit("r-2", 0.8, 2), _hit("r-3", 0.7, 3)])

    resultados = _buscador(motor).buscar("taladro")

    assert [r.rank for r in resultados] == [1, 2, 3]


# ─────────── los casos borde que §3.3 exige tratar explícitamente ────────────


def test_una_coleccion_vacia_devuelve_lista_vacia():
    """Decisión de NB05: a nivel de usuario, si no se encuentra nada, no aparece
    nada. No es un error."""
    assert _buscador(MotorFalso([])).buscar("taladro") == ()


def test_un_filtro_sin_resultados_devuelve_lista_vacia_nunca_excepcion():
    """Esta no se decide: la impone el enunciado."""
    assert _buscador(MotorFalso([])).buscar("taladro", marca="Einhell") == ()


def test_pedir_mas_de_lo_que_hay_devuelve_lo_que_hay():
    motor = MotorFalso([_hit("r-1"), _hit("r-2", rank=2)])

    assert len(_buscador(motor).buscar("taladro", top_k=50)) == 2


def test_el_motor_caido_levanta_una_excepcion_tipada_con_contexto():
    """NB04 midió que Qdrant sube un error de gRPC, cuyo tipo depende del
    transporte. Se clasifica por lo que el error dice, no por su clase."""
    class InactiveRpcError(Exception):
        __module__ = "grpc._channel"

        def code(self):
            return "StatusCode.UNAVAILABLE"

    with pytest.raises(MotorNoDisponible, match="no respondió"):
        _buscador(MotorFalso(error=InactiveRpcError("failed to connect"))).buscar("x")


def test_el_timeout_se_distingue_del_motor_caido():
    """Un motor que no está exige revisar el despliegue; uno que tarda de más,
    la consulta o la carga. Tratarlos igual borra esa diferencia."""
    class Deadline(Exception):
        def code(self):
            return "StatusCode.DEADLINE_EXCEEDED"

    with pytest.raises(BusquedaAgotada, match="30 s"):
        _buscador(MotorFalso(error=Deadline("deadline")), timeout_s=30).buscar("x")


def test_el_timeout_tambien_se_reconoce_por_el_mensaje():
    """Por REST no hay `code()`: el cliente sube un error de httpx."""
    with pytest.raises(BusquedaAgotada):
        _buscador(MotorFalso(error=RuntimeError("Read timed out"))).buscar("x")


# ────────────────────────── validación de entrada ────────────────────────────


@pytest.mark.parametrize("consulta", ["", "   "])
def test_una_consulta_vacia_es_un_error_no_una_busqueda_sin_resultados(consulta):
    with pytest.raises(ValueError, match="vacía"):
        _buscador(MotorFalso([_hit()])).buscar(consulta)


@pytest.mark.parametrize("k", [0, -1, 2.5, True])
def test_un_top_k_absurdo_se_rechaza(k):
    with pytest.raises(ValueError, match="top_k"):
        _buscador(MotorFalso([_hit()])).buscar("taladro", top_k=k)


def test_el_buscador_cumple_el_protocolo_retriever():
    """Para que NB09 lo meta en la misma tabla que el baseline léxico."""
    motor = MotorFalso([_hit()])
    buscador = _buscador(motor)

    resultados = buscador.search("taladro", k=1)

    assert isinstance(resultados[0], SearchResult) and buscador.name
    assert motor.ultima_llamada["top_k"] == 1


# ──────── los instrumentos de medida de NB05, probados sin motor ─────────────

from aurum.busqueda import (  # noqa: E402
    auditar_casos_borde,
    auditar_filtro_de_marca,
    comparar_con_post_filtro,
)

CASOS = [{
    "workload_id": "FILTER-001",
    "query_text": "herramienta inalambrica para perforar",
    "filter_value": "Einhell",
}]


def _hits_de(marca, n, empieza=1):
    return [
        SearchHit(
            record_id=f"r-{i}", score=1.0 - i / 100, rank=i,
            payload={
                "product_id": f"B{i}", "title": f"producto {i}",
                "brand": marca, "brand_normalized": marca.lower(),
            },
        )
        for i in range(empieza, empieza + n)
    ]


def test_la_pureza_completa_se_reconoce_como_tal():
    motor = MotorFalso(_hits_de("Einhell", 10))

    fila = auditar_filtro_de_marca(
        _buscador(motor), CASOS, alcance={"Einhell": 30}
    ).iloc[0]

    assert fila["pureza"] == "100%" and fila["de_la_marca"] == 10
    assert "pureza 100 %" in fila["veredicto"]


def test_una_sola_marca_intrusa_rompe_la_pureza():
    """§8: las consultas filtradas nunca devuelven otra marca."""
    motor = MotorFalso(_hits_de("Einhell", 9) + _hits_de("Bosch", 1, empieza=90))

    fila = auditar_filtro_de_marca(
        _buscador(motor), CASOS, alcance={"Einhell": 30}
    ).iloc[0]

    assert "contamina" in fila["veredicto"] and fila["pureza"] == "90%"


def test_un_filtro_roto_no_se_cuela_como_pureza_perfecta():
    """El fallo que esta tabla existe para cazar: cero resultados cumple la
    pureza de forma vacía, y sin el oráculo se leería como un ✅."""
    fila = auditar_filtro_de_marca(
        _buscador(MotorFalso([])), CASOS, alcance={"Einhell": 30}
    ).iloc[0]

    assert "FILTRO ROTO" in fila["veredicto"] and fila["pureza"] == "—"


def test_un_cero_legitimo_se_distingue_de_un_filtro_roto():
    """Si el catálogo tampoco tiene ninguno, el cero es la respuesta correcta."""
    fila = auditar_filtro_de_marca(
        _buscador(MotorFalso([])), CASOS, alcance={"Einhell": 0}
    ).iloc[0]

    assert "ausencia real" in fila["veredicto"]


def test_la_cobertura_corta_delata_un_post_filtro():
    """Con 30 productos en catálogo y top_k=10 deberían salir 10. Salen 3."""
    motor = MotorFalso(_hits_de("Einhell", 3))

    fila = auditar_filtro_de_marca(
        _buscador(motor), CASOS, alcance={"Einhell": 30}
    ).iloc[0]

    assert "cobertura corta" in fila["veredicto"]


def test_pocos_productos_en_catalogo_no_es_cobertura_corta():
    """Si la marca solo tiene 3 productos, devolver 3 es cobertura completa."""
    motor = MotorFalso(_hits_de("Einhell", 3))

    fila = auditar_filtro_de_marca(
        _buscador(motor), CASOS, alcance={"Einhell": 3}
    ).iloc[0]

    assert fila["veredicto"].startswith("✅")


def test_el_post_filtro_se_mide_contra_el_nativo_en_la_misma_tabla():
    """Un motor donde la marca es rara: de cada 10 candidatos sin filtrar, uno."""
    class MotorDiluido(MotorFalso):
        def search(self, vector, *, top_k, filters=()):
            super().search(vector, top_k=top_k, filters=filters)
            if filters:                       # el nativo los trae todos
                return _hits_de("Einhell", top_k)
            mezcla = []                       # sin filtrar, uno de cada diez
            for i in range(top_k):
                marca = "Einhell" if i % 10 == 0 else "Bosch"
                mezcla += _hits_de(marca, 1, empieza=i + 1)
            return mezcla

    tabla = comparar_con_post_filtro(
        _buscador(MotorDiluido()), "taladro", "Einhell", factores=(1, 10)
    )

    assert tabla.iloc[0]["llega_a_10"]                    # el nativo, sí
    assert not tabla.iloc[1]["llega_a_10"]                # ×1 sin filtrar, no
    assert tabla.iloc[2]["de_la_marca"] == 10             # ×10 hacen falta
    assert (tabla["ms"] >= 0).all()


def test_los_casos_borde_registran_tambien_lo_que_levanta():
    """En dos de los cuatro la respuesta correcta ES una excepción, así que la
    tabla tiene que poder anotarla en vez de romperse."""
    def revienta():
        raise MotorNoDisponible("el motor no respondió")

    tabla = auditar_casos_borde([
        ("sin resultados", "lista vacía", lambda: ()),
        ("motor caído", "excepción tipada", revienta),
    ])

    assert tabla.iloc[0]["observado"] == "0 resultados"
    assert "MotorNoDisponible" in tabla.iloc[1]["observado"]


def test_un_caso_que_revienta_no_interrumpe_a_los_demas():
    tabla = auditar_casos_borde([
        ("uno", "algo", lambda: 1 / 0),
        ("dos", "algo", lambda: ("a", "b")),
    ])

    assert len(tabla) == 2 and tabla.iloc[1]["observado"] == "2 resultados"


def test_un_caso_borde_se_prueba_con_varias_frases():
    """Una sola consulta no distingue "la colección está vacía" de "esa consulta
    no casaba con nada": la frase tiene que ir en la tabla y ser varias."""
    tabla = auditar_casos_borde([
        ("colección vacía", "taladro sin cable", "lista vacía", lambda: ()),
        ("colección vacía", "lentejas sin gluten", "lista vacía", lambda: ()),
    ])

    assert list(tabla["consulta"]) == ["taladro sin cable", "lentejas sin gluten"]
    assert set(tabla["caso"]) == {"colección vacía"}


def test_los_casos_borde_ensenan_los_ids_que_salieron():
    """"3 resultados" hay que creérselo; con los `product_id` delante, no."""
    devueltos = _buscador(MotorFalso(_hits_de("Einhell", 3))).buscar("taladro")

    fila = auditar_casos_borde([
        ("top_k mayor que los puntos", "devuelve lo que haya", lambda: devueltos),
    ]).iloc[0]

    assert "B1 (0.990)" in fila["lo_que_devolvio"] and "B3" in fila["lo_que_devolvio"]


def test_una_lista_vacia_se_ve_como_vacia_y_una_excepcion_como_excepcion():
    tabla = auditar_casos_borde([
        ("vacía", "lista vacía", lambda: ()),
        ("caído", "excepción", lambda: (_ for _ in ()).throw(MotorNoDisponible("no hay"))),
    ])

    assert tabla.iloc[0]["lo_que_devolvio"] == "(lista vacía)"
    assert "levantó" in tabla.iloc[1]["lo_que_devolvio"]


# ────────── enseñar lo que sale: las tablas que un corrector mira ────────────

from aurum.busqueda import (  # noqa: E402
    auditar_forma_de_los_resultados,
    auditar_post_filtro,
    solapamiento_entre_consultas,
    tabla_de_resultados,
)

CATALOGO = {f"B{i}" for i in range(1, 51)}


def test_la_tabla_de_resultados_lleva_una_fila_por_producto_recuperado():
    resultados = _buscador(MotorFalso(_hits_de("Einhell", 3))).buscar("taladro")

    tabla = tabla_de_resultados([("FILTER-001", "herramienta para perforar", resultados)])

    assert list(tabla["product_id"]) == ["B1", "B2", "B3"]
    assert list(tabla["posicion"]) == [1, 2, 3]
    assert set(tabla["marca"]) == {"Einhell"}


def test_una_consulta_sin_resultados_no_desaparece_de_la_tabla():
    """Ver un cero y no ver nada no son lo mismo: la fila lo dice."""
    tabla = tabla_de_resultados([("FILTER-009", "marca inexistente", ())])

    assert len(tabla) == 1 and tabla.iloc[0]["product_id"] == "(sin resultados)"


def test_la_columna_del_score_dice_en_que_direccion_se_lee():
    """Weaviate devolvería distancia: la columna tiene que cambiar de nombre,
    no seguir llamándose `score` y leerse al revés sin avisar."""
    distancias = _buscador(MotorFalso([
        _hit("r-1", 0.10, 1, score_kind="distance", higher_is_better=False),
    ])).buscar("taladro")

    tabla = tabla_de_resultados([("EVAL-1", "taladro", distancias)])

    assert "score_menor_mejor" in tabla.columns


def test_la_tabla_no_mezcla_similitudes_con_distancias():
    """§3.2: en una sola columna se leerían en direcciones opuestas."""
    similitud = _buscador(MotorFalso([_hit("r-1", 0.9, 1)])).buscar("taladro")
    distancia = _buscador(MotorFalso([
        _hit("r-2", 0.1, 1, score_kind="distance", higher_is_better=False),
    ])).buscar("taladro")

    with pytest.raises(ValueError, match="direcciones opuestas"):
        tabla_de_resultados([("a", "a", similitud), ("b", "b", distancia)])


def test_el_solapamiento_compara_las_formulaciones_dos_a_dos():
    """La misma necesidad escrita de tres formas: cuánto se mueve el top-k."""
    directa = _buscador(MotorFalso(_hits_de("Einhell", 4))).buscar("taladro 24v")
    semantica = _buscador(MotorFalso(_hits_de("Einhell", 4, empieza=3))).buscar(
        "herramienta inalámbrica para perforar"
    )

    fila = solapamiento_entre_consultas([
        ("EVAL-direct", "taladro 24v", directa),
        ("EVAL-semantic", "herramienta inalámbrica para perforar", semantica),
    ]).iloc[0]

    assert fila["en_comun"] == 2 and fila["de"] == 4        # B3 y B4
    assert fila["solapamiento"] == "50 %" and not fila["mismo_primero"]


def test_la_forma_ensena_el_numero_y_no_un_booleano():
    """`k_respetado = True` no deja ver si vinieron 10 o 3; la cadena sí."""
    fila = auditar_forma_de_los_resultados(
        _buscador(MotorFalso(_hits_de("Einhell", 10))),
        CASOS, ids_del_catalogo=CATALOGO, n_puntos=15000, alcance={"Einhell": 30},
    ).iloc[0]

    assert fila["devueltos"] == "10 de 10 posibles (pedidos 10)"
    assert fila["ids_distintos"] == "10 de 10"
    assert fila["score_primero_ultimo"] == "0.9900 → 0.9000"
    assert fila["veredicto"].startswith("✅")


def test_una_marca_con_menos_de_diez_no_se_lee_como_fallo():
    """3 de 3 disponibles es correcto; un `k_respetado=False` habría mentido."""
    fila = auditar_forma_de_los_resultados(
        _buscador(MotorFalso(_hits_de("Einhell", 3))),
        CASOS, ids_del_catalogo=CATALOGO, n_puntos=15000, alcance={"Einhell": 3},
    ).iloc[0]

    assert fila["devueltos"] == "3 de 3 posibles (pedidos 10)"
    assert fila["veredicto"].startswith("✅")


def test_la_forma_caza_un_id_que_no_esta_en_el_catalogo():
    fila = auditar_forma_de_los_resultados(
        _buscador(MotorFalso(_hits_de("Einhell", 2, empieza=99))),
        CASOS, ids_del_catalogo=CATALOGO, n_puntos=15000, alcance={"Einhell": 30},
    ).iloc[0]

    assert "B99" in fila["fuera_del_catalogo"] and fila["veredicto"].startswith("❌")


def test_la_forma_caza_el_orden_invertido():
    """Scores ascendentes declarándose similitud: el ranking está del revés."""
    motor = MotorFalso(list(reversed(_hits_de("Einhell", 2))))   # 0.98 y luego 0.99

    fila = auditar_forma_de_los_resultados(
        _buscador(motor), CASOS,
        ids_del_catalogo=CATALOGO, n_puntos=15000, alcance={"Einhell": 30},
    ).iloc[0]

    assert "al revés" in fila["orden"] and "orden" in fila["veredicto"]


def test_el_post_filtro_se_mide_en_las_cuatro_consultas_no_en_una():
    """Con una sola marca la conclusión depende de cuál se eligiera."""
    casos = [
        {"workload_id": "FILTER-001", "query_text": "perforar", "filter_value": "Einhell"},
        {"workload_id": "FILTER-002", "query_text": "tableta", "filter_value": "Apple"},
    ]

    tabla = auditar_post_filtro(
        _buscador(MotorFalso(_hits_de("Einhell", 10))), casos,
        alcance={"Einhell": 30, "Apple": 400}, factores=(1, 10), n_catalogo=15000,
    )

    assert list(tabla["caso"].unique()) == ["FILTER-001", "FILTER-002"]
    assert len(tabla) == 2 * 3                       # nativo + dos factores
    assert list(tabla["pct_del_catalogo"].unique()) == ["0.20 %", "2.67 %"]
    assert (tabla["descartados"] >= 0).all()
