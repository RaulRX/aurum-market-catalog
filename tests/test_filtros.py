"""Pruebas del filtro de marca (§6: "pruebas minimas de ... filtros").

El enunciado exige este fichero por nombre, y lo que protege es la mitad del
filtro que no se puede comprobar contra el motor: que la condicion **viaje
nativa dentro de la consulta** en vez de aplicarse despues en Python, y que una
entrada mal formada no lo convierta en un filtro que no filtra.

La otra mitad -que las cuatro consultas del §5 devuelvan solo la marca pedida-
se mide contra Qdrant en el notebook, porque solo el motor puede demostrar que
filtro de verdad.
"""
import pytest

from aurum.busqueda import (
    BuscadorVectorial,
    auditar_variantes_de_marca,
    variantes_de_escritura,
)
from aurum.motores import SearchHit

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
    """Motor en memoria que anota con que filtros se le llamo."""

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


def test_el_filtro_viaja_dentro_de_la_consulta_y_no_se_aplica_despues():
    """Es la diferencia que el enunciado exige: `Einhell` son 30 productos de
    15.000 (0,2 %), asi que recuperar sin filtrar y descartar en Python se
    quedaria sin candidatos. El filtro tiene que salir en la llamada."""
    motor = MotorFalso([_hit()])

    _buscador(motor).buscar("herramienta inalambrica para perforar", marca="Einhell")

    assert motor.ultima_llamada["filters"], "el filtro no llego al motor"
    assert motor.ultima_llamada["top_k"] == 10, "no se sobre-recupera para filtrar luego"


def test_un_filtro_sin_resultados_devuelve_lista_vacia_nunca_excepcion():
    """Lo impone el enunciado; no es una decision del proyecto."""
    assert _buscador(MotorFalso([])).buscar("taladro", marca="Einhell") == ()


def test_la_marca_se_normaliza_al_buscar_como_decidio_d03():
    """El valor se guarda crudo y se normaliza al buscar: quien llama pasa
    `Einhell` y el motor recibe `einhell`."""
    motor = MotorFalso([_hit()])

    _buscador(motor).buscar("taladro", marca="Einhell")

    (condicion,) = motor.ultima_llamada["filters"]
    assert (condicion.field, condicion.value, condicion.operator) == (
        "brand", "einhell", "equals",
    )


def test_una_marca_en_blanco_es_entrada_invalida_y_no_un_filtro():
    """D14 dejó los huecos como cadena vacía: un filtro construido sin validar
    dejaría de filtrar en silencio, o devolvería justo los productos sin marca."""
    for vacia in ("", "   "):
        with pytest.raises(ValueError, match="en blanco"):
            _buscador(MotorFalso([_hit()])).buscar("taladro", marca=vacia)


def test_sin_marca_no_se_manda_ningun_filtro():
    motor = MotorFalso([_hit()])

    _buscador(motor).buscar("taladro")

    assert motor.ultima_llamada["filters"] == ()


def test_el_color_no_se_expone_en_la_interfaz():
    """Decisión de NB05: el almacén sabe filtrarlo, la interfaz no lo ofrece."""
    with pytest.raises(TypeError):
        _buscador(MotorFalso()).buscar("taladro", color="negro")


# ───────── la marca escrita como la escribiría un usuario, no el CSV ─────────
# El catalogo guarda `NIKE` y `SAMSUNG` en mayusculas, y nadie busca asi. D03
# decidio guardar el valor crudo y normalizar al buscar; lo que sigue protege
# esa mitad: que la escritura no cambie el resultado, y que normalizar no sea
# lo mismo que corregir faltas.

CASO_NIKE = [{"workload_id": "FILTER-003", "query_text": "zapatillas", "filter_value": "NIKE"}]


class MotorPorMarca(MotorFalso):
    """Solo devuelve productos si el filtro trae la marca que espera."""

    def __init__(self, marca_normalizada, n=10, ignora_el_filtro=False):
        super().__init__()
        self.esperada, self.n = marca_normalizada, n
        self.ignora_el_filtro = ignora_el_filtro

    def search(self, vector, *, top_k, filters=()):
        super().search(vector, top_k=top_k, filters=filters)
        casa = self.ignora_el_filtro or any(c.value == self.esperada for c in filters)
        if not casa:
            return []
        return [
            _hit(
                record_id=f"r-{i}", score=1.0 - i / 100, rank=i,
                payload={**PAYLOAD, "product_id": f"B{i}",
                         "brand": "NIKE", "brand_normalized": "nike"},
            )
            for i in range(1, min(top_k, self.n) + 1)
        ]


def test_las_variantes_cubren_caja_acentos_y_espacios():
    variantes = {etiqueta: valor for etiqueta, valor, _ in variantes_de_escritura("NIKE")}

    assert variantes["todo en minúsculas"] == "nike"
    assert variantes["Capitalizada"] == "Nike"
    assert variantes["con espacios alrededor"] == "  NIKE "
    assert variantes["con un acento colado"] == "NÍKE"


def test_las_dos_ultimas_variantes_son_faltas_que_no_deben_casar():
    """Normalizar iguala la caja y los acentos; no corrige faltas. Sin estas dos
    filas, una tabla en la que todo devuelve diez no distinguiría un filtro que
    normaliza de uno que casa con cualquier cosa."""
    deben_fallar = {v for _, v, casa in variantes_de_escritura("Einhell") if not casa}

    assert deben_fallar == {"Ei nhell", "Einhel"}


def test_no_se_repite_una_variante_que_coincide_con_otra():
    """Para `Apple`, "capitalizada" es la misma cadena que la del CSV."""
    valores = [valor for _, valor, _ in variantes_de_escritura("Apple")]

    assert len(valores) == len(set(valores))


def test_todas_las_escrituras_de_la_marca_devuelven_lo_mismo():
    """La prueba realista: `nike`, `Nike`, `NÍKE` y `  NIKE ` son la misma marca."""
    tabla = auditar_variantes_de_marca(
        _buscador(MotorPorMarca("nike")), CASO_NIKE
    )
    equivalentes = tabla[tabla["esperado"].str.startswith("los mismos")]

    assert (equivalentes["veredicto"] == "✅ idéntica a la canónica").all()
    assert (tabla[~tabla.index.isin(equivalentes.index)]["n_resultados"] == 0).all()


def test_la_tabla_ensena_el_valor_que_llega_al_motor():
    """Es donde ocurre D03: el usuario escribe `  NIKE ` y la base recibe `nike`."""
    tabla = auditar_variantes_de_marca(_buscador(MotorPorMarca("nike")), CASO_NIKE)
    fila = tabla[tabla["variante"] == "con espacios alrededor"].iloc[0]

    assert fila["marca_pedida"] == "'  NIKE '" and fila["viaja_al_motor"] == "'nike'"


def test_una_normalizacion_incompleta_se_ve_en_la_variante_con_acento():
    """El fallo que esta tabla existe para cazar. Con `casefold` en vez de
    `unaccent`, `NÍKE` viaja como `níke`, el motor no lo reconoce y el usuario ve
    una lista vacía mientras la pureza del §8 sigue saliendo perfecta."""
    buscador = _buscador(MotorPorMarca("nike"), modo_normalizacion="casefold")

    tabla = auditar_variantes_de_marca(buscador, CASO_NIKE)
    fila = tabla[tabla["variante"] == "con un acento colado"].iloc[0]

    assert fila["n_resultados"] == 0
    assert "no la reconoce" in fila["veredicto"]


def test_un_motor_que_ignora_el_filtro_se_delata_en_las_faltas():
    """Si el adaptador se comiera la condición, todas las variantes devolverían
    diez —incluidas las mal escritas—, y eso es lo que marca la fila en rojo."""
    buscador = _buscador(MotorPorMarca("nike", ignora_el_filtro=True))

    tabla = auditar_variantes_de_marca(buscador, CASO_NIKE)
    faltas = tabla[tabla["esperado"].str.startswith("0:")]

    assert faltas["veredicto"].str.startswith("❌ casa").all()
