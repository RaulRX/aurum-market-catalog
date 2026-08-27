"""Tests del puerto y del guion de humo (`aurum.motores`).

Lo que protegen: que el guion detecte lo que dice detectar. Un motor falso en
memoria hace de doble, y encima de él se montan las averías que la prueba tiene
que cazar —ingesta que duplica, filtro que no filtra, borrado que no borra— para
comprobar que el paso correspondiente falla en vez de dar por bueno el motor.

No hay Docker aquí a propósito: los adaptadores reales se prueban contra sus
contenedores en el notebook, y eso lo ejecuta el usuario.
"""
from __future__ import annotations

from collections.abc import Sequence

import pytest

from aurum.motores import (
    FilterCondition,
    FilterProbe,
    Point,
    SearchHit,
    Snapshot,
    UnsupportedFilterError,
    VectorStore,
    audited_key,
    contains_level,
    ensure_reset_allowed,
    error_quality,
    guard_collection_name,
    load_smoke,
    parse_docker_stats,
    parse_volume_sizes,
    persistence_check,
    probe_filters,
    read_snapshot,
    record_manual,
    reset_allowed,
    resource_note,
    resource_row,
    resource_table,
    run_smoke_test,
    save_smoke,
    smoke_differences,
    smoke_table,
)

DIM = 4
CONSULTA = [1.0, 0.0, 0.0, 0.0]

PUNTOS = [
    Point("r-1", [1.0, 0.0, 0.0, 0.0], {"brand": "Einhell", "color": "negro/rojo"}),
    Point("r-2", [0.9, 0.1, 0.0, 0.0], {"brand": "Einhell", "color": "blanco"}),
    Point("r-3", [0.0, 1.0, 0.0, 0.0], {"brand": "NIKE", "color": "negro"}),
    Point("r-4", [0.0, 0.0, 1.0, 0.0], {"brand": "Apple", "color": None}),
]


class FakeStore:
    """Motor en memoria que cumple el puerto. El doble contra el que se prueba
    el guion, no un motor de mentira que se parezca a uno real."""

    name = "fake"

    def __init__(self, *, soporta_contains: bool = True) -> None:
        self.soporta_contains = soporta_contains
        self.puntos: dict[str, Point] = {}
        self.lotes: list[int] = []
        self.cerrado = False

    def server_version(self) -> str:
        return "0.0.0-fake"

    def create_collection(self, *, dim: int, metric: str, recreate: bool = False) -> None:
        self.dim, self.metric = dim, metric
        if recreate:
            self.puntos.clear()

    def upsert(self, points: Sequence[Point], *, batch_size: int) -> int:
        for inicio in range(0, len(points), batch_size):
            lote = points[inicio:inicio + batch_size]
            self.lotes.append(len(lote))
            for punto in lote:
                self.puntos[punto.record_id] = punto   # idempotente por id
        return len(points)

    def count(self) -> int:
        return len(self.puntos)

    def search(self, vector, *, top_k, filters=()) -> list[SearchHit]:
        if any(c.operator == "contains" for c in filters) and not self.soporta_contains:
            raise UnsupportedFilterError("solo admite igualdad sobre metadatos")
        candidatos = [p for p in self.puntos.values() if all(self._casa(p, c) for c in filters)]
        candidatos.sort(key=lambda p: -sum(a * b for a, b in zip(p.vector, vector)))
        # Por nombre y no por posición: al añadir `score_kind` y `rank` en medio,
        # la construcción posicional metía el payload en el campo equivocado sin
        # que ningún tipo se quejara.
        return [
            SearchHit(
                record_id=p.record_id,
                score=sum(a * b for a, b in zip(p.vector, vector)),
                score_kind="similarity",
                higher_is_better=True,
                rank=posicion,
                payload=p.payload,
            )
            for posicion, p in enumerate(candidatos[:top_k], start=1)
        ]

    @staticmethod
    def _casa(punto: Point, condicion: FilterCondition) -> bool:
        # Filtra contra la clave derivada si el punto la lleva, igual que hacen
        # los tres adaptadores reales (`f"{field}_normalized"`). Antes miraba el
        # campo crudo, y ese era el agujero: el doble se comportaba distinto que
        # los motores, así que ningún test podía destapar el fallo de auditoría.
        valor = punto.payload.get(audited_key(punto.payload, condicion.field))
        if valor is None:
            return False
        return (
            str(valor) == condicion.value if condicion.operator == "equals"
            else condicion.value in str(valor)
        )

    def get(self, record_id: str) -> Point | None:
        return self.puntos.get(record_id)

    def delete(self, record_id: str) -> None:
        self.puntos.pop(record_id, None)

    def close(self) -> None:
        self.cerrado = True


def ejecutar(store, **kwargs):
    resultados = run_smoke_test(
        store, PUNTOS, query_vector=CONSULTA, dim=DIM, top_k=2, batch_size=2, **kwargs
    )
    return {r.step: r for r in resultados}


# ───────────────────────────── el puerto ─────────────────────────────────────


def test_el_motor_falso_cumple_el_puerto():
    """Si el doble deja de encajar en el protocolo, el guion no prueba nada."""
    assert isinstance(FakeStore(), VectorStore)


def test_la_condicion_rechaza_un_operador_inventado():
    with pytest.raises(ValueError, match="operator debe ser"):
        FilterCondition(field="brand", value="Einhell", operator="parecido")


@pytest.mark.parametrize("nombre", ["aurum_humo_qdrant", "AURUM-HUMO-milvus", "aurum_humo"])
def test_el_guardia_deja_pasar_las_colecciones_de_la_prueba(nombre):
    assert guard_collection_name(nombre) == nombre


@pytest.mark.parametrize("nombre", ["catalogo_productos", "humo_aurum", "", "aurum"])
def test_el_guardia_bloquea_lo_que_no_es_de_la_prueba(nombre):
    """El guion borra y recrea en cada pasada; sin este freno una errata en el
    nombre convierte esa comodidad en un borrado de la colección equivocada."""
    with pytest.raises(ValueError, match="prefijo protegido"):
        guard_collection_name(nombre)


def test_el_reset_esta_desactivado_si_la_variable_no_existe(monkeypatch):
    """Copiado de S03_ALLOW_RESET: la ausencia de la variable no habilita nada.
    Un borrado no puede depender de que alguien se acordara de prohibirlo."""
    monkeypatch.delenv("AURUM_ALLOW_RESET", raising=False)

    assert reset_allowed() is False
    with pytest.raises(PermissionError, match="AURUM_ALLOW_RESET"):
        ensure_reset_allowed("aurum_humo_qdrant")


@pytest.mark.parametrize("valor", ["true", "TRUE", " sí ", "1", "yes"])
def test_el_reset_se_activa_a_proposito(monkeypatch, valor):
    monkeypatch.setenv("AURUM_ALLOW_RESET", valor)

    assert reset_allowed() is True
    ensure_reset_allowed("aurum_humo_qdrant")   # no levanta


@pytest.mark.parametrize("valor", ["false", "", "0", "no", "quizá"])
def test_cualquier_otro_valor_deja_el_reset_apagado(monkeypatch, valor):
    """Un valor que no se reconoce apaga, no enciende: es la dirección segura."""
    monkeypatch.setenv("AURUM_ALLOW_RESET", valor)

    assert reset_allowed() is False


# ─────────────────────────── el guion completo ───────────────────────────────


def test_un_motor_correcto_pasa_los_siete_pasos():
    pasos = ejecutar(FakeStore(), filters=[FilterCondition("brand", "Einhell")])

    assert all(pasos[n].passed for n in range(1, 8)), {
        n: pasos[n].observed for n in range(1, 8) if not pasos[n].passed
    }


def test_los_tres_pasos_manuales_quedan_como_huecos():
    """Sin ellos la comparativa terminaría en el 7 y la persistencia —que es
    requisito del enunciado— desaparecería de la tabla sin que se note."""
    pasos = ejecutar(FakeStore())

    assert [pasos[n].passed for n in (8, 9, 10)] == [None, None, None]
    # El comando va en `action` -qué hay que ejecutar- y el criterio en
    # `expected` -qué tiene que salir-. Separarlos es lo que deja la fila
    # accionable sin volver al markdown de la sección.
    assert "make motor-down" in pasos[8].action
    assert "count()" in pasos[8].expected


def test_cada_paso_dice_con_que_llamada_lo_comprobo():
    """La tabla tiene que leerse sin abrir el código: que el paso 5 "pase" no
    dice contra qué filtro ni con qué top_k, y son justo los parámetros que
    deciden si el paso prueba algo."""
    pasos = ejecutar(FakeStore(), filters=[FilterCondition("brand", "Einhell")])

    assert "create_collection(dim=4" in pasos[1].action
    assert "batch_size=2" in pasos[2].action
    assert "LOS MISMOS" in pasos[3].action          # la repetición, explícita
    assert "filters=[]" in pasos[4].action          # la global, sin filtro
    assert "brand equals 'Einhell'" in pasos[5].action
    assert pasos[6].action.startswith("get(")
    assert pasos[7].action.startswith("delete(")


def test_la_auditoria_compara_contra_la_clave_normalizada_no_contra_la_cruda():
    """Regresión del caso real. D13 guarda `brand` crudo y `brand_normalized`
    derivado; los adaptadores filtran contra el segundo y el notebook pide el
    valor ya normalizado. Auditar contra el campo crudo compararía `"Einhell"`
    con `"einhell"`, daría siempre falso, y el paso 5 marcaría como roto un
    filtro perfectamente funcional en los TRES motores.

    Los tests anteriores no lo cazaban porque construían la condición con el
    valor sin normalizar: el doble coincidía con el crudo por casualidad."""
    store = FakeStore()
    store.puntos = {
        "r-1": Point("r-1", [1.0, 0.0, 0.0, 0.0],
                     {"brand": "Einhell", "brand_normalized": "einhell"}),
    }

    pasos = {
        r.step: r for r in run_smoke_test(
            store, list(store.puntos.values()), query_vector=CONSULTA, dim=DIM,
            top_k=1, batch_size=1, recreate=False,
            # El valor va normalizado, como lo manda la celda F.2 del notebook.
            filters=[FilterCondition("brand", "einhell")],
        )
    }

    assert pasos[5].passed
    assert "1 cumplen" in pasos[5].observed
    # La clave auditada va en la fila, no solo en el código: es lo que distingue
    # "el filtro falló" de "comparé la clave equivocada".
    assert pasos[5].detail["claves_auditadas"] == {"brand": "brand_normalized"}
    assert "auditado contra brand_normalized = einhell" in pasos[5].observed
    assert "el motor filtra por brand_normalized" in pasos[5].action


def test_sin_clave_derivada_la_auditoria_usa_el_campo_crudo():
    """Un punto que no la lleve -otro esquema, o un motor que no la devuelva- no
    puede quedarse sin auditar: se cae al campo crudo, que es lo único que hay."""
    assert audited_key({"brand": "Einhell"}, "brand") == "brand"
    assert audited_key({"brand": "Einhell", "brand_normalized": None}, "brand") == "brand"
    assert audited_key(
        {"brand": "Einhell", "brand_normalized": "einhell"}, "brand"
    ) == "brand_normalized"


# ───── la tabla de filtros: las 4 consultas + el `contains` que falta probar ──


COLORIDOS = [
    Point("c-1", [1.0, 0.0, 0.0, 0.0],
          {"brand": "Einhell", "brand_normalized": "einhell",
           "color": "Negro", "color_normalized": "negro"}),
    Point("c-2", [0.9, 0.1, 0.0, 0.0],
          {"brand": "Einhell", "brand_normalized": "einhell",
           "color": "Negro/Rojo", "color_normalized": "negro/rojo"}),
    Point("c-3", [0.0, 1.0, 0.0, 0.0],
          {"brand": "NIKE", "brand_normalized": "nike",
           "color": "Blanco", "color_normalized": "blanco"}),
]


def _tienda_con_colores(*, subcadena: bool):
    """Un doble que resuelve `contains` como subcadena literal o por palabras.

    Es la diferencia que separa el nivel 3 del nivel 2, y modelarla es lo único
    que permite comprobar que `contains_level` la detecta.
    """
    class Tienda(FakeStore):
        def _casa(self, punto, condicion):   # type: ignore[override]
            valor = punto.payload.get(audited_key(punto.payload, condicion.field))
            if valor is None:
                return False
            if condicion.operator == "equals":
                return str(valor) == condicion.value
            if subcadena:
                return condicion.value in str(valor)
            # Por palabras: el valor se parte en tokens y se compara entero.
            import re
            return condicion.value in re.split(r"[^0-9a-z]+", str(valor))

    tienda = Tienda()
    tienda.puntos = {p.record_id: p for p in COLORIDOS}
    return tienda


def _sondas_de_color(valor="negro", fragmento="negr"):
    return [
        FilterProbe("color · igualdad", CONSULTA,
                    [FilterCondition("color", valor, "equals")], role="referencia"),
        FilterProbe("color · contiene palabra", CONSULTA,
                    [FilterCondition("color", valor, "contains")], role="palabra"),
        FilterProbe("color · contiene fragmento", CONSULTA,
                    [FilterCondition("color", fragmento, "contains")], role="fragmento"),
    ]


def test_la_tabla_cubre_las_cuatro_consultas_del_enunciado():
    """§5 pide "resultados que cumplan la marca en las CUATRO consultas", y el
    paso 5 del guion solo ejerce una. Sin esta tabla, tres quedarían sin medir."""
    store = _tienda_con_colores(subcadena=True)
    sondas = [
        FilterProbe(f"FILTER-00{i}", CONSULTA,
                    [FilterCondition("brand", marca, "equals")])
        for i, marca in enumerate(("einhell", "apple", "nike", "samsung"), start=1)
    ]

    tabla = probe_filters(store, sondas, top_k=5)

    assert list(tabla["caso"]) == ["FILTER-001", "FILTER-002", "FILTER-003", "FILTER-004"]
    assert (tabla["papel"] == "obligatorio").all()
    # einhell y nike existen en el doble; apple y samsung no. Una marca sin
    # resultados NO es un fallo del filtro -es que no hay productos-, y por eso
    # `todos_cumplen` mira lo devuelto, no que devuelva algo.
    assert bool(tabla.set_index("caso").loc["FILTER-001", "todos_cumplen"])
    assert int(tabla.set_index("caso").loc["FILTER-002", "n_resultados"]) == 0


def test_el_contains_por_palabras_se_distingue_del_de_subcadena():
    """El discriminador: `negr` es subcadena de `negro` pero no una palabra
    suelta. Es lo que separa el nivel 2 del 3 sin creerse la documentación."""
    por_palabras = probe_filters(_tienda_con_colores(subcadena=False), _sondas_de_color())
    subcadena = probe_filters(_tienda_con_colores(subcadena=True), _sondas_de_color())

    fragmento = lambda t: int(t.set_index("papel").loc["fragmento", "n_resultados"])
    assert fragmento(por_palabras) == 0     # no lo encuentra: tokeniza
    assert fragmento(subcadena) > 0         # lo encuentra: compara caracteres

    assert "NIVEL 2" in contains_level(por_palabras)
    assert "NIVEL 3" in contains_level(subcadena)


def test_el_contiene_alcanza_los_compuestos_que_la_igualdad_deja_fuera():
    """La razón por la que el color usa `contains`: `Negro/Rojo` también es
    negro. Medido contra el motor, no solo sobre el catálogo en la sección B."""
    tabla = probe_filters(_tienda_con_colores(subcadena=False), _sondas_de_color())
    por_papel = tabla.set_index("papel")["n_resultados"]

    assert int(por_papel["referencia"]) == 1   # solo "Negro"
    assert int(por_papel["palabra"]) == 2      # "Negro" y "Negro/Rojo"


def test_un_motor_sin_contains_queda_descartado_por_el_requisito_duro():
    store = _tienda_con_colores(subcadena=True)
    store.soporta_contains = False

    tabla = probe_filters(store, _sondas_de_color())

    assert "NO SOPORTADO" in tabla.set_index("papel").loc["palabra", "valores"]
    assert "NIVEL 1" in contains_level(tabla)
    # La tabla no se interrumpe: la fila de igualdad sigue midiéndose.
    assert int(tabla.set_index("papel").loc["referencia", "n_resultados"]) == 1


def test_el_oraculo_distingue_la_ausencia_real_del_filtro_roto():
    """Sin el oráculo, "0 resultados" no dice nada: puede ser que esa marca no
    tenga productos o que el filtro esté roto, y las dos cosas se leen igual.
    Es el mismo problema que resolvió `cero_por_falta_de_dato` en B.5."""
    store = _tienda_con_colores(subcadena=True)

    tabla = probe_filters(store, [
        # El catálogo tampoco tiene Apple: el cero del motor es correcto.
        FilterProbe("ausencia real", CONSULTA,
                    [FilterCondition("brand", "apple", "equals")], expected=0),
        # El catálogo tiene 2 Einhell y el motor no devuelve ninguno → roto.
        FilterProbe("filtro roto", CONSULTA,
                    [FilterCondition("brand", "inexistente", "equals")], expected=2),
        # Coincide con el catálogo.
        FilterProbe("correcto", CONSULTA,
                    [FilterCondition("brand", "einhell", "equals")], expected=2),
    ], top_k=10).set_index("caso")

    assert "ausencia real" in tabla.loc["ausencia real", "veredicto"]
    assert "FILTRO ROTO" in tabla.loc["filtro roto", "veredicto"]
    assert "coincide con el catálogo" in tabla.loc["correcto", "veredicto"]


def test_el_oraculo_topa_en_top_k_y_no_lo_cuenta_como_fallo():
    """El motor devuelve como mucho `top_k`. Comparar contra el total del
    catálogo sin topar marcaría como roto un filtro perfecto."""
    tabla = probe_filters(_tienda_con_colores(subcadena=True), [
        FilterProbe("einhell", CONSULTA,
                    [FilterCondition("brand", "einhell", "equals")], expected=500),
    ], top_k=2).set_index("caso")

    assert "coincide" in tabla.loc["einhell", "veredicto"]


def test_la_sonda_de_fragmento_clasifica_y_nunca_suspende():
    """Regresión. La sonda `fragmento` no aprueba ni suspende: sus dos resultados
    son correctos y dicen cosas distintas. Un 0 significa que el motor tokeniza
    -nivel 2, el mejor- y marcarlo como FILTRO ROTO acusaría al motor justo
    cuando se comporta como queremos. Fue exactamente lo que pasó con Qdrant."""
    por_palabras = probe_filters(_tienda_con_colores(subcadena=False), [
        FilterProbe("fragmento", CONSULTA,
                    [FilterCondition("color", "negr", "contains")],
                    role="fragmento", expected=2),
    ], top_k=10).iloc[0]

    assert "FILTRO ROTO" not in por_palabras["veredicto"]
    assert "nivel 2" in por_palabras["veredicto"]

    subcadena = probe_filters(_tienda_con_colores(subcadena=True), [
        FilterProbe("fragmento", CONSULTA,
                    [FilterCondition("color", "negr", "contains")],
                    role="fragmento", expected=2),
    ], top_k=10).iloc[0]

    assert "nivel 3" in subcadena["veredicto"]


def test_devolver_de_menos_en_contains_no_se_sentencia_como_fallo():
    """El oráculo cuenta `contains` como SUBCADENA. Un motor que filtre por
    palabras devolverá menos, y eso no es un fallo: es justo la propiedad que lo
    hace mejor. La fila lo describe en vez de sentenciarlo."""
    parcial = probe_filters(_tienda_con_colores(subcadena=False), [
        FilterProbe("palabra", CONSULTA,
                    [FilterCondition("color", "negro", "contains")],
                    role="palabra", expected=3),
    ], top_k=10).iloc[0]

    assert "legítimo si el motor filtra por palabras" in parcial["veredicto"]


def test_un_contains_de_palabra_a_cero_sigue_siendo_sospechoso():
    """La otra cara: si el motor no encuentra ni una palabra entera que el
    catálogo sí tiene, el índice de texto es el sospechoso — y eso sí hay que
    señalarlo, aunque sin llamarlo filtro roto sin más."""
    store = _tienda_con_colores(subcadena=False)
    fila = probe_filters(store, [
        FilterProbe("palabra", CONSULTA,
                    [FilterCondition("color", "turquesa", "contains")],
                    role="palabra", expected=7),
    ], top_k=10).iloc[0]

    assert "revisar el índice de texto" in fila["veredicto"]


def test_sin_oraculo_la_fila_lo_dice_en_vez_de_callarse():
    tabla = probe_filters(_tienda_con_colores(subcadena=True), [
        FilterProbe("sin oraculo", CONSULTA, [FilterCondition("brand", "einhell")]),
    ])

    assert tabla["n_en_catalogo"].iloc[0] is None
    assert "sin oráculo" in tabla["veredicto"].iloc[0]


def test_una_sonda_declara_su_papel_y_sus_condiciones():
    with pytest.raises(ValueError, match="role debe ser"):
        FilterProbe("x", CONSULTA, [FilterCondition("brand", "a")], role="inventado")
    with pytest.raises(ValueError, match="al menos una condición"):
        FilterProbe("x", CONSULTA, [])


def test_el_filtro_informa_de_los_valores_que_devolvio():
    """§8 exige que "las consultas filtradas nunca devuelvan otra marca". Ver los
    valores devueltos en la propia fila es lo que permite comprobarlo de un
    vistazo, sin ir a inspeccionar los resultados aparte."""
    pasos = ejecutar(FakeStore(), filters=[FilterCondition("brand", "Einhell")])

    # Estos puntos no llevan clave derivada, así que la auditoría cae al campo
    # crudo — y la fila lo dice, en vez de dejarlo a la imaginación.
    assert "auditado contra brand = Einhell" in pasos[5].observed


def test_la_ingesta_que_duplica_falla_en_el_paso_de_idempotencia():
    """La avería que el paso 3 existe para cazar: un motor que inserta en vez de
    sobrescribir da el count() correcto la primera vez y el doble la segunda."""
    class DuplicaStore(FakeStore):
        def upsert(self, points, *, batch_size):
            for i, punto in enumerate(points):
                self.puntos[f"{punto.record_id}#{len(self.puntos)}#{i}"] = punto
            return len(points)

    pasos = ejecutar(DuplicaStore())

    assert pasos[2].passed          # la primera ingesta cuadra
    assert not pasos[3].passed      # la segunda destapa el problema


def test_un_motor_sin_contains_falla_el_requisito_duro_sin_romper_el_guion():
    """El caso que decide D12: queda registrado como fallo del paso 5 y la
    prueba continúa, para que la fila del motor esté completa."""
    pasos = ejecutar(
        FakeStore(soporta_contains=False),
        filters=[FilterCondition("color", "negro", operator="contains")],
    )

    assert not pasos[5].passed
    assert "NO SOPORTADO" in pasos[5].observed
    assert pasos[5].detail["requisito_duro"] is True
    assert pasos[6].passed and pasos[7].passed   # el guion no se interrumpió


def test_un_filtro_que_devuelve_lo_que_no_cumple_se_detecta():
    """Un motor puede aceptar el filtro y aun así devolver resultados que no lo
    cumplen. El paso 5 audita el payload de lo devuelto, no se fía.

    El filtro es `Apple` y no `Einhell` a propósito: los dos vecinos más cercanos
    a la consulta son Einhell, así que con esa marca un filtro ignorado devolvería
    resultados que la cumplen igualmente y la trampa pasaría desapercibida. Es la
    condición que documenta `run_smoke_test`: el filtro tiene que dejar fuera lo
    que la búsqueda sin filtrar traería."""
    class FiltroFalsoStore(FakeStore):
        def search(self, vector, *, top_k, filters=()):
            return super().search(vector, top_k=top_k)   # ignora el filtro

    pasos = ejecutar(FiltroFalsoStore(), filters=[FilterCondition("brand", "Apple")])

    assert not pasos[5].passed
    assert "cumplen" in pasos[5].observed


def test_la_auditoria_del_filtro_no_ve_la_trampa_si_el_filtro_no_es_selectivo():
    """La cara B del test anterior, escrita para que la limitación no se
    descubra en la comparativa. Con una marca que ya domina el top-k, un motor
    que ignora el filtro pasa el paso 5: por eso el guion se ejecuta con
    `Einhell`, que es el 0,2 % del catálogo."""
    class FiltroFalsoStore(FakeStore):
        def search(self, vector, *, top_k, filters=()):
            return super().search(vector, top_k=top_k)

    pasos = ejecutar(FiltroFalsoStore(), filters=[FilterCondition("brand", "Einhell")])

    assert pasos[5].passed   # falso positivo conocido y acotado


def test_el_borrado_que_no_borra_se_detecta_desde_la_lectura():
    class NoBorraStore(FakeStore):
        def delete(self, record_id: str) -> None:
            pass

    pasos = ejecutar(NoBorraStore())

    assert not pasos[7].passed
    assert "sigue presente" in pasos[7].observed


def test_una_excepcion_del_sdk_no_tumba_la_prueba():
    """Con el motor a medias, los pasos siguientes tienen que seguir dando
    información en vez de dejar la fila a medio rellenar."""
    class RevientaAlBuscar(FakeStore):
        def search(self, vector, *, top_k, filters=()):
            raise ConnectionError("connection refused")

    pasos = ejecutar(RevientaAlBuscar())

    assert not pasos[4].passed
    assert "ConnectionError" in pasos[4].observed
    assert pasos[6].passed        # la lectura por id sí funciona y se registra


def test_la_ingesta_respeta_el_tamano_de_lote():
    """D15 se decide midiendo, así que el lote tiene que llegar al motor tal y
    como se pidió y no como el adaptador prefiera."""
    store = FakeStore()
    ejecutar(store)

    assert store.lotes == [2, 2, 2, 2]   # 4 puntos en lotes de 2, dos ingestas


# ─────────────────────────── validaciones de entrada ─────────────────────────


def test_la_dimension_declarada_tiene_que_cuadrar_con_la_consulta():
    """Comprobación barata que evita un fallo confuso dentro del SDK."""
    with pytest.raises(ValueError, match="dimensiones"):
        run_smoke_test(FakeStore(), PUNTOS, query_vector=[1.0, 0.0], dim=DIM)


def test_la_prueba_exige_puntos():
    with pytest.raises(ValueError, match="necesita puntos"):
        run_smoke_test(FakeStore(), [], query_vector=CONSULTA, dim=DIM)


# ───────── la semántica del score, que §3.2 prohíbe expresamente aplanar ─────


def test_el_resultado_declara_si_su_score_es_distancia_o_similitud():
    """Sin esto, el score de Weaviate -distancia, menor es mejor- acabaría en la
    misma columna que el de Qdrant -similitud, mayor es mejor- y la comparativa
    ordenaría al revés uno de los dos sin que nada fallara."""
    class PorDistanciaStore(FakeStore):
        def search(self, vector, *, top_k, filters=()):
            return [
                SearchHit(
                    record_id=h.record_id, score=h.score, score_kind="distance",
                    higher_is_better=False, rank=h.rank, payload=h.payload,
                )
                for h in super().search(vector, top_k=top_k, filters=filters)
            ]

    pasos = ejecutar(PorDistanciaStore())

    assert pasos[4].detail["score_kind"] == "distance"
    assert pasos[4].detail["higher_is_better"] is False
    assert "menor es mejor" in pasos[4].observed


def test_un_score_kind_inventado_se_rechaza():
    with pytest.raises(ValueError, match="score_kind debe ser"):
        SearchHit("r-1", 0.9, score_kind="parecido")


def test_el_resultado_lleva_su_posicion():
    """§3.3 pide que cada resultado incluya la posición, no solo el orden de la
    lista: el consumidor puede reordenar sin darse cuenta de que la perdió."""
    pasos = ejecutar(FakeStore())

    assert pasos[4].passed   # el paso valida que rank vaya 1..n


def test_una_posicion_mal_numerada_falla_el_paso():
    class SinRankStore(FakeStore):
        def search(self, vector, *, top_k, filters=()):
            return [
                SearchHit(
                    record_id=h.record_id, score=h.score, score_kind=h.score_kind,
                    higher_is_better=h.higher_is_better, rank=0, payload=h.payload,
                )
                for h in super().search(vector, top_k=top_k, filters=filters)
            ]

    assert not ejecutar(SinRankStore())[4].passed


# ─────────── el estado de indexación, que §3.2 pide verificar antes ──────────


def test_un_motor_que_sigue_indexando_no_pasa_el_paso_de_ingesta():
    """El recuento cuadrando no basta: con el índice a medias la búsqueda
    devuelve menos de lo escrito y el fallo parece del modelo."""
    class IndexandoStore(FakeStore):
        def index_ready(self) -> bool:
            return False

    pasos = ejecutar(IndexandoStore())

    assert not pasos[3].passed
    assert "AÚN INDEXANDO" in pasos[3].observed


def test_un_motor_que_no_reporta_el_estado_no_se_penaliza():
    """`FakeStore` no tiene `index_ready`, como los motores que no lo exponen.
    Se anota que no lo reporta en vez de inventarse un `True`."""
    pasos = ejecutar(FakeStore())

    assert pasos[3].passed
    assert pasos[3].detail["index_ready"] is None
    assert "no lo reporta" in pasos[3].observed


def test_la_tabla_lleva_una_fila_por_paso_y_marca_los_manuales():
    tabla = smoke_table(
        run_smoke_test(FakeStore(), PUNTOS, query_vector=CONSULTA, dim=DIM, top_k=2),
        motor="fake",
    )

    assert len(tabla) == 10
    assert list(tabla["paso"]) == list(range(1, 11))
    assert (tabla.tail(3)["resultado"] == "✍️ manual").all()


# ──────── paso 8 · qué significa "sobrevivió al reinicio" ────────────────────
# Las tres preguntas que un `==` entre listas junta en un solo booleano.


def _instantanea(*pares, count=None):
    """Una lectura con los ids y scores que se le pasen: `("r-1", 0.9), ...`."""
    return Snapshot(count=len(pares) if count is None else count, hits=tuple(pares))


def test_lo_identico_pasa_y_lo_dice():
    lectura = _instantanea(("r-1", 0.9), ("r-2", 0.8), count=1500)
    check = persistence_check(lectura, lectura)

    assert check.passed
    assert check.same_count and check.same_set and check.same_order
    assert "idéntico" in check.verdict()


def test_perder_puntos_suspende_aunque_el_top_k_coincida():
    """El recuento es la pregunta que no admite matices: si baja, se perdió
    dato. Que el top-k siga igual no lo compensa."""
    antes = _instantanea(("r-1", 0.9), ("r-2", 0.8), count=1500)
    despues = _instantanea(("r-1", 0.9), ("r-2", 0.8), count=1499)

    check = persistence_check(antes, despues)

    assert not check.passed
    assert "el recuento no sobrevive: 1500 → 1499" in check.verdict()


def test_los_mismos_ids_en_otro_orden_no_se_confunden_con_ids_distintos():
    """El caso real de Milvus. Una búsqueda aproximada no promete un orden total
    entre vecinos que empatan, así que reordenar el top-k con el mismo conjunto
    y el mismo score es un cambio de desempate, no una pérdida del índice — y el
    criterio de la fila 8, escrito antes de medir, habla de ids, no de orden."""
    antes = _instantanea(("r-1", 0.9), ("r-2", 0.9), ("r-3", 0.5), count=1500)
    despues = _instantanea(("r-2", 0.9), ("r-1", 0.9), ("r-3", 0.5), count=1500)

    check = persistence_check(antes, despues)

    assert check.same_set and not check.same_order
    assert check.passed                      # el criterio previo pedía ids
    assert check.moved == 2
    assert check.max_score_shift == pytest.approx(0.0)
    assert "distinto orden" in check.verdict()
    assert "cambia el desempate, no el índice" in check.verdict()


def test_un_reordenamiento_con_el_score_movido_no_se_explica_por_el_desempate():
    """Mismos ids, otro orden, pero las distancias cambiaron: entonces lo que
    volvió no es el mismo índice y el veredicto no puede sonar tranquilizador."""
    antes = _instantanea(("r-1", 0.90), ("r-2", 0.80), count=10)
    despues = _instantanea(("r-2", 0.95), ("r-1", 0.70), count=10)

    check = persistence_check(antes, despues)

    assert check.same_set and check.max_score_shift == pytest.approx(0.20)
    assert "el score se mueve" in check.verdict()


def test_cambiar_de_ids_suspende_y_cuantifica_el_solapamiento():
    antes = _instantanea(("r-1", 0.9), ("r-2", 0.8), ("r-3", 0.7), ("r-4", 0.6), count=1500)
    despues = _instantanea(("r-1", 0.9), ("r-2", 0.8), ("r-9", 0.7), ("r-8", 0.6), count=1500)

    check = persistence_check(antes, despues)

    assert not check.passed
    assert check.lost == ("r-3", "r-4") and check.gained == ("r-9", "r-8")
    assert check.overlap == pytest.approx(0.5)
    assert "solapamiento 50%" in check.verdict()


def test_una_lectura_vacia_no_se_da_por_buena():
    """Sin nada en el «antes» no hay nada que sobreviva: un ✅ aquí sería un
    motor vacío pasando el paso de persistencia."""
    check = persistence_check(_instantanea(count=0), _instantanea(count=0))

    assert not check.passed
    assert "no había nada que comprobar" in check.verdict()


def test_la_lectura_del_paso_8_solo_lee():
    """`read_snapshot` no puede recrear la colección: eso borraría justo los
    datos cuya supervivencia se quiere comprobar."""
    store = FakeStore()
    store.upsert(PUNTOS, batch_size=2)

    lectura = read_snapshot(store, CONSULTA, top_k=2)

    assert lectura.count == len(PUNTOS)
    assert lectura.ids == ("r-1", "r-2")
    assert store.count() == len(PUNTOS)          # nada se borró
    assert lectura.scores["r-1"] == pytest.approx(1.0)


# ──────── paso 9 · la calidad del error, con tres orígenes distintos ─────────


def test_una_excepcion_generica_suspende_el_paso_9():
    observado, pasa = error_quality(ConnectionError("connection refused"), sdk_package="qdrant_client")

    assert not pasa
    assert observado.startswith("❌ genérica")


def test_una_excepcion_del_propio_sdk_es_la_buena():
    class WeaviateConnectionError(Exception):
        __module__ = "weaviate.exceptions"

    observado, pasa = error_quality(WeaviateConnectionError("no reachable"), sdk_package="weaviate")

    assert pasa
    assert "tipada por el SDK" in observado


def test_un_error_del_transporte_pasa_pero_queda_señalado():
    """El caso de Qdrant: `grpc._channel._InactiveRpcError` es tipada y cumple el
    criterio escrito, pero capturarla en NB05 ataría el manejo de errores a la
    capa de red del SDK en vez de a su jerarquía pública."""
    class _InactiveRpcError(Exception):
        __module__ = "grpc._channel"

    observado, pasa = error_quality(_InactiveRpcError("UNAVAILABLE"), sdk_package="qdrant_client")

    assert pasa
    assert "tipada pero de grpc, no del SDK" in observado


def test_el_mensaje_del_error_se_recorta_y_se_aplana():
    """Un traceback de varias líneas dentro de una celda de la tabla la vuelve
    ilegible, y la tabla es el artefacto."""
    observado, _ = error_quality(
        ValueError("línea uno\n   línea dos " + "x" * 300), sdk_package="qdrant_client"
    )

    assert "\n" not in observado and "línea uno línea dos" in observado


# ──────── anotar lo manual: contra el criterio, no en lugar de él ────────────


def test_anotar_un_paso_manual_rellena_la_fila_sin_perder_lo_esperado():
    resultados = run_smoke_test(FakeStore(), PUNTOS, query_vector=CONSULTA, dim=DIM, top_k=2)

    record_manual(resultados, 8, observed="✅ idéntico", passed=True)

    fila = next(r for r in resultados if r.step == 8)
    assert fila.passed is True and fila.observed == "✅ idéntico"
    assert "mismo count()" in fila.expected      # el criterio previo sigue ahí


@pytest.mark.parametrize("paso", [1, 5, 7])
def test_no_se_puede_anotar_a_mano_un_paso_que_mide_el_guion(paso):
    """Escribir un ✅ encima de un paso automático taparía lo que se midió."""
    resultados = run_smoke_test(FakeStore(), PUNTOS, query_vector=CONSULTA, dim=DIM, top_k=2)

    with pytest.raises(ValueError, match="no es manual"):
        record_manual(resultados, paso, observed="✅ pasa", passed=True)


def test_anotar_un_paso_inexistente_falla():
    with pytest.raises(ValueError, match="ninguna fila con el paso"):
        record_manual([], 8, observed="lo que sea", passed=True)


# ──────── la pasada, en disco: el artefacto no depende del kernel ────────────


def test_la_pasada_guardada_vuelve_igual(tmp_path):
    """Sin esto, reescribir la comparativa obliga a levantar los tres motores
    otra vez: el §8 pide que los artefactos se regeneren solos."""
    store = FakeStore()
    resultados = run_smoke_test(store, PUNTOS, query_vector=CONSULTA, dim=DIM, top_k=2)
    record_manual(resultados, 8, observed="✅ idéntico", passed=True)
    tabla_filtros = probe_filters(store, _sondas_de_color(), top_k=2)

    save_smoke(tmp_path, motor="qdrant", results=resultados, filters=tabla_filtros)
    pasos, filtros = load_smoke(tmp_path)

    assert list(pasos) == ["qdrant"]
    assert smoke_table(pasos["qdrant"], motor="qdrant").equals(
        smoke_table(resultados, motor="qdrant")
    )
    assert contains_level(filtros["qdrant"]) == contains_level(tabla_filtros)


def test_la_tabla_de_diferencias_deja_solo_lo_que_separa_a_los_motores():
    """Diez ✅ en los tres motores no eligen ninguno. Lo que decide está en
    `observado` -si reporta el estado de indexación, si el score es distancia o
    similitud-, y esas son las filas que hay que mirar."""
    uno = run_smoke_test(FakeStore(), PUNTOS, query_vector=CONSULTA, dim=DIM, top_k=2)
    otro = run_smoke_test(FakeStore(), PUNTOS, query_vector=CONSULTA, dim=DIM, top_k=2)
    record_manual(uno, 9, observed="✅ tipada por el SDK", passed=True)
    record_manual(otro, 9, observed="⚠️ tipada pero del transporte", passed=True)

    diferencias = smoke_differences({"uno": uno, "otro": otro})

    assert [paso for paso, _ in diferencias.index] == [9]
    assert diferencias.loc[(9, "Calidad del error con el motor apagado"), "uno"] == (
        "✅ tipada por el SDK"
    )


def test_sin_diferencias_la_tabla_sale_vacia_en_vez_de_repetirse():
    pasos = run_smoke_test(FakeStore(), PUNTOS, query_vector=CONSULTA, dim=DIM, top_k=2)

    assert smoke_differences({"uno": pasos, "otro": pasos}).empty
    assert smoke_differences({}).empty


def test_cargar_de_un_directorio_que_no_existe_no_es_un_error(tmp_path):
    """La primera vez todavía no hay nada medido, y eso no es un fallo."""
    pasos, filtros = load_smoke(tmp_path / "todavia-no")

    assert pasos == {} and filtros == {}


# ──────── paso 10 · los recursos, leídos del transcrito de la terminal ───────

_STATS_MILVUS = """docker stats --no-stream
CONTAINER ID   NAME                        CPU %     MEM USAGE / LIMIT   MEM %     NET I/O           BLOCK I/O         PIDS
8f6500c8d17b   aurum-market-milvus         7.36%     244.1MiB / 3GiB     7.95%     40.4MB / 28.4MB   461MB / 10.5MB    54
21b80d62b1f8   aurum-market-milvus-etcd    0.98%     32.07MiB / 512MiB   6.26%     226kB / 157kB     41.8MB / 2.47MB   10
41ed0fddf375   aurum-market-milvus-attu    0.10%     100.0MiB / 512MiB   19.53%    1MB / 1MB         1MB / 1MB         10

--- volumenes ---
docker system df -v | grep aurum-market || true
8f6500c8d17b   milvusdb/milvus:v2.6.18                                "/tini -- milvus run…"   1               86kB      2 minutes ago   Up 2 minutes (healthy)      aurum-market-milvus
21b80d62b1f8   quay.io/coreos/etcd:v3.5.18                            "etcd --advertise-cl…"   1               24.6kB    2 minutes ago   Up 2 minutes (healthy)      aurum-market-milvus-etcd
aurum-market-milvus-data                                           1         10.32MB
aurum-market-milvus-etcd                                           1         144.9MB
TOTAL VOL. = 155.22MB
"""


def test_el_transcrito_se_lee_entero_sin_recortarlo_a_mano():
    """Se le pasa lo pegado tal cual —comando, cabecera, separadores— porque
    recortarlo a mano es el primer sitio donde se cuela un número que ya no
    corresponde a lo que se midió."""
    stats = parse_docker_stats(_STATS_MILVUS)

    assert list(stats["contenedor"]) == [
        "aurum-market-milvus", "aurum-market-milvus-etcd", "aurum-market-milvus-attu",
    ]
    assert stats.loc[0, "ram_mib"] == pytest.approx(244.1)
    assert stats.loc[0, "limite_mib"] == pytest.approx(3072.0)   # 3GiB, binario


def test_los_volumenes_no_se_confunden_con_las_filas_de_contenedor():
    """Las dos salidas vienen pegadas y las filas de contenedor de `df -v`
    también llevan un tamaño (el de su capa de escritura)."""
    volumenes = parse_volume_sizes(_STATS_MILVUS)

    assert list(volumenes["volumen"]) == [
        "aurum-market-milvus-data", "aurum-market-milvus-etcd",
    ]
    assert volumenes["mb"].sum() == pytest.approx(155.22)


def test_los_tamanos_de_docker_no_mezclan_binario_y_decimal():
    """`docker stats` da MiB y `docker system df` da MB. Tratarlos igual es un
    error del 5 % en MB y del 7 % en GB, que en una comparativa se nota."""
    assert parse_docker_stats(
        "abcdef123456   x   0.0%   1GiB / 2GiB   50.0%"
    ).loc[0, "ram_mib"] == pytest.approx(1024.0)
    assert parse_volume_sizes("v-uno   1   1GB").loc[0, "mb"] == pytest.approx(1000.0)


def test_el_panel_de_inspeccion_no_cuenta_como_coste_del_motor():
    """Attu es una herramienta de inspección, no parte del motor: sumarlo haría
    parecer a Milvus más caro frente a Qdrant, que sirve su panel desde el mismo
    proceso. La regla ya está escrita en el compose; aquí se aplica nombrando al
    contenedor excluido para que se vea qué se dejó fuera."""
    fila = resource_row(
        _STATS_MILVUS, motor="milvus", exclude=("aurum-market-milvus-attu",)
    ).iloc[0]

    assert fila["n_contenedores"] == 2
    assert fila["ram_mib"] == pytest.approx(276.2)         # 244.1 + 32.07, sin attu
    assert fila["excluidos"] == "aurum-market-milvus-attu"
    assert fila["mayor_consumidor"] == "aurum-market-milvus"
    assert fila["en_marcha"] == "Up 2 minutes (healthy)"


def test_el_criterio_del_paso_10_es_el_mem_limit_declarado_en_el_compose():
    """Inventar ahora un umbral de MiB con los números delante sería ponerle la
    vara al ganador. El límite ya estaba escrito en cada compose antes de medir,
    y `mas_apretado` señala cuál va más justo dentro del suyo."""
    fila = resource_row(_STATS_MILVUS, motor="milvus").iloc[0]

    assert fila["dentro_del_limite"]
    assert fila["mas_apretado"] == "aurum-market-milvus-attu al 20%"


def test_un_contenedor_por_encima_de_su_limite_suspende_la_fila():
    fila = resource_row(
        "abcdef123456   ahogado   0.0%   600MiB / 512MiB   117.19%", motor="x"
    ).iloc[0]

    assert not fila["dentro_del_limite"]


def test_la_nota_del_paso_10_declara_que_el_volumen_no_compara():
    fila = resource_row(_STATS_MILVUS, motor="milvus").iloc[0]

    nota = resource_note(fila)

    assert "RAM 376.2 MiB en 3 contenedores" in nota
    assert "no comparable" in nota
    assert "mem_limit" in nota


def test_un_transcrito_sin_filas_de_stats_falla_en_vez_de_dar_ceros():
    """Una tabla con 0 MiB se leería como «este motor no consume nada»."""
    with pytest.raises(ValueError, match="ninguna fila"):
        resource_row("--- volumenes ---\nv-uno   1   10MB", motor="qdrant")


def test_solo_cuentan_los_volumenes_del_motor_medido():
    """`docker system df -v` los lista todos, y los volúmenes con nombre de los
    otros motores sobreviven a su `down`: aparecen en el transcrito con LINKS=0.
    Sumarlos daba el total de los tres y no el del motor que se está midiendo —
    que es exactamente el fallo que inflaba el volumen del índice a 712 MB."""
    transcrito = """docker stats --no-stream
abcdef123456   aurum-market-qdrant   0.10%   28.73MiB / 2GiB   1.40%
--- volumenes ---
aurum-market-milvus-etcd                     0         81.47MB
aurum-market-weaviate-data                   0         11.11MB
aurum-market-qdrant-data                     1         289.5MB
"""

    fila = resource_row(transcrito, motor="qdrant").iloc[0]

    assert fila["volumen_mb"] == pytest.approx(289.5)
    assert fila["volumenes"] == "aurum-market-qdrant-data"
    assert "aurum-market-milvus-etcd" in fila["excluidos"]


def test_los_tres_volumenes_de_milvus_si_suman_entre_ellos():
    """El filtro es por prefijo del motor, no por volumen único: Milvus tiene
    tres y los tres son suyos."""
    transcrito = """docker stats --no-stream
abcdef123456   aurum-market-milvus   1.0%   100MiB / 3GiB   3.3%
--- volumenes ---
aurum-market-milvus-data                     1         10.32MB
aurum-market-milvus-etcd                     1         144.9MB
aurum-market-qdrant-data                     0         289.5MB
"""

    fila = resource_row(transcrito, motor="milvus").iloc[0]

    assert fila["volumen_mb"] == pytest.approx(155.2)   # la fila redondea a 1 decimal


def test_un_motor_sin_transcrito_no_aparece_inventado(tmp_path):
    (tmp_path / "milvus.txt").write_text(_STATS_MILVUS, encoding="utf-8")

    tabla = resource_table(tmp_path, ("qdrant", "weaviate", "milvus"))

    assert list(tabla["motor"]) == ["milvus"]
