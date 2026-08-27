"""Tests del guion del índice definitivo (`aurum.motores.aceptacion`).

Mismo planteamiento que `test_motores.py`: un motor falso en memoria hace de
doble, y encima se montan las averías que el guion tiene que cazar. Lo que se
protege aquí es distinto, eso sí — no es "¿sirve este motor?" sino "¿el índice
que acabo de construir es el que creo que es?" —, y la avería más importante es
la que ningún recuento detecta: que los vectores y los payloads se hayan
desalineado durante la ingesta por lotes.
"""
from __future__ import annotations

import pytest

from aurum.motores import (
    ACCEPTANCE_MANUAL_NUMBERS,
    CATALOG_PREFIX,
    Point,
    catalog_collection_name,
    guard_collection_name,
    record_manual,
    run_acceptance,
    self_retrieval_canaries,
    smoke_table,
    wait_until_indexed,
)
from test_motores import FakeStore

DIM = 4

# Cuatro puntos ortogonales: cada uno es su propio vecino más próximo y ninguno
# se parece a otro, así que un canario que falle solo puede ser desalineación.
PUNTOS = [
    Point("r-1", [1.0, 0.0, 0.0, 0.0], {"product_id": "B1"}),
    Point("r-2", [0.0, 1.0, 0.0, 0.0], {"product_id": "B2"}),
    Point("r-3", [0.0, 0.0, 1.0, 0.0], {"product_id": "B3"}),
    Point("r-4", [0.0, 0.0, 0.0, 1.0], {"product_id": "B4"}),
]


def ejecutar(store=None, **kwargs):
    resultados = run_acceptance(
        store or FakeStore(), PUNTOS, dim=DIM, batch_size=2, top_k=2, **kwargs
    )
    return {r.step: r for r in resultados}


# ─────────────────── el nombre lleva el contrato dentro ──────────────────────


def test_el_nombre_de_la_coleccion_lleva_modelo_plantilla_y_dimension():
    """Los tres forman parte del contrato del índice: cambiar cualquiera
    invalida los vectores guardados, y un nombre que no lo diga deja convivir
    dos colecciones incompatibles sin que nada avise."""
    nombre = catalog_collection_name(model="gemini-embedding-2", template="A4", dim=768)

    assert nombre == "aurum_catalogo__gemini_embedding_2__A4__768"
    assert "768" in nombre and "A4" in nombre


def test_el_nombre_sobrevive_a_un_modelo_con_barra():
    assert catalog_collection_name(
        model="ibm-granite/granite-embedding-311m", template="A0", dim=512
    ) == "aurum_catalogo__ibm_granite_granite_embedding_311m__A0__512"


def test_una_dimension_absurda_se_rechaza_al_nombrar():
    with pytest.raises(ValueError, match="positiva"):
        catalog_collection_name(model="x", template="A0", dim=0)


def test_el_indice_bueno_vive_bajo_otro_prefijo_que_la_prueba_de_humo():
    """Es la protección de verdad: el guion de humo recrea su colección en cada
    pasada, y con un prefijo común una errata podría llevarse los 15.000
    puntos. Con dos prefijos, ni equivocándose."""
    nombre = catalog_collection_name(model="m", template="A4", dim=768)

    assert guard_collection_name(nombre, prefix=CATALOG_PREFIX) == nombre
    with pytest.raises(ValueError, match="prefijo protegido"):
        guard_collection_name(nombre)          # el de humo no lo acepta
    with pytest.raises(ValueError, match="prefijo protegido"):
        guard_collection_name("aurum_humo_qdrant", prefix=CATALOG_PREFIX)


# ─────────────────────────── el guion completo ───────────────────────────────


def test_un_indice_sano_pasa_los_seis_pasos():
    pasos = ejecutar()

    assert all(pasos[n].passed for n in range(1, 7))
    assert len(pasos) == 8            # 6 automáticos + 2 manuales


def test_los_dos_pasos_manuales_quedan_como_huecos():
    """Persistencia y recursos los mide una persona. Si no aparecieran, se
    perderían del artefacto sin que se note."""
    pasos = ejecutar()

    assert set(ACCEPTANCE_MANUAL_NUMBERS) == {7, 8}
    assert all(pasos[n].passed is None for n in (7, 8))
    assert "(pendiente)" in pasos[7].observed


def test_la_calidad_del_error_no_se_repite_aqui():
    """Es propiedad del SDK, no del índice, y ya se midió en la prueba de humo."""
    pasos = ejecutar()

    assert not any("motor apagado" in p.name for p in pasos.values())


def test_la_ingesta_anota_los_vectores_por_segundo():
    """El número va al README como tiempo aproximado; si no se guarda aquí, se
    acaba estimando de memoria."""
    pasos = ejecutar()

    assert pasos[2].detail["vectores_por_segundo"] > 0
    assert "vectores/s" in pasos[2].observed


def test_reingerir_lo_mismo_no_puede_sumar():
    pasos = ejecutar()

    assert pasos[5].passed
    assert pasos[5].observed == f"count() = {len(PUNTOS)}"


def test_una_ingesta_que_duplica_se_caza_en_el_paso_de_idempotencia():
    class Duplicadora(FakeStore):
        def upsert(self, points, *, batch_size):     # ignora el id: acumula
            for i, punto in enumerate(points):
                self.puntos[f"{punto.record_id}#{len(self.puntos)}#{i}"] = punto
            return len(points)

    pasos = ejecutar(Duplicadora())

    assert not pasos[5].passed
    assert "eran" in pasos[5].observed


# ───────── los canarios: la avería que ningún recuento detecta ───────────────


def test_los_canarios_se_reparten_y_son_deterministas():
    """Repartidos para no mirar siempre la misma zona del corpus; deterministas
    para poder repetir la comprobación contra los mismos puntos tras reiniciar."""
    canarios = self_retrieval_canaries(PUNTOS, n=3)

    assert [c.record_id for c in canarios] == ["r-1", "r-3", "r-4"]
    assert canarios == self_retrieval_canaries(PUNTOS, n=3)


def test_se_piden_mas_canarios_de_los_que_hay():
    assert len(self_retrieval_canaries(PUNTOS[:2], n=5)) == 2
    assert len(self_retrieval_canaries(PUNTOS[:1], n=3)) == 1


def test_sin_puntos_no_hay_canarios_que_elegir():
    with pytest.raises(ValueError, match="No hay puntos"):
        self_retrieval_canaries([])


def test_cada_canario_se_recupera_a_si_mismo_el_primero():
    pasos = ejecutar()

    assert pasos[6].passed
    assert all(c["posicion"] == 1 for c in pasos[6].detail["canarios"])


def test_un_payload_desalineado_lo_caza_el_canario_y_no_el_recuento():
    """La avería clásica de una ingesta por lotes: el recuento cuadra, la
    dimensión cuadra, la idempotencia cuadra, y cada vector lleva el payload de
    otro producto. Solo la búsqueda de sí mismo lo ve."""
    class Desalineada(FakeStore):
        def upsert(self, points, *, batch_size):
            rotados = [
                Point(p.record_id, points[(i + 1) % len(points)].vector, p.payload)
                for i, p in enumerate(points)
            ]
            return super().upsert(rotados, batch_size=batch_size)

    pasos = ejecutar(Desalineada())

    assert pasos[2].passed and pasos[5].passed      # recuento e idempotencia, bien
    assert not pasos[6].passed                      # el canario, no
    assert "alineación" in pasos[6].observed


# ─────────────── el esquema: lo que se cree frente a lo que hay ──────────────


def test_la_dimension_se_le_pregunta_a_la_coleccion():
    class ConDimension(FakeStore):
        def collection_dim(self):
            return DIM

    assert ejecutar(ConDimension())[4].passed


def test_una_coleccion_con_otra_dimension_suspende():
    """El caso real: una colección preexistente creada con otro modelo. El
    notebook diría 768 igualmente, porque es su variable."""
    class Impostora(FakeStore):
        def collection_dim(self):
            return 1536

    paso = ejecutar(Impostora())[4]

    assert not paso.passed and "1536" in paso.observed


def test_un_motor_que_no_reporta_la_dimension_no_se_penaliza():
    paso = ejecutar()[4]

    assert paso.passed and "no lo reporta" in paso.observed


def test_un_indice_que_nunca_termina_no_acepta_consultas():
    """§3.2: verificar el estado de indexación ANTES de aceptar consultas. Si al
    agotarse la espera sigue sin estar listo, la fila suspende: no se acepta un
    índice a medias por cansancio."""
    class Indexando(FakeStore):
        def index_ready(self):
            return False

    paso = ejecutar(Indexando(), index_timeout=0, index_poll=0)[3]

    assert not paso.passed and "AÚN INDEXANDO tras" in paso.observed


def test_el_indice_que_tarda_en_estar_listo_dice_las_dos_cosas():
    """El caso real de los 15.000: al terminar la ingesta el HNSW seguía
    construyéndose. Las dos mitades importan — que no estaba listo, que es lo
    que avisa de no aceptar tráfico; y cuánto tardó, que es lo que dice cuándo
    sí y evita tener que dormir un rato a ojo."""
    class TardaUnPoco(FakeStore):
        sondeos = 0

        def index_ready(self):
            TardaUnPoco.sondeos += 1
            return TardaUnPoco.sondeos > 3

    paso = ejecutar(TardaUnPoco(), index_poll=0)[3]

    assert paso.passed
    assert paso.detail["listo_al_terminar_la_ingesta"] is False
    assert "AÚN INDEXANDO al terminar la ingesta" in paso.observed
    assert "listo tras" in paso.observed


def test_un_indice_listo_de_inmediato_no_espera_nada():
    class Rapido(FakeStore):
        def index_ready(self):
            return True

    paso = ejecutar(Rapido())[3]

    assert paso.passed and paso.observed == "listo ya al terminar la ingesta"
    assert paso.detail["segundos_de_espera"] == 0.0


def test_no_se_espera_a_una_senal_que_el_motor_no_da():
    """Esperar a algo que nadie va a decir es dormir con otro nombre."""
    listo, segundos, sondeos = wait_until_indexed(FakeStore(), timeout=99, poll=0)

    assert (listo, segundos, sondeos) == (None, 0.0, 0)
    assert ejecutar()[3].detail["segundos_de_espera"] == 0.0


# ─────────────────────────── entradas y anotación ────────────────────────────


def test_el_guion_exige_que_el_vector_cuadre_con_la_dimension():
    with pytest.raises(ValueError, match="dimensiones"):
        run_acceptance(FakeStore(), PUNTOS, dim=8, batch_size=2)


def test_el_guion_exige_puntos():
    with pytest.raises(ValueError, match="No hay puntos"):
        run_acceptance(FakeStore(), [], dim=DIM)


def test_no_recrea_la_coleccion_salvo_que_se_pida():
    """Al revés que la prueba de humo: aquí la colección es la buena."""
    store = FakeStore()
    store.upsert(PUNTOS, batch_size=2)

    run_acceptance(store, PUNTOS, dim=DIM, batch_size=2, top_k=2)

    assert store.count() == len(PUNTOS)     # no se vació por el camino


def test_las_filas_manuales_de_este_guion_se_anotan_con_su_propia_numeracion():
    resultados = run_acceptance(FakeStore(), PUNTOS, dim=DIM, batch_size=2, top_k=2)

    record_manual(resultados, 7, observed="✅ idéntico", passed=True,
                  manual_steps=ACCEPTANCE_MANUAL_NUMBERS)

    assert smoke_table(resultados, motor="qdrant").iloc[6]["resultado"] == "✅ pasa"
    with pytest.raises(ValueError, match="no es manual"):
        record_manual(resultados, 2, observed="✅", passed=True,
                      manual_steps=ACCEPTANCE_MANUAL_NUMBERS)
