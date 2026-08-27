"""Pruebas de NB08: clasificacion, espera activa y verificacion de eventos (src/aurum/mutaciones.py)."""
import pandas as pd
import pytest

from aurum.motores.base import Point, SearchHit
from aurum.mutaciones import (
    aplicar_secuencia,
    clasificar_evento,
    clasificar_eventos,
    esperar_visibilidad,
    verificar_evento,
)

# ────────────────────────────── clasificar_evento ─────────────────────────────


def _fila(event_id, operation, catalog_version):
    return pd.Series(
        {"event_id": event_id, "operation": operation, "catalog_version": catalog_version}
    )


def test_delete_es_baja_sin_importar_la_version():
    assert clasificar_evento(_fila("EVT-009", "DELETE", 2)) == "baja"


def test_upsert_con_version_2_es_actualizacion():
    assert clasificar_evento(_fila("EVT-001", "UPSERT", 2)) == "actualizacion"


def test_upsert_con_version_1_es_alta():
    assert clasificar_evento(_fila("EVT-017", "UPSERT", 1)) == "alta"


def test_upsert_con_version_desconocida_falla():
    with pytest.raises(ValueError, match="EVT-099"):
        clasificar_evento(_fila("EVT-099", "UPSERT", 3))


def test_operation_desconocida_falla():
    with pytest.raises(ValueError, match="PATCH"):
        clasificar_evento(_fila("EVT-099", "PATCH", 1))


# ───────────────────────────── clasificar_eventos ─────────────────────────────


def _eventos(filas):
    """`filas`: lista de (sequence, event_id, operation, catalog_version)."""
    return pd.DataFrame(
        [
            {"sequence": s, "event_id": e, "operation": o, "catalog_version": v}
            for s, e, o, v in filas
        ]
    )


def test_clasifica_el_reparto_8_8_8_y_ordena_por_sequence():
    filas = (
        [(i, f"EVT-{i:03d}", "UPSERT", 2) for i in range(1, 9)]
        + [(i, f"EVT-{i:03d}", "DELETE", 2) for i in range(9, 17)]
        + [(i, f"EVT-{i:03d}", "UPSERT", 1) for i in range(17, 25)]
    )
    # Desordenados a proposito: la funcion debe reordenar por sequence.
    eventos = _eventos(filas).sample(frac=1, random_state=0)

    resultado = clasificar_eventos(eventos)

    assert list(resultado["sequence"]) == list(range(1, 25))
    assert resultado["tipo"].value_counts().to_dict() == {
        "actualizacion": 8, "baja": 8, "alta": 8,
    }


def test_clasificar_eventos_falla_si_el_reparto_no_es_8_8_8():
    filas = [(i, f"EVT-{i:03d}", "UPSERT", 2) for i in range(1, 8)]  # solo 7
    with pytest.raises(ValueError, match="actualizacion"):
        clasificar_eventos(_eventos(filas))


def test_clasificar_eventos_falla_con_dataframe_vacio():
    with pytest.raises(ValueError, match="No hay eventos"):
        clasificar_eventos(pd.DataFrame(columns=["sequence", "event_id", "operation", "catalog_version"]))


# ───────────────────────────── esperar_visibilidad ────────────────────────────


def test_visible_a_la_primera():
    resultado = esperar_visibilidad(lambda: True, timeout_s=1.0, intervalo_s=0.01)

    assert resultado == {"visible": True, "segundos": pytest.approx(resultado["segundos"]), "intentos": 1}


def test_visible_tras_varios_intentos():
    intentos_restantes = [3]  # falla 2 veces, acierta a la 3ª

    def comprobar():
        intentos_restantes[0] -= 1
        return intentos_restantes[0] <= 0

    resultado = esperar_visibilidad(comprobar, timeout_s=1.0, intervalo_s=0.01)

    assert resultado["visible"] is True
    assert resultado["intentos"] == 3


def test_no_visible_agota_el_timeout_y_lo_dice():
    resultado = esperar_visibilidad(lambda: False, timeout_s=0.05, intervalo_s=0.01)

    assert resultado["visible"] is False
    assert resultado["segundos"] >= 0.05


def test_timeout_no_positivo_falla():
    with pytest.raises(ValueError, match="timeout_s"):
        esperar_visibilidad(lambda: True, timeout_s=0, intervalo_s=0.01)


def test_intervalo_no_positivo_falla():
    with pytest.raises(ValueError, match="intervalo_s"):
        esperar_visibilidad(lambda: True, timeout_s=1.0, intervalo_s=0)


# ────────────────────────────── verificar_evento ──────────────────────────────


class StoreFalso:
    def __init__(self, existentes=None, hits=None):
        self.existentes = dict(existentes or {})
        self._hits = list(hits or [])

    def get(self, record_id):
        return self.existentes.get(record_id)

    def search(self, vector, *, top_k=10):
        return self._hits[:top_k]


def test_verificar_alta_visible_por_id_y_busqueda():
    punto = Point(record_id="r1", vector=[0.1], payload={"catalog_version": 1})
    store = StoreFalso(
        existentes={"r1": punto},
        hits=[SearchHit(record_id="r1", score=0.99)],
    )

    resultado = verificar_evento(
        store, "alta", record_id="r1", vector=[0.1], timeout_s=0.2, intervalo_s=0.01
    )

    assert resultado["por_id"]["visible"] is True
    assert resultado["por_busqueda"]["visible"] is True


def test_verificar_actualizacion_exige_la_version_nueva_no_solo_existencia():
    punto_version_vieja = Point(record_id="r1", vector=[0.1], payload={"catalog_version": 1})
    store = StoreFalso(existentes={"r1": punto_version_vieja})

    resultado = verificar_evento(
        store, "actualizacion", record_id="r1", vector=None,
        catalog_version_esperado=2, timeout_s=0.05, intervalo_s=0.01,
    )

    assert resultado["por_id"]["visible"] is False


def test_verificar_baja_visible_cuando_el_punto_ya_no_existe():
    store = StoreFalso(existentes={}, hits=[])

    resultado = verificar_evento(
        store, "baja", record_id="r1", vector=[0.1], timeout_s=0.2, intervalo_s=0.01
    )

    assert resultado["por_id"]["visible"] is True
    assert resultado["por_busqueda"]["visible"] is True


def test_verificar_baja_no_visible_si_la_busqueda_todavia_lo_devuelve():
    store = StoreFalso(existentes={}, hits=[SearchHit(record_id="r1", score=0.99)])

    resultado = verificar_evento(
        store, "baja", record_id="r1", vector=[0.1], timeout_s=0.05, intervalo_s=0.01
    )

    assert resultado["por_busqueda"]["visible"] is False


def test_verificar_evento_sin_vector_no_comprueba_por_busqueda():
    store = StoreFalso(existentes={"r1": Point(record_id="r1", vector=[], payload={})})

    resultado = verificar_evento(
        store, "alta", record_id="r1", vector=None, timeout_s=0.05, intervalo_s=0.01
    )

    assert resultado["por_busqueda"] == {"visible": None, "segundos": 0.0, "intentos": 0}


def test_verificar_evento_tipo_invalido_falla():
    store = StoreFalso()
    with pytest.raises(ValueError, match="tipo"):
        verificar_evento(store, "renombrado", record_id="r1", vector=None)


# ───────────────────────────────── aplicar_secuencia ──────────────────────────


class StoreDeEscritura:
    def __init__(self):
        self.upserts = []
        self.deletes = []

    def upsert(self, points, *, batch_size):
        self.upserts.append((list(points), batch_size))
        return len(points)

    def delete(self, record_id):
        self.deletes.append(record_id)


def test_aplicar_secuencia_hace_upsert_y_delete():
    store = StoreDeEscritura()
    puntos = [Point(record_id="a", vector=[0.1], payload={}), Point(record_id="b", vector=[0.2], payload={})]

    resultado = aplicar_secuencia(store, puntos, ["c", "d"], batch_size=128)

    assert resultado["n_upsert"] == 2
    assert resultado["n_delete"] == 2
    assert store.deletes == ["c", "d"]
    assert store.upserts[0][1] == 128


def test_aplicar_secuencia_sin_puntos_no_llama_a_upsert():
    store = StoreDeEscritura()

    resultado = aplicar_secuencia(store, [], ["c"], batch_size=128)

    assert resultado["n_upsert"] == 0
    assert store.upserts == []
