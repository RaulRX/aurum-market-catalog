"""NB08: aplicar y verificar los 24 eventos de eventos_catalogo.csv (D18-D19).

D19 fija el orden real de trabajo: NB07 se calibra contra la coleccion ANTES
de que este notebook la mute. D18 fija la estrategia de visibilidad: la
escritura ya es sincrona (`QdrantStore.upsert(..., wait=True)`, desde NB04);
este modulo anade la espera activa con timeout del lado de la LECTURA, que es
la mitad que el enunciado (S4.1) pide y que NB04 no necesitaba resolver.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

from .motores.base import Point

TIPOS_EVENTO = ("actualizacion", "alta", "baja")
CONTEO_ESPERADO = {"actualizacion": 8, "alta": 8, "baja": 8}


def clasificar_evento(fila: pd.Series) -> str:
    """La regla verificada del plan (NB08 Procedimiento, paso 2):
    `UPSERT` + `catalog_version==2` es actualizacion (el registro ya
    existia), `UPSERT` + `catalog_version==1` es alta (no existia), y
    `DELETE` es baja -su `catalog_version` no distingue nada, el borrado no
    depende de la version-."""
    operacion = fila["operation"]
    if operacion == "DELETE":
        return "baja"
    if operacion == "UPSERT":
        version = int(fila["catalog_version"])
        if version == 2:
            return "actualizacion"
        if version == 1:
            return "alta"
        raise ValueError(
            f"{fila['event_id']}: catalog_version {version} inesperado para UPSERT "
            f"(se esperaba 1 o 2)."
        )
    raise ValueError(f"{fila['event_id']}: operation {operacion!r} desconocida.")


def clasificar_eventos(eventos: pd.DataFrame) -> pd.DataFrame:
    """Ordena por `sequence` -el orden es parte del contrato- y anade la
    columna `tipo`. Levanta si el reparto no es 8/8/8: es una comprobacion
    ya verificada contra el dato, asi que si no cuadra, el problema esta en
    el fichero o en la lectura, no en la regla."""
    if eventos.empty:
        raise ValueError("No hay eventos que clasificar.")
    resultado = eventos.sort_values("sequence").reset_index(drop=True).copy()
    resultado["tipo"] = [clasificar_evento(fila) for _, fila in resultado.iterrows()]
    conteo = resultado["tipo"].value_counts()
    for tipo, esperado in CONTEO_ESPERADO.items():
        obtenido = int(conteo.get(tipo, 0))
        if obtenido != esperado:
            raise ValueError(
                f"Se esperaban {esperado} eventos de tipo {tipo!r}, hay {obtenido}."
            )
    return resultado


def esperar_visibilidad(
    comprobar: Callable[[], bool],
    *,
    timeout_s: float = 10.0,
    intervalo_s: float = 0.5,
) -> dict[str, Any]:
    """D18: reintenta `comprobar()` hasta que devuelva `True` o se agote
    `timeout_s`. Nunca un `sleep` a ciegas -cuenta intentos y segundos
    reales, y si no aparece a tiempo lo dice (`visible=False`) en vez de
    asumir que si: es la diferencia entre "saber esperar" y "esperar y
    cruzar los dedos" que pide el enunciado (S4.1)."""
    if timeout_s <= 0:
        raise ValueError("timeout_s debe ser positivo.")
    if intervalo_s <= 0:
        raise ValueError("intervalo_s debe ser positivo.")
    inicio = time.perf_counter()
    intentos = 0
    while True:
        intentos += 1
        if comprobar():
            return {
                "visible": True,
                "segundos": time.perf_counter() - inicio,
                "intentos": intentos,
            }
        transcurrido = time.perf_counter() - inicio
        if transcurrido >= timeout_s:
            return {"visible": False, "segundos": transcurrido, "intentos": intentos}
        time.sleep(intervalo_s)


def verificar_evento(
    store: Any,
    tipo: str,
    *,
    record_id: str,
    vector: Sequence[float] | None,
    catalog_version_esperado: int | None = None,
    top_k: int = 5,
    timeout_s: float = 10.0,
    intervalo_s: float = 0.5,
) -> dict[str, Any]:
    """Verifica un evento ya aplicado por las dos rutas exigidas: lectura
    por ID y busqueda vectorial (NB08, paso 5).

    `tipo` decide que se espera en cada ruta: altas y actualizaciones deben
    **existir** y **aparecer**; bajas deben **no existir** y **no
    aparecer**. Para una actualizacion, `catalog_version_esperado=2` exige
    ademas que el dato leido sea el nuevo, no solo que el punto exista -un
    upsert que fallara a medias podria dejar el punto ahi con la version
    vieja, y una comprobacion de solo existencia no lo detectaria-.

    `vector=None` (los DELETE no traen texto que reencodear, salvo
    que el notebook recupere el vector original de la cache) deja
    `por_busqueda` sin verificar en vez de fallar: la ruta de ID sigue
    siendo valida por si sola.
    """
    if tipo not in TIPOS_EVENTO:
        raise ValueError(f"tipo debe ser uno de {TIPOS_EVENTO}, no {tipo!r}.")
    debe_existir = tipo != "baja"

    def comprobar_id() -> bool:
        punto = store.get(record_id)
        if not debe_existir:
            return punto is None
        if punto is None:
            return False
        if catalog_version_esperado is not None:
            return punto.payload.get("catalog_version") == catalog_version_esperado
        return True

    por_id = esperar_visibilidad(comprobar_id, timeout_s=timeout_s, intervalo_s=intervalo_s)

    if vector is None:
        por_busqueda: dict[str, Any] = {"visible": None, "segundos": 0.0, "intentos": 0}
    else:
        def comprobar_busqueda() -> bool:
            hits = store.search(vector, top_k=top_k)
            encontrado = any(hit.record_id == record_id for hit in hits)
            return encontrado if debe_existir else not encontrado

        por_busqueda = esperar_visibilidad(
            comprobar_busqueda, timeout_s=timeout_s, intervalo_s=intervalo_s
        )

    return {"record_id": record_id, "tipo": tipo, "por_id": por_id, "por_busqueda": por_busqueda}


def aplicar_secuencia(
    store: Any,
    puntos_upsert: Sequence[Point],
    ids_borrar: Sequence[str],
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Aplica altas+actualizaciones (`upsert`) y bajas (`delete`) de una
    pasada. Idempotente por diseno -no por que esta funcion lo garantice,
    sino porque lo son sus dos primitivas-: `upsert` sobrescribe por
    `record_id` (D19/NB04) y borrar un id ya ausente no es un error en
    Qdrant. Repetir la llamada con los mismos argumentos dos veces deja el
    mismo estado, que es lo que el paso 4 del plan pide comprobar."""
    inicio = time.perf_counter()
    n_upsert = store.upsert(puntos_upsert, batch_size=batch_size) if puntos_upsert else 0
    for record_id in ids_borrar:
        store.delete(record_id)
    return {
        "n_upsert": n_upsert,
        "n_delete": len(ids_borrar),
        "segundos": time.perf_counter() - inicio,
    }
