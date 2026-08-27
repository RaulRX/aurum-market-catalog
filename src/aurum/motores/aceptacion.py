"""El guion del índice definitivo: qué hay que comprobar antes de aceptar consultas.

La prueba de humo (`humo.py`) respondía *"¿sirve este motor?"* sobre 1.500
puntos que se podían tirar. Esto responde otra cosa: *"¿el índice que acabo de
construir es el que creo que es?"*, sobre los 15.000 que van a usar NB05, NB06 y
NB08. Son guiones distintos y por eso están en ficheros distintos, pero
comparten `SmokeResult` y su tabla: la forma de mirar no tiene por qué cambiar
solo porque cambie la pregunta.

Las comprobaciones salen de la tabla "Métricas de verificación de NB04" del plan
y de §3.2 del enunciado, que pide literalmente *"verificad el recuento final y el
estado de indexación antes de aceptar consultas"*.

El canario merece explicación aparte; ver `self_retrieval_canaries`.
"""
from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter, sleep

from .base import Point, VectorStore
from .humo import SmokeResult

# Los pasos que mide una persona desde la terminal. Son dos y no tres: la
# calidad del error con el motor caído ya se midió en la prueba de humo y es una
# propiedad del SDK, no del índice — repetirla aquí no añadiría nada. Los otros
# dos sí cambian con los 15.000 puntos dentro, y el de recursos es además el
# número que de verdad va al README.
ACCEPTANCE_MANUAL_STEPS: tuple[tuple[int, str, str, str], ...] = (
    (
        7, "Persistencia tras reinicio",
        "make motor-down MOTOR=qdrant && make motor-up MOTOR=qdrant  (+ relectura)",
        "mismo count() y los mismos canarios en la posición 1",
    ),
    (
        8, "Recursos con el índice completo",
        "make motor-stats  (con el motor vivo y los 15.000 ingeridos)",
        "RAM del contenedor y tamaño del volumen, esta vez sobre volumen limpio",
    ),
)

ACCEPTANCE_MANUAL_NUMBERS = frozenset(step for step, *_ in ACCEPTANCE_MANUAL_STEPS)


def self_retrieval_canaries(points: Sequence[Point], *, n: int = 3) -> list[Point]:
    """Los puntos que se van a buscar con su propio vector.

    **Por qué la búsqueda de sí mismo y no una consulta de desarrollo.** El plan
    pide "3 consultas con IDs conocidos que deben aparecer en el top-10", y la
    tentación es usar una consulta juzgada y su producto relevante. Sería un mal
    canario: que un producto relevante entre en el top-10 es una pregunta de
    **calidad** —el Recall@10 medido es 0,26, así que fallaría a menudo con el
    índice perfectamente sano— y un canario que falla por lo que no vigila no
    sirve para vigilar nada.

    Buscar un punto con su propio vector, en cambio, tiene una respuesta que no
    depende de la calidad del modelo: debe volver él, el primero, con similitud
    1. Si no vuelve, o vuelve otro, lo que hay roto es el índice — y en concreto
    la alineación entre vector y payload, que es el fallo silencioso clásico de
    una ingesta por lotes y el que el patrón `_validate_alignment` de la sesión 2
    perseguía.

    Se eligen de forma determinista y repartidos —primero, medio y último—: para
    poder repetir la comprobación contra los mismos puntos después de un
    reinicio, y para no mirar siempre la misma zona del corpus.
    """
    if not points:
        raise ValueError("No hay puntos entre los que elegir canarios.")
    n = min(n, len(points))
    if n == 1:
        return [points[0]]
    paso = (len(points) - 1) / (n - 1)
    return [points[round(i * paso)] for i in range(n)]


def wait_until_indexed(
    store: VectorStore, *, timeout: float = 120.0, poll: float = 1.0
) -> tuple[bool | None, float, int]:
    """Espera a que el motor declare su índice construido, con tope.

    Devuelve `(listo, segundos, sondeos)`. `listo` es `None` cuando el motor no
    reporta el estado: entonces no se espera nada, porque esperar a una señal
    que no existe es dormir con otro nombre.

    **Por qué una espera y no un `sleep`.** El enunciado pide que el sistema sepa
    *"esperar, fallar o informar"*, y las tres cosas son distintas de dormir un
    rato fijo: esta espera termina en cuanto el motor dice que está listo,
    devuelve cuánto tardó —que es el dato que hace falta para saber cuándo se
    puede medir latencia— y se rinde con un tope en vez de colgarse.
    """
    listo = getattr(store, "index_ready", None)
    if not callable(listo):
        return None, 0.0, 0
    inicio = perf_counter()
    sondeos = 0
    while True:
        sondeos += 1
        if listo():
            return True, perf_counter() - inicio, sondeos
        if perf_counter() - inicio >= timeout:
            return False, perf_counter() - inicio, sondeos
        sleep(poll)


def run_acceptance(
    store: VectorStore,
    points: Sequence[Point],
    *,
    dim: int,
    metric: str = "cosine",
    batch_size: int = 128,
    top_k: int = 10,
    n_canarios: int = 3,
    recreate: bool = False,
    index_timeout: float = 180.0,
    index_poll: float = 2.0,
) -> list[SmokeResult]:
    """Construye el índice definitivo y comprueba que es el que se cree.

    `recreate=False` por defecto, al revés que en el guion de humo: aquí la
    colección es la buena, y recrearla es una operación que hay que pedir a
    propósito (y que además exige `AURUM_ALLOW_RESET`).
    """
    if not points:
        raise ValueError("No hay puntos que ingerir.")
    if len(points[0].vector) != dim:
        raise ValueError(
            f"Los vectores tienen {len(points[0].vector)} dimensiones y la "
            f"colección se declara con {dim}."
        )

    esperado = len({punto.record_id for punto in points})
    resultados: list[SmokeResult] = []

    def cronometrar(fn):
        inicio = perf_counter()
        valor = fn()
        return valor, perf_counter() - inicio

    # ── 1 · La colección, con su contrato explícito ──────────────────────────
    _, s = cronometrar(
        lambda: store.create_collection(dim=dim, metric=metric, recreate=recreate)
    )
    resultados.append(SmokeResult(
        1, "Colección con dimensión y métrica explícitas",
        f"dim={dim}, métrica={metric}", passed=True,
        action=f"create_collection(dim={dim}, metric={metric!r}, recreate={recreate})",
        observed=(
            f"{getattr(store, 'collection', 'la colección')!r} lista "
            f"(dim={dim}, métrica={metric})"
        ),
        seconds=s,
    ))

    # ── 2 · La ingesta, cronometrada: el número va al README ─────────────────
    _, s = cronometrar(lambda: store.upsert(points, batch_size=batch_size))
    n = store.count()
    por_segundo = round(len(points) / s, 1) if s else None
    resultados.append(SmokeResult(
        2, "Ingesta por lotes", f"count() == {esperado}",
        passed=n == esperado,
        action=f"upsert({len(points)} puntos, batch_size={batch_size}) → count()",
        observed=f"count() = {n} · {por_segundo} vectores/s en {s:.1f} s",
        seconds=s, detail={"vectores_por_segundo": por_segundo},
    ))

    # ── 3 · El estado de indexación, que §3.2 pide ANTES de consultar ────────
    # Dos datos, no uno: si el índice estaba listo al terminar la ingesta, y
    # -si no lo estaba- cuánto tardó en estarlo. El primero es el que avisa de
    # que ese no es momento de aceptar tráfico; el segundo es el que dice
    # cuándo sí, y sin él la única alternativa es dormir un rato a ojo.
    sonda = getattr(store, "index_ready", None)
    de_inmediato = sonda() if callable(sonda) else None
    espera, sondeos = 0.0, 0
    estado = de_inmediato
    if de_inmediato is False:
        estado, espera, sondeos = wait_until_indexed(
            store, timeout=index_timeout, poll=index_poll
        )
    resultados.append(SmokeResult(
        3, "Índice al día antes de aceptar consultas",
        "el motor declara su índice construido", passed=estado is not False,
        action=f"index_ready() · espera con tope de {index_timeout:.0f} s",
        observed={
            True: (
                "listo ya al terminar la ingesta" if de_inmediato
                else f"AÚN INDEXANDO al terminar la ingesta · listo tras "
                     f"{espera:.1f} s de espera ({sondeos} sondeos)"
            ),
            False: (
                f"AÚN INDEXANDO tras {espera:.1f} s de espera — no aceptar "
                f"consultas todavía"
            ),
            None: "no lo reporta el motor",
        }[estado],
        seconds=espera or None,
        detail={
            "index_ready": estado,
            "listo_al_terminar_la_ingesta": de_inmediato,
            "segundos_de_espera": round(espera, 1),
        },
    ))

    # ── 4 · La dimensión, preguntándosela a la colección ─────────────────────
    declarada = getattr(store, "collection_dim", None)
    real = declarada() if callable(declarada) else None
    resultados.append(SmokeResult(
        4, "Dimensión declarada == dimensión real", f"la colección dice {dim}",
        passed=real in (dim, None),
        action="collection_dim()",
        observed=(
            "no lo reporta el motor" if real is None
            else f"la colección declara {real}"
        ),
        detail={"dim_declarada": real},
    ))

    # ── 5 · Idempotencia: reingerir lo mismo no puede sumar ──────────────────
    _, s = cronometrar(lambda: store.upsert(points, batch_size=batch_size))
    repetido = store.count()
    resultados.append(SmokeResult(
        5, "Ingesta repetida sin duplicar", f"count() sigue en {esperado}",
        passed=repetido == esperado,
        action=f"upsert(LOS MISMOS {len(points)} puntos) → count()",
        observed=f"count() = {repetido}" + (
            "" if repetido == esperado else f" — eran {n} antes de repetir"
        ),
        seconds=s,
    ))

    # ── 6 · Canarios: cada punto se encuentra a sí mismo el primero ──────────
    canarios = self_retrieval_canaries(points, n=n_canarios)
    aciertos, detalle = 0, []
    for canario in canarios:
        hits = store.search(canario.vector, top_k=top_k)
        posicion = next(
            (h.rank for h in hits if h.record_id == canario.record_id), None
        )
        aciertos += posicion == 1
        detalle.append({"record_id": canario.record_id, "posicion": posicion})
    fallos = [d for d in detalle if d["posicion"] != 1]
    resultados.append(SmokeResult(
        6, "Canarios: cada punto se recupera a sí mismo",
        f"los {len(canarios)} vuelven en la posición 1",
        passed=aciertos == len(canarios),
        action=f"search(vector del propio punto, top_k={top_k}) × {len(canarios)}",
        observed=(
            f"{aciertos}/{len(canarios)} en la posición 1"
            + ("" if not fallos else f" · fallan {fallos} — revisar la alineación "
                                     f"entre vector y payload")
        ),
        detail={"canarios": detalle},
    ))

    for step, name, action, expected in ACCEPTANCE_MANUAL_STEPS:
        resultados.append(SmokeResult(
            step, name, expected, passed=None, action=action, observed="(pendiente)"
        ))

    return resultados
