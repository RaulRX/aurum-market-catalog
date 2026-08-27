"""El guion de la prueba de humo: los mismos pasos contra cada candidato.

De los diez pasos del guion de NB04, aquí se automatizan **los siete
primeros**, que son los que caben dentro del proceso de Python. Los tres últimos
—reinicio del contenedor, motor apagado y consumo de recursos— los ejecuta una
persona desde la terminal y se anotan a mano: `SMOKE_MANUAL_STEPS` deja los
huecos escritos para que no se olviden al rellenar la comparativa.

Un paso que falla **no interrumpe el guion**. Un motor que no sabe filtrar por
`contains` tiene que llegar igualmente al paso de persistencia: la tabla vale
para comparar precisamente porque todas las filas están rellenas, y parar en el
primer fallo dejaría al motor peor descrito en vez de peor valorado.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from .base import FilterCondition, Point, UnsupportedFilterError, VectorStore

# Los tres pasos que no puede dar el proceso de Python: necesitan la terminal y
# los mide una persona. Se declaran aquí para que la tabla de humo salga con sus
# filas presentes y vacías, en vez de terminar en el paso 7 como si no existieran.
SMOKE_MANUAL_STEPS: tuple[tuple[int, str, str, str], ...] = (
    (
        8, "Persistencia tras reinicio",
        "make motor-down MOTOR=… && make motor-up MOTOR=…  (+ celda de solo lectura)",
        "mismo count() y mismos ids en el top-10 que antes del reinicio",
    ),
    (
        9, "Calidad del error con el motor apagado",
        "docker compose -f docker/…/compose.yaml stop  →  search()",
        "excepción tipada y legible, no un error genérico",
    ),
    (
        10, "Recursos",
        "make motor-stats  (con el motor vivo y ya ingerido)",
        "RAM del contenedor · tamaño del volumen",
    ),
)


@dataclass(slots=True)
class SmokeResult:
    """El resultado de un paso: qué se ejecutó, qué se esperaba y qué pasó.

    `action` existe porque la tabla tiene que poder leerse sin abrir el código:
    saber que el paso 5 "pasa" no dice contra qué filtro ni con qué `top_k`, y
    esos son justo los parámetros que hacen que el paso pruebe algo o no.
    """

    step: int
    name: str
    expected: str
    passed: bool | None          # None = paso manual, aún sin rellenar
    action: str = ""             # la llamada concreta, con sus parámetros
    observed: str = ""
    seconds: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)


def _timed(fn):
    started = perf_counter()
    value = fn()
    return value, perf_counter() - started


def run_smoke_test(
    store: VectorStore,
    points: Sequence[Point],
    *,
    query_vector: Sequence[float],
    dim: int,
    metric: str = "cosine",
    top_k: int = 10,
    batch_size: int = 128,
    filters: Sequence[FilterCondition] = (),
    recreate: bool = True,
) -> list[SmokeResult]:
    """Ejecuta los pasos 1–7 contra un motor ya levantado.

    No levanta ni para ningún contenedor: eso es de quien ejecuta. Espera
    encontrar el motor en marcha y falla con un error legible si no lo está.

    ⚠️ `filters` tiene que ser **selectivo** para que el paso 5 valga. Ese paso
    comprueba que lo devuelto cumpla la condición, y esa comprobación solo
    destapa a un motor que ignora el filtro si la búsqueda sin filtrar habría
    traído algo que no lo cumple. Con una condición que ya domina el top-k, un
    filtro ignorado pasaría por bueno. Por eso el guion usa `brand="Einhell"`,
    que es el 0,2 % del catálogo.
    """
    if not points:
        raise ValueError("La prueba de humo necesita puntos que ingerir.")
    if len(query_vector) != dim:
        raise ValueError(
            f"El vector de consulta tiene {len(query_vector)} dimensiones y la "
            f"colección se declara con {dim}."
        )

    results: list[SmokeResult] = []
    esperado_n = len({point.record_id for point in points})

    # ── 1 · Crear la colección con dimensión y métrica explícitas ────────────
    def paso_1() -> SmokeResult:
        _, s = _timed(lambda: store.create_collection(dim=dim, metric=metric, recreate=recreate))
        return SmokeResult(
            1, "Crear colección", f"dim={dim}, métrica={metric}",
            passed=True,
            action=f"create_collection(dim={dim}, metric={metric!r}, recreate={recreate})",
            observed=f"creada (dim={dim}, métrica={metric})", seconds=s,
        )

    # ── 2 · Ingesta por lotes ────────────────────────────────────────────────
    def paso_2() -> SmokeResult:
        _, s = _timed(lambda: store.upsert(points, batch_size=batch_size))
        n = store.count()
        return SmokeResult(
            2, "Ingesta por lotes", f"count() == {esperado_n}",
            passed=n == esperado_n,
            action=f"upsert({len(points)} puntos, batch_size={batch_size}) → count()",
            observed=f"count() = {n}", seconds=s,
            detail={"vectores_por_segundo": round(len(points) / s, 1) if s else None},
        )

    # ── 3 · Repetir la ingesta: el count() no puede moverse ──────────────────
    def paso_3() -> SmokeResult:
        _, s = _timed(lambda: store.upsert(points, batch_size=batch_size))
        n = store.count()
        # §3.2 exige verificar el recuento **y el estado de indexación** antes de
        # aceptar consultas. El recuento cuadrando no basta: en un motor que
        # indexa en segundo plano, buscar aquí devolvería menos de lo que hay
        # escrito y el fallo parecería del modelo.
        listo = getattr(store, "index_ready", None)
        estado = listo() if callable(listo) else None
        return SmokeResult(
            3, "Ingesta repetida + índice listo", f"count() sigue en {esperado_n} y el índice está al día",
            passed=n == esperado_n and estado is not False,
            action=(
                f"upsert(LOS MISMOS {len(points)} puntos, batch_size={batch_size}) "
                f"→ count() + index_ready()"
            ),
            observed=(
                f"count() = {n} · índice: "
                + {True: "listo", False: "AÚN INDEXANDO", None: "no lo reporta el motor"}[estado]
            ),
            seconds=s,
            detail={"index_ready": estado},
        )

    # ── 4 · Búsqueda global ──────────────────────────────────────────────────
    def paso_4() -> SmokeResult:
        hits, s = _timed(lambda: store.search(query_vector, top_k=top_k))
        # §3.2 pide conservar la semántica del score nativo. Se anota aquí para
        # que la comparativa no acabe con tres números en la misma columna sin
        # decir que uno de ellos se ordena al revés.
        semantica = (
            f"{hits[0].score_kind} ({'mayor' if hits[0].higher_is_better else 'menor'} es mejor)"
            if hits else "sin resultados"
        )
        posiciones_ok = [hit.rank for hit in hits] == list(range(1, len(hits) + 1))
        return SmokeResult(
            4, "Búsqueda global", f"{top_k} resultados, con posición y score tipado",
            passed=len(hits) == top_k and posiciones_ok,
            action=f"search(vector[{dim}], top_k={top_k}, filters=[])  · sin filtro",
            observed=f"{len(hits)} resultados · score: {semantica}", seconds=s,
            detail={
                "top_1": hits[0].record_id if hits else None,
                "score_kind": hits[0].score_kind if hits else None,
                "higher_is_better": hits[0].higher_is_better if hits else None,
            },
        )

    # ── 5 · Búsqueda con filtro, ejecutado por el motor ──────────────────────
    def paso_5() -> SmokeResult:
        descripcion = " y ".join(f"{c.field} {c.operator} {c.value!r}" for c in filters)
        accion = f"search(vector[{dim}], top_k={top_k}, filters=[{descripcion}])"
        try:
            hits, s = _timed(lambda: store.search(query_vector, top_k=top_k, filters=filters))
        except UnsupportedFilterError as error:
            return SmokeResult(
                5, "Filtro nativo", descripcion,
                passed=False, action=accion, observed=f"NO SOPORTADO: {error}",
                detail={"requisito_duro": True},
            )
        cumplen = [
            hit for hit in hits
            if all(_hit_matches(hit.payload, condition) for condition in filters)
        ]
        # Contra qué clave se auditó cada campo, y qué valores trajo. Va a la
        # tabla y no solo al código porque es lo que distingue "el filtro falló"
        # de "el filtro funcionó pero comparé la clave equivocada" -que es
        # exactamente el fallo que tuvo esta función-. El §8 pide comprobar que
        # "las consultas filtradas nunca devuelven otra marca", y sin saber qué
        # clave se miró, la comprobación no se sostiene.
        auditado = {
            c.field: (
                audited_key(hits[0].payload, c.field) if hits
                else f"{c.field}_normalized"
            )
            for c in filters
        }
        devueltos = {
            campo: sorted({str(hit.payload.get(clave)) for hit in hits})[:5]
            for campo, clave in auditado.items()
        }
        detalle = " · ".join(
            f"{auditado[campo]} = {', '.join(valores) or '(ninguno)'}"
            for campo, valores in devueltos.items()
        )
        return SmokeResult(
            5, "Filtro nativo", descripcion,
            passed=len(hits) > 0 and len(cumplen) == len(hits),
            action=f"{accion}  → el motor filtra por {', '.join(auditado.values())}",
            observed=(
                f"{len(hits)} resultados, {len(cumplen)} cumplen · "
                f"auditado contra {detalle}"
            ),
            seconds=s,
            detail={"claves_auditadas": auditado, "valores_devueltos": devueltos},
        )

    # ── 6 · Lectura directa por record_id ────────────────────────────────────
    def paso_6() -> SmokeResult:
        objetivo = points[0].record_id
        punto, s = _timed(lambda: store.get(objetivo))
        return SmokeResult(
            6, "Lectura por record_id", f"devuelve {objetivo}",
            passed=punto is not None and punto.record_id == objetivo,
            action=f"get({objetivo!r})",
            observed=(
                f"encontrado · payload con {len(punto.payload)} claves"
                if punto else "no encontrado"
            ),
            seconds=s,
        )

    # ── 7 · Borrado, comprobado desde la búsqueda y no desde el borrado ──────
    def paso_7() -> SmokeResult:
        objetivo = points[0].record_id
        _, s = _timed(lambda: store.delete(objetivo))
        sigue = store.get(objetivo) is not None
        n = store.count()
        return SmokeResult(
            7, "Borrado", f"{objetivo} desaparece y count() baja a {esperado_n - 1}",
            passed=not sigue and n == esperado_n - 1,
            action=f"delete({objetivo!r}) → get(mismo id) → count()",
            observed=f"{'sigue presente' if sigue else 'borrado'}, count() = {n}",
            seconds=s,
        )

    for paso in (paso_1, paso_2, paso_3, paso_4, paso_5, paso_6, paso_7):
        try:
            results.append(paso())
        except Exception as error:  # el guion sigue: la fila vale como evidencia
            numero = len(results) + 1
            results.append(SmokeResult(
                numero, f"Paso {numero}", "—",
                passed=False, action="(la llamada lanzó antes de completarse)",
                observed=f"{type(error).__module__}.{type(error).__name__}: {error}",
            ))

    for step, name, action, expected in SMOKE_MANUAL_STEPS:
        results.append(SmokeResult(
            step, name, expected, passed=None, action=action, observed="(pendiente)"
        ))

    return results


# ═══════════════ Los tres pasos manuales, con su forma de anotarse ═══════════
# Ejecutarlos es de quien tiene la terminal, pero decidir si el motor los pasa
# no puede quedar en una impresión: cada uno tiene aquí la función que convierte
# lo observado en la fila de la tabla, contra el `expected` que se escribió
# antes de medir.

MANUAL_STEP_NUMBERS = frozenset(step for step, *_ in SMOKE_MANUAL_STEPS)


def record_manual(
    results: list[SmokeResult],
    step: int,
    *,
    observed: str,
    passed: bool,
    manual_steps: frozenset[int] | set[int] | None = None,
) -> list[SmokeResult]:
    """Rellena una de las tres filas manuales con lo que se observó fuera.

    Anotar pasando por aquí y no editando la tabla tiene una consecuencia que se
    ve poco y vale mucho: la fila ya trae escrito su `expected` desde antes de
    medir, así que lo observado se apunta **contra el criterio**, no en lugar de
    él. Una tabla editada a mano pierde esa mitad y deja un ✅ sin decir contra
    qué se comparó.

    Falla si el paso no es de los manuales: un ✅ escrito encima de un paso
    automático taparía justo lo que el guion había medido. `manual_steps` existe
    porque el guion del índice definitivo (`aceptacion.py`) numera los suyos de
    otra forma; por defecto valen los de la prueba de humo.
    """
    permitidos = MANUAL_STEP_NUMBERS if manual_steps is None else manual_steps
    if step not in permitidos:
        raise ValueError(
            f"El paso {step} no es manual en este guion. Los manuales son "
            f"{sorted(permitidos)}; el resto los mide el guion."
        )
    for result in results:
        if result.step == step:
            result.observed = observed
            result.passed = passed
            return results
    raise ValueError(f"No hay ninguna fila con el paso {step} en estos resultados.")


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Lo que se lee de una colección en un instante. No escribe nada.

    Dos de estas —una antes del reinicio y otra después— son todo el paso 8.
    Guarda el score junto al id y no solo el id porque es lo que separa las dos
    explicaciones posibles de un top-k reordenado; ver `PersistenceCheck`.
    """

    count: int
    hits: tuple[tuple[str, float], ...] = ()

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(record_id for record_id, _ in self.hits)

    @property
    def scores(self) -> dict[str, float]:
        return dict(self.hits)


def read_snapshot(
    store: VectorStore,
    query_vector: Sequence[float],
    *,
    top_k: int = 10,
    filters: Sequence[FilterCondition] = (),
) -> Snapshot:
    """La lectura del paso 8: solo lee.

    Volver a llamar a `run_smoke_test` NO sirve para esto: recrea la colección y
    borraría justo los datos cuya supervivencia se quiere comprobar.
    """
    hits = store.search(query_vector, top_k=top_k, filters=filters)
    return Snapshot(
        count=store.count(),
        hits=tuple((hit.record_id, float(hit.score)) for hit in hits),
    )


@dataclass(frozen=True, slots=True)
class PersistenceCheck:
    """El paso 8 con el recuento, el conjunto y el orden separados.

    Compararlo todo con un `==` entre listas mete tres preguntas distintas en un
    solo booleano, y las tres tienen consecuencias distintas:

    | Qué cambia | Qué significa |
    |---|---|
    | El recuento | Se perdieron puntos. La persistencia falla y no hay más que hablar |
    | El conjunto de ids | El índice vuelve distinto: el vecindario del mismo vector ya no es el mismo |
    | Solo el orden | El vecindario sobrevive y lo que cambia es el desempate |

    Lo tercero **no es un fallo de persistencia**. Una búsqueda aproximada no
    promete un orden total entre vecinos que empatan, y al reiniciar, un motor
    que recarga o recompacta sus segmentos puede recorrerlos en otro orden. La
    prueba de que es eso y no otra cosa es `max_score_shift`: si vuelven los
    mismos ids con el mismo score y en otro orden, lo que cambió fue el
    desempate; si el score se mueve, el índice no es el mismo.

    Por eso `passed` sigue el criterio que la fila 8 llevaba escrito **antes** de
    medir —*"mismo count() y mismos ids en el top-10"*—, que habla de ids y no de
    orden. Endurecerlo ahora para que un reordenamiento suspenda sería cambiar la
    vara al ver el resultado.
    """

    before: Snapshot
    after: Snapshot

    @property
    def same_count(self) -> bool:
        return self.before.count == self.after.count

    @property
    def same_set(self) -> bool:
        return set(self.before.ids) == set(self.after.ids)

    @property
    def same_order(self) -> bool:
        return self.before.ids == self.after.ids

    @property
    def lost(self) -> tuple[str, ...]:
        """Ids que estaban en el top-k y ya no aparecen."""
        return tuple(i for i in self.before.ids if i not in set(self.after.ids))

    @property
    def gained(self) -> tuple[str, ...]:
        return tuple(i for i in self.after.ids if i not in set(self.before.ids))

    @property
    def overlap(self) -> float:
        """Solapamiento entre los dos top-k. Es la misma cuenta que el recall
        contra un oráculo, con la lectura de «antes» haciendo de oráculo."""
        if not self.before.ids:
            return 0.0
        comunes = set(self.before.ids) & set(self.after.ids)
        return len(comunes) / len(self.before.ids)

    @property
    def moved(self) -> int:
        """Cuántas posiciones ocupa otro id después del reinicio."""
        return sum(
            1 for a, b in zip(self.before.ids, self.after.ids, strict=False) if a != b
        )

    @property
    def max_score_shift(self) -> float | None:
        """Cuánto se movió el score de los ids que están en las dos lecturas.

        Cerca de cero con el orden cambiado ⇒ el motor devuelve las mismas
        distancias y solo desempata distinto. Devuelve `None` si no hay ningún
        id en común, porque entonces no hay nada que comparar.
        """
        antes, despues = self.before.scores, self.after.scores
        comunes = set(antes) & set(despues)
        if not comunes:
            return None
        return max(abs(antes[i] - despues[i]) for i in comunes)

    @property
    def passed(self) -> bool:
        return bool(self.before.hits) and self.same_count and self.same_set

    def verdict(self) -> str:
        """La frase que va a la fila 8 de la comparativa."""
        if not self.before.hits:
            return "⚠️ no había nada que comprobar: la lectura de «antes» salió vacía"
        if not self.same_count:
            return (
                f"❌ el recuento no sobrevive: {self.before.count} → {self.after.count}"
            )
        k = len(self.before.ids)
        if self.same_order:
            return (
                f"✅ idéntico — count() {self.after.count} y los mismos {k} ids "
                f"en el mismo orden"
            )
        if self.same_set:
            desvio = self.max_score_shift
            explicacion = (
                "y el score no se mueve: cambia el desempate, no el índice"
                if desvio is not None and desvio < 1e-6
                else f"y el score se mueve hasta {desvio:.2e}"
            )
            return (
                f"⚠️ los mismos {k} ids en distinto orden — {self.moved} posiciones "
                f"cambian {explicacion}"
            )
        return (
            f"❌ el top-{k} cambia de ids — salen {len(self.lost)} y entran "
            f"{len(self.gained)} (solapamiento {self.overlap:.0%})"
        )


def persistence_check(before: Snapshot, after: Snapshot) -> PersistenceCheck:
    """El paso 8, a partir de las dos lecturas."""
    return PersistenceCheck(before=before, after=after)


def error_quality(error: BaseException, *, sdk_package: str) -> tuple[str, bool]:
    """El paso 9: qué cuenta el SDK cuando el motor no está.

    No mide que el motor se caiga —se cae porque lo tiras—, sino qué te llega
    cuando pasa. Y distingue tres cosas que la palabra "tipada" junta:

    | Origen de la excepción | Qué implica para NB05 |
    |---|---|
    | `builtins` | Genérica: hay que adivinar por el mensaje. Suspende |
    | El paquete del SDK | El motor tiene su jerarquía de errores y se puede capturar por tipo |
    | Otro paquete (el transporte) | Tipada, pero el SDK deja subir el error de red tal cual |

    El criterio escrito en la fila 9 antes de medir era *"excepción tipada y
    legible"*, así que las dos últimas la pasan. La diferencia entre ellas no
    cambia el veredicto pero sí el trabajo de §3.3: capturar
    `grpc._channel._InactiveRpcError` ata el manejo de errores a la capa de
    transporte del SDK, que no es parte de su contrato público.
    """
    modulo = type(error).__module__
    raiz = modulo.split(".")[0]
    nombre = f"{modulo}.{type(error).__name__}"
    mensaje = " ".join(str(error).split())[:160]
    if raiz == "builtins":
        return f"❌ genérica — {nombre}: {mensaje}", False
    if raiz == sdk_package:
        return f"✅ tipada por el SDK — {nombre}: {mensaje}", True
    return (
        f"⚠️ tipada pero de {raiz}, no del SDK ({sdk_package}) — {nombre}: {mensaje}",
        True,
    )


def audited_key(payload: Mapping[str, Any], field: str) -> str:
    """Contra qué clave del payload hay que auditar el filtro de `field`.

    D03 guarda el valor **crudo** y materializa además una clave normalizada, y
    los adaptadores filtran contra la segunda. La auditoría tiene que mirar esa
    misma: comparar `payload["brand"] == "Einhell"` con el valor pedido
    `"einhell"` da siempre falso y marcaría como roto un filtro que funciona.

    Si el punto no lleva clave derivada —un esquema que no la materialice, o un
    motor que no la devuelva— se cae al campo crudo, que es lo único que hay.
    """
    derivada = f"{field}_normalized"
    return derivada if payload.get(derivada) is not None else field


def _hit_matches(payload: dict[str, Any], condition: FilterCondition) -> bool:
    """¿El resultado cumple de verdad lo que se pidió?

    Comprobar el filtro **sobre lo devuelto** es lo que distingue un motor que
    filtra de uno que dice filtrar. No sustituye al filtro del motor: lo audita,
    y por eso compara contra la misma clave que el motor usó.
    """
    valor = payload.get(audited_key(payload, condition.field))
    if valor is None:
        return False
    if condition.operator == "equals":
        return str(valor) == condition.value
    return condition.value in str(valor)


# PROBE_ROLES: qué papel juega cada sonda en la tabla de filtros. No es
# decorativo: `contains_level` los usa para deducir el nivel del motor, y
# `probe_filters` para no mezclar lo obligatorio con lo exploratorio.
PROBE_ROLES = ("obligatorio", "referencia", "palabra", "fragmento")


@dataclass(frozen=True, slots=True)
class FilterProbe:
    """Un caso de filtro que se le pide al motor, con cómo hay que leerlo."""

    name: str
    query_vector: Sequence[float]
    conditions: Sequence[FilterCondition]
    role: str = "obligatorio"
    note: str = ""
    # Cuántos productos cumplen la condición según el CATÁLOGO, calculado en
    # pandas sin motor de por medio. Es el oráculo que convierte un cero en
    # información: sin él, "0 resultados" no distingue «esa marca no tiene
    # productos» de «el filtro está roto», y las dos cosas se leen igual.
    expected: int | None = None

    def __post_init__(self) -> None:
        if self.role not in PROBE_ROLES:
            raise ValueError(f"role debe ser uno de {PROBE_ROLES}")
        if not self.conditions:
            raise ValueError("Una sonda de filtro necesita al menos una condición.")
        if self.expected is not None and self.expected < 0:
            raise ValueError("El oráculo no puede ser negativo.")


def probe_filters(
    store: VectorStore, probes: Sequence[FilterProbe], *, top_k: int = 10
) -> pd.DataFrame:
    """Todos los filtros que hay que verificar, en una sola tabla.

    El paso 5 del guion prueba **una** consulta con **igualdad**, que es el
    mínimo del enunciado. Esta tabla cubre lo que ese paso deja fuera:

    - **Las cuatro consultas de `consultas_filtradas.csv`**, no solo la primera.
      El §5 las pide como evidencia mínima —*"resultados que cumplan la marca en
      las cuatro consultas"*— y el §8 lo repite como criterio de corrección:
      *"las consultas filtradas nunca devuelven otra marca"*.
    - **El `contains` sobre color**, que la sección B declaró requisito duro y
      que ningún paso del guion ejerce. Sin esto, el requisito queda afirmado
      desde la documentación de cada SDK en vez de medido.

    Un motor que no soporte una condición no interrumpe la tabla: la fila queda
    con `n_resultados` a nulo y el motivo en `valores`.
    """
    rows = []
    for probe in probes:
        descripcion = " y ".join(
            f"{c.field} {c.operator} {c.value!r}" for c in probe.conditions
        )
        try:
            hits = store.search(
                probe.query_vector, top_k=top_k, filters=probe.conditions
            )
        except UnsupportedFilterError as error:
            rows.append({
                "caso": probe.name, "papel": probe.role, "filtro": descripcion,
                "n_resultados": None, "cumplen": None, "todos_cumplen": None,
                "clave_auditada": "—", "valores": f"NO SOPORTADO: {error}",
                "como_se_lee": probe.note,
            })
            continue
        cumplen = [
            hit for hit in hits
            if all(_hit_matches(hit.payload, c) for c in probe.conditions)
        ]
        claves = {
            audited_key(hits[0].payload, c.field) if hits else f"{c.field}_normalized"
            for c in probe.conditions
        }
        valores = sorted({
            str(hit.payload.get(audited_key(hit.payload, c.field)))
            for hit in hits for c in probe.conditions
        })
        rows.append({
            "caso": probe.name,
            "papel": probe.role,
            "filtro": descripcion,
            "n_en_catalogo": probe.expected,
            # Tope en `top_k`: esta tabla clasifica el motor y verifica que lo
            # devuelto cumpla. El alcance real ya lo midió la sección B sobre el
            # catálogo entero, sin motor de por medio.
            "n_resultados": len(hits),
            "cumplen": len(cumplen),
            "todos_cumplen": len(hits) > 0 and len(cumplen) == len(hits),
            "veredicto": _verdict(probe, len(hits), top_k),
            "clave_auditada": ", ".join(sorted(claves)),
            "valores": ", ".join(valores[:4]) or "(ninguno)",
            "como_se_lee": probe.note,
        })
    return pd.DataFrame(rows)


def _verdict(probe: FilterProbe, devueltos: int, top_k: int) -> str:
    """Confronta lo que devolvió el motor con lo que dice el catálogo.

    Es lo que convierte un cero en información. El enunciado usa esta misma idea
    para la fidelidad ANN —*"comparar IDs con un oráculo exacto"*—; aquí el
    oráculo es pandas sobre el mismo corpus que se ingirió.

    Un matiz que hay que tener presente al leerlo: **el oráculo calcula
    `contains` como subcadena**. Un motor que filtre por palabras (nivel 2)
    devolverá legítimamente menos en las sondas de `contains`, y eso no es un
    fallo sino la propiedad que lo hace mejor. Por eso la discrepancia se
    describe en vez de sentenciarse cuando la sonda no es de igualdad.
    """
    if probe.expected is None:
        return "— sin oráculo"

    esperado_visible = min(probe.expected, top_k)
    if probe.expected == 0:
        return (
            "✅ ausencia real — el catálogo tampoco tiene ninguno"
            if devueltos == 0
            else f"❌ el catálogo no tiene ninguno y el motor devolvió {devueltos}"
        )
    # La sonda `fragmento` no aprueba ni suspende: CLASIFICA. Buscar un trozo que
    # no es palabra suelta debe fallar en un motor que tokeniza y acertar en uno
    # de subcadena, así que sus dos resultados son correctos y dicen cosas
    # distintas. Tratarla como al resto la marcaría como filtro roto justo cuando
    # el motor se comporta como queremos.
    if probe.role == "fragmento":
        return (
            f"✅ {devueltos} de {probe.expected} — encuentra el fragmento ⇒ "
            f"subcadena literal (nivel 3)"
            if devueltos > 0 else
            f"✅ 0 de {probe.expected} — NO encuentra el fragmento ⇒ "
            f"filtra por palabras (nivel 2)"
        )

    es_contains = any(c.operator == "contains" for c in probe.conditions)
    if devueltos == 0:
        if es_contains:
            return (
                f"❌ el catálogo tiene {probe.expected} y el motor devolvió 0 — "
                f"revisar el índice de texto antes de culpar al filtro"
            )
        return f"❌ FILTRO ROTO — el catálogo tiene {probe.expected} y el motor devolvió 0"
    if devueltos == esperado_visible:
        return f"✅ coincide con el catálogo ({probe.expected}, topado en {top_k})"
    if devueltos < esperado_visible:
        if es_contains:
            return (
                f"⚠️ {devueltos} de {esperado_visible} — legítimo si el motor filtra "
                f"por palabras: el oráculo cuenta subcadenas"
            )
        return f"❌ devuelve de menos: {devueltos} de {esperado_visible}"
    return f"❌ devuelve de más: {devueltos} sobre {esperado_visible} del catálogo"


def contains_level(tabla: pd.DataFrame) -> str:
    """Qué clase de `contains` sabe hacer el motor, deducido de la tabla.

    Hay tres niveles y la diferencia decide cuánto vale el motor aquí:

    1. **No soportado** — descarta al motor por el requisito duro de la sección B.
    2. **Por palabras** — un índice de texto tokeniza el valor. Alcanza los
       compuestos de B.1 (`Negro/Rojo`) y **no** produce los falsos positivos de
       B.3 (`rosa` no casa con `rosado`). Es el mejor de los tres.
    3. **Subcadena literal** (`LIKE '%x%'`) — alcanza lo mismo que el nivel 2 y
       además trae los falsos positivos que B.3 cuantificó.

    Distinguir 2 de 3 sin creerse la documentación es lo que hace la sonda con
    papel `fragmento`: un trozo de un valor almacenado que **no es una palabra
    suelta** —`"negr"` de `"negro"`—. Un motor de subcadena lo encuentra; uno de
    palabras, no.
    """
    por_papel = tabla.set_index("papel")["n_resultados"].to_dict()
    palabra, fragmento = por_papel.get("palabra"), por_papel.get("fragmento")
    if palabra is None or pd.isna(palabra):
        return "❌ NIVEL 1 · no soporta `contains` sobre metadatos → descartado por el requisito duro"
    if int(palabra) == 0:
        return "⚠️ acepta el filtro pero no encuentra nada — revisar el índice de texto antes de concluir"
    if fragmento is not None and not pd.isna(fragmento) and int(fragmento) > 0:
        return "✅ NIVEL 3 · subcadena literal — alcanza B.1 y trae los falsos positivos de B.3"
    return "✅ NIVEL 2 · por palabras — alcanza B.1 SIN los falsos positivos de B.3"


def smoke_display(results: Sequence[SmokeResult], *, motor: str, ancho_px: int = 360):
    """La misma tabla, pero legible dentro del notebook.

    pandas recorta las celdas largas con `...`, y en esta tabla lo largo es
    precisamente lo que hay que leer: `esperado` dice qué exige el guion y
    `observado` trae el mensaje de la excepción cuando un paso falla. Un fallo
    que se lee como `ConnectionError: connection ref...` no sirve de evidencia.

    Devuelve un `Styler` que ajusta el texto en varias líneas en lugar de
    cortarlo. El dato es el mismo que da `smoke_table`, que es el que va al
    artefacto: esto solo cambia cómo se muestra.
    """
    return (
        smoke_table(results, motor=motor)
        .style
        .hide(axis="index")
        .set_properties(**{
            "white-space": "pre-wrap",     # respeta saltos y envuelve
            "text-align": "left",
            "vertical-align": "top",
        })
        .set_properties(
            subset=["que_ha_hecho", "esperado", "observado"],
            **{"max-width": f"{ancho_px}px"},
        )
        .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}])
    )


def smoke_table(results: Sequence[SmokeResult], *, motor: str) -> pd.DataFrame:
    """Los resultados como la fila de la comparativa que va al artefacto."""
    return pd.DataFrame([
        {
            "motor": motor,
            "paso": result.step,
            "comprobacion": result.name,
            # Va justo después de `comprobacion` a propósito: primero qué se
            # comprueba, luego con qué llamada y con qué parámetros. Sin esto,
            # "el paso 5 pasa" no dice contra qué filtro ni con qué top_k.
            "que_ha_hecho": result.action,
            "esperado": result.expected,
            "observado": result.observed,
            "resultado": {True: "✅ pasa", False: "❌ falla", None: "✍️ manual"}[result.passed],
            "segundos": round(result.seconds, 3) if result.seconds is not None else None,
        }
        for result in results
    ])


def smoke_differences(
    runs: Mapping[str, Sequence[SmokeResult]]
) -> pd.DataFrame:
    """Los pasos en los que los motores NO se comportaron igual.

    La tabla de ✅/❌ puede salir llena de aprobados y aun así esconder tres
    motores muy distintos: aprobar el paso 3 no distingue al que informa de su
    estado de indexación del que no lo reporta, y aprobar el 4 no distingue una
    similitud de una distancia. Eso vive en `observado`, no en `resultado`.

    Esta tabla deja **solo** las filas donde `observado` difiere entre motores,
    que son las únicas que pueden decidir R03. Las que coinciden no separan a
    nadie y ocupan sitio.
    """
    if not runs:
        return pd.DataFrame()
    tabla = pd.concat(
        [smoke_table(pasos, motor=motor) for motor, pasos in runs.items()]
    )
    ancha = tabla.pivot_table(
        index=["paso", "comprobacion"], columns="motor",
        values="observado", aggfunc="first",
    )
    difieren = ancha.apply(lambda fila: fila.nunique(dropna=False) > 1, axis=1)
    return ancha[difieren]


# ═════════════════════ La pasada de un motor, en disco ═══════════════════════
# Hasta aquí el resultado de la prueba vivía solo en la memoria del kernel:
# cerrar el notebook obligaba a volver a levantar los tres motores para poder
# reescribir la comparativa. El §8 pide lo contrario —que los artefactos se
# regeneren desde un único comando— y el motor no está para eso: guardar la
# pasada en cuanto se produce es lo que hace que el artefacto no dependa de una
# sesión abierta.


def save_smoke(
    directory: Path | str,
    *,
    motor: str,
    results: Sequence[SmokeResult],
    filters: pd.DataFrame | None = None,
) -> Path:
    """Escribe la pasada de un motor en `directory/{motor}.json`.

    Se sobrescribe a propósito: la pasada buena es la última, y guardar un
    histórico invitaría a mezclar mediciones de dos configuraciones distintas en
    la misma tabla.
    """
    destino = Path(directory)
    destino.mkdir(parents=True, exist_ok=True)
    fichero = destino / f"{motor}.json"
    fichero.write_text(
        json.dumps(
            {
                "motor": motor,
                "pasos": [asdict(result) for result in results],
                "filtros": (
                    None if filters is None else filters.to_dict(orient="records")
                ),
            },
            ensure_ascii=False,
            indent=2,
            default=str,   # nada raro debería llegar aquí, pero no vale perder la pasada
        ),
        encoding="utf-8",
    )
    return fichero


def load_smoke(
    directory: Path | str,
) -> tuple[dict[str, list[SmokeResult]], dict[str, pd.DataFrame]]:
    """Devuelve lo guardado por `save_smoke`, listo para volver a la comparativa.

    Si el directorio no existe devuelve dos diccionarios vacíos en vez de fallar:
    la primera vez que se ejecuta el notebook todavía no hay nada medido, y eso
    no es un error.
    """
    origen = Path(directory)
    pasos: dict[str, list[SmokeResult]] = {}
    filtros: dict[str, pd.DataFrame] = {}
    if not origen.is_dir():
        return pasos, filtros
    for fichero in sorted(origen.glob("*.json")):
        guardado = json.loads(fichero.read_text(encoding="utf-8"))
        motor = guardado["motor"]
        pasos[motor] = [SmokeResult(**fila) for fila in guardado["pasos"]]
        if guardado.get("filtros") is not None:
            filtros[motor] = pd.DataFrame(guardado["filtros"])
    return pasos, filtros
