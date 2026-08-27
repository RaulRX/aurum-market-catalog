"""El puerto: lo que la prueba de humo necesita de cualquier motor.

Los diez pasos del guion de NB04 piden exactamente siete
operaciones. Están aquí y no en cada adaptador para que el guion no pueda
escribirse a medida de un motor concreto.

Dos decisiones de esta interfaz merecen explicación, porque son las que hacen
que la comparativa mida el motor y no el código que lo envuelve:

- **El filtro viaja como dato, no como código.** `FilterCondition` describe la
  condición en los términos de la sección B —campo, valor y si se compara con
  igualdad o con *contiene*—, y cada adaptador la traduce a su lenguaje nativo.
  Esa traducción es justo donde se ve qué motor sabe hacer qué: quien no pueda
  expresar `contains` sobre un metadato lanza `UnsupportedFilterError` y queda
  descartado por el requisito duro, sin necesidad de interpretar su documentación.
- **El vector y el payload van juntos en `Point`.** El identificador es
  `record_id` (UUIDv5, lo impone `README_DATOS`), que es lo que hace la
  idempotencia gratis: reingerir el mismo punto lo sobrescribe en vez de
  duplicarlo. Un motor que no acepte un id propio rompe el paso 3 del guion.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# Las dos políticas que decidió la sección B: `equals` para la marca (vocabulario
# cerrado) y `contains` para el color (texto libre con valores compuestos).
FILTER_OPERATORS = ("equals", "contains")

# Qué significa el número que devuelve cada motor. `unknown` existe para que un
# adaptador que aún no lo haya averiguado lo diga en vez de suponer `similarity`,
# que es la suposición que invertiría el orden en la mitad de los casos.
SCORE_KINDS = ("similarity", "distance", "unknown")

# Prefijo obligatorio de las colecciones de la prueba de humo. Adaptado de
# `validate_resource_name` de sesion_03 (condición 4 del plan).
COLLECTION_PREFIX = "aurum_humo"

# El índice de verdad vive bajo OTRO prefijo, y esa separación es la protección
# de verdad: el guion de humo recrea su colección en cada pasada, y con un solo
# prefijo común una errata en su nombre podría llevarse por delante los 15.000
# puntos del índice bueno. Con dos prefijos, ni siquiera equivocándose.
CATALOG_PREFIX = "aurum_catalogo"


def catalog_collection_name(
    *, model: str, template: str, dim: int, prefix: str = CATALOG_PREFIX
) -> str:
    """El nombre del índice definitivo, con su contrato dentro.

    Lleva modelo, plantilla y dimensión porque los tres forman parte del
    contrato del índice: cambiar cualquiera de ellos invalida los vectores
    guardados, y un nombre que no lo diga permite que convivan dos colecciones
    incompatibles sin que nada avise.

    Es además la mitad del control de versionado que `config.yaml` se
    compromete a mantener (criterio 6 de sesión_01): migrar es **construir la
    colección nueva al lado y cambiar el puntero**, nunca reindexar sobre la
    viva. Con el contrato en el nombre, las dos pueden coexistir mientras dure
    la migración.

    >>> catalog_collection_name(model="gemini-embedding-2", template="A4", dim=768)
    'aurum_catalogo__gemini_embedding_2__A4__768'
    """
    if dim <= 0:
        raise ValueError(f"La dimensión tiene que ser positiva, no {dim}.")
    limpio = "".join(c if c.isalnum() else "_" for c in model).strip("_")
    return f"{prefix}__{limpio}__{template}__{dim}"


def guard_collection_name(name: str, *, prefix: str = COLLECTION_PREFIX) -> str:
    """Impide que un `recreate=True` se lleve por delante lo que no es de la prueba.

    El guion borra y recrea la colección en cada pasada, que es lo que hace la
    prueba repetible. Con un nombre mal escrito eso deja de ser una comodidad y
    pasa a ser una operación destructiva contra la colección equivocada —la de
    otro motor, o la del índice bueno— sin ninguna confirmación de por medio.

    Es la primera de dos barreras; la otra es `reset_allowed`.
    """
    normalizado = "".join(c for c in name.lower() if c.isalnum())
    esperado = "".join(c for c in prefix.lower() if c.isalnum())
    if not normalizado.startswith(esperado):
        raise ValueError(
            f"La colección {name!r} no empieza por el prefijo protegido {prefix!r}. "
            f"El guion de humo borra y recrea lo que toca, así que solo opera "
            f"sobre colecciones suyas."
        )
    return name


def reset_allowed(*, env_var: str = "AURUM_ALLOW_RESET") -> bool:
    """¿Está habilitado borrar y recrear? Copiado de `S03_ALLOW_RESET` (sesión 3).

    La sesión 3 no deja que una limpieza se active por descuido, y aquí vale
    igual: el prefijo protege de borrar *otra cosa*, pero no de borrar la
    colección correcta en el momento equivocado —después de una ingesta de
    15.000 puntos, por ejemplo—. Esta segunda barrera exige decirlo a propósito.

    Por defecto es `false`: la ausencia de la variable no habilita nada.
    """
    import os

    return os.getenv(env_var, "false").strip().lower() in {"1", "true", "yes", "si", "sí"}


def ensure_reset_allowed(collection: str, *, env_var: str = "AURUM_ALLOW_RESET") -> None:
    """Falla con un mensaje que dice qué hacer, en vez de con un permiso denegado."""
    if not reset_allowed(env_var=env_var):
        raise PermissionError(
            f"Recrear {collection!r} borra su contenido y {env_var} no está activa. "
            f"Ponla a `true` en el .env mientras dure la prueba de humo, o llama "
            f"con recreate=False para reutilizar la colección existente."
        )


class UnsupportedFilterError(NotImplementedError):
    """El motor no sabe ejecutar esta condición sobre un campo de payload.

    No es un fallo del adaptador: es el resultado de la comprobación. Se lanza
    para que la tabla de humo registre *qué* no puede el motor en vez de
    esconderlo tras un filtrado en Python, que es precisamente lo que el
    enunciado prohíbe.
    """


@dataclass(frozen=True, slots=True)
class FilterCondition:
    """Una condición de filtro, en los términos de la decisión y no del motor."""

    field: str
    value: str
    operator: str = "equals"

    def __post_init__(self) -> None:
        if self.operator not in FILTER_OPERATORS:
            raise ValueError(f"operator debe ser uno de {FILTER_OPERATORS}")
        if not self.field:
            raise ValueError("La condición necesita un campo.")


@dataclass(frozen=True, slots=True)
class Point:
    """Un producto tal y como se guarda: id estable, vector y payload."""

    record_id: str
    vector: Sequence[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchHit:
    """Un resultado, con la puntuación **nativa** del motor sin reinterpretar.

    El enunciado (§3.2) lo pide con estas palabras: *"conservad la semántica del
    score nativo; no comparéis como si fueran equivalentes una distancia y una
    similitud"*. Y no es una precaución teórica: con la misma métrica coseno,
    Qdrant y Milvus devuelven **similitud** —mayor es mejor— y Weaviate devuelve
    **distancia** —menor es mejor—. Un campo `score` a secas los pondría en la
    misma columna de la comparativa y el orden saldría invertido para uno de los
    tres sin que nada avisara.

    Por eso el número viaja siempre con qué es (`score_kind`) y en qué dirección
    se lee (`higher_is_better`). Convertirlos a una escala común se hará donde
    haga falta y a la vista, nunca aquí y de tapadillo.

    `rank` cubre la otra exigencia, la de §3.3: el resultado lleva su posición.
    """

    record_id: str
    score: float
    score_kind: str = "unknown"        # similarity | distance | unknown
    higher_is_better: bool = True
    rank: int = 0                      # 1-indexado, como lo lee una persona
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.score_kind not in SCORE_KINDS:
            raise ValueError(f"score_kind debe ser uno de {SCORE_KINDS}")


@runtime_checkable
class VectorStore(Protocol):
    """Lo que el guion de humo necesita saber hacer a un motor.

    Los pasos 8, 9 y 10 del guion —reinicio, motor apagado y consumo de
    recursos— no aparecen aquí a propósito: se ejecutan desde fuera del proceso
    y los mide una persona, no una llamada al SDK.
    """

    name: str
    """Identificador corto del motor, el que va a la fila de la tabla."""

    def server_version(self) -> str:
        """Versión que reporta el servidor. Va al informe: una tabla de humo sin
        versión no es reproducible."""
        ...

    def create_collection(self, *, dim: int, metric: str, recreate: bool = False) -> None:
        """Paso 1. Dimensión y métrica **explícitas**, nunca por defecto."""
        ...

    def upsert(self, points: Sequence[Point], *, batch_size: int) -> int:
        """Pasos 2 y 3. Escribe por lotes y devuelve cuántos puntos envió.

        Tiene que ser idempotente por `record_id`: repetir la llamada con los
        mismos puntos deja el mismo `count()`.
        """
        ...

    def count(self) -> int:
        """Cuántos puntos hay. Es el número que compara los pasos 2 y 3."""
        ...

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        filters: Sequence[FilterCondition] = (),
    ) -> list[SearchHit]:
        """Pasos 4 y 5. Con `filters`, la condición la ejecuta **el motor**.

        Lanza `UnsupportedFilterError` si no sabe expresar alguna condición. No
        se filtra en Python como alternativa: eso convertiría un motor que no
        cumple el requisito en uno que sí, que es el error que la prueba busca
        evitar.
        """
        ...

    def get(self, record_id: str) -> Point | None:
        """Paso 6. Lectura directa por id, sin pasar por la búsqueda."""
        ...

    def delete(self, record_id: str) -> None:
        """Paso 7. Después, el punto no debe aparecer en la búsqueda."""
        ...

    def close(self) -> None:
        """Suelta la conexión. Importante con 7,9 GB y los motores de uno en uno."""
        ...

    # ── opcional ─────────────────────────────────────────────────────────────
    # `index_ready` no está en el Protocol a propósito: no todos los motores
    # exponen el estado del índice, y exigirlo obligaría a los que no pueden a
    # devolver un `True` inventado. El guion lo llama con `getattr` y distingue
    # tres respuestas —listo, aún indexando, no lo reporta—, que es lo que §3.2
    # pide poder verificar "antes de aceptar consultas".
