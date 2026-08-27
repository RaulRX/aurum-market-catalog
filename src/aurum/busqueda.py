"""Interfaz común de recuperación y buscador denso exacto.

El enunciado (§3.3) pide *"una interfaz común -función, clase, API o comando-
que reciba una consulta y devuelva resultados normalizados"*. Ese contrato vive
aquí, no en cada implementación: `SearchResult` y `stable_top_k_indices` los
comparten el baseline léxico (`aurum.lexico`), este buscador denso y, más
adelante, el motor vectorial de NB04. Así una tabla comparativa puede mezclar
filas de sistemas distintos sin traducir formatos.

`DenseRetriever` es el **oráculo exacto**: recorre todos los vectores sin
índice aproximado. En NB02 es el buscador que compara modelos (la calidad de la
representación no debe medirse a través de la pérdida de un ANN); en NB06 es la
referencia contra la que se mide la fidelidad del ANN del motor.
"""
from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

DEFAULT_TOP_K = 10

# Regla 5 de experimentación: `cosine` y `dot` son similitudes (mayor = mejor);
# `l2` es una distancia (menor = mejor). El contrato de `SearchResult` obliga a
# declarar cuál de las dos cosas es el score, para no mezclarlas en una tabla.
METRICS = ("cosine", "dot", "l2")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Un documento recuperado, con su posición y su puntuación."""

    rank: int
    document_id: str
    score: float
    score_es_similitud: bool = True


class Retriever(Protocol):
    """Lo mínimo que debe ofrecer cualquier buscador del proyecto."""

    name: str

    def search(
        self, query_text: str, *, k: int = DEFAULT_TOP_K
    ) -> tuple[SearchResult, ...]:
        """Devuelve los `k` documentos mejor puntuados para la consulta."""
        ...


def stable_top_k_indices(scores: NDArray[Any], *, k: int) -> NDArray[np.intp]:
    """Índices del top-k por puntuación, con el orden original como desempate.

    El desempate determinista es lo que hace que dos ejecuciones den métricas
    idénticas — un criterio de verificación explícito en NB09."""
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scores debe ser un vector unidimensional no vacío.")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k debe ser un entero positivo.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contiene NaN o infinito.")

    effective_k = min(k, scores.size)
    # np.lexsort ordena por la última clave: primero por -score, y a igualdad
    # de score, por el índice original ascendente.
    order = np.lexsort((np.arange(scores.size), -scores))
    return order[:effective_k]


def build_results(
    indices: NDArray[np.intp],
    scores: NDArray[Any],
    document_ids: Sequence[str],
    *,
    score_es_similitud: bool = True,
) -> tuple[SearchResult, ...]:
    """Empaqueta un top-k ya ordenado en el contrato común."""
    return tuple(
        SearchResult(
            rank=rank,
            document_id=document_ids[int(index)],
            score=float(scores[int(index)]),
            score_es_similitud=score_es_similitud,
        )
        for rank, index in enumerate(indices, start=1)
    )


def _validate_matrix(
    vectors: NDArray[Any], document_ids: Sequence[str]
) -> tuple[NDArray[np.float32], tuple[str, ...]]:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("vectors debe ser una matriz 2D no vacía.")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors contiene NaN o infinito.")
    ids = tuple(str(document_id) for document_id in document_ids)
    if len(ids) != matrix.shape[0]:
        raise ValueError(
            f"document_ids y vectores deben alinearse: {len(ids)} != {matrix.shape[0]}."
        )
    if len(set(ids)) != len(ids):
        raise ValueError("document_ids contiene identificadores duplicados.")
    return matrix, ids


def _l2_normalize(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Un vector nulo no se puede normalizar: se deja como está en vez de
    # producir NaN, que rompería el top-k aguas abajo.
    return np.divide(matrix, np.where(norms == 0, 1.0, norms)).astype(np.float32)


class DenseRetriever:
    """Búsqueda densa exacta por producto escalar o distancia euclídea.

    Recorre los `n` vectores en cada consulta. Con 15.000 × 768 en `float32`
    (~46 MB) eso son unos milisegundos, así que no compensa un índice: lo que
    aquí interesa es que el resultado sea **exacto**, para que la comparación
    entre modelos de NB02 no arrastre la pérdida de ningún ANN.

    `metric` cambia qué significa el score y, sin normalizar, también el orden:

    - `cosine` normaliza documentos y consulta antes del producto escalar.
    - `dot` no normaliza nada, así que premia los vectores de norma grande.
    - `l2` ordena por distancia euclídea ascendente.

    Con vectores ya L2-normalizados las tres dan el **mismo ranking**. Que se
    cumpla es la verificación de normalización que pide NB02; que deje de
    cumplirse significa que la normalización no se está aplicando."""

    def __init__(
        self,
        vectors: NDArray[Any],
        document_ids: Sequence[str],
        *,
        metric: str = "cosine",
        name: str = "dense",
    ) -> None:
        if metric not in METRICS:
            raise ValueError(f"metric debe ser uno de {METRICS}")
        matrix, self._document_ids = _validate_matrix(vectors, document_ids)
        self.name = name
        self.metric = metric
        self._matrix = _l2_normalize(matrix) if metric == "cosine" else matrix

    @property
    def document_ids(self) -> tuple[str, ...]:
        """IDs indexados, en el mismo orden que las filas de la matriz."""
        return self._document_ids

    @property
    def dim(self) -> int:
        """Dimensión de los vectores indexados."""
        return int(self._matrix.shape[1])

    @property
    def score_es_similitud(self) -> bool:
        """`False` para `l2`: su score es una distancia, no una similitud."""
        return self.metric != "l2"

    def _scores(self, query_vector: NDArray[Any]) -> NDArray[np.float32]:
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.size != self._matrix.shape[1]:
            raise ValueError(
                f"La consulta tiene dimensión {query.size} y el índice {self._matrix.shape[1]}."
            )
        if not np.isfinite(query).all():
            raise ValueError("El vector de consulta contiene NaN o infinito.")

        if self.metric == "l2":
            return np.linalg.norm(self._matrix - query, axis=1).astype(np.float32)
        if self.metric == "cosine":
            query = _l2_normalize(query.reshape(1, -1))[0]
        return (self._matrix @ query).astype(np.float32)

    def search_vector(
        self, query_vector: NDArray[Any], *, k: int = DEFAULT_TOP_K
    ) -> tuple[SearchResult, ...]:
        """Top-k para una consulta **ya codificada**.

        Es la entrada natural del buscador denso: codificar las consultas en
        un solo lote fuera de aquí evita una llamada al modelo (o a la API) por
        consulta."""
        scores = self._scores(query_vector)
        # Con `l2` se ordena por distancia ascendente, pero el score reportado
        # sigue siendo la distancia: negar solo sirve para ordenar.
        ranking_scores = -scores if self.metric == "l2" else scores
        indices = stable_top_k_indices(ranking_scores, k=k)
        return build_results(
            indices, scores, self._document_ids, score_es_similitud=self.score_es_similitud
        )


def rank_queries_dense(
    retriever: DenseRetriever,
    query_ids: Sequence[str],
    query_vectors: NDArray[Any],
    *,
    k: int = DEFAULT_TOP_K,
) -> dict[str, list[str]]:
    """Ejecuta un lote de consultas ya codificadas → `{query_id: [doc_id, ...]}`.

    Es la forma que espera `evaluacion.evaluate_rankings`. Guardar los IDs y no
    solo la métrica es lo que permite atribuir errores en NB09 (Regla 3)."""
    matrix = np.asarray(query_vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(query_ids):
        raise ValueError(
            f"Se esperaban {len(query_ids)} vectores de consulta, llegaron {matrix.shape}."
        )
    return {
        str(query_id): [
            result.document_id for result in retriever.search_vector(vector, k=k)
        ]
        for query_id, vector in zip(query_ids, matrix, strict=True)
    }


def results_frame(
    results: Mapping[str, Sequence[SearchResult]],
    *,
    metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> pd.DataFrame:
    """Aplana resultados al formato normalizado que exige el enunciado §3.3.

    Cada fila lleva `query_id`, posición, `product_id`, score y los metadatos
    que se le adjunten por documento (título y marca, típicamente). Es la
    estructura de la que salen los CSV de entrega de NB09."""
    filas = []
    for query_id, resultados in results.items():
        for result in resultados:
            fila: dict[str, object] = {
                "query_id": query_id,
                "rank": result.rank,
                "product_id": result.document_id,
                "score": round(result.score, 6),
                "score_es_similitud": result.score_es_similitud,
            }
            if metadata is not None:
                fila.update(metadata.get(result.document_id, {}))
            filas.append(fila)
    return pd.DataFrame(filas)


# ═════════ NB05 · la puerta de entrada al sistema, sobre el motor ════════════
# §3.3 pide "una interfaz comun -funcion, clase, API o comando- que reciba una
# consulta y devuelva resultados normalizados", con `product_id`, posicion,
# titulo, metadatos y score. Lo que sigue es ese contrato y su implementacion
# contra la base vectorial.


class BusquedaError(Exception):
    """Algo impidió responder a la consulta.

    Existe para que quien llame pueda distinguir **no hay resultados** —que es
    una respuesta legítima y se devuelve como lista vacía— de **no he podido
    buscar**, que es un fallo. El enunciado (§3.3) pide tratar los dos casos
    explícitamente, y meterlos en el mismo canal los confunde.
    """


class MotorNoDisponible(BusquedaError):
    """La base vectorial no responde: caída, puerto cerrado o red."""


class BusquedaAgotada(BusquedaError):
    """La búsqueda superó el tiempo declarado y se cortó.

    Se separa de `MotorNoDisponible` a propósito: un motor que no está exige
    revisar el despliegue, y uno que tarda de más exige revisar la consulta o la
    carga. Tratarlos igual borra esa diferencia justo cuando hace falta.
    """


@dataclass(frozen=True, slots=True)
class Resultado(SearchResult):
    """Un resultado de búsqueda con todo lo que §3.3 exige devolver.

    **Hereda de `SearchResult` y eso no es cosmética:** un `Resultado` *es* un
    `SearchResult`, así que `evaluacion.py`, las gráficas y la tabla comparativa
    de NB09 lo aceptan sin traducir nada. Un tipo aparte habría obligado a
    convertir en cada frontera, que es donde se pierden los campos.

    Los dos identificadores conviven porque son cosas distintas y las dos hacen
    falta: `document_id` es el **`product_id`**, que es lo que juzgan los qrels y
    lo que piden los CSV de entrega; `record_id` es el id del punto en la base,
    que es lo que permite volver a leerlo o borrarlo.
    """

    record_id: str = ""
    titulo: str = ""
    metadatos: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # `frozen` protege la asignación, no el contenido: sin esta copia, el
        # payload que devuelve el motor viajaría compartido y quien recorriera
        # los resultados podría modificarlo sin querer. Se copia y se cierra.
        object.__setattr__(self, "metadatos", MappingProxyType(dict(self.metadatos)))


def resultado_desde_hit(hit: Any, *, posicion: int | None = None) -> Resultado:
    """Traduce un `SearchHit` del motor al contrato de §3.3.

    Es la costura donde el payload deja de ser un diccionario opaco y pasa a ser
    los campos que el enunciado exige. Y es también donde se decide qué falta es
    un error: **sin `product_id` no hay entrega posible** —los dos CSV de salida
    se identifican con él y los qrels lo usan para juzgar—, así que su ausencia
    levanta en vez de colarse como cadena vacía hasta NB09.

    El título, en cambio, puede faltar sin romper nada: se devuelve vacío, que es
    lo que D14 escribió en el payload de los productos sin ese dato.
    """
    payload = dict(getattr(hit, "payload", {}) or {})
    product_id = payload.get("product_id")
    if not product_id:
        raise ValueError(
            f"El punto {getattr(hit, 'record_id', '?')!r} no trae `product_id` en el "
            f"payload. Es el identificador que piden los CSV de entrega y los "
            f"juicios de relevancia: sin él el resultado no se puede usar."
        )
    return Resultado(
        rank=posicion if posicion is not None else hit.rank,
        document_id=str(product_id),
        score=float(hit.score),
        # Del motor, no supuesto: Qdrant y Milvus devuelven similitud y Weaviate
        # distancia. §3.2 prohíbe expresamente aplanar esa diferencia.
        score_es_similitud=bool(getattr(hit, "higher_is_better", True)),
        record_id=str(getattr(hit, "record_id", "")),
        titulo=str(payload.get("title") or ""),
        metadatos=payload,
    )


class BuscadorVectorial:
    """La interfaz común de recuperación, contra la base vectorial (§3.3).

    Recibe una consulta en texto y devuelve resultados normalizados. Tres
    decisiones de diseño que se ven en la firma:

    - **El codificador se inyecta.** Este módulo no sabe de Gemini ni de
      Hugging Face: recibe una función que convierte texto en vector. Así el
      buscador se prueba sin red y cambiar de modelo no toca este fichero.
    - **El filtro se pide por su valor crudo** —`marca="Einhell"`— y se
      normaliza aquí dentro, que es lo que D03 decidió: el dato se guarda tal
      cual y la normalización ocurre al buscar.
    - **El color no se expone.** El almacén sabe filtrarlo y el índice de texto
      está declarado, pero la interfaz pública no lo ofrece todavía: el
      enunciado solo pide marca (decisión de NB05).

    El `timeout` no lo aplica esta clase sino el cliente del motor, que es quien
    tiene el socket. Se declara aquí para que el mensaje del error pueda decir
    contra qué límite se agotó en vez de dejarlo en "tardó demasiado".
    """

    name = "denso_motor"

    def __init__(
        self,
        store: Any,
        codificar_consulta: Callable[[str], Sequence[float]],
        *,
        top_k: int = DEFAULT_TOP_K,
        modo_normalizacion: str = "unaccent",
        timeout_s: float = 30.0,
        ef: int | None = None,
    ) -> None:
        self.store = store
        self.codificar_consulta = codificar_consulta
        self.top_k = top_k
        self.modo_normalizacion = modo_normalizacion
        self.timeout_s = timeout_s
        # NB06 (D16): un `ef` por instancia, no por llamada -así el barrido crea
        # un `BuscadorVectorial` por cada punto de la curva, igual que NB05 ya
        # creaba uno distinto para la colección vacía o el motor caído. `None`
        # no manda nada al motor: es exactamente el buscador que ya medía NB05.
        self.ef = ef

    def buscar(
        self, consulta: str, *, top_k: int | None = None, marca: str | None = None
    ) -> tuple[Resultado, ...]:
        """Busca y devuelve como mucho `top_k` resultados, ya ordenados.

        Devuelve **lista vacía** cuando no hay nada que devolver —colección sin
        puntos, o un filtro que no casa con ninguno—, y **levanta** cuando no ha
        podido buscar. Son las dos mitades que §3.3 pide separar.
        """
        from .datos import normalize_brand

        if not consulta or not consulta.strip():
            raise ValueError("La consulta está vacía: no hay nada que buscar.")
        k = self.top_k if top_k is None else top_k
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(f"top_k debe ser un entero positivo, no {top_k!r}.")

        filtros = ()
        if marca is not None:
            # D14 dejó los huecos como cadena vacía, así que un filtro construido
            # sin validar la entrada deja de filtrar en silencio: `contains ""`
            # casaría con todo y `equals ""` devolvería justo los 658 productos
            # sin marca. Una marca en blanco es entrada inválida, no un filtro.
            if not marca.strip():
                raise ValueError(
                    "La marca del filtro está en blanco. Para buscar sin filtrar, "
                    "pasa `marca=None`; una cadena vacía haría que el filtro "
                    "dejara de filtrar sin avisar."
                )
            from .motores import FilterCondition

            filtros = (FilterCondition(
                field="brand",
                value=str(normalize_brand(marca, self.modo_normalizacion)),
                operator="equals",   # D03: vocabulario cerrado, una marca por ficha
            ),)

        vector = self.codificar_consulta(consulta)
        # `ef` solo se envía si se fijó: los motores/dobles de prueba que no lo
        # conocen (MotorFalso, Milvus, Weaviate) nunca ven el argumento, así que
        # nada de lo que ya funciona en NB05 puede romperse por esto.
        argumentos_extra = {"ef": self.ef} if self.ef is not None else {}
        try:
            hits = self.store.search(vector, top_k=k, filters=filtros, **argumentos_extra)
        except BusquedaError:
            raise
        except Exception as error:   # el SDK lanza lo suyo; aquí se traduce
            raise self._traducir(error) from error
        return tuple(
            resultado_desde_hit(hit, posicion=posicion)
            for posicion, hit in enumerate(hits, start=1)
        )

    def search(
        self, query_text: str, *, k: int = DEFAULT_TOP_K
    ) -> tuple[SearchResult, ...]:
        """El método del protocolo `Retriever`, para que NB09 pueda mezclarlo
        con el baseline léxico en la misma tabla sin traducir nada."""
        return self.buscar(query_text, top_k=k)

    def _traducir(self, error: Exception) -> BusquedaError:
        """Convierte lo que sube el SDK en la jerarquía del proyecto.

        NB04 midió que el tipo depende del transporte: con el motor caído,
        Qdrant sube `grpc._channel._InactiveRpcError` por gRPC y algo de `httpx`
        por REST. Capturar el tipo concreto ataría este código a la capa de red
        del cliente, así que se clasifica por lo que el error *dice* —su código
        de estado si lo trae, y su mensaje si no—, que es estable entre ambos.
        """
        codigo = ""
        obtener = getattr(error, "code", None)
        if callable(obtener):
            try:
                codigo = str(obtener())
            except Exception:       # noqa: BLE001 — un SDK raro no puede tumbar el error
                codigo = ""
        texto = f"{codigo} {error}".lower()
        contexto = (
            f"colección {getattr(self.store, 'collection', '?')!r} · "
            f"{type(error).__module__}.{type(error).__name__}"
        )
        if "deadline" in texto or "timeout" in texto or "timed out" in texto:
            return BusquedaAgotada(
                f"La búsqueda superó el límite de {self.timeout_s:g} s "
                f"declarado para la consulta. {contexto}"
            )
        return MotorNoDisponible(
            f"La base vectorial no respondió. {contexto}. Comprueba que el motor "
            f"está levantado (`make motor-up MOTOR=qdrant`)."
        )


# ─────────── Lo que hay que medir contra el motor, no contra un doble ────────
# Tres cosas que ningún test unitario puede demostrar, porque dependen de que el
# motor filtre de verdad: la pureza del filtro, lo que costaría filtrar después
# en Python, y qué hace el sistema en los cuatro casos borde de §3.3.


def auditar_filtro_de_marca(
    buscador: Any,
    casos: Sequence[Mapping[str, Any]],
    *,
    alcance: Mapping[str, int],
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """Pureza y cobertura de las consultas filtradas del §5.

    §8 lo pide como criterio de corrección: *"las consultas filtradas nunca
    devuelven otra marca"*. Pero con eso solo no basta, y por un motivo que ya
    apareció en NB04: **una respuesta vacía cumple la pureza de forma vacía**.
    Un filtro roto que no devuelve nada saca el mismo 100 % que uno perfecto.

    Por eso la tabla lleva las dos columnas juntas. `alcance` es el oráculo
    —cuántos productos de esa marca hay en el catálogo, contado con pandas— y
    es lo que distingue un cero legítimo de un filtro que no funciona, y una
    cobertura corta de una marca con pocos productos.

    Una cobertura por debajo de `min(top_k, alcance)` es además la señal que el
    plan da para detectar un post-filtro: quien recupera y descarta se queda sin
    candidatos justo en las marcas raras.
    """
    from .datos import normalize_brand

    filas = []
    for caso in casos:
        marca = str(caso["filter_value"])
        esperados = int(alcance.get(marca, 0))
        buscados = min(top_k, esperados) if esperados else 0
        resultados = buscador.buscar(caso["query_text"], top_k=top_k, marca=marca)
        pedida = normalize_brand(marca, buscador.modo_normalizacion)
        # Contra la clave normalizada, igual que en NB04: comparar el valor
        # crudo del payload con el valor pedido daría siempre falso y marcaría
        # como roto un filtro que funciona.
        de_la_marca = sum(
            1 for r in resultados
            if str(r.metadatos.get("brand_normalized", r.metadatos.get("brand"))) == pedida
        )
        filas.append({
            "caso": caso.get("workload_id", ""),
            "consulta": caso["query_text"],
            "marca": marca,
            "n_en_catalogo": esperados,
            "n_resultados": len(resultados),
            "de_la_marca": de_la_marca,
            "pureza": (
                f"{100 * de_la_marca / len(resultados):.0f}%" if resultados else "—"
            ),
            "veredicto": _veredicto_filtro(
                len(resultados), de_la_marca, esperados, buscados, top_k
            ),
        })
    return pd.DataFrame(filas)


def _veredicto_filtro(
    devueltos: int, de_la_marca: int, en_catalogo: int, buscados: int, top_k: int
) -> str:
    if devueltos == 0:
        return (
            "✅ ausencia real: el catálogo tampoco tiene ninguno" if en_catalogo == 0
            else f"❌ FILTRO ROTO — el catálogo tiene {en_catalogo} y devolvió 0"
        )
    if de_la_marca < devueltos:
        return f"❌ contamina: {devueltos - de_la_marca} de otra marca"
    if devueltos < buscados:
        return (
            f"⚠️ cobertura corta: {devueltos} de {buscados} disponibles — "
            f"la señal de que se está post-filtrando"
        )
    return f"✅ pureza 100 % y cobertura completa ({devueltos} de {top_k})"


def comparar_con_post_filtro(
    buscador: Any,
    consulta: str,
    marca: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    factores: Sequence[int] = (1, 5, 10, 50, 100),
) -> pd.DataFrame:
    """Qué pasaría si el filtro se aplicara en Python en vez de en la base.

    El plan lo llama *"el gráfico de una línea que justifica la decisión"*, y la
    línea es esta: recuperar `top_k × factor` sin filtrar y quedarse con los de
    la marca. Con `Einhell` al 0,2 % del catálogo, la aritmética dice que harían
    falta del orden de 5.000 candidatos para esperar diez suyos; la tabla lo
    mide en vez de suponerlo, y de paso cuenta lo que cuesta cada intento.

    La primera fila es el filtro nativo, para tener contra qué comparar.
    """
    from .datos import normalize_brand

    pedida = normalize_brand(marca, buscador.modo_normalizacion)

    def de_la_marca(resultados) -> int:
        return sum(
            1 for r in resultados
            if str(r.metadatos.get("brand_normalized", r.metadatos.get("brand"))) == pedida
        )

    inicio = perf_counter()
    nativo = buscador.buscar(consulta, top_k=top_k, marca=marca)
    ms_nativo = 1000 * (perf_counter() - inicio)
    filas = [{
        "estrategia": f"filtro nativo (top_k={top_k})",
        "candidatos": len(nativo),
        "de_la_marca": de_la_marca(nativo),
        "descartados": len(nativo) - de_la_marca(nativo),
        "llega_a_10": de_la_marca(nativo) >= top_k,
        "ms": round(ms_nativo, 1),
    }]

    for factor in factores:
        candidatos = top_k * factor
        inicio = perf_counter()
        sin_filtrar = buscador.buscar(consulta, top_k=candidatos)
        ms = 1000 * (perf_counter() - inicio)
        encontrados = de_la_marca(sin_filtrar)
        filas.append({
            "estrategia": f"post-filtro ×{factor}",
            "candidatos": len(sin_filtrar),
            "de_la_marca": encontrados,
            # Lo que costaría la estrategia, con la resta ya hecha: son los
            # candidatos que el motor trae y Python tira a la basura.
            "descartados": len(sin_filtrar) - encontrados,
            "llega_a_10": encontrados >= top_k,
            "ms": round(ms, 1),
        })
    return pd.DataFrame(filas)


_VOCALES_ACENTUADAS = {
    "a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú",
    "A": "Á", "E": "É", "I": "Í", "O": "Ó", "U": "Ú",
}


def variantes_de_escritura(marca: str) -> tuple[tuple[str, str, bool], ...]:
    """Las formas en que un usuario escribiría esa marca, y si deben casar.

    Cada variante es `(etiqueta, valor, debe_casar)`. Las seis primeras son la
    **misma marca escrita distinto** y tienen que devolver exactamente lo mismo,
    porque D03 decidió guardar el valor crudo y normalizar al buscar: si el
    `casefold` o el `unaccent` no se aplicaran, `nike` no encontraría los 295
    productos de `NIKE` y el filtro solo funcionaría para quien escribiera la
    marca igual que el catálogo.

    Las dos últimas son **errores que no deben casar** —un espacio de más dentro
    y una letra de menos—, y están para marcar el límite: la normalización iguala
    la caja y los acentos, no corrige faltas. Sin ellas, una tabla en la que todo
    devuelve diez resultados no distinguiría un filtro que normaliza de un filtro
    que casa con cualquier cosa.

    Se eliminan las variantes que coinciden en valor con otra anterior: para
    `Apple`, "capitalizada" y "tal como está en el CSV" son la misma cadena y la
    fila repetida no añadiría nada.
    """
    marca = str(marca)
    candidatas: list[tuple[str, str | None, bool]] = [
        ("tal como está en el CSV", marca, True),
        ("todo en minúsculas", marca.lower(), True),
        ("TODO EN MAYÚSCULAS", marca.upper(), True),
        ("Capitalizada", marca.capitalize(), True),
        ("con espacios alrededor", f"  {marca} ", True),
        ("con un acento colado", _con_acento(marca), True),
        ("con un espacio dentro", _con_espacio_dentro(marca), False),
        ("con una letra de menos", marca[:-1] if len(marca) > 2 else None, False),
    ]
    vistas: set[str] = set()
    variantes = []
    for etiqueta, valor, debe_casar in candidatas:
        if valor is None or valor in vistas:
            continue
        vistas.add(valor)
        variantes.append((etiqueta, valor, debe_casar))
    return tuple(variantes)


def _con_acento(marca: str) -> str | None:
    for posicion, letra in enumerate(marca):
        if letra in _VOCALES_ACENTUADAS:
            return marca[:posicion] + _VOCALES_ACENTUADAS[letra] + marca[posicion + 1:]
    return None


def _con_espacio_dentro(marca: str) -> str | None:
    return marca[:2] + " " + marca[2:] if len(marca) > 3 else None


def auditar_variantes_de_marca(
    buscador: Any,
    casos: Sequence[Mapping[str, Any]],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """La misma consulta filtrada por la misma marca, escrita de varias formas.

    Es la prueba realista del filtro: nadie escribe `SAMSUNG` en mayúsculas ni
    sabe que el catálogo guarda `NIKE` así. La columna `viaja_al_motor` enseña
    dónde ocurre D03 —lo que sale de `normalize_brand` y llega a la base— y
    `iguales_a_la_canonica` compara el top-k de cada variante con el de la
    escritura del CSV, id a id.

    Una fila en rojo aquí es un fallo caro y silencioso: el filtro seguiría
    devolviendo resultados puros —solo que ninguno— y la pureza del §8 saldría
    perfecta mientras el usuario ve una lista vacía.
    """
    from .datos import normalize_brand

    filas = []
    for caso in casos:
        canonica = str(caso["filter_value"])
        consulta = caso["query_text"]
        esperada = normalize_brand(canonica, buscador.modo_normalizacion)
        ids_canonicos = [
            r.document_id
            for r in buscador.buscar(consulta, top_k=top_k, marca=canonica)
        ]
        for etiqueta, valor, debe_casar in variantes_de_escritura(canonica):
            resultados = buscador.buscar(consulta, top_k=top_k, marca=valor)
            ids = [r.document_id for r in resultados]
            de_la_marca = sum(
                1 for r in resultados
                if str(r.metadatos.get("brand_normalized", r.metadatos.get("brand")))
                == esperada
            )
            comunes = len(set(ids) & set(ids_canonicos))
            filas.append({
                "marca_en_el_csv": canonica,
                "variante": etiqueta,
                "marca_pedida": repr(valor),
                "viaja_al_motor": repr(
                    normalize_brand(valor, buscador.modo_normalizacion)
                ),
                "n_resultados": len(ids),
                "de_la_marca": de_la_marca,
                "iguales_a_la_canonica": (
                    f"{comunes} de {len(ids_canonicos)}"
                    + (" · mismo orden" if ids == ids_canonicos else "")
                ),
                "esperado": (
                    f"los mismos {len(ids_canonicos)}" if debe_casar
                    else "0: no es esa marca"
                ),
                "veredicto": _veredicto_variante(
                    debe_casar, ids, ids_canonicos, de_la_marca
                ),
            })
    return pd.DataFrame(filas)


def _veredicto_variante(
    debe_casar: bool, ids: Sequence[str], ids_canonicos: Sequence[str], de_la_marca: int
) -> str:
    if not debe_casar:
        return (
            "✅ no casa, y no debía" if not ids
            else f"❌ casa {len(ids)} con una marca mal escrita"
        )
    if list(ids) == list(ids_canonicos):
        return "✅ idéntica a la canónica"
    if not ids:
        return "❌ vacía: la normalización no la reconoce"
    if de_la_marca < len(ids):
        return f"❌ contamina: {len(ids) - de_la_marca} de otra marca"
    return (
        f"⚠️ misma marca, distinto top-k: {len(set(ids) & set(ids_canonicos))} "
        f"de {len(ids_canonicos)} en común"
    )


def auditar_post_filtro(
    buscador: Any,
    casos: Sequence[Mapping[str, Any]],
    *,
    alcance: Mapping[str, int],
    top_k: int = DEFAULT_TOP_K,
    factores: Sequence[int] = (1, 5, 10, 50, 100),
    n_catalogo: int | None = None,
) -> pd.DataFrame:
    """El mismo barrido de `comparar_con_post_filtro`, en las cuatro consultas.

    Con una sola marca la conclusión depende de cuál se eligiera: `Einhell` es
    el 0,2 % del catálogo y no llega a diez ni con mil candidatos, pero una
    marca frecuente sí llegaría, y quien leyera solo esa fila podría concluir
    que el post-filtro *funciona con un poco de sobre-recuperación*.

    Pasando las cuatro se ve lo que de verdad pasa: que el post-filtro no falla
    siempre, sino que **falla justo donde el filtro hace falta** —en las marcas
    raras— y que cuando acierta lo hace pagando candidatos que tira. Por eso la
    tabla trae `n_en_catalogo` y `descartados` al lado del `llega_a_10`.
    """
    tablas = []
    for caso in casos:
        marca = str(caso["filter_value"])
        en_catalogo = int(alcance.get(marca, 0))
        tabla = comparar_con_post_filtro(
            buscador, caso["query_text"], marca, top_k=top_k, factores=factores
        )
        tabla.insert(0, "marca", marca)
        tabla.insert(1, "n_en_catalogo", en_catalogo)
        if n_catalogo:
            tabla.insert(
                2, "pct_del_catalogo", f"{100 * en_catalogo / n_catalogo:.2f} %"
            )
        tabla.insert(0, "caso", str(caso.get("workload_id", "")))
        tablas.append(tabla)
    return pd.concat(tablas, ignore_index=True) if tablas else pd.DataFrame()


def auditar_casos_borde(
    casos: Sequence[tuple[str, ...]]
) -> pd.DataFrame:
    """Ejecuta los casos borde de §3.3 y anota qué hizo el sistema en cada uno.

    Cada caso es `(nombre, esperado, función)`, o `(nombre, consulta, esperado,
    función)` cuando el mismo caso se prueba con varias frases —que es como hay
    que probarlo: *"colección vacía devuelve lista vacía"* medido con una sola
    consulta no distingue el caso borde de una consulta que no casaba con nada.

    La función se ejecuta y se registra lo que devolvió **o lo que levantó**,
    porque en dos de los cuatro casos la respuesta correcta es precisamente una
    excepción. Un caso que revienta no interrumpe a los demás: la tabla vale
    porque están todas las filas, igual que en el guion de humo.

    `lo_que_devolvio` lleva los `product_id` y los scores que salieron de
    verdad. Sin esa columna la tabla dice *"3 resultados"* y hay que creérselo;
    con ella se ve **cuáles** tres, que es lo que permite contradecirla.
    """
    filas = []
    for caso in casos:
        if len(caso) == 4:
            nombre, consulta, esperado, funcion = caso
        else:
            nombre, esperado, funcion = caso
            consulta = "—"
        try:
            valor = funcion()
        except Exception as error:   # la excepción ES el resultado que se audita
            observado = f"{type(error).__module__}.{type(error).__name__}: {error}"
            devuelto = "—  (levantó, no devolvió)"
        else:
            observado = (
                f"{len(valor)} resultados" if hasattr(valor, "__len__")
                else repr(valor)
            )
            devuelto = _muestra_de_resultados(valor)
        filas.append({
            "caso": nombre,
            "consulta": consulta,
            "esperado": esperado,
            "observado": " ".join(observado.split())[:200],
            "lo_que_devolvio": devuelto,
        })
    return pd.DataFrame(filas)


def _muestra_de_resultados(valor: Any, *, cuantos: int = 3) -> str:
    """Los primeros `product_id` con su score, para poder mirarlos."""
    if not hasattr(valor, "__len__") or not hasattr(valor, "__iter__"):
        return repr(valor)[:80]
    if len(valor) == 0:
        return "(lista vacía)"
    piezas = []
    for elemento in list(valor)[:cuantos]:
        identificador = getattr(elemento, "document_id", None)
        if identificador is None:
            piezas.append(str(elemento)[:40])
            continue
        piezas.append(f"{identificador} ({float(elemento.score):.3f})")
    if len(valor) > cuantos:
        piezas.append(f"… y {len(valor) - cuantos} más")
    return " · ".join(piezas)


# ────────────── enseñar lo que sale, no solo declarar que cumple ─────────────
# Las tablas de arriba resumen; estas enseñan. La diferencia importa para quien
# corrige: un "✅ pureza 100 %" es una afirmación del código sobre sí mismo,
# mientras que las diez filas con su `product_id`, su marca y su score son la
# prueba con la que cualquiera puede contradecirlo.


def _recortar(texto: str, limite: int) -> str:
    texto = " ".join(str(texto).split())
    return texto if len(texto) <= limite else texto[: limite - 1].rstrip() + "…"


def tabla_de_resultados(
    bloques: Sequence[tuple[str, str, Sequence[Any]]],
    *,
    metadatos: Mapping[str, str] = MappingProxyType({"brand": "marca"}),
    top: int | None = None,
    recorte_titulo: int = 60,
) -> pd.DataFrame:
    """Los resultados de varias consultas, una fila por resultado.

    `bloques` son tripletas `(caso, consulta, resultados)`. Una fila por
    producto recuperado, con su posición, su `product_id`, los metadatos que se
    pidan y el score. Una consulta sin resultados **no desaparece de la tabla**:
    deja una fila que lo dice, que es la diferencia entre ver un cero y no ver
    nada.

    La columna del score se llama `score_mayor_mejor` o `score_menor_mejor`
    según lo que declaren los resultados, para que no haya que recordar en qué
    dirección se lee. Y por eso mismo **mezclar similitudes con distancias en la
    misma tabla levanta**: en una sola columna se leerían en direcciones
    opuestas, que es justo lo que §3.2 prohíbe aplanar.
    """
    similitudes = {bool(r.score_es_similitud) for _, _, res in bloques for r in res}
    if len(similitudes) > 1:
        raise ValueError(
            "Los bloques mezclan scores de similitud y de distancia: en una sola "
            "columna se leerían en direcciones opuestas (§3.2). Sepáralos en dos "
            "tablas o normaliza la métrica antes."
        )
    columna_score = "score_menor_mejor" if similitudes == {False} else "score_mayor_mejor"

    filas: list[dict[str, Any]] = []
    for caso, consulta, resultados in bloques:
        visibles = list(resultados if top is None else list(resultados)[:top])
        if not visibles:
            filas.append({
                "caso": caso,
                "consulta": consulta,
                "posicion": "—",
                "product_id": "(sin resultados)",
                **{nombre: "" for nombre in metadatos.values()},
                "titulo": "",
                columna_score: None,
            })
            continue
        for resultado in visibles:
            filas.append({
                "caso": caso,
                "consulta": consulta,
                "posicion": resultado.rank,
                "product_id": resultado.document_id,
                **{
                    nombre: _recortar(resultado.metadatos.get(campo, "") or "", 24)
                    for campo, nombre in metadatos.items()
                },
                "titulo": _recortar(resultado.titulo, recorte_titulo),
                columna_score: round(float(resultado.score), 4),
            })
    return pd.DataFrame(filas)


def solapamiento_entre_consultas(
    bloques: Sequence[tuple[str, str, Sequence[Any]]]
) -> pd.DataFrame:
    """Cuántos productos comparten los top-k de dos formulaciones distintas.

    Sirve para lo que el enunciado llama *"consultas de distinto tipo"*: la
    misma necesidad escrita como palabras clave, como frase natural y como
    contexto tiene que entrar por la misma puerta y salir por ella. Aquí no se
    juzga si lo recuperado es bueno —eso es NB09 y necesita los juicios de
    relevancia—, sino **cuánto se mueve el resultado al reformular**, que es una
    propiedad de la representación y se mide sin qrels.
    """
    filas = []
    for i, (caso_a, texto_a, res_a) in enumerate(bloques):
        for caso_b, texto_b, res_b in list(bloques)[i + 1:]:
            ids_a = [r.document_id for r in res_a]
            ids_b = [r.document_id for r in res_b]
            comunes = set(ids_a) & set(ids_b)
            de = min(len(ids_a), len(ids_b))
            filas.append({
                "consulta_a": caso_a,
                "consulta_b": caso_b,
                "texto_a": _recortar(texto_a, 45),
                "texto_b": _recortar(texto_b, 45),
                "en_comun": len(comunes),
                "de": de,
                "solapamiento": f"{100 * len(comunes) / de:.0f} %" if de else "—",
                "mismo_primero": bool(ids_a and ids_b and ids_a[0] == ids_b[0]),
            })
    return pd.DataFrame(filas)


def auditar_forma_de_los_resultados(
    buscador: Any,
    casos: Sequence[Mapping[str, Any]],
    *,
    ids_del_catalogo: Collection[str],
    n_puntos: int,
    alcance: Mapping[str, int] | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """Las comprobaciones de forma del plan, con el número medido a la vista.

    Son cuatro —`k` respetado, `product_id` sin repetir, orden monótono en la
    dirección que declara el score, y nada recuperado que no esté en el
    catálogo— y todas se podrían resolver con un `True`. La tabla enseña **el
    valor** en su lugar: cuántos se pidieron y cuántos vinieron, cuántos ids
    distintos de cuántos, y de qué score a qué score.

    El motivo no es estético. Un `k_respetado = True` con `top_k=10` y una marca
    de 3 productos es correcto y parece un fallo; `3 de 3 disponibles (pedidos
    10)` se entiende sin ir a mirar el código. Y al revés: un `True` no deja ver
    que el rango de scores se ha aplanado, que sí se ve en la columna del score.

    `alcance` es el oráculo por marca para los casos filtrados; `n_puntos`, el
    de la colección entera para los que no filtran.
    """
    alcance = alcance or {}
    validos = set(ids_del_catalogo)
    filas = []
    for caso in casos:
        marca = caso.get("filter_value") or None
        resultados = buscador.buscar(
            caso["query_text"], top_k=top_k, marca=str(marca) if marca else None
        )
        disponibles = int(alcance.get(str(marca), 0)) if marca else int(n_puntos)
        esperados = min(top_k, disponibles)
        ids = [r.document_id for r in resultados]
        scores = [float(r.score) for r in resultados]
        es_similitud = bool(resultados and resultados[0].score_es_similitud)
        descendente = all(a >= b for a, b in zip(scores, scores[1:]))
        bien_ordenado = descendente if es_similitud else not descendente
        intrusos = [i for i in ids if i not in validos]
        filas.append({
            "caso": caso.get("workload_id", ""),
            "consulta": _recortar(caso["query_text"], 45),
            "marca": str(marca) if marca else "— (sin filtro)",
            "devueltos": f"{len(ids)} de {esperados} posibles (pedidos {top_k})",
            "posiciones": f"{resultados[0].rank}→{resultados[-1].rank}" if resultados else "—",
            "ids_distintos": f"{len(set(ids))} de {len(ids)}",
            "score_primero_ultimo": (
                f"{scores[0]:.4f} → {scores[-1]:.4f}" if scores else "—"
            ),
            "orden": (
                "—" if not resultados else
                f"{'descendente' if descendente else 'ascendente'} "
                f"({'similitud' if es_similitud else 'distancia'}: "
                f"{'✅' if bien_ordenado else '❌ al revés'})"
            ),
            "fuera_del_catalogo": (
                f"0 de {len(ids)}" if not intrusos
                else f"❌ {len(intrusos)}: {', '.join(intrusos[:3])}"
            ),
            "veredicto": _veredicto_forma(
                len(ids), esperados, len(set(ids)), bien_ordenado, len(intrusos)
            ),
        })
    return pd.DataFrame(filas)


def _veredicto_forma(
    devueltos: int, esperados: int, distintos: int, bien_ordenado: bool, intrusos: int
) -> str:
    if intrusos:
        return f"❌ {intrusos} id fuera del catálogo"
    if distintos < devueltos:
        return f"❌ {devueltos - distintos} `product_id` repetidos en la misma consulta"
    if devueltos and not bien_ordenado:
        return "❌ el orden contradice la dirección que declara el score"
    if devueltos < esperados:
        return f"⚠️ faltan {esperados - devueltos}: había {esperados} disponibles"
    if devueltos > esperados:
        return f"❌ devolvió {devueltos} y solo se pedían {esperados}"
    return "✅ k, ids, orden y catálogo"
