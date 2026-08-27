"""Codificación de textos y medición de longitud en tokens.

Dos capas que comparten fichero porque comparten el mismo objeto de estudio,
el modelo de embeddings:

1. **Medición de longitud** (`token_lengths` y derivadas): evidencia para
   **D07** (¿hace falta chunking?) y para la verificación de **D09** (¿la
   ventana del modelo elegido cubre el catálogo?).
2. **Codificación** (`Encoder` y sus implementaciones): convierte texto en
   vectores para los tres candidatos de D09, con su contrato de entrada,
   el truncado Matryoshka de D09/D10 y una caché en disco.

La caché no es comodidad: codificar 1.500 registros con `jina-embeddings-v3`
en 4 núcleos son decenas de minutos, y los ejes de D10 que *no* obligan a
recodificar (dimensión, normalización, métrica) tienen que poder barrerse en
segundos sobre los mismos vectores. Codificar una vez y barrer gratis es lo
que hace viable el experimento en este hardware.

Por qué no vale contar caracteres ni palabras: cada modelo parte el texto con su
propio vocabulario de subpalabras, así que "cuántos tokens ocupa este registro del catálogo" no
tiene una respuesta única — tiene una por modelo. Las longitudes medidas en NB01
(`datos.document_length_stats`) cuentan **palabras** con una expresión regular
propia; el límite de contexto de un modelo se mide en **piezas de su
vocabulario**. Son dos unidades distintas y no se pueden comparar.

El tokenizador se recibe inyectado (cualquier objeto con `.encode()`), de modo
que la lógica se puede probar sin red y sin descargar ningún modelo.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from tokenizers import Tokenizer as FastTokenizer


class Tokenizer(Protocol):
    """Contrato mínimo: convertir un texto en su secuencia de tokens."""

    def encode(self, text: str) -> Sequence[int]:
        """Devuelve los IDs de token del texto, incluidos los especiales."""
        ...


class HubTokenizer:
    """Adaptador de un `tokenizers.Tokenizer` al Protocol `Tokenizer`."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def encode(self, text: str) -> Sequence[int]:
        return self._tokenizer.encode(text).ids


class CountingTokenizer:
    """Adapta un contador remoto (una API) al Protocol `Tokenizer`.

    Un servicio como `count_tokens` de Gemini devuelve **cuántos** tokens hay,
    no cuáles. Como toda la medición de este módulo solo usa la longitud, basta
    con devolver una secuencia de ese tamaño: así los modelos por API entran en
    la misma tabla que los locales sin duplicar código."""

    def __init__(self, count: Callable[[str], int]) -> None:
        self._count = count

    def encode(self, text: str) -> Sequence[int]:
        return [0] * self._count(text)


def gemini_token_counter(model: str, *, api_key: str) -> Callable[[str], int]:
    """Contador de tokens contra la API de Gemini, para envolver en `CountingTokenizer`.

    ⚠️ Cada llamada es una petición de red: mide un subconjunto de registros del catálogo, no el
    catálogo entero. `count_tokens` no consume cuota de facturación, pero sí
    tiempo y límite de peticiones por minuto."""
    from google import genai  # dependencia opcional: solo la necesitan los modelos por API

    cliente = genai.Client(api_key=api_key)

    def contar(text: str) -> int:
        return int(cliente.models.count_tokens(model=model, contents=text).total_tokens)

    return contar


def load_hub_tokenizer(repo_id: str, *, token: str | None = None) -> HubTokenizer:
    """Descarga **solo** el `tokenizer.json` del repo y lo carga con `tokenizers`.

    Por qué no `AutoTokenizer`: los modelos con código propio en el repo
    (`custom_code`, como `jina-embeddings-v3` o los `nomic`) exigen
    `trust_remote_code=True`, y ese módulo importa `torch`. Contar tokens no
    debería obligar a instalar el framework de inferencia entero ni a ejecutar
    código de terceros — para medir longitudes basta el vocabulario.
    """
    ruta = hf_hub_download(repo_id, "tokenizer.json", token=token)
    return HubTokenizer(FastTokenizer.from_file(ruta))


def token_lengths(texts: Iterable[object], tokenizer: Tokenizer) -> NDArray[np.int64]:
    """Longitud en tokens de cada texto, con el vocabulario de `tokenizer`.

    Los valores nulos cuentan como 0 tokens en lugar de romper la medición: en
    el catálogo `text` no tiene vacíos, pero los eventos DELETE sí llegan con
    campos vacíos (NB08)."""
    lengths = [
        0 if pd.isna(text) else len(tokenizer.encode(str(text))) for text in texts
    ]
    if not lengths:
        raise ValueError("No hay textos que medir.")
    return np.asarray(lengths, dtype=np.int64)


def token_length_stats(
    texts: Iterable[object],
    tokenizer: Tokenizer,
    *,
    model_id: str,
    window: int,
) -> dict[str, object]:
    """Distribución de longitud en tokens y porcentaje que excede la ventana.

    `pct_supera_ventana` es el número que decide D07: si es 0, el chunking no
    resuelve ningún problema real y la familia C del plan queda descartada por
    los datos, no por conveniencia."""
    return _stats_from_lengths(
        token_lengths(texts, tokenizer), model_id=model_id, window=window
    )


def _stats_from_lengths(
    lengths: NDArray[np.int64], *, model_id: str, window: int
) -> dict[str, object]:
    """Estadísticos a partir de longitudes ya calculadas.

    Existe para que `token_length_report` tokenice **una sola vez** por modelo:
    con 15.000 registros, cada pasada extra son ~30 s tirados."""
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window debe ser un entero positivo.")

    n_supera = int((lengths > window).sum())
    return {
        "modelo": model_id,
        "ventana": window,
        "n_docs": int(lengths.size),
        "tokens_media": round(float(lengths.mean()), 1),
        "tokens_p50": int(np.percentile(lengths, 50)),
        "tokens_p90": int(np.percentile(lengths, 90)),
        "tokens_p95": int(np.percentile(lengths, 95)),
        "tokens_max": int(lengths.max()),
        "n_supera_ventana": n_supera,
        "pct_supera_ventana": round(100 * n_supera / lengths.size, 2),
        "margen_p95": window - int(np.percentile(lengths, 95)),
    }


def chars_per_token(texts: Iterable[object], tokenizer: Tokenizer) -> float:
    """Caracteres por token del corpus con este tokenizador.

    Mide lo que separa una estimación en caracteres de la cuenta real. Un valor
    bajo significa que el vocabulario fragmenta mucho este texto: códigos como
    `160x200` o `B0818K237B` se llevan varias piezas cada uno."""
    materialized = [("" if pd.isna(text) else str(text)) for text in texts]
    return _chars_per_token(materialized, token_lengths(materialized, tokenizer))


def _chars_per_token(texts: Sequence[str], lengths: NDArray[np.int64]) -> float:
    """Ratio a partir de longitudes ya calculadas (ver `_stats_from_lengths`)."""
    total_tokens = int(lengths.sum())
    if total_tokens == 0:
        raise ValueError("El corpus no produjo ningún token.")
    return round(sum(len(text) for text in texts) / total_tokens, 2)


def token_length_report(
    texts: Iterable[object],
    tokenizers: dict[str, tuple[Tokenizer, int]],
) -> pd.DataFrame:
    """Una fila por modelo candidato: distribución, truncado y ratio chars/token.

    `tokenizers` mapea `model_id -> (tokenizer, ventana)`. Compara los
    candidatos de D09 sobre el mismo corpus, que es lo que exige la Regla 2."""
    if not tokenizers:
        raise ValueError("Hay que pasar al menos un tokenizador.")

    materialized = [("" if pd.isna(text) else str(text)) for text in texts]
    filas = []
    for model_id, (tokenizer, window) in tokenizers.items():
        # Una sola tokenización por modelo: los estadísticos y el ratio salen
        # ambos del mismo vector de longitudes.
        lengths = token_lengths(materialized, tokenizer)
        fila = _stats_from_lengths(lengths, model_id=model_id, window=window)
        fila["chars_por_token"] = _chars_per_token(materialized, lengths)
        filas.append(fila)
    return pd.DataFrame(filas)


# ───────────────────────── Capa de codificación (D09 · D10) ──────────────────

# Un texto se codifica como documento del catálogo o como consulta. Varios
# modelos tratan los dos casos de forma distinta, y confundirlos degrada la
# recuperación en silencio.
KINDS = ("document", "query")

# Eje "prefijos" de D10. `nativo` aplica el contrato de entrada que el modelo
# declara; `sin_contrato` lo omite deliberadamente para medir cuánto aporta.
CONTRACTS = ("nativo", "sin_contrato")


def _validate_kind(kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"kind debe ser uno de {KINDS}")
    return kind


def _validate_contract(contract: str) -> str:
    if contract not in CONTRACTS:
        raise ValueError(f"contract debe ser uno de {CONTRACTS}")
    return contract


def _materialize(texts: Iterable[object]) -> list[str]:
    return [("" if pd.isna(text) else str(text)) for text in texts]


def safe_l2_normalize(matrix: NDArray[Any], *, epsilon: float = 1e-12) -> NDArray[np.float32]:
    """Normaliza cada fila a norma 1 sin dividir por cero.

    Un vector nulo se deja intacto en vez de convertirse en `NaN`: un `NaN`
    envenena el producto escalar de toda una consulta y aparecería como un
    fallo del ranking, no como lo que es — un vector degenerado."""
    vectors = np.asarray(matrix, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("Se espera una matriz 2D de vectores.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, np.where(norms < epsilon, 1.0, norms)).astype(np.float32)


def truncate_dim(
    matrix: NDArray[Any], dim: int, *, renormalize: bool = True
) -> NDArray[np.float32]:
    """Trunca vectores Matryoshka (MRL) a `dim` y los renormaliza.

    Los tres candidatos de D09 están entrenados con MRL: las primeras
    componentes concentran la mayor parte de la información, así que quedarse
    con un prefijo del vector es una reducción de dimensión válida y **gratis**
    (no hay que recodificar nada).

    La renormalización no es opcional aunque el proveedor afirme devolver
    vectores ya normalizados: truncar un vector unitario deja una norma < 1, y
    el coseno dejaría de ser un coseno. Si ya llegara normalizado, renormalizar
    es idempotente — el coste de dejarlo puesto es cero y el de quitarlo es una
    métrica sutilmente equivocada."""
    vectors = np.asarray(matrix, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("Se espera una matriz 2D de vectores.")
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
        raise ValueError("dim debe ser un entero positivo.")
    if dim > vectors.shape[1]:
        raise ValueError(
            f"No se puede truncar a {dim}: los vectores tienen {vectors.shape[1]} dimensiones."
        )
    truncated = vectors[:, :dim]
    return safe_l2_normalize(truncated) if renormalize else truncated.copy()


def vector_health(matrix: NDArray[Any]) -> dict[str, object]:
    """Comprobaciones de sanidad de una matriz de embeddings.

    Son las verificaciones que NB02 exige antes de creerse ninguna métrica: sin
    `NaN`/`inf`, normas coherentes y sin filas duplicadas. Una matriz con filas
    idénticas suele significar que el modelo recibió textos vacíos."""
    vectors = np.asarray(matrix, dtype=np.float32)
    if vectors.ndim != 2 or vectors.size == 0:
        raise ValueError("Se espera una matriz 2D no vacía.")
    norms = np.linalg.norm(vectors, axis=1)
    return {
        "n_vectores": int(vectors.shape[0]),
        "dim": int(vectors.shape[1]),
        "dtype": str(vectors.dtype),
        "finito": bool(np.isfinite(vectors).all()),
        "norma_min": round(float(norms.min()), 6),
        "norma_max": round(float(norms.max()), 6),
        "normalizado": bool(np.allclose(norms, 1.0, atol=1e-4)),
        "n_filas_duplicadas": int(vectors.shape[0] - len(np.unique(vectors, axis=0))),
        "bytes_por_vector": int(vectors.shape[1] * vectors.dtype.itemsize),
    }


@dataclass(frozen=True, slots=True)
class EncodingStats:
    """Coste de una codificación. Va junto a la calidad, nunca después (Regla 3)."""

    model_id: str
    kind: str
    contract: str
    n_textos: int
    dim: int
    segundos: float
    desde_cache: bool = False

    @property
    def textos_por_segundo(self) -> float:
        """Ritmo de codificación; con `desde_cache` no significa nada."""
        return round(self.n_textos / self.segundos, 2) if self.segundos > 0 else 0.0

    def as_row(self) -> dict[str, object]:
        """Fila plana para la tabla comparativa de NB02."""
        return {
            "modelo": self.model_id,
            "tipo": self.kind,
            "contrato": self.contract,
            "n_textos": self.n_textos,
            "dim": self.dim,
            "segundos": round(self.segundos, 2),
            "textos_por_segundo": self.textos_por_segundo,
            "desde_cache": self.desde_cache,
        }


class Encoder(Protocol):
    """Contrato mínimo de un modelo de embeddings para este proyecto."""

    model_id: str
    native_dim: int
    window: int

    @property
    def has_contract(self) -> bool:
        """¿El modelo distingue documento de consulta en su entrada?"""
        ...

    def encode(
        self,
        texts: Sequence[str],
        *,
        kind: str = "document",
        contract: str = "nativo",
        batch_size: int = 16,
    ) -> NDArray[np.float32]:
        """Codifica los textos y devuelve una matriz `(n, native_dim)`."""
        ...


def _enable_transformers_v4_remote_code() -> None:
    """Permite cargar código remoto escrito para transformers 4.x sobre la 5.x.

    `jina-embeddings-v3` sirve su propia implementación, y el `__init__` de
    `XLMRobertaLoRA` no llama a `post_init()`. En transformers 4.x era
    inofensivo; en la 5.x ahí es donde nace `all_tied_weights_keys`, que
    `from_pretrained` lee al finalizar la carga (`modeling_utils`), así que el
    modelo revienta **después** de cargar los pesos con:

        AttributeError: 'XLMRobertaLoRA' object has no attribute
        'all_tied_weights_keys'

    El valor por defecto es un dict vacío porque estos encoders no atan pesos
    entre sí. Es un atributo de clase, pero todo modelo que sí llama a
    `post_init()` se crea el suyo de instancia y lo tapa, así que este se queda
    vacío. Alternativa descartada: bajar a transformers 4.x arrastraría también
    `huggingface_hub` y `tokenizers` hacia atrás, y con ellos el resto del
    proyecto, que sí funciona en la 5.x."""
    from transformers import PreTrainedModel

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}


class SentenceTransformerEncoder:
    """Modelos locales servidos por `sentence-transformers` (jina-v3, granite-r2).

    `tasks` mapea `kind -> valor` del argumento que el modelo usa para
    distinguir documento de consulta, y es lo que separa a los dos candidatos
    locales:

    - `jina-embeddings-v3` conmuta **adaptadores LoRA** (`retrieval.passage`,
      `retrieval.query`, `text-matching`). No son prefijos de texto: son pesos
      distintos, así que el eje "con/sin contrato" de D10 sí cambia el vector.
    - `granite-embedding-311m-multilingual-r2` declara en su
      `config_sentence_transformers.json` los prompts `query` y `document`
      **ambos como cadena vacía**: no tiene contrato de entrada. Se construye
      con `tasks=None` y el eje de D10 es degenerado por diseño, no por
      omisión — inventarle un prefijo sería medir un modelo que no existe.

    Los vectores se devuelven **sin normalizar** (`normalize_embeddings=False`)
    aunque el modelo traiga un módulo `Normalize`: la normalización se aplica
    aquí arriba, donde es observable y donde el eje de D10 puede activarla y
    desactivarla."""

    def __init__(
        self,
        repo_id: str,
        *,
        window: int,
        native_dim: int | None = None,
        tasks: Mapping[str, str] | None = None,
        task_arg: str = "task",
        trust_remote_code: bool = False,
        device: str = "cpu",
        token: str | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_id = repo_id
        self.window = window
        self._tasks = dict(tasks) if tasks else None
        self._task_arg = task_arg
        if self._tasks is not None and set(self._tasks) != set(KINDS):
            raise ValueError(f"tasks debe cubrir exactamente {KINDS}")

        def cargar() -> Any:
            # Import perezoso: `sentence_transformers` arrastra torch entero, y
            # medir longitudes en tokens no debería pagar ese arranque.
            from sentence_transformers import SentenceTransformer

            if trust_remote_code:
                _enable_transformers_v4_remote_code()

            return SentenceTransformer(
                repo_id,
                trust_remote_code=trust_remote_code,
                device=device,
                token=token,
            )

        self._cargar = cargar
        self._model = model
        # Los pesos se cargan en el primer `encode`, no aquí. Construir el
        # encoder de un modelo cuyos vectores ya están todos en caché no puede
        # costar 2,3 GB de RAM y varios minutos para no llegar a usarlo: es
        # exactamente lo que ocurre al re-ejecutar el notebook. La excepción es
        # no declarar `native_dim`, porque entonces el único que sabe la
        # dimensión es el propio modelo y hay que abrirlo.
        if native_dim is not None:
            self.native_dim = int(native_dim)
        else:
            if self._model is None:
                self._model = cargar()
            self.native_dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def model(self) -> Any:
        """Los pesos, cargados la primera vez que hacen falta de verdad."""
        if self._model is None:
            self._model = self._cargar()
        return self._model

    @property
    def has_contract(self) -> bool:
        """`False` en granite: sus dos prompts declarados son cadena vacía."""
        return self._tasks is not None

    def encode(
        self,
        texts: Sequence[str],
        *,
        kind: str = "document",
        contract: str = "nativo",
        batch_size: int = 16,
        show_progress: bool = True,
    ) -> NDArray[np.float32]:
        """Codifica en lotes con el adaptador que corresponda al `kind`."""
        _validate_kind(kind)
        _validate_contract(contract)
        materialized = _materialize(texts)
        if not materialized:
            raise ValueError("No hay textos que codificar.")

        kwargs: dict[str, Any] = {}
        if contract == "nativo" and self._tasks is not None:
            kwargs[self._task_arg] = self._tasks[kind]

        vectors = self.model.encode(
            materialized,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=show_progress,
            **kwargs,
        )
        return np.asarray(vectors, dtype=np.float32)


class GeminiEncoder:
    """`gemini-embedding-2` por API.

    Su contrato de entrada no es un `task_type` (eso pertenece a
    `gemini-embedding-001`) sino una **instrucción dentro del prompt**, así que
    el eje "con/sin contrato" de D10 se implementa anteponiendo o no esa
    instrucción al texto.

    Sobre la dimensión: se pide `output_dimensionality` a la API cuando se
    quiere una salida reducida, pero el barrido MRL de NB02 se hace en local
    con `truncate_dim` sobre la salida nativa. Motivo: una petición por
    dimensión multiplicaría las llamadas de red por cuatro para obtener,
    salvo detalles de redondeo, el mismo vector truncado."""

    INSTRUCCIONES = {
        "document": (
            "Ficha de producto de un catálogo de comercio electrónico en español."
        ),
        "query": (
            "Consulta de una persona buscando un producto en un catálogo de "
            "comercio electrónico en español."
        ),
    }

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_id: str = "gemini-embedding-2",
        native_dim: int = 3072,
        window: int = 8192,
        output_dim: int | None = None,
        instructions: Mapping[str, str] | None = None,
        client: Any | None = None,
        max_reintentos: int = 3,
    ) -> None:
        self.model_id = model_id
        self.native_dim = native_dim
        self.window = window
        self.output_dim = output_dim
        self.max_reintentos = max_reintentos
        self._instructions = dict(instructions or self.INSTRUCCIONES)
        if set(self._instructions) != set(KINDS):
            raise ValueError(f"instructions debe cubrir exactamente {KINDS}")

        if client is None:
            if not api_key:
                raise ValueError(
                    "Falta la clave de la API: pásala en api_key o inyecta un client."
                )
            from google import genai  # dependencia solo de los modelos por API

            client = genai.Client(api_key=api_key)
        self._client = client

    @property
    def has_contract(self) -> bool:
        """Siempre `True`: la instrucción de tarea es su contrato de entrada."""
        return True

    def _prepare(self, texts: Sequence[str], kind: str, contract: str) -> list[str]:
        materialized = _materialize(texts)
        if contract == "sin_contrato":
            return materialized
        instruccion = self._instructions[kind]
        return [f"{instruccion}\n{text}" for text in materialized]

    def _embed_batch(self, batch: Sequence[str]) -> list[list[float]]:
        """Una llamada a la API, con reintentos ante fallos transitorios.

        Cada texto viaja **envuelto en su propia lista**, y eso no es un
        capricho de estilo. Con `gemini-embedding-2` el SDK normaliza
        `contents` con `t_contents` antes de enviar nada, y esa función agrupa
        las cadenas sueltas consecutivas en un **único `Content` multi-parte**:
        es la semántica de un turno de chat (varias partes, un interlocutor),
        no la de un lote. Un lote de 32 cadenas se convertiría así en un solo
        `Content` y la API devolvería **un** vector — el del lote concatenado —
        en vez de 32. Envolviendo cada texto, el SDK lo trata como un `Content`
        independiente y `batchEmbedContents` devuelve un vector por texto.

        El backoff es exponencial y sin jitter porque aquí solo hay un cliente:
        el jitter resuelve la sincronización entre clientes concurrentes, que
        no es un problema de un notebook."""
        config: dict[str, Any] = {}
        if self.output_dim is not None:
            config["output_dimensionality"] = self.output_dim

        contents = [[text] for text in batch]
        ultimo_error: Exception | None = None
        for intento in range(self.max_reintentos):
            try:
                respuesta = self._client.models.embed_content(
                    model=self.model_id,
                    contents=contents,
                    config=config or None,
                )
            except Exception as error:  # cuota, red o 5xx del proveedor
                ultimo_error = error
                if intento == self.max_reintentos - 1:
                    break
                time.sleep(2.0**intento)
                continue

            vectores = [list(item.values) for item in respuesta.embeddings]
            if len(vectores) != len(batch):
                # No es transitorio: reintentar daría exactamente lo mismo. Se
                # corta aquí, junto a la llamada, y no en `encode_corpus`, donde
                # el desajuste ya solo se ve como un total que no cuadra.
                raise RuntimeError(
                    f"{self.model_id} devolvió {len(vectores)} vectores para un lote "
                    f"de {len(batch)} textos: el SDK agrupó el lote en un solo Content."
                )
            return vectores
        raise RuntimeError(
            f"La API de {self.model_id} falló tras {self.max_reintentos} intentos: {ultimo_error}"
        ) from ultimo_error

    def encode(
        self,
        texts: Sequence[str],
        *,
        kind: str = "document",
        contract: str = "nativo",
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> NDArray[np.float32]:
        """Codifica por lotes contra la API."""
        _validate_kind(kind)
        _validate_contract(contract)
        preparados = self._prepare(texts, kind, contract)
        if not preparados:
            raise ValueError("No hay textos que codificar.")

        vectores: list[list[float]] = []
        for inicio in range(0, len(preparados), batch_size):
            vectores.extend(self._embed_batch(preparados[inicio : inicio + batch_size]))
            if show_progress:
                print(f"  {min(inicio + batch_size, len(preparados))}/{len(preparados)}")
        return np.asarray(vectores, dtype=np.float32)


# ─────────────────────────────── Caché en disco ──────────────────────────────


def corpus_fingerprint(texts: Sequence[str]) -> str:
    """SHA-256 corto del corpus, para invalidar la caché si cambia el texto.

    Sin esto, cambiar la plantilla (A0 → A3) reutilizaría en silencio vectores
    del texto anterior y el experimento compararía cosas distintas creyendo
    que compara la misma."""
    digest = hashlib.sha256()
    for text in _materialize(texts):
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")  # separador: evita que dos corpus distintos colisionen
    return digest.hexdigest()[:16]


def cache_key(
    *, model_id: str, kind: str, contract: str, corpus_id: str, fingerprint: str
) -> str:
    """Nombre de fichero legible y único por experimento."""
    modelo = model_id.replace("/", "__")
    return f"{modelo}__{corpus_id}__{kind}__{contract}__{fingerprint}"


@dataclass(frozen=True, slots=True)
class EncodedCorpus:
    """Vectores de un corpus más su procedencia y su coste."""

    vectors: NDArray[np.float32]
    stats: EncodingStats
    metadata: dict[str, object] = field(default_factory=dict)


def encode_corpus(
    encoder: Encoder,
    texts: Sequence[str],
    *,
    corpus_id: str,
    kind: str = "document",
    contract: str = "nativo",
    batch_size: int = 16,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> EncodedCorpus:
    """Codifica un corpus reutilizando la caché en disco si existe.

    Devuelve siempre los vectores **sin normalizar y en dimensión nativa**: la
    normalización y el truncado MRL son ejes de D10 que se aplican después,
    sobre estos mismos vectores, sin volver a pagar la codificación.

    El artefacto en disco es el que exige el plan: `.npy` con los vectores más
    un `.json` con `model_id`, dimensión, dtype, contrato y el SHA-256 del
    corpus de origen."""
    _validate_kind(kind)
    _validate_contract(contract)
    materialized = _materialize(texts)
    fingerprint = corpus_fingerprint(materialized)
    clave = cache_key(
        model_id=encoder.model_id,
        kind=kind,
        contract=contract,
        corpus_id=corpus_id,
        fingerprint=fingerprint,
    )

    destino = Path(cache_dir) / f"{clave}.npy" if cache_dir is not None else None
    if destino is not None and destino.exists() and not force:
        vectors = np.load(destino)
        metadata = json.loads(destino.with_suffix(".json").read_text(encoding="utf-8"))
        return EncodedCorpus(
            vectors=vectors,
            stats=EncodingStats(
                model_id=encoder.model_id,
                kind=kind,
                contract=contract,
                n_textos=int(vectors.shape[0]),
                dim=int(vectors.shape[1]),
                segundos=float(metadata.get("segundos", 0.0)),
                desde_cache=True,
            ),
            metadata=metadata,
        )

    inicio = time.perf_counter()
    vectors = encoder.encode(
        materialized, kind=kind, contract=contract, batch_size=batch_size
    )
    segundos = time.perf_counter() - inicio

    if vectors.shape[0] != len(materialized):
        raise ValueError(
            f"El modelo devolvió {vectors.shape[0]} vectores para {len(materialized)} textos."
        )

    stats = EncodingStats(
        model_id=encoder.model_id,
        kind=kind,
        contract=contract,
        n_textos=int(vectors.shape[0]),
        dim=int(vectors.shape[1]),
        segundos=segundos,
    )
    metadata = {
        **stats.as_row(),
        "corpus_id": corpus_id,
        "sha256_corpus": fingerprint,
        "dtype": str(vectors.dtype),
        "normalizado_en_origen": bool(
            np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
        ),
        "ventana_tokens": getattr(encoder, "window", None),
    }

    if destino is not None:
        destino.parent.mkdir(parents=True, exist_ok=True)
        np.save(destino, vectors)
        destino.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return EncodedCorpus(vectors=vectors, stats=stats, metadata=metadata)


# ─────────────── Coste, latencia y deriva: el encoder en producción ──────────
#
# `EncodingStats` mide el coste de *indexar*: un lote grande, una vez. Lo que
# sigue mide el coste de *buscar*: una consulta suelta, en cada búsqueda y para
# siempre. Son dos regímenes distintos y el segundo no se deduce del primero —
# un lote de 8 consultas amortiza el viaje de red entre las ocho y esconde
# justo lo que domina la latencia online.


def measure_encode_latency(
    encoder: Encoder,
    texts: Sequence[str],
    *,
    kind: str = "query",
    contract: str = "nativo",
    repeticiones: int = 20,
    calentamiento: int = 2,
) -> dict[str, object]:
    """Latencia de codificar **una** consulta, con calentamiento y repeticiones.

    Codifica de una en una a propósito: es lo que hace el buscador cuando llega
    una consulta de usuario. Las repeticiones recorren `texts` en ciclo para que
    ni la longitud de un texto concreto ni una caché del proveedor dominen la
    medición.

    El calentamiento no se contabiliza: la primera llamada paga el
    establecimiento de la conexión TLS en los modelos por API y, en los locales,
    la carga perezosa de los pesos más la reserva de memoria. Nada de eso se
    repite en cada consulta, así que contarlo mediría el arranque del proceso y
    no el coste de buscar.
    """
    _validate_kind(kind)
    _validate_contract(contract)
    materialized = _materialize(texts)
    if not materialized:
        raise ValueError("No hay textos con los que medir.")
    if repeticiones < 1:
        raise ValueError("repeticiones debe ser >= 1.")

    for indice in range(max(0, calentamiento)):
        encoder.encode(
            [materialized[indice % len(materialized)]], kind=kind, contract=contract
        )

    muestras: list[float] = []
    for indice in range(repeticiones):
        texto = materialized[indice % len(materialized)]
        inicio = time.perf_counter()
        encoder.encode([texto], kind=kind, contract=contract)
        muestras.append((time.perf_counter() - inicio) * 1000)

    tiempos = np.asarray(muestras, dtype=np.float64)
    return {
        "modelo": encoder.model_id,
        "kind": kind,
        "contrato": contract,
        "n_llamadas": int(repeticiones),
        "ms_p50": round(float(np.percentile(tiempos, 50)), 2),
        "ms_p95": round(float(np.percentile(tiempos, 95)), 2),
        "ms_media": round(float(tiempos.mean()), 2),
        "ms_min": round(float(tiempos.min()), 2),
        "ms_max": round(float(tiempos.max()), 2),
    }


def api_cost(n_tokens: float, *, precio_por_millon: float) -> float:
    """Coste en dólares de codificar `n_tokens` con un modelo de pago por token."""
    if n_tokens < 0 or precio_por_millon < 0:
        raise ValueError("Ni los tokens ni el precio pueden ser negativos.")
    return n_tokens * precio_por_millon / 1_000_000


def api_cost_report(
    *,
    tokens_indexacion: float,
    tokens_por_consulta: float,
    precio_por_millon: float,
    consultas_mes: int | None = None,
) -> dict[str, float]:
    """Coste de indexar una vez frente al coste de buscar siempre.

    La cifra por consulta suelta es tan pequeña que no dice nada, así que el
    informe la da también por cada mil consultas y, sobre todo, traduce la
    reindexación completa a **cuántas consultas cuesta lo mismo**: ese número sí
    ordena las dos partidas y no depende de la escala del negocio.

    Cada clave lleva su unidad en el nombre, y no es manía de nomenclatura: una
    tabla de costes en la que hay dólares, consultas y ratios mezclados se lee
    mal si hay que deducir de cuál es cada columna.

    | Clave | Unidad | Qué es |
    |---|---|---|
    | `indexacion_completa_usd` | USD | codificar el catálogo entero, una vez |
    | `por_consulta_usd` | USD | codificar **una** consulta de usuario |
    | `por_1000_consultas_usd` | USD | lo mismo × 1.000, para que se lea |
    | `consultas_equivalentes_a_reindexar` | consultas | cuántas búsquedas cuestan lo que una reindexación |
    | `mes_consultas_usd` | USD/mes | solo si se pasa `consultas_mes` |
    """
    indexacion = api_cost(tokens_indexacion, precio_por_millon=precio_por_millon)
    por_consulta = api_cost(tokens_por_consulta, precio_por_millon=precio_por_millon)

    reporte = {
        "indexacion_completa_usd": round(indexacion, 4),
        "por_consulta_usd": round(por_consulta, 8),
        "por_1000_consultas_usd": round(por_consulta * 1000, 4),
        "consultas_equivalentes_a_reindexar": (
            round(indexacion / por_consulta) if por_consulta > 0 else float("inf")
        ),
    }
    if consultas_mes is not None:
        reporte["mes_consultas_usd"] = round(por_consulta * consultas_mes, 2)
    return reporte


def drift_check(
    encoder: Encoder,
    texts: Sequence[str],
    reference: NDArray[Any],
    *,
    kind: str = "document",
    contract: str = "nativo",
    tolerancia: float = 1e-3,
) -> dict[str, object]:
    """¿El modelo sigue produciendo los mismos vectores que cuando se indexó?

    Existe por una asimetría entre los dos regímenes de modelo. Con pesos
    locales, la versión la fija el fichero: si no se descarga otro, el vector de
    hoy es el de ayer. Con un modelo servido por API, el proveedor puede
    actualizarlo bajo el mismo identificador, y entonces las altas que se
    codifiquen después dejan de vivir en el mismo espacio que el catálogo ya
    indexado. **Eso no lanza ninguna excepción**: los vecinos simplemente
    empeoran, y sin una comprobación como esta el síntoma se confunde con una
    mala representación.

    Recodifica un puñado de textos ya indexados y los compara con sus vectores
    guardados. Con vectores unitarios, coseno ≈ 1 significa que el modelo no ha
    cambiado; cualquier desviación por encima de `tolerancia` es una señal para
    reindexar antes de seguir escribiendo en el índice.
    """
    materialized = _materialize(texts)
    referencia = np.asarray(reference, dtype=np.float32)
    if referencia.ndim != 2 or referencia.shape[0] != len(materialized):
        raise ValueError(
            f"Hacen falta {len(materialized)} vectores de referencia, "
            f"llegaron {referencia.shape}."
        )

    actuales = encoder.encode(materialized, kind=kind, contract=contract)
    if actuales.shape[1] != referencia.shape[1]:
        # Cambio de dimensión nativa: es deriva, y de la peor, pero el coseno no
        # llega ni a calcularse. Se informa en vez de reventar con un error de
        # álgebra que no diría qué ha pasado.
        return {
            "modelo": encoder.model_id,
            "n_textos": len(materialized),
            "sin_deriva": False,
            "motivo": (
                f"la dimensión nativa cambió de {referencia.shape[1]} a "
                f"{actuales.shape[1]}"
            ),
        }

    cosenos = np.sum(
        safe_l2_normalize(actuales) * safe_l2_normalize(referencia), axis=1
    )
    desviados = int(np.sum(cosenos < 1.0 - tolerancia))
    return {
        "modelo": encoder.model_id,
        "n_textos": len(materialized),
        "coseno_min": round(float(cosenos.min()), 6),
        "coseno_medio": round(float(cosenos.mean()), 6),
        "n_desviados": desviados,
        "tolerancia": tolerancia,
        "sin_deriva": desviados == 0,
    }
