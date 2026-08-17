"""Pruebas de la medición de longitud en tokens y de la codificación.

Ningún test descarga un modelo ni llama a una API: los encoders reciben un
modelo o un cliente falso, que es justo para lo que se inyectan."""
import json

import numpy as np
import pytest

from aurum.embeddings import (
    CountingTokenizer,
    GeminiEncoder,
    HubTokenizer,
    SentenceTransformerEncoder,
    cache_key,
    chars_per_token,
    corpus_fingerprint,
    encode_corpus,
    safe_l2_normalize,
    token_length_report,
    token_length_stats,
    token_lengths,
    truncate_dim,
    vector_health,
)


class TokenizadorFalso:
    """Tokenizador de juguete: una pieza por palabra, más `n_especiales`.

    Permite verificar la lógica sin red ni descargas. `n_especiales` imita los
    tokens `<s>`/`</s>` que los tokenizadores reales añaden y que también
    consumen ventana de contexto."""

    def __init__(self, n_especiales: int = 0) -> None:
        self.n_especiales = n_especiales

    def encode(self, text: str) -> list[int]:
        return [0] * (len(text.split()) + self.n_especiales)


TEXTOS = ["uno dos tres", "uno", "uno dos tres cuatro cinco"]


def test_token_lengths_cuenta_los_tokens_especiales():
    lengths = token_lengths(TEXTOS, TokenizadorFalso(n_especiales=2))

    assert lengths.tolist() == [5, 3, 7]
    assert lengths.dtype == np.int64


def test_token_lengths_trata_los_nulos_como_cero():
    assert token_lengths(["uno dos", None], TokenizadorFalso()).tolist() == [2, 0]


def test_token_lengths_rechaza_un_corpus_vacio():
    with pytest.raises(ValueError, match="No hay textos"):
        token_lengths([], TokenizadorFalso())


def test_token_length_stats_cuenta_las_fichas_que_superan_la_ventana():
    stats = token_length_stats(
        TEXTOS, TokenizadorFalso(), model_id="falso", window=3
    )

    # Longitudes 3, 1 y 5: solo la última supera una ventana de 3.
    assert stats["n_supera_ventana"] == 1
    assert stats["pct_supera_ventana"] == pytest.approx(33.33)
    assert stats["tokens_max"] == 5
    assert stats["ventana"] == 3


def test_token_length_stats_no_cuenta_la_ficha_que_cabe_justa():
    """Una ficha de exactamente `window` tokens entra: el corte es `>`, no `>=`."""
    stats = token_length_stats(
        ["uno dos tres"], TokenizadorFalso(), model_id="falso", window=3
    )

    assert stats["n_supera_ventana"] == 0
    assert stats["margen_p95"] == 0


def test_token_length_stats_rechaza_una_ventana_invalida():
    with pytest.raises(ValueError, match="entero positivo"):
        token_length_stats(TEXTOS, TokenizadorFalso(), model_id="falso", window=0)


def test_chars_per_token_divide_caracteres_totales_entre_tokens_totales():
    # "uno dos tres" = 12 chars / 3 tokens; "uno" = 3 chars / 1 token.
    assert chars_per_token(["uno dos tres", "uno"], TokenizadorFalso()) == 3.75


def test_token_length_report_devuelve_una_fila_por_modelo():
    informe = token_length_report(
        TEXTOS,
        {
            "compacto": (TokenizadorFalso(), 4),
            "fragmentador": (TokenizadorFalso(n_especiales=2), 4),
        },
    )

    assert list(informe["modelo"]) == ["compacto", "fragmentador"]
    # El que añade especiales trunca más con la misma ventana.
    assert list(informe["n_supera_ventana"]) == [1, 2]
    assert "chars_por_token" in informe.columns


def test_token_length_report_rechaza_una_lista_vacia_de_modelos():
    with pytest.raises(ValueError, match="al menos un tokenizador"):
        token_length_report(TEXTOS, {})


class _EncodingFalso:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _TokenizerRapidoFalso:
    """Imita `tokenizers.Tokenizer`: `.encode()` devuelve un objeto con `.ids`."""

    def encode(self, text: str) -> _EncodingFalso:
        return _EncodingFalso([7] * len(text.split()))


def test_hub_tokenizer_adapta_encoding_a_lista_de_ids():
    """El adaptador traduce el `.ids` de `tokenizers` al Protocol de este módulo."""
    adaptado = HubTokenizer(_TokenizerRapidoFalso())

    assert adaptado.encode("uno dos tres") == [7, 7, 7]
    assert token_lengths(["uno dos", "uno"], adaptado).tolist() == [2, 1]


def test_counting_tokenizer_entra_en_la_misma_tabla_que_los_locales():
    """Un contador remoto que solo devuelve cuántos tokens hay basta para medir."""
    llamadas = []

    def contador_remoto(text: str) -> int:
        llamadas.append(text)
        return len(text.split()) * 2  # imita un vocabulario que fragmenta más

    stats = token_length_stats(
        TEXTOS, CountingTokenizer(contador_remoto), model_id="remoto", window=5
    )

    assert llamadas == TEXTOS  # una llamada por ficha: es red, no CPU
    assert stats["tokens_max"] == 10
    assert stats["n_supera_ventana"] == 2


# ──────────────────────── Normalización y truncado MRL ───────────────────────


def test_safe_l2_normalize_deja_norma_uno():
    normalizados = safe_l2_normalize(np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32))

    assert np.allclose(np.linalg.norm(normalizados, axis=1), 1.0)


def test_safe_l2_normalize_no_produce_nan_con_el_vector_nulo():
    """Un vector nulo se deja intacto: un NaN contaminaría toda la consulta."""
    normalizados = safe_l2_normalize(np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32))

    assert np.isfinite(normalizados).all()
    assert normalizados[0].tolist() == [0.0, 0.0]


def test_truncate_dim_renormaliza_el_prefijo():
    """Truncar un vector unitario deja norma < 1: sin renormalizar el coseno miente."""
    unitarios = safe_l2_normalize(np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32))

    truncados = truncate_dim(unitarios, 2)

    assert truncados.shape == (1, 2)
    assert np.allclose(np.linalg.norm(truncados, axis=1), 1.0)


def test_truncate_dim_sin_renormalizar_conserva_el_prefijo_crudo():
    matriz = np.array([[3.0, 4.0, 9.0]], dtype=np.float32)

    assert truncate_dim(matriz, 2, renormalize=False).tolist() == [[3.0, 4.0]]


def test_truncate_dim_rechaza_ampliar_la_dimension():
    with pytest.raises(ValueError, match="No se puede truncar"):
        truncate_dim(np.zeros((2, 4), dtype=np.float32), 8)


def test_vector_health_detecta_una_matriz_sin_normalizar():
    salud = vector_health(np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32))

    assert salud["normalizado"] is False
    assert salud["finito"] is True
    assert salud["bytes_por_vector"] == 8  # 2 dims x 4 bytes de float32


def test_vector_health_cuenta_las_filas_duplicadas():
    """Filas idénticas suelen significar que el modelo recibió textos vacíos."""
    repetida = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    assert vector_health(repetida)["n_filas_duplicadas"] == 1


# ──────────────────────────────── Encoders ───────────────────────────────────


class ModeloFalso:
    """Imita `SentenceTransformer.encode`: registra los kwargs que recibe."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.llamadas: list[dict] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, texts, **kwargs):
        self.llamadas.append({"n": len(texts), **kwargs})
        # Vectores deterministas y distintos entre sí, con normas != 1 para
        # comprobar que el encoder NO normaliza por su cuenta.
        return np.array(
            [[float(i + 1)] * self.dim for i in range(len(texts))], dtype=np.float32
        )


TAREAS_JINA = {"document": "retrieval.passage", "query": "retrieval.query"}


def test_encoder_local_aplica_el_adaptador_del_kind():
    modelo = ModeloFalso()
    encoder = SentenceTransformerEncoder(
        "jinaai/jina-embeddings-v3", window=8192, tasks=TAREAS_JINA, model=modelo
    )

    encoder.encode(["a", "b"], kind="query")

    assert encoder.has_contract is True
    assert modelo.llamadas[0]["task"] == "retrieval.query"
    assert modelo.llamadas[0]["normalize_embeddings"] is False


def test_encoder_local_omite_el_adaptador_sin_contrato():
    """El eje 'prefijos' de D10: la misma llamada, sin el argumento de tarea."""
    modelo = ModeloFalso()
    encoder = SentenceTransformerEncoder(
        "jinaai/jina-embeddings-v3", window=8192, tasks=TAREAS_JINA, model=modelo
    )

    encoder.encode(["a"], kind="document", contract="sin_contrato")

    assert "task" not in modelo.llamadas[0]


def test_encoder_sin_tareas_declara_que_no_tiene_contrato():
    """granite-r2: sus prompts 'query' y 'document' son ambos cadena vacía."""
    modelo = ModeloFalso(dim=768)
    encoder = SentenceTransformerEncoder(
        "ibm-granite/granite-embedding-311m-multilingual-r2", window=32768, model=modelo
    )

    encoder.encode(["a"], kind="query")

    assert encoder.has_contract is False
    assert encoder.native_dim == 768
    assert "task" not in modelo.llamadas[0]


def test_encoder_local_rechaza_un_kind_desconocido():
    encoder = SentenceTransformerEncoder("x", window=512, model=ModeloFalso())

    with pytest.raises(ValueError, match="kind debe ser"):
        encoder.encode(["a"], kind="passage")


class ClienteGeminiFalso:
    """Imita `client.models.embed_content` con la semántica real del SDK.

    El detalle que un falso ingenuo se salta: `google-genai` normaliza
    `contents` con `t_contents` **antes** de llamar a la API, y esa función
    agrupa las cadenas sueltas consecutivas en un único `Content` multi-parte.
    La API devuelve un vector por `Content`, no por cadena. Un falso que
    devolviera `len(contents)` vectores daría por bueno un lote de cadenas
    desnudas que en producción colapsa a un solo vector — exactamente el fallo
    que este falso existe para detectar."""

    def __init__(self, dim: int = 3, fallos: int = 0) -> None:
        self.dim = dim
        self.fallos_pendientes = fallos
        self.llamadas: list[list[list[str]]] = []  # los Contents de cada llamada
        self.models = self

    @staticmethod
    def _agrupar(contents) -> list[list[str]]:
        """Réplica reducida de `google.genai._transformers.t_contents`."""
        grupos: list[list[str]] = []
        sueltas: list[str] = []
        for item in contents:
            if isinstance(item, str):
                sueltas.append(item)
                continue
            if sueltas:
                grupos.append(sueltas)
                sueltas = []
            grupos.append(list(item))
        if sueltas:
            grupos.append(sueltas)
        return grupos

    @property
    def textos_enviados(self) -> list[str]:
        """Todos los textos, en orden y sin la estructura de `Content`."""
        return [texto for llamada in self.llamadas for grupo in llamada for texto in grupo]

    def embed_content(self, *, model, contents, config=None):
        if self.fallos_pendientes:
            self.fallos_pendientes -= 1
            raise RuntimeError("503 del proveedor")
        grupos = self._agrupar(contents)
        self.llamadas.append(grupos)
        vectores = [
            type("E", (), {"values": [float(i)] * self.dim})() for i in range(len(grupos))
        ]
        return type("R", (), {"embeddings": vectores})()


def test_gemini_antepone_la_instruccion_de_tarea():
    cliente = ClienteGeminiFalso()
    encoder = GeminiEncoder(client=cliente)

    encoder.encode(["taladro 24v"], kind="query")

    assert cliente.textos_enviados[0].startswith("Consulta de una persona")
    assert cliente.textos_enviados[0].endswith("taladro 24v")


def test_gemini_sin_contrato_envia_el_texto_desnudo():
    cliente = ClienteGeminiFalso()

    GeminiEncoder(client=cliente).encode(["taladro 24v"], contract="sin_contrato")

    assert cliente.textos_enviados == ["taladro 24v"]


def test_gemini_trocea_en_lotes():
    cliente = ClienteGeminiFalso()

    vectores = GeminiEncoder(client=cliente).encode(["a", "b", "c"], batch_size=2)

    assert [len(llamada) for llamada in cliente.llamadas] == [2, 1]
    assert vectores.shape == (3, 3)


def test_gemini_envia_cada_texto_como_su_propio_content():
    """Regresión: enviar cadenas desnudas hace que el SDK las agrupe en un
    único `Content` y la API devuelva 1 vector por lote en vez de uno por
    texto (47 vectores para 1.500 textos con `batch_size=32`)."""
    cliente = ClienteGeminiFalso()

    vectores = GeminiEncoder(client=cliente).encode(
        ["a", "b", "c"], batch_size=2, contract="sin_contrato"
    )

    assert cliente.llamadas == [[["a"], ["b"]], [["c"]]]
    assert vectores.shape[0] == 3


def test_gemini_no_acepta_un_lote_colapsado_en_un_solo_vector():
    """Si la API devolviera menos vectores que textos, hay que enterarse en la
    llamada — no cuatro capas más arriba con un total que no cuadra."""

    class ClienteQueColapsa(ClienteGeminiFalso):
        def embed_content(self, *, model, contents, config=None):
            respuesta = super().embed_content(model=model, contents=contents, config=config)
            respuesta.embeddings = respuesta.embeddings[:1]
            return respuesta

    with pytest.raises(RuntimeError, match="un solo Content"):
        GeminiEncoder(client=ClienteQueColapsa()).encode(["a", "b"], batch_size=2)


def test_gemini_reintenta_los_fallos_transitorios(monkeypatch):
    monkeypatch.setattr("aurum.embeddings.time.sleep", lambda _: None)
    cliente = ClienteGeminiFalso(fallos=2)

    vectores = GeminiEncoder(client=cliente, max_reintentos=3).encode(["a"])

    assert vectores.shape == (1, 3)


def test_gemini_se_rinde_y_lo_dice_tras_agotar_los_reintentos(monkeypatch):
    monkeypatch.setattr("aurum.embeddings.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="falló tras 2 intentos"):
        GeminiEncoder(client=ClienteGeminiFalso(fallos=9), max_reintentos=2).encode(["a"])


def test_gemini_exige_clave_o_cliente():
    with pytest.raises(ValueError, match="Falta la clave"):
        GeminiEncoder()


# ──────────────────────────────── Caché ──────────────────────────────────────


def test_corpus_fingerprint_distingue_corpus_que_concatenan_igual():
    """Sin separador, ['ab','c'] y ['a','bc'] tendrían la misma huella."""
    assert corpus_fingerprint(["ab", "c"]) != corpus_fingerprint(["a", "bc"])


def test_cache_key_no_mezcla_documentos_y_consultas():
    comun = {"model_id": "a/b", "corpus_id": "muestra", "fingerprint": "ff"}

    documento = cache_key(kind="document", contract="nativo", **comun)
    consulta = cache_key(kind="query", contract="nativo", **comun)

    assert documento != consulta
    assert "/" not in documento  # el model_id lleva barra y sería una subcarpeta


def test_encode_corpus_guarda_vectores_y_metadatos(tmp_path):
    modelo = ModeloFalso()
    encoder = SentenceTransformerEncoder("a/b", window=512, tasks=TAREAS_JINA, model=modelo)

    resultado = encode_corpus(
        encoder, ["uno", "dos"], corpus_id="muestra", cache_dir=tmp_path
    )

    assert resultado.vectors.shape == (2, 4)
    assert resultado.stats.desde_cache is False
    npy = list(tmp_path.glob("*.npy"))
    metadatos = json.loads(list(tmp_path.glob("*.json"))[0].read_text(encoding="utf-8"))
    assert len(npy) == 1
    assert metadatos["modelo"] == "a/b"
    assert metadatos["sha256_corpus"] == corpus_fingerprint(["uno", "dos"])
    assert metadatos["normalizado_en_origen"] is False


def test_encode_corpus_reutiliza_la_cache_sin_volver_a_codificar(tmp_path):
    modelo = ModeloFalso()
    encoder = SentenceTransformerEncoder("a/b", window=512, model=modelo)
    argumentos = {"corpus_id": "muestra", "cache_dir": tmp_path}

    primero = encode_corpus(encoder, ["uno", "dos"], **argumentos)
    segundo = encode_corpus(encoder, ["uno", "dos"], **argumentos)

    assert len(modelo.llamadas) == 1  # la segunda vez no se llamó al modelo
    assert segundo.stats.desde_cache is True
    assert np.array_equal(primero.vectors, segundo.vectors)


def test_encode_corpus_invalida_la_cache_si_cambia_el_texto(tmp_path):
    """Cambiar de plantilla (A0 -> A3) no puede reutilizar vectores del texto viejo."""
    modelo = ModeloFalso()
    encoder = SentenceTransformerEncoder("a/b", window=512, model=modelo)
    argumentos = {"corpus_id": "muestra", "cache_dir": tmp_path}

    encode_corpus(encoder, ["uno", "dos"], **argumentos)
    encode_corpus(encoder, ["uno", "DOS"], **argumentos)

    assert len(modelo.llamadas) == 2


def test_encode_corpus_no_mezcla_contratos_en_la_cache(tmp_path):
    modelo = ModeloFalso()
    encoder = SentenceTransformerEncoder("a/b", window=512, tasks=TAREAS_JINA, model=modelo)
    argumentos = {"corpus_id": "muestra", "cache_dir": tmp_path}

    encode_corpus(encoder, ["uno"], contract="nativo", **argumentos)
    resultado = encode_corpus(encoder, ["uno"], contract="sin_contrato", **argumentos)

    assert len(modelo.llamadas) == 2
    assert resultado.stats.desde_cache is False


def test_encode_corpus_funciona_sin_cache():
    encoder = SentenceTransformerEncoder("a/b", window=512, model=ModeloFalso())

    resultado = encode_corpus(encoder, ["uno"], corpus_id="muestra")

    assert resultado.vectors.shape == (1, 4)
    assert resultado.stats.n_textos == 1
