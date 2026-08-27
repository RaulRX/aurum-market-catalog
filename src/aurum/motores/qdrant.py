"""Adaptador de Qdrant para el guion de humo.

El SDK se importa **dentro** de los métodos y no arriba: así `aurum.motores`
sigue siendo importable con solo dos de los tres clientes instalados, y probar
un motor no obliga a instalar los otros dos.

Lo que este adaptador deja ver de Qdrant, que es de lo que va la comparativa:

- **El filtro es nativo y va dentro de la consulta**, no después. `equals` sale
  con `MatchValue` sin preparar nada.
- **`contains` exige declarar un índice de texto sobre el campo.** No basta con
  pedirlo en la consulta: sin `create_payload_index(..., TextIndexParams)` el
  `MatchText` no encuentra nada. Es configuración explícita, no un detalle, y es
  justo el tipo de control que el enunciado premia frente a un motor que lo
  esconde.
- **El id del punto puede ser el `record_id`** porque es UUID. Eso es lo que hace
  la idempotencia gratis: `upsert` sobrescribe por id y el paso 3 sale solo.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base import (
    COLLECTION_PREFIX,
    FilterCondition,
    Point,
    SearchHit,
    UnsupportedFilterError,
    ensure_reset_allowed,
    guard_collection_name,
)

# Qdrant nombra las métricas a su manera; el guion las pide en el vocabulario
# del proyecto. La traducción vive aquí y no en el notebook.
_METRICS = {"cosine": "Cosine", "dot": "Dot", "l2": "Euclid"}


class QdrantStore:
    """Qdrant sobre gRPC, que es el transporte que su SDK recomienda para ingesta."""

    name = "qdrant"

    def __init__(
        self,
        *,
        collection: str,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        grpc_port: int = 6334,
        prefer_grpc: bool = True,
        text_fields: Sequence[str] = ("color",),
        timeout: int = 60,
        prefix: str = COLLECTION_PREFIX,
    ) -> None:
        from qdrant_client import QdrantClient

        # `prefix` por defecto es el de la prueba de humo. El índice definitivo
        # pasa `CATALOG_PREFIX`, y esa separación es lo que impide que una
        # errata en el guion -que recrea en cada pasada- alcance a los 15.000
        # puntos del índice bueno.
        self.collection = guard_collection_name(collection, prefix=prefix)
        # Campos que van a recibir `contains`: necesitan índice de texto y hay
        # que declararlos al crear la colección, no al consultarla.
        self.text_fields = tuple(text_fields)
        # Lo fija `create_collection`; hasta entonces no se sabe qué significa el
        # score, y decir "similitud" por defecto sería inventárselo.
        self._metric: str | None = None
        self.client = QdrantClient(
            url=url,
            # Igual que la sesión 3: la credencial vacía viaja como `None`. En
            # local no hay autenticación y pasar "" la rompería.
            api_key=(api_key or "").strip() or None,
            grpc_port=grpc_port, prefer_grpc=prefer_grpc, timeout=timeout,
        )

    def server_version(self) -> str:
        return str(self.client.info().version)

    def create_collection(
        self,
        *,
        dim: int,
        metric: str,
        recreate: bool = False,
        hnsw_m: int | None = None,
        hnsw_ef_construct: int | None = None,
    ) -> None:
        """`hnsw_m`/`hnsw_ef_construct` no están en `VectorStore` por el mismo
        motivo que `ef` en `search`: son de Qdrant, no de cualquier motor.

        NB06 (D16) los deja en `None` -el `m=16`/`ef_construct=100` por defecto
        del propio Qdrant, ya lo que tiene el índice de NB04- y solo barre `ef`
        en consulta, porque eso no reconstruye nada. Tocar estos dos sí lo
        haría: es la mejora que queda anotada para una iteración futura con
        varios tamaños de catálogo, no la que se mide aquí."""
        from qdrant_client import models

        if metric not in _METRICS:
            raise ValueError(f"métrica no soportada por este adaptador: {metric!r}")
        self._metric = metric
        if recreate and self.client.collection_exists(self.collection):
            ensure_reset_allowed(self.collection)
            self.client.delete_collection(self.collection)
        if not self.client.collection_exists(self.collection):
            hnsw_config = (
                models.HnswConfigDiff(m=hnsw_m, ef_construct=hnsw_ef_construct)
                if hnsw_m is not None or hnsw_ef_construct is not None
                else None
            )
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=dim, distance=models.Distance[_METRICS[metric].upper()]
                ),
                hnsw_config=hnsw_config,
            )

        # Índice de payload sobre `brand`: el plan lo recomienda con 9.054 marcas
        # y Einhell al 0,2 %. Es una decisión de esquema, declarada aquí.
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="brand_normalized",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        # Índice de texto para el `contains` del color. Sin esto el filtro no
        # falla: devuelve vacío, que es peor porque parece un resultado.
        for field in self.text_fields:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=f"{field}_normalized",
                field_schema=models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.WORD,
                    lowercase=True,
                ),
            )

    def upsert(self, points: Sequence[Point], *, batch_size: int) -> int:
        from qdrant_client import models

        for inicio in range(0, len(points), batch_size):
            lote = points[inicio:inicio + batch_size]
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    models.PointStruct(
                        id=punto.record_id,          # UUIDv5: idempotencia por id
                        vector=list(punto.vector),
                        payload=punto.payload,
                    )
                    for punto in lote
                ],
                wait=True,   # D18 asomando: sin esto el count() del paso 2 corre antes que la escritura
            )
        return len(points)

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        filters: Sequence[FilterCondition] = (),
        ef: int | None = None,
    ) -> list[SearchHit]:
        """`ef` es `hnsw_ef` de Qdrant: un parámetro de **consulta**, no del
        índice -NB06 lo barre sin reconstruir nada-. `None` no manda
        `search_params` y deja el comportamiento de siempre, el que ya mide
        NB05: Qdrant decide su propio `ef` por defecto."""
        from qdrant_client import models

        respuesta = self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            limit=top_k,
            query_filter=self._build_filter(filters, models),
            search_params=models.SearchParams(hnsw_ef=ef) if ef is not None else None,
            with_payload=True,
        )
        return [
            SearchHit(
                record_id=str(punto.id),
                # Qdrant devuelve SIMILITUD con Cosine y Dot (mayor es mejor), y
                # DISTANCIA con Euclid (menor es mejor). Lo declara la métrica de
                # la colección, no el resultado, así que se resuelve aquí.
                score=float(punto.score),
                score_kind="distance" if self._metric == "l2" else "similarity",
                higher_is_better=self._metric != "l2",
                rank=posicion,
                payload=dict(punto.payload or {}),
            )
            for posicion, punto in enumerate(respuesta.points, start=1)
        ]

    def get(self, record_id: str) -> Point | None:
        encontrados = self.client.retrieve(
            collection_name=self.collection, ids=[record_id],
            with_payload=True, with_vectors=True,
        )
        if not encontrados:
            return None
        punto = encontrados[0]
        return Point(str(punto.id), list(punto.vector or []), dict(punto.payload or {}))

    def delete(self, record_id: str) -> None:
        from qdrant_client import models

        self.client.delete(
            collection_name=self.collection,
            points_selector=models.PointIdsList(points=[record_id]),
            wait=True,
        )

    def index_ready(self) -> bool:
        """¿Terminó Qdrant de construir el índice, o sigue en `yellow`/`grey`?

        Qdrant indexa en segundo plano: el `count()` puede cuadrar mientras el
        HNSW aún se construye, y una búsqueda en ese momento devuelve menos de lo
        que hay escrito. Es lo que §3.2 llama "estado de indexación".
        """
        return str(self.client.get_collection(self.collection).status).lower().endswith("green")

    def collection_dim(self) -> int:
        """La dimensión que la colección tiene declarada, preguntándosela a ella.

        §3.2 pide comprobar que el esquema es el que se cree. Leerla del motor y
        no de la variable con la que se creó es la diferencia entre verificar y
        repetirse: una colección preexistente con otra dimensión daría el mismo
        `dim` en el notebook y otra cosa aquí.
        """
        return int(self.client.get_collection(self.collection).config.params.vectors.size)

    def close(self) -> None:
        self.client.close()

    # ── traducción del filtro: aquí es donde se ve qué sabe hacer el motor ────

    def _build_filter(self, filters: Sequence[FilterCondition], models: Any):
        if not filters:
            return None
        condiciones = []
        for condicion in filters:
            campo = f"{condicion.field}_normalized"   # D03: se filtra por la clave derivada
            if condicion.operator == "equals":
                condiciones.append(models.FieldCondition(
                    key=campo, match=models.MatchValue(value=condicion.value)
                ))
            elif condicion.field in self.text_fields:
                # MatchText es coincidencia por **palabras**, no subcadena literal:
                # `rosa` no casa con `rosado`. Es el nivel 2 del eje de filtros, y
                # da el alcance de B.1 sin los falsos positivos de B.3. Anotarlo en
                # la comparativa: es una ventaja que no estaba prevista.
                condiciones.append(models.FieldCondition(
                    key=campo, match=models.MatchText(text=condicion.value)
                ))
            else:
                raise UnsupportedFilterError(
                    f"`contains` sobre {condicion.field!r} necesita un índice de texto "
                    f"declarado al crear la colección; este adaptador solo lo declara "
                    f"para {self.text_fields}."
                )
        return models.Filter(must=condiciones)
