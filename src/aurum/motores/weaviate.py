"""Adaptador de Weaviate para el guion de humo.

Lo que este adaptador deja ver de Weaviate:

- **Esquema tipado y declarado por adelantado.** Hay que enumerar las propiedades
  con su tipo antes de escribir nada. Es más ceremonia que Qdrant, y a cambio el
  motor rechaza un payload que no encaje en vez de tragárselo.
- **`contains` se resuelve con `like` y comodines** —`*negro*`—, que es
  **subcadena literal**: el nivel 3 del eje de filtros. Alcanza lo mismo que
  midió B.1 y trae también los falsos positivos que contó B.3 (`rosa` casa con
  `rosado`). Es la diferencia con Qdrant, que trabaja por palabras.
- **La idempotencia sale del UUID.** Un objeto escrito con un UUID que ya existe
  se sobrescribe, así que `record_id` como UUID hace el paso 3 sin esfuerzo.
"""
from __future__ import annotations

from collections.abc import Sequence

from .base import (
    FilterCondition,
    Point,
    SearchHit,
    ensure_reset_allowed,
    guard_collection_name,
)

_METRICS = {"cosine": "cosine", "dot": "dot", "l2": "l2-squared"}


class WeaviateStore:
    """Weaviate con el vectorizador apagado: los vectores los trae el proyecto."""

    name = "weaviate"

    def __init__(
        self,
        *,
        collection: str,
        host: str = "localhost",
        port: int = 8080,
        grpc_port: int = 50051,
        api_key: str | None = None,
        payload_fields: Sequence[str] = (
            "product_id", "title", "brand", "color",
            "brand_normalized", "color_normalized", "catalog_version", "active",
        ),
    ) -> None:
        import weaviate

        # Weaviate exige que el nombre de la clase empiece por mayúscula. El
        # prefijo se valida ANTES de capitalizar, sobre el nombre que se pidió.
        guard_collection_name(collection)
        self.collection = collection[:1].upper() + collection[1:]
        self.payload_fields = tuple(payload_fields)
        # Credencial opcional y vacía en local, como en la sesión 3: el compose
        # de docker/ arranca con acceso anónimo habilitado.
        credencial = (api_key or "").strip()
        self.client = weaviate.connect_to_local(
            host=host, port=port, grpc_port=grpc_port,
            auth_credentials=(
                weaviate.classes.init.Auth.api_key(credencial) if credencial else None
            ),
        )

    def server_version(self) -> str:
        return str(self.client.get_meta()["version"])

    def create_collection(self, *, dim: int, metric: str, recreate: bool = False) -> None:
        from weaviate.classes.config import Configure, DataType, Property, VectorDistances

        if metric not in _METRICS:
            raise ValueError(f"métrica no soportada por este adaptador: {metric!r}")
        if recreate and self.client.collections.exists(self.collection):
            ensure_reset_allowed(self.collection)
            self.client.collections.delete(self.collection)
        if self.client.collections.exists(self.collection):
            return

        # `dim` no se declara: Weaviate la fija con el primer vector escrito. Se
        # comprueba después, en la verificación de "dimensión declarada == real".
        self.client.collections.create(
            name=self.collection,
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances(_METRICS[metric]),
            ),
            properties=[
                Property(
                    name=campo,
                    data_type=DataType.BOOL if campo == "active"
                    else DataType.INT if campo == "catalog_version"
                    else DataType.TEXT,
                    # Filtrable sin tokenizar: el `like` trabaja sobre el valor entero.
                    tokenization=None,
                )
                for campo in self.payload_fields
            ],
        )

    def upsert(self, points: Sequence[Point], *, batch_size: int) -> int:
        coleccion = self.client.collections.get(self.collection)
        for inicio in range(0, len(points), batch_size):
            lote = points[inicio:inicio + batch_size]
            with coleccion.batch.fixed_size(batch_size=batch_size) as batch:
                for punto in lote:
                    batch.add_object(
                        uuid=punto.record_id,      # UUIDv5: escribir dos veces sobrescribe
                        properties={
                            campo: punto.payload[campo]
                            for campo in self.payload_fields
                            if campo in punto.payload
                        },
                        vector=list(punto.vector),
                    )
        return len(points)

    def count(self) -> int:
        coleccion = self.client.collections.get(self.collection)
        return int(coleccion.aggregate.over_all(total_count=True).total_count)

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        filters: Sequence[FilterCondition] = (),
    ) -> list[SearchHit]:
        from weaviate.classes.query import MetadataQuery

        coleccion = self.client.collections.get(self.collection)
        respuesta = coleccion.query.near_vector(
            near_vector=list(vector),
            limit=top_k,
            filters=self._build_filter(filters),
            return_metadata=MetadataQuery(distance=True),
        )
        return [
            SearchHit(
                record_id=str(obj.uuid),
                # ⚠️ Weaviate devuelve DISTANCIA, no similitud: menor es mejor.
                # Es la asimetría que el enunciado (§3.2) avisa de no aplanar —
                # meter esto en la misma columna que el score de Qdrant daría el
                # orden invertido sin que nada fallara.
                score=float(obj.metadata.distance or 0.0),
                score_kind="distance",
                higher_is_better=False,
                rank=posicion,
                payload=dict(obj.properties),
            )
            for posicion, obj in enumerate(respuesta.objects, start=1)
        ]

    def get(self, record_id: str) -> Point | None:
        coleccion = self.client.collections.get(self.collection)
        obj = coleccion.query.fetch_object_by_id(record_id, include_vector=True)
        if obj is None:
            return None
        vector = obj.vector.get("default", []) if isinstance(obj.vector, dict) else (obj.vector or [])
        return Point(str(obj.uuid), list(vector), dict(obj.properties))

    def delete(self, record_id: str) -> None:
        self.client.collections.get(self.collection).data.delete_by_id(record_id)

    def close(self) -> None:
        self.client.close()

    # ── traducción del filtro ────────────────────────────────────────────────

    def _build_filter(self, filters: Sequence[FilterCondition]):
        from weaviate.classes.query import Filter

        if not filters:
            return None
        condiciones = []
        for condicion in filters:
            propiedad = Filter.by_property(f"{condicion.field}_normalized")
            condiciones.append(
                propiedad.equal(condicion.value) if condicion.operator == "equals"
                # Subcadena literal, con los falsos positivos que B.3 cuantifica.
                else propiedad.like(f"*{condicion.value}*")
            )
        return Filter.all_of(condiciones) if len(condiciones) > 1 else condiciones[0]
