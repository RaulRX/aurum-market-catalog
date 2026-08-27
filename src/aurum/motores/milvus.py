"""Adaptador de Milvus para el guion de humo.

Lo que este adaptador deja ver de Milvus:

- **El esquema es el más explícito de los tres.** Tipo de la clave primaria,
  longitud máxima de cada campo de texto, tipo de índice y métrica se declaran a
  mano. Es el motor que menos decide por ti, y también el que más ceremonia pide.
- **Hay que cargar la colección en memoria antes de buscar** (`load_collection`).
  Es un paso que los otros dos no tienen y que cuenta para el paso 10: lo que
  Milvus ocupa depende de qué esté cargado, no solo de qué haya escrito.
- **`contains` se expresa con `like "%valor%"`**, subcadena literal como Weaviate.
  ⚠️ El comodín **por delante** es el que hay que verificar en la prueba de humo:
  las versiones antiguas de Milvus solo resolvían prefijos (`"valor%"`), y con
  eso el requisito duro no se cumpliría. La imagen fijada es la 2.6.18.
- **Son tres contenedores.** Eso no se ve en el código, pero sí en la factura de
  RAM del paso 10, y es el criterio por el que el plan lo marca como pesado.
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

_METRICS = {"cosine": "COSINE", "dot": "IP", "l2": "L2"}

# Milvus exige longitud máxima en cada VARCHAR. Salen del perfil de bytes de
# NB04 sección C, con margen: `bytes_max` de `title` manda sobre el resto.
_MAX_LEN = {"record_id": 64, "product_id": 32, "title": 512, "brand": 128, "color": 128}
_DEFAULT_MAX_LEN = 128


class MilvusStore:
    """Milvus standalone, con el esquema declarado campo a campo."""

    name = "milvus"

    def __init__(
        self,
        *,
        collection: str,
        uri: str = "http://localhost:19530",
        token: str | None = None,
        text_fields: Sequence[str] = (
            "product_id", "title", "brand", "color", "brand_normalized", "color_normalized",
        ),
        timeout: float = 60.0,
    ) -> None:
        from pymilvus import MilvusClient

        self.collection = guard_collection_name(collection)
        self.text_fields = tuple(text_fields)
        self._metric: str | None = None   # lo fija create_collection
        # Vacío en local, con valor solo contra Zilliz Cloud o Milvus con auth.
        self.client = MilvusClient(
            uri=uri, token=(token or "").strip() or None, timeout=timeout
        )

    def server_version(self) -> str:
        from pymilvus import __version__ as cliente

        return f"pymilvus {cliente}"

    def create_collection(self, *, dim: int, metric: str, recreate: bool = False) -> None:
        from pymilvus import DataType

        if metric not in _METRICS:
            raise ValueError(f"métrica no soportada por este adaptador: {metric!r}")
        self._metric = metric
        if recreate and self.client.has_collection(self.collection):
            ensure_reset_allowed(self.collection)
            self.client.drop_collection(self.collection)
        if self.client.has_collection(self.collection):
            self.client.load_collection(self.collection)
            return

        esquema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        esquema.add_field(
            "record_id", DataType.VARCHAR, is_primary=True, max_length=_MAX_LEN["record_id"]
        )
        esquema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
        for campo in self.text_fields:
            esquema.add_field(
                campo, DataType.VARCHAR,
                max_length=_MAX_LEN.get(campo, _DEFAULT_MAX_LEN),
                nullable=True,   # D14: el campo vacío existe y vale NULL, no cadena inventada
            )
        esquema.add_field("catalog_version", DataType.INT64, nullable=True)
        esquema.add_field("active", DataType.BOOL, nullable=True)

        indices = self.client.prepare_index_params()
        indices.add_index(
            field_name="vector", index_type="HNSW", metric_type=_METRICS[metric],
            params={"M": 16, "efConstruction": 200},
        )
        # Índice escalar sobre la marca: lo que el plan recomienda con 9.054
        # marcas y Einhell al 0,2 %.
        indices.add_index(field_name="brand_normalized", index_type="INVERTED")

        self.client.create_collection(
            collection_name=self.collection, schema=esquema, index_params=indices
        )
        self.client.load_collection(self.collection)

    def upsert(self, points: Sequence[Point], *, batch_size: int) -> int:
        for inicio in range(0, len(points), batch_size):
            lote = points[inicio:inicio + batch_size]
            self.client.upsert(
                collection_name=self.collection,
                data=[
                    {"record_id": punto.record_id, "vector": list(punto.vector),
                     **{campo: punto.payload.get(campo) for campo in self._campos_payload()}}
                    for punto in lote
                ],
            )
        # Sin esto, el count() del paso 2 puede leer antes de que el segmento sea
        # visible: en Milvus la escritura y la visibilidad son dos momentos (D18).
        self.client.flush(self.collection)
        return len(points)

    def count(self) -> int:
        filas = self.client.query(
            collection_name=self.collection, filter="", output_fields=["count(*)"]
        )
        return int(filas[0]["count(*)"]) if filas else 0

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        filters: Sequence[FilterCondition] = (),
    ) -> list[SearchHit]:
        respuesta = self.client.search(
            collection_name=self.collection,
            data=[list(vector)],
            limit=top_k,
            filter=self._build_filter(filters),
            output_fields=["record_id", *self._campos_payload()],
        )
        return [
            SearchHit(
                record_id=str(hit["entity"]["record_id"]),
                # El campo se llama `distance` en la respuesta, pero con COSINE e
                # IP contiene una SIMILITUD (mayor es mejor); solo con L2 es de
                # verdad una distancia. El nombre engaña y por eso se resuelve
                # desde la métrica declarada, no desde la clave del diccionario.
                score=float(hit["distance"]),
                score_kind="distance" if self._metric == "l2" else "similarity",
                higher_is_better=self._metric != "l2",
                rank=posicion,
                payload={k: v for k, v in hit["entity"].items() if k != "record_id"},
            )
            for posicion, hit in enumerate(respuesta[0] if respuesta else [], start=1)
        ]

    def get(self, record_id: str) -> Point | None:
        filas = self.client.get(collection_name=self.collection, ids=[record_id])
        if not filas:
            return None
        fila = dict(filas[0])
        return Point(
            str(fila.pop("record_id")), list(fila.pop("vector", [])), fila
        )

    def delete(self, record_id: str) -> None:
        self.client.delete(collection_name=self.collection, ids=[record_id])
        self.client.flush(self.collection)

    def close(self) -> None:
        self.client.close()

    # ── traducción del filtro ────────────────────────────────────────────────

    def _campos_payload(self) -> tuple[str, ...]:
        return (*self.text_fields, "catalog_version", "active")

    def _build_filter(self, filters: Sequence[FilterCondition]) -> str:
        """Milvus filtra con una expresión de texto, no con objetos.

        El valor se cita con `json.dumps` y no interpolando comillas a mano: un
        color con comilla dentro rompería la expresión, y en un motor que recibe
        expresiones eso no es un error de sintaxis, es una inyección.
        """
        import json

        if not filters:
            return ""
        partes = []
        for condicion in filters:
            campo = f"{condicion.field}_normalized"
            valor = json.dumps(condicion.value, ensure_ascii=False)
            partes.append(
                f"{campo} == {valor}" if condicion.operator == "equals"
                else f"{campo} like {json.dumps(f'%{condicion.value}%', ensure_ascii=False)}"
            )
        return " and ".join(partes)
