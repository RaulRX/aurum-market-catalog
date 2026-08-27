# Prueba de humo · comparativa de motores (D12 → R03)

Corpus: `catalogo_muestra.csv` (1500 puntos) · dim 768 · métrica cosine · lote 128 (D15)
Payload: `completo` (D13) · nulos: `cadena_vacia` (D14)
Filtro del paso 5: `brand equals 'einhell'` (FILTER-001)
|                                               | milvus   | qdrant   | weaviate   |
|:----------------------------------------------|:---------|:---------|:-----------|
| (1, 'Crear colección')                        | ✅ pasa  | ✅ pasa  | ✅ pasa    |
| (2, 'Ingesta por lotes')                      | ✅ pasa  | ✅ pasa  | ✅ pasa    |
| (3, 'Ingesta repetida + índice listo')        | ✅ pasa  | ✅ pasa  | ✅ pasa    |
| (4, 'Búsqueda global')                        | ✅ pasa  | ✅ pasa  | ✅ pasa    |
| (5, 'Filtro nativo')                          | ✅ pasa  | ✅ pasa  | ✅ pasa    |
| (6, 'Lectura por record_id')                  | ✅ pasa  | ✅ pasa  | ✅ pasa    |
| (7, 'Borrado')                                | ✅ pasa  | ✅ pasa  | ✅ pasa    |
| (8, 'Persistencia tras reinicio')             | ✅ pasa  | ✅ pasa  | ✅ pasa    |
| (9, 'Calidad del error con el motor apagado') | ✅ pasa  | ✅ pasa  | ✅ pasa    |
| (10, 'Recursos')                              | ✅ pasa  | ✅ pasa  | ✅ pasa    |

## Nivel de `contains` sobre metadatos (requisito duro · sección B)

- **milvus** — ✅ NIVEL 3 · subcadena literal — alcanza B.1 y trae los falsos positivos de B.3
- **qdrant** — ✅ NIVEL 2 · por palabras — alcanza B.1 SIN los falsos positivos de B.3
- **weaviate** — ✅ NIVEL 3 · subcadena literal — alcanza B.1 y trae los falsos positivos de B.3

## Las cuatro consultas filtradas (§5)

|                                          |   milvus |   qdrant |   weaviate |
|:-----------------------------------------|---------:|---------:|-----------:|
| ('FILTER-001', "brand equals 'einhell'") |        1 |        1 |          1 |
| ('FILTER-002', "brand equals 'apple'")   |        0 |        0 |          0 |
| ('FILTER-003', "brand equals 'nike'")    |        1 |        1 |          1 |
| ('FILTER-004', "brand equals 'samsung'") |        1 |        1 |          1 |

## Donde los motores NO se comportaron igual

Solo los pasos con `observado` distinto entre motores: los que coinciden no separan a nadie.

|                                               | milvus                                                                                                                                                                                                                                                      | qdrant                                                                                                                                                                                                                                                 | weaviate                                                                                                                                                                                                        |
|:----------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| (3, 'Ingesta repetida + índice listo')        | count() = 1500 · índice: no lo reporta el motor                                                                                                                                                                                                             | count() = 1500 · índice: listo                                                                                                                                                                                                                         | count() = 1500 · índice: no lo reporta el motor                                                                                                                                                                 |
| (4, 'Búsqueda global')                        | 10 resultados · score: similarity (mayor es mejor)                                                                                                                                                                                                          | 10 resultados · score: similarity (mayor es mejor)                                                                                                                                                                                                     | 10 resultados · score: distance (menor es mejor)                                                                                                                                                                |
| (8, 'Persistencia tras reinicio')             | ✅ idéntico — count() 1499 y los mismos 10 ids en el mismo orden                                                                                                                                                                                            | ✅ idéntico — count() 1500 y los mismos 10 ids en el mismo orden                                                                                                                                                                                       | ✅ idéntico — count() 1500 y los mismos 10 ids en el mismo orden                                                                                                                                                |
| (9, 'Calidad del error con el motor apagado') | ✅ tipada por el SDK — pymilvus.exceptions.MilvusException: <MilvusException: (code=2, message=Fail connecting to server on localhost:19530, illegal connection params or server unavailable)>                                                              | ⚠️ tipada pero de grpc, no del SDK (qdrant_client) — grpc._channel._InactiveRpcError: <_InactiveRpcError of RPC that terminated with: status = StatusCode.UNAVAILABLE details = "failed to connect to all addresses; last error: UNAVAILABLE: ipv4:127 | ✅ tipada por el SDK — weaviate.exceptions.WeaviateConnectionError: Connection to Weaviate failed. Details: Error: timed out. Is Weaviate running and reachable at http://localhost:8080?                       |
| (10, 'Recursos')                              | RAM 530.8 MiB en 3 contenedores, el mayor aurum-market-milvus-minio · volumen 182.9 MB (no comparable entre motores) · dentro del `mem_limit` declarado, el más apretado aurum-market-milvus-minio al 50% · foto con el contenedor «Up 2 minutes (healthy)» | RAM 36.4 MiB en 1 contenedor · volumen 293.8 MB (no comparable entre motores) · dentro del `mem_limit` declarado, el más apretado aurum-market-qdrant al 2% · foto con el contenedor «Up 6 minutes (healthy)»                                          | RAM 50.9 MiB en 1 contenedor · volumen 10.9 MB (no comparable entre motores) · dentro del `mem_limit` declarado, el más apretado aurum-market-weaviate al 2% · foto con el contenedor «Up 10 minutes (healthy)» |


## Segundos por paso

Una sola pasada sobre 1.500 puntos, no un banco de pruebas: sirven para ver ordenes de magnitud, no para afinar.

|                                        |   milvus |   qdrant |   weaviate |
|:---------------------------------------|---------:|---------:|-----------:|
| (1, 'Crear colección')                 |    3.664 |    1.156 |      0.324 |
| (2, 'Ingesta por lotes')               |    2.641 |    3.836 |      3.991 |
| (3, 'Ingesta repetida + índice listo') |    9.683 |    3.693 |      3.279 |
| (4, 'Búsqueda global')                 |    0.036 |    0.022 |      0.01  |
| (5, 'Filtro nativo')                   |    0.088 |    0.01  |      0.005 |
| (6, 'Lectura por record_id')           |    0.014 |    0.005 |      0.006 |
| (7, 'Borrado')                         |   11.551 |    0.009 |      0.006 |


## Paso 10 · recursos, y en qué condiciones se midieron

| motor    |   n_contenedores |   ram_mib |   volumen_mb | mayor_consumidor          | dentro_del_limite   | mas_apretado                     | en_marcha               | excluidos   |
|:---------|-----------------:|----------:|-------------:|:--------------------------|:--------------------|:---------------------------------|:------------------------|:------------|
| qdrant   |                1 |      36.4 |        293.8 | aurum-market-qdrant       | True                | aurum-market-qdrant al 2%        | Up 6 minutes (healthy)  |             |
| weaviate |                1 |      50.9 |         10.9 | aurum-market-weaviate     | True                | aurum-market-weaviate al 2%      | Up 10 minutes (healthy) |             |
| milvus   |                3 |     530.8 |        182.9 | aurum-market-milvus-minio | True                | aurum-market-milvus-minio al 50% | Up 2 minutes (healthy)  |             |

- La RAM es una foto en reposo tras ingerir, no un pico bajo carga: mide el coste base del proceso, que con 4,6 MB de vectores en la muestra es casi todo lo que hay.
- El tamaño del volumen NO es comparable entre motores: los volúmenes con nombre sobreviven a `down` y arrastran todo lo que cada motor escribió mientras se desarrollaba su adaptador, más la preasignación de WAL.
- Cada motor se midió con un tiempo en marcha distinto; la columna `en_marcha` lo declara en vez de dejarlo suponer.

Transcritos íntegros de `docker stats` y `docker system df -v` en `artifacts/recursos/`; la tabla de arriba se deriva de ellos.
