# El índice del catálogo · aceptación (NB04 § G)

Colección: `aurum_catalogo__gemini_embedding_2__A4__768`  ·  motor: **qdrant** (R03)

| Elemento | Valor | De dónde sale |
|---|---|---|
| Id del punto | `record_id` (UUIDv5) | README_DATOS · idempotencia por id |
| Modelo | `gemini-embedding-2` [sin_contrato] | R02 |
| Plantilla | `A4` | R01 |
| Dimensión | 768 (truncada de 3.072 y renormalizada) | D09b |
| Métrica | cosine | D10 |
| Payload | `completo` · nulos `cadena_vacia` | D13 · D14 |
| Lote de ingesta | 128 | D15 |
| Índices de payload | brand_normalized (keyword) · color_normalized (texto) | Sección B |
| Puntos | 15.000 | catalogo_productos.csv |

## Cómo leer los números del paso 8

- La **RAM** es en reposo tras ingerir, no bajo carga. Vale solo si el `NET I/O` del transcrito no está a cero: un contenedor recién reiniciado no ha tocado sus datos y da una cifra que no es la del índice. La huella sirviendo consultas se mide en NB06.
- El **volumen** no lo llena el índice. Con los 1.500 de la prueba de humo eran 293,8 MB y con los 15.000 son 289,5: diez veces más puntos y no crece. Los vectores son 46 MB; el resto es asignación del motor —WAL— y por eso escribir esos 46 MB costó 665 MB de `BLOCK I/O`.

## Las ocho comprobaciones

|   paso | comprobacion                                 | que_ha_hecho                                                              | esperado                                                               | observado                                                                                                                                                                                                      | resultado   |   segundos |
|-------:|:---------------------------------------------|:--------------------------------------------------------------------------|:-----------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------|-----------:|
|      1 | Colección con dimensión y métrica explícitas | create_collection(dim=768, metric='cosine', recreate=False)               | dim=768, métrica=cosine                                                | 'aurum_catalogo__gemini_embedding_2__A4__768' lista (dim=768, métrica=cosine)                                                                                                                                  | ✅ pasa     |      1.37  |
|      2 | Ingesta por lotes                            | upsert(15000 puntos, batch_size=128) → count()                            | count() == 15000                                                       | count() = 15000 · 317.2 vectores/s en 47.3 s                                                                                                                                                                   | ✅ pasa     |     47.293 |
|      3 | Índice al día antes de aceptar consultas     | index_ready() · espera con tope de 180 s                                  | el motor declara su índice construido                                  | AÚN INDEXANDO al terminar la ingesta · listo tras 4.1 s de espera (3 sondeos)                                                                                                                                  | ✅ pasa     |      4.139 |
|      4 | Dimensión declarada == dimensión real        | collection_dim()                                                          | la colección dice 768                                                  | la colección declara 768                                                                                                                                                                                       | ✅ pasa     |    nan     |
|      5 | Ingesta repetida sin duplicar                | upsert(LOS MISMOS 15000 puntos) → count()                                 | count() sigue en 15000                                                 | count() = 15000                                                                                                                                                                                                | ✅ pasa     |     70.761 |
|      6 | Canarios: cada punto se recupera a sí mismo  | search(vector del propio punto, top_k=10) × 3                             | los 3 vuelven en la posición 1                                         | 3/3 en la posición 1                                                                                                                                                                                           | ✅ pasa     |    nan     |
|      7 | Persistencia tras reinicio                   | make motor-down MOTOR=qdrant && make motor-up MOTOR=qdrant  (+ relectura) | mismo count() y los mismos canarios en la posición 1                   | ✅ idéntico — count() 15000 y los mismos 10 ids en el mismo orden                                                                                                                                              | ✅ pasa     |    nan     |
|      8 | Recursos con el índice completo              | make motor-stats  (con el motor vivo y los 15.000 ingeridos)              | RAM del contenedor y tamaño del volumen, esta vez sobre volumen limpio | RAM 106.6 MiB en 1 contenedor · volumen 289.5 MB (no comparable entre motores) · dentro del `mem_limit` declarado, el más apretado aurum-market-qdrant al 5% · foto con el contenedor «Up 4 minutes (healthy)» | ✅ pasa     |    nan     |
