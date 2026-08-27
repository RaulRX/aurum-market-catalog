# Despliegue de los motores candidatos (D12)

Un `compose.yaml` por motor, a propósito: **los tres no caben a la vez** en 7,9 GB
compartidos con el notebook, y la comparativa exige medir la RAM de cada uno sin
que la contaminen los otros (paso 10 del guion de humo).

Adaptados de `../../sesion_03/deploy/`. Los nombres
de proyecto, contenedor y volumen llevan prefijo `aurum-market-` para que
levantar esto **no toque** los contenedores ni los datos de la sesión.

## Comandos

Se ejecutan desde la terminal, fuera del notebook. El `Makefile` de la raíz los envuelve:

```bash
make motor-up   MOTOR=qdrant   # levanta y espera a que el healthcheck pase
make motor-ps   MOTOR=qdrant   # ¿está listo para ejecutar el notebook?
make motor-stats               # paso 10 — RAM de los contenedores y volúmenes
make motor-down MOTOR=qdrant   # apaga y libera la máquina
```

O a mano, desde la raíz del repo:

```bash
# Levantar uno
docker compose -f docker/qdrant/compose.yaml up -d

# Comprobar que está listo antes de ejecutar el notebook
docker compose -f docker/qdrant/compose.yaml ps

# Paso 8 del guion — persistencia
docker compose -f docker/qdrant/compose.yaml restart

# Paso 9 del guion — calidad del error con el motor caído
docker compose -f docker/qdrant/compose.yaml stop

# Paso 10 del guion — RAM y volumen
docker stats --no-stream
docker system df -v | grep aurum-market

# Apagar y liberar la máquina antes del siguiente motor
docker compose -f docker/qdrant/compose.yaml down
```

> ⚠️ **Bájalo antes de levantar el siguiente.** El `mem_limit` de cada servicio
> mantiene la medición comparable, pero no impide que tres motores a la vez
> dejen sin memoria al kernel del notebook.

## Puertos

| Motor | Puertos | Contenedores | `mem_limit` total |
|---|---|---|---|
| Qdrant | 6333 (REST) · 6334 (gRPC) | 1 | 2 GB |
| Weaviate | 8080 (REST) · 50051 (gRPC) | 1 | 2 GB |
| Milvus | 19530 (gRPC) · 9091 (healthz) | **3** (etcd + minio + standalone) | 4 GB |

Ninguno se solapa, así que el orden en que los pruebes no importa — pero la
memoria sí, y Milvus conviene dejarlo para el final y con el resto abajo.

## Lo que estos ficheros dejan fijado, y por qué

- **Sin vectorizador en el motor.** Weaviate va con `DEFAULT_VECTORIZER_MODULE: none`.
  Los vectores los trae el proyecto ya calculados (`gemini-embedding-2` @768,
  R02 + R01). Si el motor vectorizara, la comparativa mediría su modelo y no su
  índice, que es lo que se está eligiendo aquí.
- **Volúmenes con nombre.** La persistencia del paso 8 tiene que sobrevivir a un
  `restart`; con almacenamiento efímero el paso no probaría nada.
- **Telemetría apagada** donde el motor la trae puesta.
- **`mem_limit` declarado.** No es una optimización: es lo que hace que la
  columna de RAM del paso 10 compare motores en vez de comparar en qué momento
  de la sesión se midió cada uno.
