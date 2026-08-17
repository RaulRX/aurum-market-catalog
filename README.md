# Aurum Market: búsqueda semántica y control de catálogo

**Motor de descubrimiento y control de catálogo**

Repositorio: [https://github.com/RaulRX/aurum-market-catalog](https://github.com/RaulRX/aurum-market-catalog.git)
Autor: Raúl Sánchez Serrano

## Descripción del proyecto

Aurum Market es un motor de descubrimiento de productos sobre un catálogo de 15.000
referencias en español. El sistema resuelve dos recorridos: **búsqueda semántica** con
recuperación top-k y filtrado por marca ejecutado por la base de datos, y **control de
altas**, que detecta posibles productos duplicados al ingresar nuevas registros del catálogo en el
catálogo. La solución se apoya en una base de datos vectorial (embeddings + índice ANN)
y se compara en todo momento contra un baseline léxico, midiendo calidad del ranking,
fidelidad del índice, latencia, filtros, duplicados y mutaciones sobre la colección. No
se utiliza generación de texto ni LLM en tiempo de ejecución: todas las decisiones se
justifican con métricas y experimentos reproducibles.

---

## Índice

- [Contexto y objetivo](#contexto-y-objetivo)
- [Arquitectura del sistema](#arquitectura-del-sistema)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Variables de entorno](#variables-de-entorno)
- [Datos](#datos)
- [Baseline léxico](#baseline-léxico)
- [Modelo de embeddings](#modelo-de-embeddings)
  - [D09 · Modelos comparados](#d09--modelos-comparados)
  - [D09b · Criterio de desempate](#d09b--criterio-de-desempate)
  - [D10 · Normalización y métrica](#d10--normalización-y-métrica--qué-significa-el-score)
  - [R02 · Qué está confirmado y qué no](#r02--qué-está-confirmado-y-qué-no)
- [Representación del texto](#representación-del-texto)
  - [D06 · Qué plantillas se compararon](#d06--qué-plantillas-se-compararon)
  - [D07 · Sin chunking](#d07--sin-chunking)
  - [R01 · Plantilla elegida](#r01--plantilla-elegida)
- [Configuración](#configuración)
- [Recorrido de notebooks](#recorrido-de-notebooks)
- [Comandos](#comandos)
- [Tests](#tests)
- [Resultados y artefactos](#resultados-y-artefactos)
- [Métricas y evaluación](#métricas-y-evaluación)
- [Visualizaciones](#visualizaciones)
- [Tiempos aproximados](#tiempos-aproximados)
- [Solución de problemas frecuentes](#solución-de-problemas-frecuentes)
- [Seguridad y limpieza de recursos](#seguridad-y-limpieza-de-recursos)
- [Licencia y procedencia de los datos](#licencia-y-procedencia-de-los-datos)

---

## Contexto y objetivo

TODO

## Arquitectura del sistema

TODO

## Estructura del repositorio

```
aurum-market-catalog/
├─ notebooks/            00_datos.ipynb … 10_entrega.ipynb — el núcleo de la entrega
├─ scripts/               build_notebook.py · execute_notebook.py · notebook_cells.py
│                         (construyen y ejecutan los notebooks a partir de Python plano,
│                          revisable y versionable — el .ipynb es un artefacto regenerable)
├─ src/aurum/              datos.py · embeddings.py · almacen.py · busqueda.py ·
│                          duplicados.py · evaluacion.py — la lógica que importan los notebooks
├─ tests/                 test_ids.py · test_batching.py · test_filtros.py ·
│                          test_mutaciones.py · test_formato.py — pruebas mínimas exigidas
├─ config/config.yaml     parámetros de la ejecución
├─ artifacts/             perfilado, comparativas, trazas, métricas intermedias
├─ resultados/            resultados_busqueda.csv · resultados_duplicados.csv
├─ data/                  CSVs de entrada — no se commitea, ver sección Datos
├─ docker/docker-compose.yaml
├─ requirements.txt · pytest.ini · Makefile · .env.example · README.md
```

No forman parte de la entrega y están excluidos del repositorio (`.gitignore`): `data/` (los datos en sí), `docs/` (documentación de trabajo interna: plan, memorias de sesión, el enunciado), el entorno virtual `.venv/`, las cachés de herramientas (`__pycache__/`, `.pytest_cache/`), y el material de estudio de sesiones previas (`notebooks/sesiones/`, `src/vector_index_session/`, `src/vector_search_session/`) — se consulta como referencia, pero cualquier función que se reutilice de ahí se copia y adapta dentro de `src/aurum/`, nunca se importa directamente.

## Requisitos

| | Mínimo | Notas |
|---|---|---|
| Python | 3.11+ | El desarrollo se hizo sobre 3.13 |
| RAM | 8 GB | Los modelos de embeddings y el motor vectorial compiten por ella; el motor se levanta de uno en uno |
| Disco | ~6 GB libres | ~1 GB de dependencias (`torch` incluido) + ~2,5 GB de pesos de los modelos descargados en la caché de Hugging Face + volúmenes de Docker |
| CPU | 4 núcleos | Toda la codificación es **CPU-only**; no se requiere GPU y el código no la usa |
| Docker | Desktop o Engine | Solo para el motor vectorial (a partir de NB04) |

El entorno de referencia de las mediciones que aparecen en este README es: Windows 10, Intel i5-6300HQ (4 núcleos / 4 hilos), 7,9 GB de RAM, sin GPU utilizable. Los tiempos dependen del equipo; las métricas de calidad, no.

> `torch` se instala desde PyPI sin índice adicional. En Windows la rueda oficial ya es CPU-only; en Linux, PyPI sirve la variante con CUDA, más pesada pero igualmente funcional en CPU. No hace falta configurar nada.

## Instalación

1. Clona el repositorio y sitúate en su raíz.
2. Crea el entorno virtual del proyecto y actívalo (nunca se instala nada en el intérprete global):

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. Instala las dependencias congeladas:

   ```bash
   pip install -r requirements.txt
   ```

4. Registra el kernel de Jupyter, necesario para regenerar y ejecutar los notebooks con `scripts/execute_notebook.py`:

   ```bash
   python -m ipykernel install --user --name aurum-market-catalog --display-name "Aurum Market (.venv)"
   ```

5. Copia los ficheros de datos dentro de `data/` (ver sección [Datos](#datos)) — el directorio está vacío en el repositorio por diseño.

6. Verifica la instalación ejecutando los tests:

   ```bash
   pytest
   ```

## Variables de entorno

Se leen de un fichero `.env` en la raíz, que **no está versionado** (`.gitignore`). Copia `.env.example` y rellena solo lo que vayas a usar:

```bash
cp .env.example .env
```

| Variable | ¿Obligatoria? | Para qué |
|---|---|---|
| `GEMINI_API_KEY` | Solo para `gemini-embedding-2` | Medir su longitud en tokens (`count_tokens`) y generar sus embeddings. **Sin ella, esas celdas se omiten con un aviso y el resto del notebook se ejecuta igual** |
| `HF_TOKEN` | No, salvo repos *gated* | Descargar modelos de Hugging Face. Los tres candidatos actuales son de acceso abierto, así que solo hace falta si se añade un modelo con licencia aceptada (p. ej. la familia Gemma) |
| `LOCAL_EMBEDDING_MODEL` | No | Sobrescribe el modelo local por defecto sin tocar el código |
| `FAISS_NUM_THREADS` | No | Limita los hilos de FAISS en el laboratorio ANN (NB06) |

Ninguna clave aparece en el código, en los notebooks ni en sus salidas: se leen con `os.environ` tras `load_dotenv()`, y `.env.example` se mantiene siempre **sin valores reales**.

## Datos

Decisiones de negocio tomadas en NB00 sobre `catalogo_muestra.csv` (1.500 filas), con la evidencia real que las sustenta. Detalle completo, código y celdas en `notebooks/00_datos.ipynb` (regenerable con `python scripts/build_notebook.py 00_datos.ipynb` + `python scripts/execute_notebook.py 00_datos.ipynb`).

### D01 — Qué cuenta como "relevante" para Recall@10 y MRR@10

**Decisión: Exact + Substitute (E+S).** nDCG no se ve afectado (usa siempre el mapeo `E=3, S=2, C=1, I=0` del enunciado); esta decisión solo cambia el denominador de Recall@10 y MRR@10.

Denominador por consulta de desarrollo (`relevantes_solo_E` vs `relevantes_E_mas_S`):

| query_id | consulta | solo E | E+S |
|---|---|---:|---:|
| 13357 | base tapizada 160x200 sin patas | 4 | 31 |
| 18868 | botines marrones mujer tacon medio | 7 | 9 |
| 28703 | convertibles 2 en 1 portátil táctil | 25 | 39 |
| 31224 | cámaras bridge baratas | 5 | 15 |
| 33633 | disfraz halloween talla grande hombre | 1 | 4 |
| 38249 | estantes sin taladro habitación | 30 | 35 |
| 43240 | funda ipad air 4 sin tapa | 24 | 35 |
| 61533 | lentejas sin gluten | 25 | 30 |

**Por qué elegí esto:** elegí que Exact + Substitute cuenten como relevante para Recall@10 y MRR@10. Considero que un sustituto razonable es un resultado exitoso para el cliente, no solo el acierto exacto — es una decisión de negocio legítima sobre qué significa "servir bien" una búsqueda. Además, esta elección estabiliza la métrica: con solo Exact, la consulta 33633 se queda con un único ítem relevante (su Recall@10 solo puede valer 0 o 1) y la 13357 pasa de 31 a 4 relevantes — ese ruido pesaría demasiado al comparar configuraciones distintas en los notebooks siguientes.

### D02 — Política de nulos en el texto codificado

**Decisión: omitir la sección.** Si un campo (p.ej. `color`) está vacío, no se añade su sección al texto compuesto — en vez de dejar `"Color: ."` o insertar la palabra `"desconocido"`.

Evidencia (`null_field_rates` + `text_field_label_summary` sobre la muestra):

| campo | % nulos (estructurado) | % de `text` que ya menciona su etiqueta | de esas, con el campo estructurado vacío |
|---|---:|---:|---:|
| `brand` | 2,93 % (44/1.500) | 97,07 % dice "Marca:" | 0 |
| `color` | 36,60 % (549/1.500) | 66,73 % dice "Color:" | 50 |

El propio `text` de origen ya sigue esta política para `brand`: cuando `brand` está vacío, el texto **nunca** dice "Marca: ." (0 casos) — la sección simplemente no aparece. Confirma que "omitir" es coherente con el contrato de datos existente, no una convención nueva. Para `color` hay 50 filas donde el texto sí menciona "Color:" con el campo estructurado vacío — indicio de que `color` no es un campo limpio (ver también valores como `"Negro Black 333824 067"` en la distribución de valores), a vigilar si más adelante se decide enriquecer el texto con `color`.

**Por qué elegí esto:** decidí omitir la sección en vez de dejarla en blanco o escribir "desconocido" porque lo que me importa es el significado semántico del texto, no su longitud. Insertar la palabra "desconocido" en cientos de productos (36,60 % de `color` en la muestra) crearía una señal compartida artificial: el modelo podría acercar esos productos entre sí porque comparten literalmente esa palabra, no porque se parezcan como productos — es un riesgo de contaminación semántica del embedding. Soy consciente de que, al omitir la sección, pierdo la distinción entre "este producto no tiene color" y "no hacía falta mencionar el color"; con un 36,60 % de `color` vacío es un volumen grande, pero prefiero asumir ese coste antes que arriesgarme a meter ruido en el embedding.

**Puesta a prueba, y mantenida.** Tomé esta decisión razonando, sin medir, así que al comparar representaciones añadí una plantilla de **control** con la política invertida: la misma receta pero rellenando los campos vacíos.

Sobre la muestra de 1.500 la recuperación salía mejor con el relleno, pero al mirar dónde se producía esa mejora **no aplicaba sobre los vectores que realmente se rellenaban**: las consultas cuyos productos relevantes tenían el campo vacío no eran las que mejoraban, y la que los tenía *todos* vacíos —la exposición máxima posible— apenas se movía. Si la ventaja no viene de donde el relleno actúa, no es una ventaja de la recuperación. Sobre el catálogo completo, además, es la plantilla que peor escala de las siete.

Mantengo la política de omitir, ahora con una medición detrás en lugar de solo un razonamiento.

> 📓 `03_representacion.ipynb` → *El control de la política de nulos*, *¿Es señal o es perturbación?*

### D03 — Normalización de `brand` para el filtro nativo

**Decisión: `brand` se guarda y se embebe tal cual (raw), sin normalizar. La normalización (`strip` + `casefold` + sin acentos) se aplica solo en el momento de la búsqueda/filtro, no al dato almacenado.**

Sobre la muestra (1.500 filas, 1.191 marcas distintas) **ninguno de los tres modos produce colisiones** — la muestra no aporta evidencia a favor ni en contra de normalizar (`n_marcas_crudas` de `brand_normalization_collisions` da 0 grupos en los tres modos). Repetido sobre el catálogo completo (15.000 filas, 9.054 marcas distintas) para confirmarlo con certeza:

| modo | grupos con colisión | marcas fusionadas |
|---|---:|---:|
| raw | 0 | 0 |
| casefold | 66 | 133 |
| unaccent | 77 | 156 |

`casefold` funde variantes de mayúsculas/espacios de la misma marca (`Buffalo`/`BUFFALO`, `Bugatti`/`bugatti`, `Fortnite`/`FORTNITE`, `Rc Ocio`/`Rc ocio`/`RC OCIO`). `unaccent` añade 11 grupos más que solo se detectan al quitar acentos: `L'Oréal Paris`/`L'Oreal Paris`, `Nescafé`/`Nescafe`, `Nestlé`/`Nestle`, `Química Alemana`/`Quimica Alemana`, `Reig Martí`/`Reig Marti`, `Mühle`/`Muhle`. Revisando los grupos más grandes, ninguno parece una fusión indebida (marcas realmente distintas mezcladas) — todos son variantes tipográficas de la misma marca. Esto confirma la decisión con evidencia real sobre el catálogo completo, no solo sobre la muestra.

**Por qué elegí esto:** al principio pensé que normalizar `brand` podía afectar al embedding (que "Apple" se acercara semánticamente a "apple" la fruta), pero eso no aplica: `brand` se guarda y se embebe tal cual, sin normalizar, así que esta decisión no toca el embedding en absoluto. La normalización solo vive en el filtro de búsqueda, y su único objetivo es que un cliente no se quede sin resultados por escribir "nike" en vez de "NIKE", o por una variante de acentos. Descarté tocar la puntuación (apóstrofes incluidos) precisamente para no fusionar marcas que en realidad son distintas.

### D04 — Alcance de evaluación (nDCG/Recall/MRR)

**Decisión: un producto recuperado que no está entre los juicios de la consulta cuenta como Irrelevant (0), sin recolocar posiciones.** No es lo mismo que "muestra vs. catálogo completo" (eso ya está resuelto: desarrollo sobre la muestra, ejecución final sobre el catálogo completo) — D04 decide qué grado de relevancia recibe, en la puntuación, un producto que el motor devuelve pero que ningún humano llegó a juzgar.

Antes de decidir comprobé (`qrels_coverage_in_catalog`) que los productos juzgados por las 8 consultas de desarrollo sí existen en `catalogo_muestra.csv` (cobertura 100 %), así que Recall@10/nDCG calculados sobre la muestra son fiables — el hueco de evaluación no viene de ahí, sino de que solo 248 de los 15.000 productos tienen juicio humano (pools de 16 a 40 candidatos por consulta).

**Por qué elegí esto:** decidí marcar los productos no juzgados como Irrelevant (0) porque no quiero asumir, sin evidencia, que son buenos resultados — eso sería una suposición injustificada a favor del sistema. Es la opción conservadora: si nadie lo ha verificado, no le doy crédito por defecto. Además, es la práctica estándar en evaluación de sistemas de recuperación con pools parciales, y evita que la métrica se pueda inflar artificialmente con contenido nunca revisado.

## Baseline léxico

Decisiones tomadas en NB01, medidas primero sobre `catalogo_muestra.csv` (1.500 filas) y confirmadas después sobre el catálogo completo (15.000 registros) — las cifras de esta sección son las del catálogo completo. Detalle, código y celdas en `notebooks/01_baseline.ipynb`; métricas e IDs recuperados por consulta en `artifacts/baseline_lexico.json`.

### D05 — Qué baselines léxicos se implementan

**Decisión: TF-IDF + BM25.** Ambos comparten tokenizador (`aurum.datos.tokenize`), así que ven exactamente los mismos términos: entre ellos varía únicamente la fórmula de puntuación.

Dispersión de la longitud de documento (en tokens del propio tokenizador) — es el eje en el que ambos métodos se diferencian:

| campo | p50 | p95 | máx | cv | ratio p95/p50 |
|---|---:|---:|---:|---:|---:|
| `title` | 18 | 32 | 80 | 0,465 | 1,78 |
| `text` | 150 | 493 | 603 | 0,828 | 3,29 |

Techo del emparejamiento literal — % de productos relevantes (E+S) que contienen *todos* los términos de la consulta en el título:

| consultas de desarrollo | `pct_todos_en_title` |
|---|---:|
| 13357 · 18868 · 28703 · 31224 · 33633 · 38249 · 43240 | 0,0 % |
| 61533 (`lentejas sin gluten`) | 36,7 % |

Métricas sobre las 8 consultas de desarrollo (relevante = E+S según D01; no juzgado = 0 según D04):

| baseline | nDCG@10 | Recall@10 | MRR@10 | Precision@10 |
|---|---:|---:|---:|---:|
| TF-IDF | 0,4129 | 0,1510 | 0,750 | 0,4125 |
| BM25 | **0,5088** | **0,1841** | 0,750 | **0,5500** |

**Por qué elegí esto:**

- **TF-IDF** es la referencia léxica mínima y ya estaba disponible, sin coste añadido.
- **BM25** entra porque es el estándar en recuperación de información y porque es el único que trata la longitud del documento de forma distinta, saturando la repetición de términos y comparando cada registro del catálogo con la longitud media del corpus. Según las pruebas realizadas, el campo que indexo tiene una dispersión de longitud alta, así que esa diferencia no es teórica aquí: BM25 obtiene una ventaja clara en nDCG@10 sobre el catálogo completo.
- **LSA, descartado:** comprime la misma matriz de TF-IDF, así que no aporta una señal nueva sobre el corpus, y su papel de puente hacia lo denso lo cubre ya el modelo de embeddings de los notebooks siguientes.
- **Coincidencia exacta de título, descartada:** según las pruebas realizadas, prácticamente ningún producto relevante contiene todos los términos de la consulta en su título, así que devolvería resultados vacíos en casi todas las consultas.

### D05.b — Normalización de acentos en el tokenizador

**Decisión: se normalizan los acentos, con la misma normalización al indexar y al consultar.**

Frecuencia documental de los términos afectados — a cuántos registros del catálogo apunta cada palabra de la consulta, antes y después de normalizar:

| término (consulta) | sin normalizar | normalizando |
|---|---:|---:|
| `tactil` (28703) | 10 | **326** |
| `habitacion` (38249) | 19 | **434** |
| `tacon` (18868) | 16 | **140** |
| `portátil` (28703) | 919 | 946 |
| `cámaras` (31224) | 180 | 187 |

Las dos últimas filas son el contraste: las palabras que el usuario ya escribe con su tilde apenas se mueven. El efecto se concentra en las que llegan sin ella.

**Por qué elegí esto:** elegí normalizar los acentos en el tokenizador, aplicando la misma normalización al indexar y al consultar. Varias de las consultas de desarrollo se escriben sin tilde mientras el catálogo sí las lleva, y según las pruebas realizadas eso deja a esos términos apuntando a un puñado de registros del catálogo —casi siempre las mal escritas—, que por su rareza acaban decidiendo el ranking. Normalizar fusiona ambas listas y hace que consulta y catálogo se escriban en el mismo alfabeto.

### D05.c — Campo indexado

**Decisión: se indexa `text`, no `title`.** `text` es además la plantilla A0 con la que arranca la comparación de representaciones en NB02, de modo que léxico y denso se evalúan sobre la misma superficie textual.

La evidencia es la tabla de dispersión de D05: `title` tiene p50 de 18 tokens y un ratio p95/p50 de 1,78 (campo corto y homogéneo), frente a los 150 tokens de mediana y el ratio de 3,29 de `text`.

**Por qué elegí esto:** indexo `text` y no `title` porque `text` es la superficie completa del registro del catálogo y es la misma que usará el sistema denso como plantilla de partida, de modo que la comparación entre léxico y denso varía un único factor. Según las pruebas realizadas, `title` es además un campo corto y muy homogéneo.

## Modelo de embeddings

Elección del modelo que convierte cada producto en un vector. Compiten `gemini-embedding-2`, `jinaai/jina-embeddings-v3` e `ibm-granite/granite-embedding-311m-multilingual-r2` sobre el mismo texto, con un barrido de dimensiones. Detalle, código y celdas en `notebooks/02_modelo.ipynb`.

### Longitud en tokens y necesidad de chunking

Medida con el tokenizador real de cada modelo — no en caracteres ni en palabras, porque el límite de contexto se cuenta en piezas del vocabulario de cada modelo y esa cuenta no es la misma para todos.

Catálogo completo (15.000 registros), modelos locales:

| modelo | ventana | tokens p50 | p95 | máx | registros del catálogo que superan la ventana |
|---|---:|---:|---:|---:|---:|
| `jina-embeddings-v3` | 8.192 | 250 | 820 | 1.308 | **0 (0,00 %)** |
| `granite-311m-r2` | 32.768 | 242 | 789 | 1.972 | **0 (0,00 %)** |

Y cuántos registros del catálogo se truncarían con cada tamaño de ventana (tokenizador de jina):

| ventana | registros del catálogo que la superan |
|---:|---|
| 128 | 10.006 (66,71 %) |
| 512 | 4.167 (27,78 %) |
| 1.024 | 25 (0,17 %) |
| 2.048 | 0 (0,00 %) |

**Consecuencia:** con los tres modelos elegidos no se trunca ningún registro del catálogo, así que el chunking (multi-vector por producto) no resuelve ningún problema real. El punto de la base vectorial se mantiene en `record_id`, en relación 1:1 con el producto.

### ⚠️ Restricción de la medición de `gemini-embedding-2`

Gemini no publica su tokenizador, así que su longitud en tokens se mide llamando a `count_tokens` de la API. Eso impone dos límites que conviene conocer al leer las cifras:

1. **Es una petición de red por registro del catálogo**, de modo que se mide un subconjunto de **150 registros** (las 50 más largas en caracteres, más 100 aleatorias con semilla fija) en lugar del catálogo completo. Las 50 más largas son las únicas que podrían acercarse a la ventana, y el máximo observado (1.887 tokens) queda a más de 4× de los 8.192 disponibles: la conclusión no depende de esa precisión.
2. **Requiere `GEMINI_API_KEY`.** Sin clave, esa celda del notebook se omite con un aviso y el resto se ejecuta igual. `count_tokens` no consume cuota de facturación, pero sí cuenta para el límite de peticiones por minuto.

Sobre ese mismo subconjunto de 150 registros, los tres modelos dan: `gemini-embedding-2` máx. 1.887 tokens, `jina-v3` máx. 1.308 y `granite-311m-r2` máx. 1.875 — ninguno supera su ventana.

### D09 · Modelos comparados

**Decisión: compitieron `gemini-embedding-2`, `jinaai/jina-embeddings-v3` e `ibm-granite/granite-embedding-311m-multilingual-r2`. Gana `gemini-embedding-2`** —sin instrucción de tarea en el prompt, a 768 dimensiones— **por nDCG@10 sobre las 8 consultas de desarrollo**, la métrica primaria que fijé de antemano en el criterio de desempate.

§3.1 exige apoyar la elección en dos patas: las restricciones del caso y los resultados de desarrollo.

**Por qué compitieron estos tres.** Los filtré antes de medir, por cuatro restricciones del caso: son **multilingües** (el catálogo está en español y un modelo entrenado en inglés fragmenta el vocabulario); **caben en 4 núcleos sin GPU**, aunque los locales lo paguen en horas de CPU —incluí uno por API a propósito, para tener los dos regímenes de coste en la comparación en vez de dar el trade-off por supuesto—; están entrenados con **MRL**, que permite recortar dimensión sin recodificar; y **declaran contratos de entrada distintos**, que es lo que da contenido al eje de prefijos. La ventana de contexto no llegó a discriminar: los tres cubren el catálogo con holgura, y esa misma medición descartó el chunking.

**Por qué gana Gemini.** Es el único de los tres que **supera al baseline léxico**. Haber puesto BM25 como referencia —y no solo TF-IDF, más fácil de batir— es lo que dejó ver que la infraestructura vectorial no compensa por sí sola una representación deficiente. `jina-v3` cae por partida doble: es el peor de los tres y el único con licencia `cc-by-nc-4.0`, incompatible con un marketplace comercial.

Acepto a cambio una **dependencia de API**: cuota, condiciones que el proveedor puede cambiar y datos del catálogo saliendo de la máquina. `granite-311m-r2` era la alternativa limpia —Apache-2.0, pesos locales— pero no alcanza al léxico.

**El contrato depende de la dimensión:** retirar la instrucción ayuda con el vector grande y estorba al recortarlo mucho. La decisión vale para las 768 elegidas; habría que **revisarla, no heredarla**, si la memoria del índice obligara a bajar más.

> 📓 Medido en `02_modelo.ipynb` → *Longitud en tokens*, *Barrido de dimensión (MRL)*, *El contrato de entrada*, *El contrato no aporta lo mismo en todas las dimensiones* y *Contra el baseline léxico*


### D09b · Criterio de desempate

**Decisión: dentro de una tolerancia de 0,02 de nDCG@10, gana la configuración de menor dimensión.** Con ella, la elegida es `gemini-embedding-2` a **768 dimensiones**, cuatro veces más ligera que su dimensión nativa.

La regla se fijó **antes de codificar nada** y se aplica como función determinista cubierta por tests, no a ojo. Eso es lo que hace verificable la afirmación: el ganador sale de la tabla, no de una lectura interesada de ella.

1. `B` = mejor nDCG@10 de toda la tabla.
2. **Admisibles**: las que quedan a menos de 0,02 de `B`.
3. Entre las admisibles gana la de **menor dimensión**; a igualdad, mayor nDCG; después, menor tiempo de codificación.

**Por qué elegí esto:** con solo 8 consultas de desarrollo, una diferencia de nDCG por debajo de 0,02 no distingue dos sistemas — es ruido de muestreo, y dejar que decida el máximo sería fingir una precisión que no tengo. Prefiero declarar de antemano que, dentro de ese margen, manda el coste. La memoria del índice es una restricción real en esta máquina y se paga en cada consulta de por vida; unas milésimas de nDCG que no puedo defender estadísticamente, no.

El barrido lo confirmó: la calidad se mantiene plana mientras se recorta el vector y **solo se desploma pasado cierto punto**, así que la regla compró una reducción sustancial de memoria sin ceder calidad medible. El desempate por tiempo de codificación nunca llegó a usarse — la dimensión resolvió el orden.

> 📓 Medido en `02_modelo.ipynb` → *Barrido de dimensión (MRL)*, *Curva calidad ↔ dimensión* y *Aplicar el criterio de desempate*

### D10 · Normalización y métrica — qué significa el score

**Decisión: normalización L2 explícita, aplicada siempre al truncar, y `cosine` como métrica del motor.**

§3.1 pide *"explicar la relación entre la métrica configurada, la normalización y el significado del score"*, y §3.2 *"conservar la semántica del score nativo"*.

Con vectores de norma 1, las tres métricas candidatas —coseno, producto escalar y distancia euclídea— **producen exactamente el mismo ranking**: el producto escalar de dos unitarios *es* el coseno, y la distancia euclídea es una función decreciente de él. Sin normalizar, el producto escalar deja de medir parecido y empieza a premiar a los vectores largos; el coseno es el único inmune, porque normaliza por dentro.

**Por qué elegí esto:** si con vectores unitarios da igual cuál elija, entonces la elección no es de rendimiento sino de **seguridad**. Escojo la métrica que sigue significando lo mismo si la normalización fallara en algún punto del recorrido — al truncar, al reindexar o al cambiar de motor. Es defensa en profundidad: no gano nada hoy, pero no me juego el significado del score ante un fallo que sería invisible en las métricas.

La comprobación reveló algo que no esperaba y que justifica no delegar la normalización en el modelo: **de los tres candidatos, solo uno entrega vectores sin normalizar**, y es precisamente el único con el que este experimento demuestra algo. Si los tres normalizaran en origen, la verificación pasaría sin detectar nada. Basta una desviación de milésimas en la norma para reordenar resultados.

> 📓 Medido en `02_modelo.ipynb` → *Salud de los vectores* y *Normalización y métrica*

### R02 · Qué está confirmado y qué no

La elección del modelo se sostiene en tres comprobaciones. Dos están cerradas; la tercera, a medias — y prefiero decirlo que dejarlo implícito.

**✅ El contrato de entrada de `granite-311m-r2`.** Lo excluí del eje de prefijos porque su configuración declara los dos prompts vacíos, pero eso solo probaba que la librería no le añade nada, no que IBM lo entrenara sin instrucción — y §3.1 avisa de que no basta con citar la documentación del modelo. Revisé la model card completa, con todos sus backends y sus ejemplos de uso: **no documenta ninguna instrucción ni prefijo en ningún sitio**, y en su ejemplo de recuperación consultas y documentos se codifican por igual, a diferencia de las familias que sí llevan instrucciones. El contrato real es texto plano simétrico. `granite` no compitió en desventaja.

**✅ La fiabilidad de los tiempos.** Sospechaba de ellos como criterio de desempate (ver [Tiempos aproximados](#tiempos-aproximados)), pero nunca llegaron a usarse: la dimensión resolvió el orden antes.

**🟡 El comportamiento a escala real.** Todo el barrido corre sobre la muestra de 1.500 productos, y el enunciado evalúa sobre los 15.000. Recodifiqué al ganador sobre el catálogo completo y lo medí contra el baseline léxico: **el orden se mantiene** —el denso sigue por delante— aunque **la ventaja se estrecha**, y la mejora en cobertura es donde más se nota. La dimensión elegida se confirma: a escala real no solo ahorra memoria, rinde igual o mejor que la nativa.

Lo que **no** he confirmado a escala completa es el **orden entre los tres modelos densos**: recodificar los dos locales sobre 15.000 productos son decenas de horas de CPU. Con la distancia que les saca el ganador sobre la muestra, el riesgo de inversión es bajo — pero bajo no es cero, y lo dejo declarado como límite del experimento en lugar de darlo por supuesto.

> 📓 Medido en `02_modelo.ipynb` → *El ganador sobre el catálogo completo*

## Representación del texto

Con el modelo, la dimensión y la métrica ya congelados en el apartado anterior, aquí lo único que varía es **qué texto de cada producto se convierte en vector**. Detalle, código y celdas en `notebooks/03_representacion.ipynb`.

### D06 · Qué plantillas se compararon

**Decisión: siete recetas, desde el texto completo del producto hasta solo el título.** Entre medias, tres composiciones a partir de los campos estructurados —con etiquetas, sin ellas y sin `color`— y un recorte del texto por la mediana de longitud del catálogo.

**Por qué elegí esto:** quería cubrir el rango entero en vez de probar dos variantes parecidas. Los extremos acotan el problema —si el texto largo fuera puro relleno ganaría el título solo; si cada campo aportara, ganaría la composición más completa— y las intermedias dicen dónde está el punto de corte.

Añadí una séptima como **control, no como candidata**: la misma receta que una de las composiciones pero rellenando los campos vacíos, para poner a prueba la política de nulos que había decidido razonando y sin medir. Su papel quedó declarado antes de codificar nada, y por eso no compite por ser la elegida aunque puntúe.

El punto de corte del recorte **no lo elegí yo**: sale de la mediana de longitud del propio catálogo. Un número escrito a mano habría sido una decisión disfrazada de detalle de implementación, imposible de defender frente a cualquier otro valor.

> 📓 `03_representacion.ipynb` → *Las siete plantillas*, *De dónde sale el recorte*

### D07 · Sin chunking

**Decisión: un vector por producto, sin partir el texto en trozos.**

**Por qué elegí esto:** el chunking resuelve un problema que aquí no existe. La medición de longitudes en tokens del apartado anterior mostró que ningún registro del catálogo se acerca a la ventana de contexto de los modelos candidatos, con un margen de más de cuatro veces. Partir en trozos habría multiplicado los vectores, obligado a deduplicar por producto antes de cortar el top-10 y complicado la idempotencia con fragmentos huérfanos al actualizar — todo a cambio de nada.

Se descarta **por medición, no por falta de tiempo**, y eso deja el esquema simple: el punto de la base vectorial es `record_id`, en relación 1:1 con el producto.

> 📓 `02_modelo.ipynb` → *Longitud en tokens*, *Cuántos registros se truncarían*

### R01 · Plantilla elegida

**Decisión: el texto del producto recortado por la mediana de longitud del catálogo**, algo más de la mitad del original. Gana por **nDCG@10 sobre las 8 consultas de desarrollo** y es la única que entra en la tolerancia de 0,01 que fijé antes de ver ningún resultado.

**Por qué elegí esto:** decidí sobre el catálogo completo y no sobre la muestra, y esa distinción no es un detalle — **la muestra me habría llevado a otra plantilla**. Con 1.500 productos ganaba la más corta, solo el título; con los 15.000 reales el orden se da la vuelta y el recorte por la mediana pasa a primera. Eso solo se ve midiendo a escala real, y equivocarse ahí obliga a recodificar el catálogo entero y a reconstruir el índice.

Me quedo con el recorte porque **con la mitad del contenido conservo la información que hace falta para recuperar**: el resto era, en buena parte, prosa comercial que diluía el vector. Y el punto de corte no lo elegí yo — sale de la mediana de longitud del propio catálogo, así que es una regla que se recalcula sola si el catálogo cambia, no un número puesto a dedo que habría que defender frente a cualquier otro.

Asumo un **riesgo conocido**: recortar por longitud da por hecho que lo importante va al principio de la ficha. Si en algún producto la descripción deja para el final los detalles que lo distinguen, ese recorte los pierde. No lo he medido, y queda declarado como límite.

**Por qué el orden cambia al escalar:** con diez veces más candidatos compitiendo por las mismas diez posiciones, una representación de un centenar de caracteres se queda sin con qué separar vecinos próximos, mientras que una larga conserva los detalles que desempatan. Las dos plantillas más largas son las que menos se degradan; las cortas, las que más.

> ⚠️ La segunda clasificada queda fuera de la banda por poco más de una milésima. Son dos plantillas prácticamente empatadas y la regla las separa por un margen muy fino: conviene tenerlo presente si más adelante cambia algo del recorrido.

> 📓 `03_representacion.ipynb` → *El barrido*, *El barrido sobre el catálogo completo*


## Configuración

TODO

## Recorrido de notebooks

TODO

## Comandos

TODO

## Tests

TODO

## Resultados y artefactos

TODO

## Métricas y evaluación

TODO

## Visualizaciones

Las figuras viven en `src/aurum/graficas.py`, con sus tests. Heredan la paleta y el layout de los notebooks de sesión del máster —copiados, no importados, para que el entregable no dependa de material de clase— y añaden una sola dependencia, `plotly`.

Aplicadas hasta ahora en `02_modelo.ipynb`: la **curva de calidad frente a dimensión**, con la banda de tolerancia del criterio de desempate sombreada para que la regla se vea antes de aplicarla; la comparación del **denso frente al baseline léxico** sobre una escala común; y la **diferencia que aporta el contrato de entrada** según la dimensión, con una banda alrededor del cero que marca dónde la diferencia deja de ser distinguible.

Quedan pendientes cuatro figuras más para notebooks posteriores —latencia de consulta, proyección 2D del espacio de embeddings, mapa de calor de similitud entre duplicados y detalle de un ranking concreto—. No están porque **les falta el dato que representan**, no porque no encajen.

## Tiempos aproximados

Medidos sobre la máquina de desarrollo (4 núcleos, sin GPU) codificando los 1.500 productos de la muestra:

| Modelo | Tiempo |
|---|---:|
| `gemini-embedding-2` | menos de un minuto (API) |
| `jina-v3` | ~1 h 23 min |
| `granite-311m-r2` | ~4 h 51 min |

Codificar el catálogo completo multiplica esos tiempos por diez. La segunda ejecución de cualquiera de ellos es instantánea: los vectores se sirven de la caché en disco, cuya clave incluye el hash del texto de origen, de modo que cambiar la plantilla la invalida sola.

Esa diferencia de tres órdenes de magnitud entre la API y los modelos locales es lo que hizo viable barrer siete plantillas de texto en el notebook siguiente. Con un modelo local, ese barrido habría costado días.

> ⚠️ **Los tiempos de los dos modelos locales no son comparables entre sí.** El más pequeño tardó bastante más que el grande, lo cual está invertido respecto a lo esperable y apunta a contención de CPU o *throttling* durante la medición, no a una propiedad del modelo. Da igual para las decisiones tomadas —el desempate se resolvió por dimensión y el tiempo nunca llegó a usarse—, pero no me fiaría de esas cifras para comparar coste entre modelos locales sin remedirlas en igualdad de condiciones.

## Solución de problemas frecuentes

**`jina-v3`: `AttributeError: 'XLMRobertaLoRA' object has no attribute 'all_tied_weights_keys'`.** Su código remoto (`trust_remote_code=True`) está escrito para transformers 4.x y el entorno usa la 5.x. `SentenceTransformerEncoder` aplica el shim de compatibilidad automáticamente (`_enable_transformers_v4_remote_code` en `src/aurum/embeddings.py`); si ves este error, el kernel está usando otra ruta de carga.

## Seguridad y limpieza de recursos

TODO

## Licencia y procedencia de los datos

TODO
