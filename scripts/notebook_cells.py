"""Contenido fuente de los notebooks, en Python plano y versionable.

`build_notebook.py` convierte estas listas de celdas en los .ipynb de
`notebooks/`. Se edita aquí; el JSON del notebook nunca se toca a mano.
"""

NB00_DATOS = [
    ("markdown", "# NB00 · Datos: contrato, perfilado y decisiones de negocio"),
    (
        "markdown",
        "Evidencia real para las decisiones **D01-D04**. La validación "
        "de contrato completa y la longitud en tokens con el tokenizer del modelo elegido "
        "quedan para cuando ese modelo esté decidido (NB03).\n"
        "\n"
        "---\n"
        "\n"
        "### 📐 Convención de corpus — qué se mide sobre qué\n"
        "\n"
        "Toda cabecera de sección lleva marcado su corpus, y ninguna celda mezcla los dos:\n"
        "\n"
        "| Marca | Corpus | Fichero | Para qué |\n"
        "|---|---|---|---|\n"
        "| 🔬 **MUESTRA** | 1.500 registros | `catalogo_muestra.csv` | Desarrollo y calibración (condición 3 del plan) |\n"
        "| 📚 **COMPLETO** | 15.000 registros | `catalogo_productos.csv` | Confirmación de las decisiones antes de fijarlas |\n"
        "| ⚪ **SIN CATÁLOGO** | — | juicios y consultas | Evidencia que no depende del catálogo |\n"
        "\n"
        "⚠️ Importa para el README: las cifras de una sección 🔬 **no** son las del "
        "catálogo completo. Por ejemplo, `brand` tiene 2,93 % de nulos en la muestra y "
        "4,4 % en los 15.000 registros.",
    ),
    (
        "code",
        'import sys\n'
        'from pathlib import Path\n'
        '\n'
        'sys.path.insert(0, str(Path("..") / "src"))\n'
        '\n'
        'import pandas as pd\n'
        '\n'
        'from aurum.datos import (\n'
        '    brand_normalization_collisions,\n'
        '    esci_label_counts_per_query,\n'
        '    null_field_rates,\n'
        '    qrels_coverage_in_catalog,\n'
        '    qrels_pool_sizes,\n'
        '    text_field_label_summary,\n'
        '    value_frequency,\n'
        ')\n'
        '\n'
        'DATA = Path("..") / "data"\n'
        '\n'
        '# Los dos corpus, cargados con nombres que no se pueden confundir.\n'
        'muestra = pd.read_csv(DATA / "catalogo_muestra.csv")\n'
        'completo = pd.read_csv(DATA / "catalogo_productos.csv")\n'
        '\n'
        'consultas = pd.read_csv(DATA / "consultas_desarrollo.csv")\n'
        'relevancias = pd.read_csv(DATA / "relevancias_desarrollo.csv")\n'
        '\n'
        'print(f"🔬 MUESTRA : {len(muestra):>6} registros, {muestra[\'record_id\'].nunique():>6} record_id únicos")\n'
        'print(f"📚 COMPLETO: {len(completo):>6} registros, {completo[\'record_id\'].nunique():>6} record_id únicos")\n'
        'print(f"consultas_desarrollo: {len(consultas)} consultas · "\n'
        '      f"relevancias_desarrollo: {len(relevancias)} juicios")\n',
    ),
    ("markdown", "## 🔬 Distribución de valores — `brand` y `color` (top 15 por frecuencia)"),
    ("code", 'value_frequency(muestra, "brand").head(15)\n'),
    ("code", 'value_frequency(muestra, "color").head(15)\n'),
    (
        "markdown",
        "## D01 · ⚪ ¿Qué cuenta como relevante para Recall@10 / MRR@10? (E vs E+S)\n"
        "\n"
        "Se calcula solo sobre `relevancias_desarrollo.csv`: no depende de qué catálogo "
        "se use, así que la cifra es idéntica en muestra y en completo.",
    ),
    (
        "code",
        'esci_counts = esci_label_counts_per_query(relevancias).merge(\n'
        '    consultas[["query_id", "query_text"]], on="query_id"\n'
        ')\n'
        'esci_counts[["query_id", "query_text", "E", "S", "C", "I", "relevantes_solo_E", "relevantes_E_mas_S"]]\n',
    ),
    ("markdown", "## D02 · 🔬 Nulos en los campos que pueden entrar en el texto codificado"),
    ("code", 'null_field_rates(muestra, ["title", "brand", "color", "text"])\n'),
    (
        "markdown",
        "### 🔬 ¿El `text` ya trae las etiquetas Marca/Color (ES o EN)? — evidencia por regex",
    ),
    (
        "code",
        'text_field_label_summary(muestra)\n',
    ),
    (
        "markdown",
        "### D02 (confirmación) · 📚 Los mismos nulos sobre el catálogo completo\n"
        "\n"
        "La política de nulos se aplicará al codificar los 15.000 registros, no solo a la "
        "muestra: conviene saber si las proporciones se sostienen a escala real.",
    ),
    ("code", 'null_field_rates(completo, ["title", "brand", "color", "text"])\n'),
    ("code", 'text_field_label_summary(completo)\n'),
    ("markdown", "## D03 · 🔬 Colisiones al normalizar `brand`"),
    (
        "code",
        'n_distinct_raw = muestra["brand"].dropna().nunique()\n'
        'resumen = []\n'
        'for mode in ["raw", "casefold", "unaccent"]:\n'
        '    colisiones = brand_normalization_collisions(muestra, mode)\n'
        '    resumen.append({\n'
        '        "modo": mode,\n'
        '        "marcas_distintas_crudas": n_distinct_raw,\n'
        '        "grupos_con_colision": len(colisiones),\n'
        '        "marcas_fusionadas": int(colisiones["n_marcas_crudas"].sum()) if len(colisiones) else 0,\n'
        '    })\n'
        'pd.DataFrame(resumen)\n',
    ),
    (
        "code",
        'brand_normalization_collisions(muestra, "casefold").sort_values("n_marcas_crudas", ascending=False).head(20)\n',
    ),
    (
        "markdown",
        "### D03 (confirmación) · 📚 El mismo análisis sobre el catálogo completo\n"
        "\n"
        "Sobre la muestra no salió ninguna colisión en ningún modo: no prueba nada, ni a favor "
        "ni en contra. Se repite aquí sobre los 15.000 registros para confirmar la decisión con "
        "evidencia real.",
    ),
    (
        "code",
        'print(f"📚 COMPLETO: {len(completo)} registros, "\n'
        '      f"{completo[\'brand\'].dropna().nunique()} marcas distintas")\n',
    ),
    (
        "code",
        'n_distinct_raw_completo = completo["brand"].dropna().nunique()\n'
        'resumen_completo = []\n'
        'for mode in ["raw", "casefold", "unaccent"]:\n'
        '    colisiones = brand_normalization_collisions(completo, mode)\n'
        '    resumen_completo.append({\n'
        '        "modo": mode,\n'
        '        "marcas_distintas_crudas": n_distinct_raw_completo,\n'
        '        "grupos_con_colision": len(colisiones),\n'
        '        "marcas_fusionadas": int(colisiones["n_marcas_crudas"].sum()) if len(colisiones) else 0,\n'
        '    })\n'
        'pd.DataFrame(resumen_completo)\n',
    ),
    (
        "code",
        'brand_normalization_collisions(completo, "casefold").sort_values("n_marcas_crudas", ascending=False).head(20)\n',
    ),
    (
        "markdown",
        "📚 Grupos que solo aparecen al quitar acentos (no detectados por `casefold`):",
    ),
    (
        "code",
        'cf_normalizadas = set(brand_normalization_collisions(completo, "casefold")["normalizada"])\n'
        'ua = brand_normalization_collisions(completo, "unaccent")\n'
        'ua[~ua["normalizada"].isin(cf_normalizadas)]\n',
    ),
    (
        "markdown",
        "## D04 · ⚪ Tamaño del pool juzgado por consulta, frente al catálogo completo\n"
        "\n"
        "El tamaño del pool sale de los juicios, no del catálogo. La columna "
        "`catalogo_completo` es el universo real de búsqueda de la ejecución final.",
    ),
    (
        "code",
        'pool = qrels_pool_sizes(relevancias).merge(consultas[["query_id", "query_text"]], on="query_id")\n'
        'pool["catalogo_completo"] = len(completo)\n'
        'pool[["query_id", "query_text", "pool_size", "catalogo_completo"]]\n',
    ),
    (
        "markdown",
        "### Precondición de D04 · 🔬 ¿los `product_id` juzgados existen en la muestra?\n"
        "\n"
        "Si no están, Recall@10/nDCG de desarrollo calculados buscando sobre la muestra no son "
        "fiables sin importar qué universo de puntuación elijamos en D04.",
    ),
    (
        "code",
        'qrels_coverage_in_catalog(relevancias, muestra).merge(\n'
        '    consultas[["query_id", "query_text"]], on="query_id"\n'
        ')[["query_id", "query_text", "n_juzgados", "n_presentes", "pct_presentes"]]\n',
    ),
    (
        "markdown",
        "### Precondición de D04 (confirmación) · 📚 los mismos juicios sobre el catálogo completo",
    ),
    (
        "code",
        'qrels_coverage_in_catalog(relevancias, completo).merge(\n'
        '    consultas[["query_id", "query_text"]], on="query_id"\n'
        ')[["query_id", "query_text", "n_juzgados", "n_presentes", "pct_presentes"]]\n',
    ),
]

NB01_BASELINE = [
    ("markdown", "# NB01 · Baseline léxico"),
    (
        "markdown",
        "**Evidencia para D05** (qué baselines léxicos se implementan). Aquí no se "
        "implementa ningún retriever todavía: solo se mide qué tienen los datos que "
        "separa unas opciones de otras. La implementación llega cuando D05 esté "
        "ratificada.\n"
        "\n"
        "---\n"
        "\n"
        "### 📐 Convención de corpus — qué se mide sobre qué\n"
        "\n"
        "Toda cabecera de sección lleva marcado su corpus, y ninguna celda mezcla los dos:\n"
        "\n"
        "| Marca | Corpus | Fichero | Para qué |\n"
        "|---|---|---|---|\n"
        "| 🔬 **MUESTRA** | 1.500 registros | `catalogo_muestra.csv` | Desarrollo y calibración (condición 3 del plan) |\n"
        "| 📚 **COMPLETO** | 15.000 registros | `catalogo_productos.csv` | Confirmación de las decisiones antes de fijarlas |\n"
        "\n"
        "La **Parte A** mide sobre la muestra y la **Parte B** repite exactamente las "
        "mismas pruebas sobre el catálogo completo.",
    ),
    (
        "code",
        'import sys\n'
        'from pathlib import Path\n'
        '\n'
        'sys.path.insert(0, str(Path("..") / "src"))\n'
        '\n'
        'import pandas as pd\n'
        '\n'
        'from aurum.datos import (\n'
        '    document_length_stats,\n'
        '    literal_match_ceiling,\n'
        '    query_term_coverage,\n'
        '    query_token_frequencies,\n'
        ')\n'
        '\n'
        'DATA = Path("..") / "data"\n'
        '\n'
        '# Los dos corpus, cargados con nombres que no se pueden confundir.\n'
        'muestra = pd.read_csv(DATA / "catalogo_muestra.csv")\n'
        'completo = pd.read_csv(DATA / "catalogo_productos.csv")\n'
        '\n'
        'consultas = pd.read_csv(DATA / "consultas_desarrollo.csv")\n'
        'relevancias = pd.read_csv(DATA / "relevancias_desarrollo.csv")\n'
        'consultas_eval = pd.read_csv(DATA / "consultas_evaluacion.csv")\n'
        '\n'
        'print(f"🔬 MUESTRA : {len(muestra):>6} registros")\n'
        'print(f"📚 COMPLETO: {len(completo):>6} registros")\n'
        'print(f"consultas_desarrollo: {len(consultas)} · consultas_evaluacion: {len(consultas_eval)}")\n',
    ),
    (
        "markdown",
        "---\n"
        "\n"
        "# Parte A · Pruebas sobre la 🔬 MUESTRA (1.500 registros)",
    ),
    (
        "markdown",
        "## A.1 · 🔬 Dispersión de la longitud de documento\n"
        "\n"
        "BM25 normaliza la longitud del documento de forma explícita (parámetro `b`); "
        "TF-IDF solo la absorbe en la norma L2 del vector. Cuanto mayor sea la dispersión "
        "(`cv`, `ratio_p95_p50`), más se separan ambos métodos y más sentido tiene "
        "implementar los dos en vez de uno.",
    ),
    ("code", 'document_length_stats(muestra, ["title", "text"])\n'),
    (
        "markdown",
        "## A.2 · 🔬 Cobertura léxica de las 8 consultas de desarrollo\n"
        "\n"
        "Un término que no está en el vocabulario del corpus no aporta señal ni a TF-IDF "
        "ni a BM25. Se mide con y sin `strip_accents` porque varias consultas vienen "
        "**sin tildes** (`habitacion`, `tacon`, `tactil`) mientras el catálogo sí las "
        "lleva: es una decisión de configuración del índice, no un detalle cosmético.\n"
        "\n"
        "`df_min` es la frecuencia documental del término **más raro** de la consulta: el "
        "que más peso IDF recibe y, por tanto, el que más manda en el ranking.",
    ),
    (
        "code",
        'cobertura_con_tildes = query_term_coverage(consultas, muestra)\n'
        'cobertura_sin_tildes = query_term_coverage(consultas, muestra, strip_accents=True)\n'
        '\n'
        'comparativa_muestra = cobertura_con_tildes[\n'
        '    ["query_id", "query_text", "n_tokens", "n_oov", "tokens_oov", "df_min", "df_max"]\n'
        '].merge(\n'
        '    cobertura_sin_tildes[["query_id", "n_oov", "tokens_oov", "df_min"]],\n'
        '    on="query_id",\n'
        '    suffixes=("_con_tildes", "_sin_tildes"),\n'
        ')\n'
        'comparativa_muestra\n',
    ),
    (
        "markdown",
        "### A.2b · 🔬 La misma cobertura, palabra por palabra\n"
        "\n"
        "La vista desagregada: a qué lista de documentos apunta **cada palabra** que "
        "escribe el usuario. `df = 0` es una palabra que no existe en el corpus; un `df` "
        "muy pequeño recibe mucho peso IDF — para bien (término preciso) o para mal "
        "(errata). Comparar las dos columnas muestra qué palabras cambian de lista al "
        "normalizar acentos.",
    ),
    (
        "code",
        'tokens_dev_muestra = query_token_frequencies(consultas, muestra)\n'
        'tokens_dev_muestra["cambia_al_normalizar"] = (\n'
        '    tokens_dev_muestra["df_con_tildes"] != tokens_dev_muestra["df_sin_tildes"]\n'
        ')\n'
        'tokens_dev_muestra.merge(consultas[["query_id", "query_text"]], on="query_id")[\n'
        '    ["query_id", "query_text", "posicion", "token", "df_con_tildes", "df_sin_tildes", "cambia_al_normalizar"]\n'
        ']\n',
    ),
    (
        "markdown",
        "## A.3 · 🔬 Cobertura de las 12 consultas ciegas (`direct` · `context` · `semantic`)\n"
        "\n"
        "Sin etiquetas no se puede calcular nDCG sobre ellas, pero la cobertura léxica sí "
        "se mide: es la evidencia directa del *vocabulary gap* que justifica el coste del "
        "sistema denso.",
    ),
    (
        "code",
        'cobertura_eval_muestra = query_term_coverage(\n'
        '    consultas_eval, muestra, id_col="evaluation_id", strip_accents=True\n'
        ')\n'
        'cobertura_eval_muestra["formulacion"] = (\n'
        '    cobertura_eval_muestra["evaluation_id"].str.split("-").str[2]\n'
        ')\n'
        'cobertura_eval_muestra[\n'
        '    ["evaluation_id", "formulacion", "query_text", "n_tokens", "n_oov", "tokens_oov", "df_min", "df_max"]\n'
        '].sort_values("evaluation_id")\n',
    ),
    (
        "code",
        'cobertura_eval_muestra.groupby("formulacion")[\n'
        '    ["n_tokens", "n_oov", "df_min", "df_max"]\n'
        '].mean().round(1)\n',
    ),
    (
        "markdown",
        "### A.3b · 🔬 Las palabras de las ciegas que no existen en el corpus\n"
        "\n"
        "Qué escribe el cliente que no está en ningún registro del catálogo. Es el "
        "*vocabulary gap* en su forma más literal.",
    ),
    (
        "code",
        'tokens_eval_muestra = query_token_frequencies(\n'
        '    consultas_eval, muestra, id_col="evaluation_id"\n'
        ')\n'
        'tokens_eval_muestra["formulacion"] = (\n'
        '    tokens_eval_muestra["evaluation_id"].str.split("-").str[2]\n'
        ')\n'
        'tokens_eval_muestra[tokens_eval_muestra["df_sin_tildes"] == 0][\n'
        '    ["evaluation_id", "formulacion", "posicion", "token"]\n'
        ']\n',
    ),
    (
        "markdown",
        "## A.4 · 🔬 Techo del emparejamiento literal\n"
        "\n"
        "% de productos **relevantes** (E+S, según D01) que contienen *todos* los términos "
        "de la consulta. Es el techo de la opción *«coincidencia exacta de título»* y la "
        "medida de cuánto queda fuera del alcance de una coincidencia literal.",
    ),
    (
        "code",
        'techo_muestra = literal_match_ceiling(consultas, relevancias, muestra, strip_accents=True)\n'
        'techo_muestra[\n'
        '    ["query_id", "query_text", "n_relevantes", "n_en_catalogo",\n'
        '     "n_todos_en_title", "pct_todos_en_title", "n_todos_en_text", "pct_todos_en_text"]\n'
        ']\n',
    ),
    (
        "markdown",
        "---\n"
        "\n"
        "# Parte B · Las mismas pruebas sobre el 📚 CATÁLOGO COMPLETO (15.000 registros)\n"
        "\n"
        "Mismo código, mismas consultas, único factor que cambia: el corpus. Sirve para "
        "comprobar si las conclusiones de la Parte A se sostienen a escala real o eran un "
        "artefacto del tamaño de la muestra — el mismo procedimiento que se siguió con "
        "D03 en NB00.\n"
        "\n"
        "⏱️ Estas celdas tardan más: recorren 15.000 registros en vez de 1.500.",
    ),
    ("markdown", "## B.1 · 📚 Dispersión de la longitud de documento"),
    ("code", 'document_length_stats(completo, ["title", "text"])\n'),
    ("markdown", "## B.2 · 📚 Cobertura léxica de las 8 consultas de desarrollo"),
    (
        "code",
        'cobertura_con_tildes_completo = query_term_coverage(consultas, completo)\n'
        'cobertura_sin_tildes_completo = query_term_coverage(consultas, completo, strip_accents=True)\n'
        '\n'
        'comparativa_completo = cobertura_con_tildes_completo[\n'
        '    ["query_id", "query_text", "n_tokens", "n_oov", "tokens_oov", "df_min", "df_max"]\n'
        '].merge(\n'
        '    cobertura_sin_tildes_completo[["query_id", "n_oov", "tokens_oov", "df_min"]],\n'
        '    on="query_id",\n'
        '    suffixes=("_con_tildes", "_sin_tildes"),\n'
        ')\n'
        'comparativa_completo\n',
    ),
    ("markdown", "### B.2b · 📚 La misma cobertura, palabra por palabra"),
    (
        "code",
        'tokens_dev_completo = query_token_frequencies(consultas, completo)\n'
        'tokens_dev_completo["cambia_al_normalizar"] = (\n'
        '    tokens_dev_completo["df_con_tildes"] != tokens_dev_completo["df_sin_tildes"]\n'
        ')\n'
        'tokens_dev_completo.merge(consultas[["query_id", "query_text"]], on="query_id")[\n'
        '    ["query_id", "query_text", "posicion", "token", "df_con_tildes", "df_sin_tildes", "cambia_al_normalizar"]\n'
        ']\n',
    ),
    ("markdown", "## B.3 · 📚 Cobertura de las 12 consultas ciegas"),
    (
        "code",
        'cobertura_eval_completo = query_term_coverage(\n'
        '    consultas_eval, completo, id_col="evaluation_id", strip_accents=True\n'
        ')\n'
        'cobertura_eval_completo["formulacion"] = (\n'
        '    cobertura_eval_completo["evaluation_id"].str.split("-").str[2]\n'
        ')\n'
        'cobertura_eval_completo[\n'
        '    ["evaluation_id", "formulacion", "query_text", "n_tokens", "n_oov", "tokens_oov", "df_min", "df_max"]\n'
        '].sort_values("evaluation_id")\n',
    ),
    (
        "code",
        'cobertura_eval_completo.groupby("formulacion")[\n'
        '    ["n_tokens", "n_oov", "df_min", "df_max"]\n'
        '].mean().round(1)\n',
    ),
    ("markdown", "### B.3b · 📚 Las palabras de las ciegas que no existen en el corpus"),
    (
        "code",
        'tokens_eval_completo = query_token_frequencies(\n'
        '    consultas_eval, completo, id_col="evaluation_id"\n'
        ')\n'
        'tokens_eval_completo["formulacion"] = (\n'
        '    tokens_eval_completo["evaluation_id"].str.split("-").str[2]\n'
        ')\n'
        'tokens_eval_completo[tokens_eval_completo["df_sin_tildes"] == 0][\n'
        '    ["evaluation_id", "formulacion", "posicion", "token"]\n'
        ']\n',
    ),
    ("markdown", "## B.4 · 📚 Techo del emparejamiento literal"),
    (
        "code",
        'techo_completo = literal_match_ceiling(consultas, relevancias, completo, strip_accents=True)\n'
        'techo_completo[\n'
        '    ["query_id", "query_text", "n_relevantes", "n_en_catalogo",\n'
        '     "n_todos_en_title", "pct_todos_en_title", "n_todos_en_text", "pct_todos_en_text"]\n'
        ']\n',
    ),
    (
        "markdown",
        "---\n"
        "\n"
        "# Parte C · El baseline léxico\n"
        "\n"
        "Decisiones ratificadas y escritas en `config/config.yaml`:\n"
        "\n"
        "| ID | Decisión | Evidencia que la sostiene |\n"
        "|---|---|---|\n"
        "| **D05** | **TF-IDF + BM25** | Son los dos únicos que se diferencian en algo medible aquí: la normalización de longitud, con 3,29× de dispersión en `text` (B.1). LSA heredaría el vocabulario y la normalización de TF-IDF; la coincidencia exacta daría 0 en 7 de 8 consultas (B.4) |\n"
        "| **D05.b** | **Normalizar acentos** al indexar | `tactil` pasa de 10 a 326 registros, `habitacion` de 19 a 434 (B.2b). Las palabras ya bien escritas apenas se mueven |\n"
        "| **D05.c** | Índice sobre **`text`** | Es la misma superficie textual que usará el denso en NB02 (plantilla A0), así que la comparación léxico↔denso varía un solo factor |\n"
        "\n"
        "Los dos retrievers comparten tokenizador (`aurum.datos.tokenize`), así que "
        "ven exactamente los mismos términos: lo único que cambia entre ellos es la "
        "fórmula de puntuación (Regla 2 de experimentación).\n"
        "\n"
        "**Contrato de relevancia** (`manifest.json`): `E=3, S=2, C=1, I=0`. **D01**: "
        "relevante para Recall@10/MRR@10 es `E+S`. **D04**: un producto recuperado sin "
        "juicio puntúa 0.",
    ),
    (
        "code",
        'from aurum.evaluacion import (\n'
        '    evaluate_rankings,\n'
        '    formulation_consistency,\n'
        '    qrels_from_judgements,\n'
        ')\n'
        'from aurum.lexico import Bm25Retriever, TfidfRetriever, rank_queries\n'
        '\n'
        'CAMPO = "text"   # D05.c\n'
        'STRIP_ACCENTS = True  # D05.b\n'
        'TOP_K = 10\n'
        '\n'
        'qrels = qrels_from_judgements(relevancias)\n'
        'print(f"qrels: {len(qrels)} consultas juzgadas, "\n'
        '      f"{sum(len(v) for v in qrels.values())} juicios")\n',
    ),
    (
        "markdown",
        "## C.1 · 🔬 Construcción de los dos índices sobre la muestra\n"
        "\n"
        "Se cronometra la construcción: el coste de indexación es parte de la "
        "comparación, no una nota al pie.",
    ),
    (
        "code",
        'import time\n'
        '\n'
        'def construir(clase, catalogo):\n'
        '    inicio = time.perf_counter()\n'
        '    retriever = clase(\n'
        '        catalogo[CAMPO].tolist(),\n'
        '        catalogo["product_id"].tolist(),\n'
        '        strip_accents=STRIP_ACCENTS,\n'
        '    )\n'
        '    return retriever, time.perf_counter() - inicio\n'
        '\n'
        'tfidf_muestra, t_tfidf = construir(TfidfRetriever, muestra)\n'
        'bm25_muestra, t_bm25 = construir(Bm25Retriever, muestra)\n'
        '\n'
        'pd.DataFrame([\n'
        '    {"baseline": "tfidf", "n_docs": len(muestra), "vocabulario": tfidf_muestra.vocabulary_size,\n'
        '     "construccion_s": round(t_tfidf, 2)},\n'
        '    {"baseline": "bm25", "n_docs": len(muestra), "vocabulario": bm25_muestra.vocabulary_size,\n'
        '     "construccion_s": round(t_bm25, 2)},\n'
        '])\n',
    ),
    (
        "markdown",
        "## C.2 · 🔬 Métricas sobre las 8 consultas de desarrollo\n"
        "\n"
        "La media macro primero, y **la tabla por consulta justo después**: la "
        "consulta 33633 tiene un solo `Exact`, así que su Recall@10 solo puede valer 0 "
        "o 1 y distorsiona la media (trampa nº 8 del plan).",
    ),
    (
        "code",
        'rankings_muestra = {\n'
        '    "tfidf": rank_queries(tfidf_muestra, consultas, k=TOP_K),\n'
        '    "bm25": rank_queries(bm25_muestra, consultas, k=TOP_K),\n'
        '}\n'
        'informes_muestra = {\n'
        '    nombre: evaluate_rankings(ranking, qrels, k=TOP_K)\n'
        '    for nombre, ranking in rankings_muestra.items()\n'
        '}\n'
        '\n'
        'pd.DataFrame([\n'
        '    {"baseline": nombre, **informe.summary}\n'
        '    for nombre, informe in informes_muestra.items()\n'
        '])\n',
    ),
    ("markdown", "### C.2b · 🔬 Tabla por consulta"),
    (
        "code",
        'por_consulta = pd.concat([\n'
        '    informe.per_query_frame().assign(baseline=nombre)\n'
        '    for nombre, informe in informes_muestra.items()\n'
        '])\n'
        'por_consulta["query_id"] = por_consulta["query_id"].astype(int)\n'
        'por_consulta.merge(consultas[["query_id", "query_text"]], on="query_id").pivot(\n'
        '    index=["query_id", "query_text"], columns="baseline",\n'
        '    values=["ndcg@10", "recall@10", "mrr@10"],\n'
        ')\n',
    ),
    (
        "markdown",
        "## C.3 · 📚 Las mismas métricas sobre el catálogo completo\n"
        "\n"
        "Aquí el buscador compite contra 15.000 candidatos en vez de 1.500: es el "
        "escenario real y las métricas deberían bajar. La diferencia entre C.2 y C.3 "
        "mide cuánto de la calidad venía de que el corpus fuera pequeño.",
    ),
    (
        "code",
        'tfidf_completo, t_tfidf_c = construir(TfidfRetriever, completo)\n'
        'bm25_completo, t_bm25_c = construir(Bm25Retriever, completo)\n'
        '\n'
        'rankings_completo = {\n'
        '    "tfidf": rank_queries(tfidf_completo, consultas, k=TOP_K),\n'
        '    "bm25": rank_queries(bm25_completo, consultas, k=TOP_K),\n'
        '}\n'
        'informes_completo = {\n'
        '    nombre: evaluate_rankings(ranking, qrels, k=TOP_K)\n'
        '    for nombre, ranking in rankings_completo.items()\n'
        '}\n'
        '\n'
        'pd.DataFrame([\n'
        '    {"baseline": nombre, "n_docs": len(completo), **informe.summary}\n'
        '    for nombre, informe in informes_completo.items()\n'
        '])\n',
    ),
    ("markdown", "### C.3b · 📚 Tabla por consulta"),
    (
        "code",
        'por_consulta_completo = pd.concat([\n'
        '    informe.per_query_frame().assign(baseline=nombre)\n'
        '    for nombre, informe in informes_completo.items()\n'
        '])\n'
        'por_consulta_completo["query_id"] = por_consulta_completo["query_id"].astype(int)\n'
        'por_consulta_completo.merge(consultas[["query_id", "query_text"]], on="query_id").pivot(\n'
        '    index=["query_id", "query_text"], columns="baseline",\n'
        '    values=["ndcg@10", "recall@10", "mrr@10"],\n'
        ')\n',
    ),
    (
        "markdown",
        "## C.4 · 📚 Coherencia entre formulaciones de las 12 ciegas (Jaccard@10)\n"
        "\n"
        "Las 12 consultas de evaluación **no tienen juicios**, así que su nDCG es "
        "incalculable. Lo que sí se mide sin etiquetas: las tres formulaciones de una "
        "misma intención piden lo mismo, así que un buscador que entienda la intención "
        "debería devolver productos parecidos. Jaccard@10 = productos en ambos top-10 / "
        "productos en alguno de los dos.",
    ),
    (
        "code",
        'consistencia = {\n'
        '    nombre: formulation_consistency(\n'
        '        rank_queries(retriever, consultas_eval, id_col="evaluation_id", k=TOP_K),\n'
        '        k=TOP_K,\n'
        '    ).assign(baseline=nombre)\n'
        '    for nombre, retriever in [("tfidf", tfidf_completo), ("bm25", bm25_completo)]\n'
        '}\n'
        'pd.concat(consistencia.values()).set_index(["baseline", "intencion"]).sort_index()\n',
    ),
    (
        "markdown",
        "## C.5 · Artefacto `artifacts/baseline_lexico.json`\n"
        "\n"
        "Métricas **y los IDs recuperados por consulta**: sin los IDs no se puede "
        "atribuir errores en NB09 (Regla 3 de experimentación).",
    ),
    (
        "code",
        'import json\n'
        '\n'
        'artefacto = {\n'
        '    "configuracion": {\n'
        '        "campo_indexado": CAMPO,\n'
        '        "strip_accents": STRIP_ACCENTS,\n'
        '        "top_k": TOP_K,\n'
        '        "relevancia": {"E": 3, "S": 2, "C": 1, "I": 0},\n'
        '        "umbral_relevante_recall_mrr": 2.0,\n'
        '        "gain_ndcg": "exponential",\n'
        '    },\n'
        '    "muestra": {\n'
        '        "n_docs": len(muestra),\n'
        '        "metricas": {n: i.summary for n, i in informes_muestra.items()},\n'
        '        "por_consulta": {n: i.per_query_frame().to_dict("records") for n, i in informes_muestra.items()},\n'
        '        "rankings": rankings_muestra,\n'
        '    },\n'
        '    "completo": {\n'
        '        "n_docs": len(completo),\n'
        '        "metricas": {n: i.summary for n, i in informes_completo.items()},\n'
        '        "por_consulta": {n: i.per_query_frame().to_dict("records") for n, i in informes_completo.items()},\n'
        '        "rankings": rankings_completo,\n'
        '        "jaccard_ciegas": pd.concat(consistencia.values()).to_dict("records"),\n'
        '    },\n'
        '}\n'
        '\n'
        'destino = Path("..") / "artifacts" / "baseline_lexico.json"\n'
        'destino.write_text(json.dumps(artefacto, indent=2, ensure_ascii=False), encoding="utf-8")\n'
        'print(f"Escrito {destino} ({destino.stat().st_size / 1024:.1f} KB)")\n',
    ),
]

NB02_MODELO = [
    ("markdown", '# NB02 · ¿Con qué modelo? — modelos, prefijos, normalización, métrica'),
    (
        "markdown",
        '**D09 ya está decidida**: compiten `gemini-embedding-2`, `jinaai/jina-embeddings-v3` e `ibm-granite/granite-embedding-311m-multilingual-r2`, con la **plantilla congelada en A0** (la columna `text` tal cual).\n'
        '\n'
        'El notebook empieza por el **paso 0**: medir cuánto texto ve realmente cada candidato. Es la evidencia que decide **D07** (¿hace falta chunking?) y la que confirma que la ventana de los tres cubre el catálogo.\n'
        '\n'
        '⚠️ **Las longitudes de NB01 no sirven aquí.** Allí se midió `text` en **palabras** (p50 = 150); la ventana de un modelo se mide en **piezas de su vocabulario de subpalabras**, y un código como `160x200` se lleva varias. La única cuenta válida es la del tokenizador de cada modelo, y por eso se descarga aquí.\n'
        '\n'
        '| Marca | Corpus | Fichero |\n'
        '|---|---|---|\n'
        '| 🔬 **MUESTRA** | 1.500 registros | `catalogo_muestra.csv` |\n'
        '| 📚 **COMPLETO** | 15.000 registros | `catalogo_productos.csv` |\n'
        '\n'
        '> 📄 **Cada celda de código declara sus datos en la primera línea, incluidas las que no leen ninguno.** El mismo modelo baja de nDCG al pasar de 1.500 candidatos a 15.000 sin que nada haya empeorado, así que saber qué corpus se está mirando cambia la lectura de cualquier métrica. Además de los dos catálogos aparecen `consultas_desarrollo.csv` (8 consultas con juicios), `relevancias_desarrollo.csv` (248 juicios de relevancia) y `consultas_evaluacion.csv` (12 ciegas, sin juicios).\n'
        '\n'
        '**El cambio de corpus ocurre en la sección I**: todo lo anterior mide sobre la muestra.',
    ),
    (
        "code",
        '# 📄 DATOS · carga catalogo_muestra.csv (1.500) y catalogo_productos.csv (15.000)\n'
        'import os\n'
        'import sys\n'
        'import warnings\n'
        'from pathlib import Path\n'
        '\n'
        'warnings.filterwarnings("ignore")\n'
        'sys.path.insert(0, str(Path("..") / "src"))\n'
        '\n'
        'import pandas as pd\n'
        'from dotenv import load_dotenv\n'
        'from transformers import logging as hf_logging\n'
        '\n'
        'from aurum.embeddings import load_hub_tokenizer, token_length_report, token_lengths\n'
        '\n'
        'hf_logging.set_verbosity_error()\n'
        '# Carga HF_TOKEN y GEMINI_API_KEY en el entorno del proceso. El fichero .env no se\n'
        '# imprime ni se versiona: solo se leen los valores desde os.environ.\n'
        'load_dotenv(Path("..") / ".env")\n'
        '\n'
        'DATA = Path("..") / "data"\n'
        'muestra = pd.read_csv(DATA / "catalogo_muestra.csv")\n'
        'completo = pd.read_csv(DATA / "catalogo_productos.csv")\n'
        '\n'
        'print(f"🔬 MUESTRA : {len(muestra):>6} registros")\n'
        'print(f"📚 COMPLETO: {len(completo):>6} registros")\n'
        'print(f"HF_TOKEN cargado: {bool(os.environ.get(\'HF_TOKEN\'))}")',
    ),
    (
        "markdown",
        '## A.1 · Los tres candidatos de D09\n'
        '\n'
        'Datos de cada uno, contrastados contra su *model card* y su `config.json` (no contra la memoria de nadie). `gemini-embedding-2` es API, así que no tiene tokenizador descargable: su ventana se toma de la documentación oficial.\n'
        '\n'
        '| Modelo | Dim nativa | Ventana | Contrato de entrada | Acceso |\n'
        '|---|---:|---:|---|---|\n'
        '| `gemini-embedding-2` | 3.072 (MRL 128–3.072) | 8.192 | instrucción **en el prompt** — no admite `task_type` | API de Google (`GEMINI_API_KEY`) |\n'
        '| `jinaai/jina-embeddings-v3` | 1.024 (MRL 32–1.024) | 8.192 | **adaptadores LoRA por tarea**: `retrieval.query` · `retrieval.passage` · `text-matching` | HF abierto, licencia **`cc-by-nc-4.0`** ⚠️ |\n'
        '| `ibm-granite/granite-embedding-311m-multilingual-r2` | 768 | 32.768 | por confirmar en su model card | HF abierto (Apache-2.0) |',
    ),
    (
        "code",
        '# 📄 DATOS · ninguno: solo descarga los tokenizadores del Hub\n'
        'CANDIDATOS_HF = {  # espejo de config.yaml -> nb02_modelo.d09_modelos\n'
        '    "jina-v3": ("jinaai/jina-embeddings-v3", 8192),\n'
        '    "granite-311m-r2": ("ibm-granite/granite-embedding-311m-multilingual-r2", 32768),\n'
        '}\n'
        'VENTANA_GEMINI = 8192  # gemini-embedding-2, según la documentación de la API\n'
        '\n'
        '# `load_hub_tokenizer` baja solo el tokenizer.json: jina-v3 lleva código propio en\n'
        '# el repo y AutoTokenizer exigiría trust_remote_code, que importa torch. Para\n'
        '# contar tokens basta el vocabulario.\n'
        'tokenizadores = {}\n'
        'for alias, (repo, ventana) in CANDIDATOS_HF.items():\n'
        '    try:\n'
        '        tokenizadores[alias] = (\n'
        '            load_hub_tokenizer(repo, token=os.environ.get("HF_TOKEN")),\n'
        '            ventana,\n'
        '        )\n'
        '        print(f"✅ {alias}: tokenizador descargado")\n'
        '    except Exception as error:  # sin red, repo gated o token sin permiso\n'
        '        print(f"⛔ {alias}: {type(error).__name__} — {str(error).splitlines()[0]}")',
    ),
    (
        "markdown",
        '## A.2 · 🔬 Longitud en tokens sobre la muestra\n'
        '\n'
        '`pct_supera_ventana` es el número que decide **D07**. `chars_por_token` mide cuánto se aleja la cuenta real de una estimación en caracteres.',
    ),
    (
        "code",
        '# 📄 DATOS · 🔬 catalogo_muestra.csv (1.500 fichas)\n'
        'token_length_report(muestra["text"], tokenizadores)',
    ),
    (
        "markdown",
        '## A.3 · 📚 La misma medición sobre el catálogo completo\n'
        '\n'
        '⏱️ Tarda ~30 s: tokeniza 15.000 registros.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000 fichas)\n'
        'informe_completo = token_length_report(completo["text"], tokenizadores)\n'
        'informe_completo',
    ),
    (
        "markdown",
        '## A.4 · 📚 Cuántos registros del catálogo se truncarían con cada tamaño de ventana\n'
        '\n'
        'La pregunta de fondo de D07 no es *"¿se trunca?"* sino *"¿a partir de qué ventana deja de truncarse?"*. Esta tabla la responde de una vez para cualquier modelo, presente o futuro.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000 fichas)\n'
        'alias_referencia = next(iter(tokenizadores))\n'
        'longitudes = token_lengths(completo["text"], tokenizadores[alias_referencia][0])\n'
        '\n'
        'pd.DataFrame([\n'
        '    {\n'
        '        "ventana": ventana,\n'
        '        "registros_que_la_superan": int((longitudes > ventana).sum()),\n'
        '        "pct": round(100 * float((longitudes > ventana).mean()), 2),\n'
        '    }\n'
        '    for ventana in (128, 512, 1024, 2048, 8192)\n'
        ']).assign(tokenizador=alias_referencia)',
    ),
    (
        "markdown",
        '## A.5 · 📚 `gemini-embedding-2`: medición contra la API\n'
        '\n'
        'Los dos modelos locales se miden con su tokenizador descargado. Gemini no publica el suyo, pero la API expone `count_tokens` **para el propio modelo de embeddings** — así que no hay que estimar nada ni usar un modelo generativo como sustituto.\n'
        '\n'
        '⚠️ **Es una petición de red por registro**, así que se mide un subconjunto: los **50 más largos** en caracteres —los únicos que podrían acercarse a la ventana— más **100 al azar** para el ratio `chars_por_token`. Con un máximo local de 1.972 tokens contra una ventana de 8.192, el margen es de 4×: no hace falta más precisión para responder a D07.\n'
        '\n'
        'Sin `GEMINI_API_KEY` la celda se salta sin romper el notebook: el corrector puede ejecutar el resto sin clave.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv → subconjunto de 150 fichas (una llamada por ficha)\n'
        'from aurum.embeddings import CountingTokenizer, gemini_token_counter\n'
        '\n'
        'MODELO_GEMINI = "gemini-embedding-2"\n'
        'N_MAS_LARGAS, N_AZAR = 50, 100\n'
        '\n'
        'longitud_chars = completo["text"].fillna("").str.len()\n'
        'mas_largas = completo.loc[longitud_chars.nlargest(N_MAS_LARGAS).index, "text"]\n'
        'al_azar = completo["text"].sample(N_AZAR, random_state=42)\n'
        'subconjunto = pd.concat([mas_largas, al_azar]).drop_duplicates()\n'
        '\n'
        'clave = os.environ.get("GEMINI_API_KEY")\n'
        'if not clave:\n'
        '    print("⏭️  Sin GEMINI_API_KEY: se omite la medición de gemini-embedding-2")\n'
        'else:\n'
        '    gemini = CountingTokenizer(gemini_token_counter(MODELO_GEMINI, api_key=clave))\n'
        '    informe_gemini = token_length_report(subconjunto, {MODELO_GEMINI: (gemini, VENTANA_GEMINI)})\n'
        '    display(informe_gemini)',
    ),
    (
        "markdown",
        '### A.5b · 📚 El mismo subconjunto con los tres tokenizadores\n'
        '\n'
        'Comparar los tres sobre **los mismos registros** es lo que exige la Regla 2: si cada modelo se midiera sobre un corpus distinto, las columnas no serían comparables entre sí.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 el mismo subconjunto de 150 fichas de catalogo_productos.csv\n'
        'if clave:\n'
        '    todos = {MODELO_GEMINI: (gemini, VENTANA_GEMINI), **tokenizadores}\n'
        '    display(token_length_report(subconjunto, todos))',
    ),
    (
        "markdown",
        '## A.6 · Cómo se lee esto para D07\n'
        '\n'
        'El chunking solo tiene sentido si el modelo **no puede leer el registro entero**. Con `pct_supera_ventana = 0` en los tres candidatos, no hay información que se pierda por truncado: queda descartado **por medición**, no por falta de tiempo.\n'
        '\n'
        'Consecuencia en la base vectorial: el punto sigue siendo `record_id` (1:1 producto↔vector), el esquema de NB04 se mantiene simple y la idempotencia no necesita borrar chunks huérfanos.',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '# B · Codificación de los tres candidatos\n'
        '\n'
        'Hasta aquí se ha medido **cuánto texto ve** cada modelo. Ahora se codifica de verdad, sobre `catalogo_muestra.csv`: la muestra existe justo para esto y el catálogo completo solo se ingiere en la ejecución final.\n'
        '\n'
        '### Qué se codifica y qué no\n'
        '\n'
        '| Eje de D10 | ¿Obliga a recodificar? | Cómo se barre |\n'
        '|---|---|---|\n'
        '| **Modelo** | Sí | 3 codificaciones |\n'
        '| **Contrato de entrada** (con/sin prefijos) | Sí | ×2 **solo** en los modelos que tienen contrato |\n'
        '| **Dimensión (MRL)** | No | Truncar + renormalizar los mismos vectores |\n'
        '| **Normalización L2** | No | Post-proceso |\n'
        '| **Métrica** (`cosine`·`dot`·`l2`) | No | Cambia el buscador, no los vectores |\n'
        '\n'
        'Los tres últimos ejes son **gratis**, y por eso el barrido de la sección C cubre 15 configuraciones sin pagar 15 codificaciones.\n'
        '\n'
        '### ⏱️ Lo que cuesta en esta máquina\n'
        '\n'
        '4 núcleos, sin GPU. `jina-embeddings-v3` son 572M de parámetros: en `float32` ocupa ~2,3 GB de los 7,9 disponibles, así que **los modelos se cargan y se liberan de uno en uno**, y la primera pasada son decenas de minutos.\n'
        '\n'
        'La segunda es instantánea: `encode_corpus` cachea en `artifacts/embeddings/` con un `.json` que lleva `model_id`, dimensión, dtype y **SHA-256 del corpus**. Si el texto cambia (al pasar de A0 a otra plantilla en NB03), la huella cambia y la caché se invalida sola — lo que impide comparar en silencio vectores de dos textos distintos.',
    ),
    (
        "code",
        '# 📄 DATOS · 🔬 catalogo_muestra.csv (1.500) es el corpus de aquí en adelante\n'
        '#            + consultas_desarrollo.csv (8) · relevancias_desarrollo.csv (248 juicios)\n'
        '#            + consultas_evaluacion.csv (12 ciegas, sin juicios)\n'
        'import gc\n'
        'import time\n'
        '\n'
        'import numpy as np\n'
        'import torch\n'
        '\n'
        'from aurum.busqueda import DenseRetriever, rank_queries_dense\n'
        'from aurum.embeddings import (\n'
        '    GeminiEncoder,\n'
        '    SentenceTransformerEncoder,\n'
        '    api_cost_report,\n'
        '    drift_check,\n'
        '    encode_corpus,\n'
        '    measure_encode_latency,\n'
        '    safe_l2_normalize,\n'
        '    truncate_dim,\n'
        '    vector_health,\n'
        ')\n'
        'from aurum.evaluacion import (\n'
        '    apply_tolerance_rule,\n'
        '    evaluate_rankings,\n'
        '    formulation_consistency,\n'
        '    qrels_from_judgements,\n'
        ')\n'
        'from aurum.graficas import (\n'
        '    plot_contract_delta,\n'
        '    plot_dimension_curve,\n'
        '    plot_metric_comparison,\n'
        ')\n'
        '\n'
        'torch.set_num_threads(4)  # los 4 núcleos físicos de la máquina\n'
        '\n'
        'CORPUS_ID = "catalogo_muestra"   # condición 3 del plan\n'
        'PLANTILLA = "A0"                 # congelada: la columna `text` tal cual\n'
        'CAMPO = "text"\n'
        'TOP_K = 10\n'
        'BATCH_LOCAL = 8                  # 8 GB de RAM con un modelo de 572M cargado\n'
        'CACHE = Path("..") / "artifacts" / "embeddings"\n'
        'TOLERANCIA_D09B = 0.02           # tau de config.yaml -> d09b_criterio_desempate\n'
        '\n'
        'consultas = pd.read_csv(DATA / "consultas_desarrollo.csv")\n'
        'relevancias = pd.read_csv(DATA / "relevancias_desarrollo.csv")\n'
        'ciegas = pd.read_csv(DATA / "consultas_evaluacion.csv")\n'
        'qrels = qrels_from_judgements(relevancias)\n'
        '\n'
        'corpus_textos = muestra[CAMPO].tolist()\n'
        'corpus_ids = muestra["product_id"].tolist()\n'
        'query_ids = [str(q) for q in consultas["query_id"]]\n'
        'query_textos = consultas["query_text"].tolist()\n'
        '# Las 12 ciegas se cargan aquí, junto al resto, porque la sección J las necesita\n'
        '# codificadas con CADA modelo: si se dejaran para más abajo habría que volver a\n'
        '# cargar los modelos locales en memoria solo para 12 frases.\n'
        'ciegas_textos = ciegas["query_text"].tolist()\n'
        '\n'
        'print(f"corpus   : {len(corpus_textos)} documentos (plantilla {PLANTILLA})")\n'
        'print(f"consultas: {len(query_textos)} de desarrollo, {len(qrels)} juzgadas")\n'
        'n_intenciones = ciegas["evaluation_id"].str.split("-").str[1].nunique()\n'
        'print(f"ciegas   : {len(ciegas_textos)} = {n_intenciones} intenciones x "\n'
        '      f"{ciegas[\'query_type\'].nunique()} formulaciones")',
    ),
    (
        "markdown",
        '## B.1 · Registro de modelos\n'
        '\n'
        'Cada ficha reproduce lo verificado en A.1 más lo que el modelo necesita para codificar. La columna que más decisiones arrastra es **el contrato de entrada**:\n'
        '\n'
        '| Modelo | Mecanismo del contrato | ¿Entra en el eje con/sin de D10? |\n'
        '|---|---|---|\n'
        '| `jina-v3` | **Adaptadores LoRA** (`retrieval.passage` / `retrieval.query`) — pesos distintos, no texto | ✅ Sí, y cuesta ×2 codificaciones |\n'
        '| `granite-311m-r2` | **Ninguno** — confirmado en la model card de IBM | ❌ No: no hay contrato que retirar |\n'
        '| `gemini-2` | **Instrucción dentro del prompt** | ✅ Sí (API, barato) |\n'
        '\n'
        '> 🔎 **Por qué granite no entra en ese eje (P02, cerrado).** Su `config_sentence_transformers.json` declara `"prompts": {"query": "", "document": ""}`: codificar con y sin contrato daría **los mismos vectores**, la Δ sería 0 por construcción, e inventarle un prefijo mediría un modelo que nadie entrenó. Como §3.1 avisa de que *"no basta con citar la documentación del modelo"* —el fichero prueba lo que hace **la librería**, no lo que entrenó **IBM**—, se revisó la model card entera: ni los backends documentados (`sentence-transformers`, Transformers, ONNX, OpenVINO, vLLM, GGUF) ni las secciones *Usage* y *When to Use This Model* traen instrucción alguna, y en el ejemplo de retrieval consultas y documentos se pasan por igual a `model.encode()`. El contrato real es **texto plano simétrico**: `granite` no compitió en desventaja.\n'
        '\n'
        '> ⚠️ `jina-v3` exige `trust_remote_code=True` —se ejecuta código del repositorio de Jina, y lo hereda quien ejecute el notebook— y su licencia **`cc-by-nc-4.0`** prohíbe el uso comercial, que es justo el escenario de un marketplace. §3.1 obliga a pesar esa clase de restricciones al elegir.',
    ),
    (
        "code",
        '# 📄 DATOS · ninguno: la ficha técnica de cada modelo candidato\n'
        'REGISTRO = {\n'
        '    "jina-v3": {\n'
        '        "repo": "jinaai/jina-embeddings-v3",\n'
        '        "ventana": 8192,\n'
        '        "dim_nativa": 1024,\n'
        '        "tasks": {"document": "retrieval.passage", "query": "retrieval.query"},\n'
        '        "trust_remote_code": True,\n'
        '        "dims": [1024, 768, 512, 256, 128],\n'
        '        "licencia": "cc-by-nc-4.0",\n'
        '        # Fuera del análisis de robustez de la sección J: codificar sus 12\n'
        '        # ciegas obliga a meter 2,3 GB de pesos en memoria, y el barrido ya\n'
        '        # lo dejó último de los tres. Se declara aquí, junto a la ficha del\n'
        '        # modelo, para que la exclusión sea auditable y no una celda saltada.\n'
        '        "ciegas": False,\n'
        '    },\n'
        '    "granite-311m-r2": {\n'
        '        "repo": "ibm-granite/granite-embedding-311m-multilingual-r2",\n'
        '        "ventana": 32768,\n'
        '        "dim_nativa": 768,\n'
        '        "tasks": None,  # prompts declarados como cadena vacía: no hay contrato\n'
        '        "trust_remote_code": False,\n'
        '        "dims": [768, 512, 256, 128],\n'
        '        "licencia": "apache-2.0",\n'
        '    },\n'
        '    "gemini-2": {\n'
        '        "repo": "gemini-embedding-2",\n'
        '        "ventana": VENTANA_GEMINI,\n'
        '        "dim_nativa": 3072,\n'
        '        "api": True,\n'
        '        "dims": [3072, 1536, 768, 512, 256, 128],\n'
        '        "licencia": "servicio de terceros",\n'
        '    },\n'
        '}\n'
        '\n'
        '\n'
        'def fabricar(alias):\n'
        '    """Construye el encoder del alias. Se llama justo antes de codificar y el\n'
        '    objeto se libera después: dos modelos locales a la vez no caben en 8 GB."""\n'
        '    ficha = REGISTRO[alias]\n'
        '    if ficha.get("api"):\n'
        '        return GeminiEncoder(\n'
        '            api_key=os.environ.get("GEMINI_API_KEY"),\n'
        '            model_id=ficha["repo"],\n'
        '            native_dim=ficha["dim_nativa"],\n'
        '            window=ficha["ventana"],\n'
        '        )\n'
        '    return SentenceTransformerEncoder(\n'
        '        ficha["repo"],\n'
        '        window=ficha["ventana"],\n'
        '        native_dim=ficha["dim_nativa"],\n'
        '        tasks=ficha["tasks"],\n'
        '        trust_remote_code=ficha["trust_remote_code"],\n'
        '        device="cpu",\n'
        '        token=os.environ.get("HF_TOKEN"),\n'
        '    )\n'
        '\n'
        '\n'
        'pd.DataFrame([\n'
        '    {\n'
        '        "alias": alias,\n'
        '        "repo": f["repo"],\n'
        '        "dim_nativa": f["dim_nativa"],\n'
        '        "ventana": f["ventana"],\n'
        '        "tiene_contrato": bool(f.get("api") or f.get("tasks")),\n'
        '        "licencia": f["licencia"],\n'
        '    }\n'
        '    for alias, f in REGISTRO.items()\n'
        '])',
    ),
    (
        "markdown",
        '## B.2 · Codificar — ⏱️ **una celda por modelo, ejecutables por separado**\n'
        '\n'
        'Los tres modelos **no** se codifican en un bucle: cada uno tiene su celda, lanzable por separado y en sesiones distintas. Tres razones, ninguna estética:\n'
        '\n'
        '1. **RAM.** `jina-v3` ocupa ~2,3 GB de los 7,9 de la máquina. Cada celda construye el modelo, codifica y lo libera con `gc.collect()`: dos modelos vivos a la vez no caben.\n'
        '2. **Tiempo.** Decenas de minutos por modelo; un bucle único obliga a esperar a los tres para ver el primer número.\n'
        '3. **Aislamiento de fallos.** Si `jina-v3` revienta por su `trust_remote_code` o Gemini se queda sin cuota, los demás ya están medidos — y con dos de tres sigue habiendo las *"al menos dos configuraciones relevantes"* que pide el enunciado.\n'
        '\n'
        '### 🔑 Cada celda codifica las DOS variantes\n'
        '\n'
        '`nativo` (con el contrato de entrada que el modelo declara) y `sin_contrato` (omitiéndolo), juntas a propósito: es lo que hace que **el notebook dé el mismo resultado se ejecute como se ejecute**. La sección C evalúa todo lo que encuentre en `VECTORES`, así que si `sin_contrato` se codificara en la D —donde se analiza—, una ejecución de principio a fin llegaría a C con media tabla y **C.1, C.2, F y G decidirían sobre ella sin avisar**. La D, por tanto, no codifica nada: solo mide la diferencia entre dos ramas que ya existen.\n'
        '\n'
        '> `granite-311m-r2` no tiene contrato que retirar, así que su segunda llamada no codifica y lo dice por pantalla: se ve que se saltó a propósito y no por olvido (**P02**).\n'
        '\n'
        'Cada celda deposita sus vectores en `VECTORES`, indexado por `(modelo, contrato)`, y **todo lo que viene detrás —barrido C, contrato D, métrica E, comparación F, regla G— lee ese diccionario y trabaja con lo que encuentre**: las tablas salen con uno, dos o tres modelos.\n'
        '\n'
        '> 🔁 **Tras reiniciar el kernel** hay que reejecutar las tres celdas, pero con la caché son segundos. Reejecutar tampoco duplica nada: `COSTES` está indexado por `(modelo, contrato, tipo)` y sobrescribe en vez de acumular.',
    ),
    (
        "code",
        '# 📄 DATOS · 🔬 muestra (documentos) + las 8 de desarrollo + las 12 ciegas\n'
        'VECTORES = {}   # (alias, contrato) -> {"document", "query", "query_ciegas"}\n'
        'COSTES = {}     # (alias, contrato, tipo) -> fila de coste. Dict, no lista: así\n'
        '                # re-ejecutar la celda de un modelo sobrescribe en vez de duplicar.\n'
        'ERRORES = {}\n'
        '\n'
        '\n'
        'def codificar(alias, contrato="nativo"):\n'
        '    """Codifica documentos y consultas de un modelo, y libera la memoria."""\n'
        '    ficha = REGISTRO[alias]\n'
        '    encoder = fabricar(alias)\n'
        '    lotes = 32 if ficha.get("api") else BATCH_LOCAL\n'
        '    salida = {}\n'
        '    try:\n'
        '        # Los corpus del modelo en una sola pasada. `clave` distingue las dos\n'
        '        # tandas de consultas, que para el encoder son el mismo `kind`: las de\n'
        '        # desarrollo deciden (tienen juicios) y las ciegas miden robustez (no\n'
        '        # los tienen). Codificarlas aquí y no en la sección J evita volver a\n'
        '        # cargar los pesos del modelo para 12 frases.\n'
        '        corpus = [\n'
        '            ("document", "document", corpus_textos, CORPUS_ID),\n'
        '            ("query", "query", query_textos, "consultas_desarrollo"),\n'
        '        ]\n'
        '        if ficha.get("ciegas", True):\n'
        '            corpus.append(\n'
        '                ("query_ciegas", "query", ciegas_textos, "consultas_evaluacion")\n'
        '            )\n'
        '        for clave, kind, textos, corpus_id in corpus:\n'
        '            resultado = encode_corpus(\n'
        '                encoder, textos, corpus_id=corpus_id, kind=kind,\n'
        '                contract=contrato, batch_size=lotes, cache_dir=CACHE,\n'
        '            )\n'
        '            salida[clave] = resultado.vectors\n'
        '            COSTES[(alias, contrato, clave)] = {\n'
        '                "alias": alias, **resultado.stats.as_row(), "tipo": clave,\n'
        '            }\n'
        '    finally:\n'
        '        # El `finally` importa: si la codificación de consultas falla, el modelo\n'
        '        # se libera igual y el kernel no se queda con 2,3 GB retenidos.\n'
        '        del encoder\n'
        '        gc.collect()\n'
        '    return salida\n'
        '\n'
        '\n'
        'def ejecutar(alias, contrato="nativo"):\n'
        '    """Codifica un modelo dejando el resultado en VECTORES, sin propagar el fallo.\n'
        '\n'
        '    Cada celda de modelo llama aquí dos veces, una por rama de contrato. Un error\n'
        '    se registra y se muestra, pero no detiene el notebook: los modelos que sí\n'
        '    funcionaron siguen siendo medibles."""\n'
        '    ficha = REGISTRO[alias]\n'
        '    if contrato == "sin_contrato" and not (ficha.get("api") or ficha.get("tasks")):\n'
        '        print(f"⏭️  {alias}: no tiene contrato de entrada, el eje no aplica")\n'
        '        return\n'
        '    inicio = time.perf_counter()\n'
        '    try:\n'
        '        VECTORES[(alias, contrato)] = codificar(alias, contrato)\n'
        '        ERRORES.pop(f"{alias}[{contrato}]", None)\n'
        '        print(f"✅ {alias} [{contrato}] listo en {time.perf_counter() - inicio:.1f}s")\n'
        '    except Exception as error:\n'
        '        ERRORES[f"{alias}[{contrato}]"] = f"{type(error).__name__}: {error}"\n'
        '        print(f"⛔ {alias} [{contrato}]: {ERRORES[f\'{alias}[{contrato}]\'][:250]}")\n'
        '\n'
        '\n'
        'def estado():\n'
        '    """Qué hay codificado ahora mismo. Es lo que podrán medir las secciones C-G."""\n'
        '    filas = [\n'
        '        {\n'
        '            "modelo": alias,\n'
        '            "contrato": contrato,\n'
        '            "docs": VECTORES[(alias, contrato)]["document"].shape,\n'
        '            "consultas": VECTORES[(alias, contrato)]["query"].shape,\n'
        '            "ciegas": VECTORES[(alias, contrato)].get("query_ciegas", np.empty((0, 0))).shape,\n'
        '        }\n'
        '        for (alias, contrato) in sorted(VECTORES)\n'
        '    ]\n'
        '    return pd.DataFrame(filas) if filas else "Todavía no hay ningún modelo codificado."\n'
        '\n'
        '\n'
        'print(\n'
        '    "Listo. Ejecuta las tres celdas siguientes en el orden que prefieras.\\n"\n'
        '    "Cada una codifica su modelo en las DOS ramas de contrato: la sección D las\\n"\n'
        '    "compara, pero no las genera, así que ninguna sección posterior depende del\\n"\n'
        '    "orden en que ejecutes esto."\n'
        ')',
    ),
    (
        "markdown",
        '### B.2a · `jina-v3` — ⏱️ el más caro (572M de parámetros)\n'
        '\n'
        '**Dos codificaciones completas de los 1.500 documentos.** En jina el contrato no es un prefijo de texto sino un **adaptador LoRA**: `retrieval.passage` para documentos y `retrieval.query` para consultas. Retirarlo significa usar otros pesos, así que la variante `sin_contrato` no se puede derivar de la primera — hay que codificar de nuevo. Es el eje más caro de todo D10, y por eso se paga aquí una sola vez.\n'
        '\n'
        '⚠️ Ejecuta código del repositorio de Jina (`trust_remote_code=True`) y su licencia es **`cc-by-nc-4.0`**: no bloquea el experimento académico, pero sí la recomendación final para un marketplace real. Queda anotado para el informe.',
    ),
    (
        "code",
        '# 📄 DATOS · 🔬 muestra (1.500 documentos) + 8 consultas de desarrollo. Sin ciegas: ver J\n'
        'ejecutar("jina-v3")                            # con contrato: adaptador LoRA por tarea\n'
        'ejecutar("jina-v3", contrato="sin_contrato")   # sin él: mismos textos, otros pesos',
    ),
    (
        "markdown",
        '### B.2b · `granite-311m-r2` — Apache-2.0, sin código remoto\n'
        '\n'
        '**Una sola codificación.** La segunda llamada está puesta pero no codifica: granite declara sus dos prompts como cadena vacía, así que `nativo` y `sin_contrato` darían vectores idénticos. Imprime el motivo del salto en vez de gastar otra pasada en una Δ que es 0 por construcción.\n'
        '\n'
        'La exclusión se registró como **P02** y está **cerrada**: la model card de IBM no documenta instrucción ni prefijo en ningún backend, así que se sostiene en la fuente primaria y no en un JSON de configuración.',
    ),
    (
        "code",
        '# 📄 DATOS · 🔬 muestra (1.500 documentos) + 8 de desarrollo + 12 ciegas\n'
        'ejecutar("granite-311m-r2")\n'
        '# No codifica nada: granite no declara contrato que retirar. La llamada se deja\n'
        '# puesta para que el salto aparezca en la salida del notebook — así el informe\n'
        '# muestra que se omitió a propósito, no por olvido (P02, cerrado).\n'
        'ejecutar("granite-311m-r2", contrato="sin_contrato")',
    ),
    (
        "markdown",
        '### B.2c · `gemini-2` — por API\n'
        '\n'
        '**Dos codificaciones, pero baratas.** En Gemini el contrato es texto: una instrucción de tarea antepuesta al contenido. Retirarla no cambia los pesos, solo lo que se envía, y el trabajo lo hace la API — el eje cuesta llamadas, no horas de CPU.\n'
        '\n'
        'Necesita `GEMINI_API_KEY` en `.env`; sin ella las dos llamadas fallan de forma controlada y el notebook sigue con los modelos locales.',
    ),
    (
        "code",
        '# 📄 DATOS · 🔬 muestra (1.500 documentos) + 8 de desarrollo + 12 ciegas\n'
        'ejecutar("gemini-2")                            # con la instrucción de tarea en el prompt\n'
        'ejecutar("gemini-2", contrato="sin_contrato")   # con el texto desnudo',
    ),
    (
        "markdown",
        '### B.2d · Qué hay codificado y cuánto ha costado\n'
        '\n'
        'El coste va **junto** a la calidad, no en una nota al pie: una ventaja de nDCG que se paga con 3× de tiempo de indexación es una decisión distinta a una ventaja gratis. Es el criterio que D09b declaró de antemano.',
    ),
    (
        "code",
        '# 📄 DATOS · ninguno: resume lo que ya está codificado y lo que costó\n'
        'display(estado())\n'
        'if ERRORES:\n'
        '    display(pd.DataFrame([{"modelo": k, "error": v} for k, v in ERRORES.items()]))\n'
        'pd.DataFrame(COSTES.values())',
    ),
    (
        "markdown",
        '## B.3 · Salud de los vectores — antes de creerse ninguna métrica\n'
        '\n'
        'Un `NaN` o una matriz con filas repetidas producen métricas presentables y falsas. Se comprueban finitud, normas y duplicados antes de creerse ningún número.\n'
        '\n'
        '**Atención a la columna `normalizado`, que no resultó anecdótica.** `SentenceTransformerEncoder` pide `normalize_embeddings=False` a los dos modelos locales, pero `jina-v3` y `gemini-2` salen con norma exactamente 1 y solo `granite-311m-r2` entrega la salida cruda: ese flag **añade** normalización cuando vale `True`, no **retira** el módulo `Normalize` que jina lleva en su pipeline, ni impide que Gemini devuelva unitarios por API.\n'
        '\n'
        'Consecuencia: `granite` es el único candidato con el que la sección E puede medir algo.',
    ),
    (
        "code",
        '# 📄 DATOS · vectores de 🔬 la muestra y de sus consultas, ya en memoria\n'
        'pd.DataFrame([\n'
        '    {"alias": alias, "contrato": contrato, "tipo": kind, **vector_health(matriz)}\n'
        '    for (alias, contrato), por_tipo in VECTORES.items()\n'
        '    for kind, matriz in por_tipo.items()\n'
        '])',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '# C · Barrido de dimensión (MRL) — el eje gratis\n'
        '\n'
        'Los tres candidatos están entrenados con **Matryoshka Representation Learning**: las primeras componentes concentran la mayor parte de la información, así que quedarse con un prefijo del vector reduce la dimensión sin recodificar.\n'
        '\n'
        'Truncar **obliga a renormalizar** —el prefijo de un vector unitario tiene norma < 1, y sin renormalizar el coseno deja de ser un coseno—; `truncate_dim` lo hace siempre.\n'
        '\n'
        'Lo que se busca no es el máximo sino **dónde se cae la curva**: si 256 dimensiones pierden menos de 0,02 de nDCG@10 frente a 1.024, el ahorro es de 4× en memoria del motor y en ancho de banda por consulta. Es el compromiso que D09b declaró de antemano.',
    ),
    (
        "code",
        '# 📄 DATOS · 🔬 muestra (1.500 candidatos) · 8 consultas de desarrollo · 248 juicios\n'
        'def evaluar(vectores, dim, *, metric="cosine", normalizar=True):\n'
        '    """Evalúa una configuración concreta sobre las 8 consultas de desarrollo.\n'
        '\n'
        '    Devuelve el informe y los rankings: guardar los IDs y no solo la métrica es\n'
        '    lo que permite atribuir errores en NB09 (Regla 3 de experimentación)."""\n'
        '    docs = truncate_dim(vectores["document"], dim, renormalize=normalizar)\n'
        '    queries = truncate_dim(vectores["query"], dim, renormalize=normalizar)\n'
        '    retriever = DenseRetriever(docs, corpus_ids, metric=metric)\n'
        '    rankings = rank_queries_dense(retriever, query_ids, queries, k=TOP_K)\n'
        '    return evaluate_rankings(rankings, qrels, k=TOP_K), rankings\n'
        '\n'
        '\n'
        'costes = pd.DataFrame(COSTES.values())\n'
        'segundos_por_modelo = (\n'
        '    costes.query("tipo == \'document\'").set_index(["alias", "contrato"])["segundos"]\n'
        '    if len(costes) else pd.Series(dtype=float)\n'
        ')\n'
        '\n'
        'if not VECTORES:\n'
        '    raise RuntimeError(\n'
        '        "No hay ningún modelo codificado: ejecuta B.2a, B.2b o B.2c antes de esta celda."\n'
        '    )\n'
        '\n'
        '# El barrido recorre TODO lo codificado: cada modelo en sus dos ramas de\n'
        '# contrato. Filtrar aquí por `nativo` haría que la regla D09b de la sección G\n'
        '# eligiera al ganador dentro de una sola rama, sin llegar a ver la otra. Y no\n'
        '# cuesta ninguna codificación extra: truncar y evaluar es el eje gratis de D10.\n'
        'BARRIDO = []\n'
        'RANKINGS_DENSOS = {}\n'
        'for (alias, contrato), vectores in VECTORES.items():\n'
        '    for dim in REGISTRO[alias]["dims"]:\n'
        '        informe, rankings = evaluar(vectores, dim)\n'
        '        RANKINGS_DENSOS[(alias, contrato, dim)] = rankings\n'
        '        BARRIDO.append({\n'
        '            "modelo": alias,\n'
        '            "contrato": contrato,\n'
        '            # Etiqueta única de la configuración: `modelo` ya no la identifica,\n'
        '            # porque ahora hay dos filas por modelo y dimensión.\n'
        '            "sistema": f"{alias} [{contrato}]",\n'
        '            "dim": dim,\n'
        '            **informe.summary,\n'
        '            "bytes_por_vector": dim * 4,\n'
        '            "segundos": float(segundos_por_modelo.get((alias, contrato), float("nan"))),\n'
        '        })\n'
        '\n'
        'barrido = pd.DataFrame(BARRIDO).sort_values("ndcg_at_10", ascending=False)\n'
        '\n'
        '# Qué ha entrado en el barrido, en voz alta: es la única forma de que quien lea\n'
        '# el notebook sepa sobre qué se está decidiendo sin auditar el diccionario.\n'
        'print(f"Evaluadas {len(barrido)} configuraciones sobre {len(query_ids)} consultas:")\n'
        'for (alias, contrato), grupo in barrido.groupby(["modelo", "contrato"]):\n'
        '    dims = ", ".join(str(d) for d in sorted(grupo["dim"], reverse=True))\n'
        '    print(f"  · {alias:<16} [{contrato:<12}]  dims: {dims}")\n'
        '\n'
        'faltan_ramas = [\n'
        '    alias for alias, f in REGISTRO.items()\n'
        '    if (f.get("api") or f.get("tasks")) and (alias, "sin_contrato") not in VECTORES\n'
        ']\n'
        'if faltan_ramas:\n'
        '    print(\n'
        '        f"\\n⚠️  Sin la rama `sin_contrato`: {\', \'.join(faltan_ramas)}. Tienen contrato de"\n'
        '        "\\n    entrada, así que el barrido está incompleto y D09b decidiría sobre media"\n'
        '        "\\n    tabla. Ejecuta su celda de B.2 entera y vuelve aquí."\n'
        '    )\n'
        '\n'
        'barrido',
    ),
    (
        "markdown",
        '### C.1 · Curva calidad ↔ dimensión\n'
        '\n'
        'Los mismos números del barrido, en la forma que responde a la pregunta real: **¿dónde se cae la curva?** Si la caída es suave, MRL está funcionando; si hay un escalón, esa dimensión ya no basta para este catálogo.\n'
        '\n'
        'Tres elementos del gráfico que no son decoración:\n'
        '\n'
        '- **Color = modelo · trazo = contrato.** Son dos ejes cruzados: fundidos en el color darían cinco tonos sin relación aparente; separados, el ojo agrupa por modelo y compara la continua con la discontinua **dentro** de cada uno. Esa comparación —el mismo modelo consigo mismo— es la que cuantifica la sección D.\n'
        '- **La banda gris es la tolerancia τ = 0,02 de D09b.** Todo punto que cae dentro es admisible, y entre los admisibles gana el de menor dimensión: **el admisible más a la izquierda**. G lo calcula con `apply_tolerance_rule`; aquí se ve venir.\n'
        '- **Eje X logarítmico** porque las dimensiones se barren dividiendo por dos: en lineal, 128 y 256 se amontonarían contra el margen y la zona donde se decide el ahorro quedaría ilegible.\n'
        '\n'
        'La lógica vive en `aurum.graficas`, cubierta por `tests/test_graficas.py`: el notebook declara *qué* quiere ver, no *cómo* se dibuja.',
    ),
    (
        "code",
        '# 📄 DATOS · el barrido, medido sobre 🔬 la muestra\n'
        'plot_dimension_curve(\n'
        '    barrido,\n'
        '    model_column="modelo",     # el color agrupa por modelo\n'
        '    dash_column="contrato",    # el trazo separa con/sin contrato dentro de cada uno\n'
        '    tolerance=TOLERANCIA_D09B,\n'
        '    subtitle=(\n'
        '        f"{CORPUS_ID} ({len(corpus_textos)} docs) · {len(query_ids)} consultas · "\n'
        '        f"la banda gris es la tolerancia τ={TOLERANCIA_D09B} de D09b"\n'
        '    ),\n'
        ').show()',
    ),
    (
        "markdown",
        '### C.2 · Tabla por consulta — la media esconde el caso 33633\n'
        '\n'
        'NB00 midió que la consulta **33633** (*disfraz halloween talla grande hombre*) tiene **un solo `Exact`** en todo el pool: su Recall@10 solo puede valer 0 o 1, y una media macro sobre 8 consultas se mueve 0,125 según caiga. Reportar solo la media dejaría que esa consulta decidiera el modelo.',
    ),
    (
        "code",
        '# 📄 DATOS · el barrido sobre 🔬 la muestra, desglosado por las 8 de desarrollo\n'
        '# Mejor configuración de cada modelo, ya **entre las dos ramas de contrato**:\n'
        '# si `sin_contrato` gana, es esa la que representa al modelo de aquí en adelante.\n'
        'mejor_por_modelo = barrido.loc[barrido.groupby("modelo")["ndcg_at_10"].idxmax()]\n'
        '\n'
        'por_consulta = pd.concat([\n'
        '    # `fila.contrato` y no "nativo" a mano: si se fijara, se leerían los vectores\n'
        '    # de la otra rama sin ningún error visible —`(alias, "nativo")` también\n'
        '    # existe— y la tabla mostraría por consulta un sistema que no es el que ganó.\n'
        '    evaluar(VECTORES[(fila.modelo, fila.contrato)], int(fila.dim))[0]\n'
        '    .per_query_frame()\n'
        '    .assign(sistema=f"{fila.sistema}@{int(fila.dim)}")\n'
        '    for fila in mejor_por_modelo.itertuples()\n'
        '])\n'
        'por_consulta.pivot(index="query_id", columns="sistema", values="ndcg@10")',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '# D · El contrato de entrada (eje "prefijos" de D10)\n'
        '\n'
        '> 📌 **Esta sección no codifica nada**: las dos variantes de cada modelo se generaron en **B.2**. Aquí solo se comparan, así que corre en segundos.\n'
        '\n'
        '§3.1 pide elegir cuatro cosas y justificarlas juntas: *"Después elegid **la representación textual, el modelo de embeddings, los prefijos que requiera y la normalización**. No basta con citar la documentación del modelo: la elección debe apoyarse en los resultados de desarrollo y en las restricciones del caso."* El sujeto de "la elección" es la enumeración entera, así que la exigencia se reparte por todo NB02: la representación está congelada en A0 (NB01/NB03), el modelo lo deciden el barrido de C y la regla de G, la normalización se mide en E — y los **prefijos** son esta sección.\n'
        '\n'
        'Comparar cada modelo consigo mismo, con y sin su contrato, es lo que convierte la cita en evidencia:\n'
        '\n'
        '- Si retirar el contrato **no cambia nada** (Δ ≈ 0), es que no se estaba aplicando: un fallo de integración disfrazado de resultado.\n'
        '- Si **cambia**, queda demostrado con datos que el contrato hace algo — y el signo dice si ayuda o estorba en *este* catálogo, que no tiene por qué coincidir con lo que promete la model card.\n'
        '\n'
        'Qué significa "sin contrato" en cada uno:\n'
        '\n'
        '| Modelo | `nativo` | `sin_contrato` |\n'
        '|---|---|---|\n'
        '| `jina-v3` | Adaptador LoRA `retrieval.passage` / `retrieval.query` | Sin adaptador de tarea — **otros pesos** |\n'
        '| `gemini-2` | Instrucción de tarea antepuesta al texto | Texto desnudo |\n'
        '| `granite-311m-r2` | — | **No aplica**: sus dos prompts declarados son cadena vacía |\n'
        '\n'
        '> ✅ **P02, cerrado.** Esa última exclusión era justo la que §3.1 no admite tal cual: se apoyaba en el `config_sentence_transformers.json` de granite, es decir, en **la documentación del modelo**. Se cerró con la fuente primaria —la model card completa de IBM, incluidas las secciones *Usage* y *When to Use This Model*—, que **no documenta instrucción ni prefijo en ningún backend**: en su ejemplo de retrieval cross-lingual, `input_queries` e `input_passages` van directos a `model.encode()`, a diferencia de `e5-instruct` o los BGE. El contrato real es **texto plano simétrico**, así que granite no compitió en desventaja.',
    ),
    ("markdown", '### D.2 · Δ nDCG@10 al retirar el contrato'),
    (
        "code",
        '# 📄 DATOS · el barrido sobre 🔬 la muestra: las dos ramas de contrato enfrentadas\n'
        'filas = []\n'
        'for alias in REGISTRO:\n'
        '    if (alias, "sin_contrato") not in VECTORES:\n'
        '        continue\n'
        '    dim = REGISTRO[alias]["dim_nativa"]\n'
        '    con, _ = evaluar(VECTORES[(alias, "nativo")], dim)\n'
        '    sin, _ = evaluar(VECTORES[(alias, "sin_contrato")], dim)\n'
        '    filas.append({\n'
        '        "modelo": alias,\n'
        '        "dim": dim,\n'
        '        "ndcg_con_contrato": con.summary["ndcg_at_10"],\n'
        '        "ndcg_sin_contrato": sin.summary["ndcg_at_10"],\n'
        '        "delta": round(con.summary["ndcg_at_10"] - sin.summary["ndcg_at_10"], 4),\n'
        '    })\n'
        '\n'
        'pd.DataFrame(filas) if filas else "Sin modelos con contrato codificados"',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '# E · Normalización y métrica — qué significa el score\n'
        '\n'
        '§3.2 pide *"conservar la semántica del score nativo"* y §3.1 *"explicar la relación entre la métrica configurada, la normalización y el significado del score"*. Estas dos celdas son esa explicación, medida:\n'
        '\n'
        '- Con vectores **L2-normalizados**, `cosine` y `dot` dan el **mismo ranking** (el producto escalar de dos unitarios *es* el coseno), y `l2` también, porque `‖a−b‖² = 2 − 2·a·b` es monótona decreciente del producto escalar.\n'
        '- **Sin normalizar**, `dot` premia los vectores de norma grande y el ranking cambia. Ese es el fallo silencioso que la comprobación caza: si las tres filas normalizadas no coinciden, la normalización no se está aplicando y **todas las métricas del notebook quedan en duda**.\n'
        '\n'
        '### ⚠️ Cómo leer la tabla: solo un modelo demuestra algo\n'
        '\n'
        '`jina-v3` y `gemini-2` ya entregan unitarios (B.3), así que su fila *"sin normalizar"* sale idéntica a la normalizada: solo confirma que normalizar dos veces es idempotente. **La evidencia la aporta `granite-311m-r2`**, el único que llega crudo: ahí `cosine` no se mueve —normaliza internamente— mientras `dot` baja y `l2` sube, y las tres dejan de coincidir.\n'
        '\n'
        'Y llama la atención lo poco que hace falta para romperlo: las normas de granite se desvían milésimas de 1 y ya basta para reordenar resultados y mover el nDCG.\n'
        '\n'
        '> Si los tres modelos normalizaran en origen, esta comprobación pasaría sin detectar nada. Es el modo en que este tipo de verificación falla en silencio.',
    ),
    (
        "code",
        '# 📄 DATOS · vectores de 🔬 la muestra, evaluados con las 8 de desarrollo\n'
        'filas_semantica = []\n'
        'for (alias, contrato), vectores in VECTORES.items():\n'
        '    # Solo la rama `nativo`: lo que se demuestra aquí es una propiedad geométrica\n'
        '    # de los vectores (con norma 1, coseno·dot·l2 ordenan igual), y esa propiedad\n'
        '    # no depende del contrato con que se generaran. Recorrer las dos ramas\n'
        '    # duplicaría las filas de la tabla sin añadir ninguna información nueva.\n'
        '    if contrato != "nativo":\n'
        '        continue\n'
        '    dim = REGISTRO[alias]["dim_nativa"]\n'
        '    for normalizar in (True, False):\n'
        '        rankings_por_metrica, ndcg_por_metrica = {}, {}\n'
        '        for metric in ("cosine", "dot", "l2"):\n'
        '            informe, rankings = evaluar(vectores, dim, metric=metric, normalizar=normalizar)\n'
        '            rankings_por_metrica[metric] = rankings\n'
        '            ndcg_por_metrica[metric] = informe.summary["ndcg_at_10"]\n'
        '        iguales = (\n'
        '            rankings_por_metrica["cosine"] == rankings_por_metrica["dot"] == rankings_por_metrica["l2"]\n'
        '        )\n'
        '        # `mismo_ranking` se guarda en la fila, no solo se imprime: es la\n'
        '        # conclusión de la sección y tiene que sobrevivir a un reinicio.\n'
        '        filas_semantica += [\n'
        '            {"modelo": alias, "dim": dim, "normalizado": normalizar,\n'
        '             "metrica": metric, "ndcg_at_10": valor, "mismo_ranking": iguales}\n'
        '            for metric, valor in ndcg_por_metrica.items()\n'
        '        ]\n'
        '        print(f"{alias} · normalizado={normalizar}: ¿mismo ranking en las 3 métricas? {iguales}")\n'
        '\n'
        'semantica_score = pd.DataFrame(filas_semantica)\n'
        'semantica_score.pivot(index=["modelo", "normalizado"], columns="metrica", values="ndcg_at_10")',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '# F · Contra el baseline léxico — el requisito del enunciado §3.1\n'
        '\n'
        '> *"El trabajo debe comparar el sistema denso con, al menos, un baseline léxico o exacto."*\n'
        '\n'
        'NB01 dejó ese baseline en `artifacts/baseline_lexico.json`. Aquí se recupera **sobre el mismo corpus** (la muestra de 1.500), con las mismas 8 consultas, el mismo `k`, los mismos qrels y el mismo contrato de relevancia: sin esa igualdad no se compararían métodos sino entornos (Regla 2).\n'
        '\n'
        'La pregunta no es *"¿gana el denso?"* sino **"¿cuánto gana y a cambio de qué coste?"**: BM25 se construye en segundos sobre CPU, sin modelo ni base vectorial. Si la mejora fuera marginal, el argumento de negocio para montar esta infraestructura sería flojo — y decirlo con un número es mejor informe que esconderlo.',
    ),
    (
        "code",
        '# 📄 DATOS · artifacts/baseline_lexico.json, rama "muestra" — el mismo corpus 🔬\n'
        'import json\n'
        '\n'
        'baseline = json.loads(\n'
        '    (Path("..") / "artifacts" / "baseline_lexico.json").read_text(encoding="utf-8")\n'
        ')\n'
        'lexico_muestra = baseline["muestra"]["metricas"]\n'
        '\n'
        '# `modelo` y `contrato` viajan hasta aquí aunque no se muestren: F.1 los necesita\n'
        '# para volver a buscar los vectores del ganador en VECTORES. `sistema` es solo\n'
        '# la etiqueta legible.\n'
        'comparativa = pd.concat([\n'
        '    pd.DataFrame([\n'
        '        {"sistema": nombre, "familia": "léxico", "modelo": None, "contrato": None,\n'
        '         "dim": None, **metricas}\n'
        '        for nombre, metricas in lexico_muestra.items()\n'
        '    ]),\n'
        '    mejor_por_modelo.assign(familia="denso")[\n'
        '        ["sistema", "familia", "modelo", "contrato", "dim",\n'
        '         "precision_at_10", "recall_at_10", "mrr_at_10", "ndcg_at_10"]\n'
        '    ],\n'
        ']).sort_values("ndcg_at_10", ascending=False).reset_index(drop=True)\n'
        '\n'
        'mejor_lexico = max(m["ndcg_at_10"] for m in lexico_muestra.values())\n'
        'comparativa["delta_vs_mejor_lexico"] = (comparativa["ndcg_at_10"] - mejor_lexico).round(4)\n'
        'comparativa',
    ),
    (
        "code",
        '# 📄 DATOS · denso y léxico, ambos sobre 🔬 la muestra\n'
        '# La misma comparativa de arriba en barras agrupadas: las cuatro métricas en una\n'
        '# escala 0-1 común, que es la forma en que §3.1 pide leer denso frente a léxico.\n'
        'METRICAS = ["precision_at_10", "recall_at_10", "mrr_at_10", "ndcg_at_10"]\n'
        '\n'
        '\n'
        'def etiqueta(fila):\n'
        '    """El denso lleva su dimensión en el nombre: `granite-311m-r2@256` y `@768`\n'
        '    son sistemas distintos y la leyenda tiene que poder distinguirlos."""\n'
        '    if fila["familia"] == "léxico":\n'
        '        return fila["sistema"]\n'
        '    return f"{fila[\'sistema\']}@{int(fila[\'dim\'])}"\n'
        '\n'
        '\n'
        'sistemas = {\n'
        '    etiqueta(fila): {metrica: float(fila[metrica]) for metrica in METRICAS}\n'
        '    for _, fila in comparativa.iterrows()\n'
        '}\n'
        '\n'
        'plot_metric_comparison(\n'
        '    sistemas,\n'
        '    title="Denso frente al baseline léxico de NB01",\n'
        '    subtitle=(\n'
        '        f"{CORPUS_ID} ({len(corpus_textos)} docs) · {len(query_ids)} consultas · k={TOP_K} "\n'
        '        "· mismo corpus, mismos qrels, mismo contrato de relevancia"\n'
        '    ),\n'
        ').show()',
    ),
    (
        "markdown",
        '### F.1 · Dónde gana cada familia, consulta a consulta\n'
        '\n'
        'El agregado dice quién gana; esta tabla dice **por qué**. Lo interesante son las consultas donde el denso mejora mucho (vocabulario distinto al del catálogo) y las que empeora (el léxico acierta por coincidencia literal y el denso trae vecinos semánticamente próximos pero comercialmente distintos). Los dos casos alimentan la atribución de errores de NB09.',
    ),
    (
        "code",
        '# 📄 DATOS · 🔬 muestra · las 8 de desarrollo, una a una\n'
        'mejor_denso = comparativa.query("familia == \'denso\'").iloc[0]\n'
        'nombre_lexico = max(lexico_muestra, key=lambda n: lexico_muestra[n]["ndcg_at_10"])\n'
        'etiqueta_densa = f"{mejor_denso[\'sistema\']}@{int(mejor_denso[\'dim\'])}"\n'
        '\n'
        'ndcg_lexico = {\n'
        '    str(fila["query_id"]): fila["ndcg@10"]\n'
        '    for fila in baseline["muestra"]["por_consulta"][nombre_lexico]\n'
        '}\n'
        '# La clave de VECTORES es (modelo, contrato). `sistema` es la etiqueta legible\n'
        '# —"jina-v3 [sin_contrato]"— y no sirve para buscar aquí.\n'
        'informe_denso, _ = evaluar(\n'
        '    VECTORES[(mejor_denso["modelo"], mejor_denso["contrato"])], int(mejor_denso["dim"])\n'
        ')\n'
        '\n'
        'frente_a_frente = (\n'
        '    informe_denso.per_query_frame()\n'
        '    .assign(\n'
        '        query_id=lambda d: d["query_id"].astype(str),\n'
        '        **{nombre_lexico: lambda d: d["query_id"].map(ndcg_lexico)},\n'
        '    )\n'
        '    .rename(columns={"ndcg@10": etiqueta_densa})\n'
        '    [["query_id", nombre_lexico, etiqueta_densa]]\n'
        '    .merge(consultas.assign(query_id=consultas["query_id"].astype(str)), on="query_id")\n'
        ')\n'
        'frente_a_frente["delta"] = (\n'
        '    frente_a_frente[etiqueta_densa] - frente_a_frente[nombre_lexico]\n'
        ').round(4)\n'
        'frente_a_frente[["query_id", "query_text", nombre_lexico, etiqueta_densa, "delta"]].sort_values("delta")',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '# G · R02 · Aplicar D09b y dejar el artefacto\n'
        '\n'
        '**D09b se fijó en `config/config.yaml` antes de codificar nada.** Aplicarla como función y no a ojo es lo que hace verificable esa afirmación: el ganador sale de `apply_tolerance_rule`, que es determinista y está cubierta por tests.\n'
        '\n'
        '```yaml\n'
        'd09b_criterio_desempate:\n'
        '  forma: mas_barata_dentro_de_tolerancia\n'
        '  metrica_primaria: ndcg_at_10\n'
        '  tolerancia_tau: 0.02\n'
        '```\n'
        '\n'
        '1. `B` = mejor nDCG@10 de toda la tabla.\n'
        '2. Admisibles: las que están a menos de 0,02 de `B` — con 8 consultas, una diferencia menor no distingue dos sistemas.\n'
        '3. Entre las admisibles gana **la de menor dimensión**; a igualdad, mayor nDCG; después, menor tiempo de codificación.\n'
        '\n'
        '> ⚖️ **La celda produce la ordenación; R02 se ratifica sobre ella.** Si el resultado sorprende, el sitio para discutirlo es el criterio, no la tabla: cambiar la regla después de ver los números es exactamente lo que el enunciado penaliza.',
    ),
    (
        "code",
        '# 📄 DATOS · el barrido sobre 🔬 la muestra; la regla no vuelve a leer nada\n'
        'ordenadas = apply_tolerance_rule(\n'
        '    barrido, metrica="ndcg_at_10", tolerancia=TOLERANCIA_D09B,\n'
        '    coste="dim", desempates=("segundos",),\n'
        ')\n'
        '# `contrato` en las columnas mostradas: sin él, dos configuraciones distintas\n'
        '# del mismo modelo y dimensión aparecen como filas idénticas y no hay forma de\n'
        '# saber cuál ha ganado.\n'
        'ordenadas[["posicion_regla", "modelo", "contrato", "dim", "ndcg_at_10", "recall_at_10",\n'
        '           "mrr_at_10", "bytes_por_vector", "segundos", "admisible"]]',
    ),
    (
        "code",
        '# 📄 DATOS · escribe artifacts/comparativa_modelos.json y .md — todo de 🔬 la muestra\n'
        'def registros(frame):\n'
        '    """Filas como tipos JSON nativos: `to_dict` dejaría escalares de numpy."""\n'
        '    return json.loads(frame.to_json(orient="records"))\n'
        '\n'
        '\n'
        'artefacto = {\n'
        '    "configuracion": {\n'
        '        "corpus": CORPUS_ID,\n'
        '        "n_docs": len(corpus_textos),\n'
        '        "plantilla": PLANTILLA,\n'
        '        "campo": CAMPO,\n'
        '        "top_k": TOP_K,\n'
        '        "relevancia": {"E": 3, "S": 2, "C": 1, "I": 0},\n'
        '        "d09b": {"metrica": "ndcg_at_10", "tolerancia": TOLERANCIA_D09B, "coste": "dim"},\n'
        '    },\n'
        '    "modelos": {alias: {k: v for k, v in f.items() if k != "tasks"} for alias, f in REGISTRO.items()},\n'
        '    "errores_de_codificacion": ERRORES,\n'
        '    "costes_de_codificacion": list(COSTES.values()),\n'
        '    "modelos_codificados": [f"{a}[{c}]" for a, c in sorted(VECTORES)],\n'
        '    "barrido": registros(barrido),\n'
        '    "regla_d09b": registros(ordenadas),\n'
        '    "comparativa_con_lexico": registros(comparativa),\n'
        '    # §3.1 pide explicar la relación entre métrica, normalización y score. Sin\n'
        '    # esto, esa evidencia vivía solo en la salida de una celda.\n'
        '    "semantica_del_score": registros(semantica_score),\n'
        '    # La clave lleva el contrato: el barrido tiene ahora dos rankings por modelo\n'
        '    # y dimensión, y sin él uno sobrescribiría al otro en silencio.\n'
        '    "rankings": {\n'
        '        f"{alias}[{contrato}]@{dim}": r\n'
        '        for (alias, contrato, dim), r in RANKINGS_DENSOS.items()\n'
        '    },\n'
        '}\n'
        '\n'
        'destino = Path("..") / "artifacts" / "comparativa_modelos.json"\n'
        'destino.write_text(json.dumps(artefacto, indent=2, ensure_ascii=False, default=str), encoding="utf-8")\n'
        '\n'
        'markdown = Path("..") / "artifacts" / "comparativa_modelos.md"\n'
        'markdown.write_text(\n'
        '    "# Comparativa de modelos (NB02)\\n\\n"\n'
        '    f"Corpus: `{CORPUS_ID}` ({len(corpus_textos)} docs) · plantilla `{PLANTILLA}` · k={TOP_K}\\n\\n"\n'
        '    "## Barrido modelo x contrato x dimension\\n\\n" + barrido.to_markdown(index=False) + "\\n\\n"\n'
        '    "## Regla D09b aplicada\\n\\n" + ordenadas.to_markdown(index=False) + "\\n\\n"\n'
        '    "## Denso frente al baseline lexico de NB01\\n\\n" + comparativa.to_markdown(index=False) + "\\n",\n'
        '    encoding="utf-8",\n'
        ')\n'
        'print(f"Escrito {destino.name} ({destino.stat().st_size / 1024:.1f} KB) y {markdown.name}")',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '# H · El contrato no aporta lo mismo en todas las dimensiones\n'
        '\n'
        'La sección D midió la Δ del contrato **solo en la dimensión nativa**, y con ese único punto la conclusión parecía limpia: retirarlo mejora en los dos modelos que lo tienen. El barrido completo dice algo más incómodo.\n'
        '\n'
        '**La Δ cambia de signo al truncar.** En `gemini-2`:\n'
        '\n'
        '| dim | `nativo` | `sin_contrato` | Δ |\n'
        '|---:|---:|---:|---:|\n'
        '| 3072 | 0,7509 | 0,7765 | **−0,0256** |\n'
        '| 1536 | 0,7496 | 0,7750 | −0,0254 |\n'
        '| 768 | 0,7478 | 0,7718 | −0,0240 |\n'
        '| 512 | 0,7110 | 0,7410 | −0,0300 |\n'
        '| 256 | 0,7101 | 0,6843 | **+0,0258** |\n'
        '| 128 | 0,6648 | 0,5575 | **+0,1073** |\n'
        '\n'
        'Por encima de 512 el contrato estorba. Por debajo, ayuda — y a 128 la diferencia es de **0,107**, cinco veces la tolerancia de D09b. El cruce está entre 512 y 256.\n'
        '\n'
        '### Una hipótesis, no una conclusión\n'
        '\n'
        'La instrucción de tarea es **texto idéntico en los 1.500 documentos**, y en un modelo con MRL las primeras componentes concentran la estructura más gruesa y compartida del corpus, que es donde esa señal común pesa más.\n'
        '\n'
        '- **A dimensión completa** ese prefijo compartido es lastre: ocupa norma sin distinguir un producto de otro, así que quitarlo mejora.\n'
        '- **Al truncar fuerte** quedan casi solo esas componentes, y ahí la instrucción funciona como condicionamiento de tarea que sí orienta la búsqueda.\n'
        '\n'
        'No está comprobado —habría que mirar la energía por componente en ambas variantes—, así que queda como hipótesis explícita.\n'
        '\n'
        '### Consecuencia práctica\n'
        '\n'
        '**"Sin contrato es mejor" no es incondicional: vale a partir de 512.** El ganador de D09b (`gemini-2 [sin_contrato] @768`) cae con holgura en esa zona, así que **R02 no se ve afectada** — pero la condición hay que arrastrarla: si la memoria del índice empujara a bajar de dimensión (15.000 productos son 46 MB a 768 y 15 MB a 256), la decisión sobre el contrato habría que **revisarla, no heredarla**. Medir ese eje en un solo punto habría dejado la trampa puesta.',
    ),
    (
        "code",
        '# 📄 DATOS · el barrido sobre 🔬 la muestra, dimensión a dimensión\n'
        'from aurum.graficas import plot_contract_delta\n'
        '\n'
        'plot_contract_delta(\n'
        '    barrido,\n'
        '    tolerance=TOLERANCIA_D09B,\n'
        '    subtitle=(\n'
        '        f"{CORPUS_ID} ({len(corpus_textos)} docs) · {len(query_ids)} consultas · "\n'
        '        f"dentro de la banda gris (±{TOLERANCIA_D09B}) la diferencia no se distingue"\n'
        '    ),\n'
        ').show()\n'
        '\n'
        '# El cruce, en números: dónde deja de convenir retirar el contrato.\n'
        'cruce = (\n'
        '    barrido.pivot_table(index=["modelo", "dim"], columns="contrato", values="ndcg_at_10")\n'
        '    .dropna(subset=["nativo", "sin_contrato"])\n'
        '    .assign(delta=lambda d: (d["nativo"] - d["sin_contrato"]).round(4))\n'
        '    .assign(gana=lambda d: d["delta"].apply(\n'
        '        lambda x: "nativo" if x > TOLERANCIA_D09B\n'
        '        else ("sin_contrato" if x < -TOLERANCIA_D09B else "indistinguible")\n'
        '    ))\n'
        '    .reset_index()\n'
        '    .sort_values(["modelo", "dim"], ascending=[True, False])\n'
        ')\n'
        'cruce[["modelo", "dim", "nativo", "sin_contrato", "delta", "gana"]]',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '# I · P01 · El ganador sobre el catálogo completo\n'
        '\n'
        'Todo lo anterior está medido sobre **1.500 documentos**. El enunciado (§6, *Condiciones de comparabilidad*) dice que *"el catálogo completo es el recorrido evaluado; la muestra sirve para desarrollar y depurar"*, y esta sección cierra esa distancia para el ganador de D09b.\n'
        '\n'
        '### Por qué esto no es un detalle\n'
        '\n'
        'NB01 ya midió lo que pasa al multiplicar por diez los candidatos, con **los mismos juicios de relevancia**:\n'
        '\n'
        '| baseline | muestra (1.500) | completo (15.000) | caída |\n'
        '|---|---:|---:|---:|\n'
        '| BM25 | 0,6512 | 0,5088 | **−0,1424** |\n'
        '| TF-IDF | 0,5654 | 0,4129 | −0,1525 |\n'
        '\n'
        'Los qrels no cambian: aparecen 13.500 productos más compitiendo por las 10 posiciones que, al no estar juzgados, puntúan 0 (D04). Un sistema que los sube se desploma; uno que mantiene arriba los juzgados aguanta. **Es un test de precisión bajo distracción**, y no hay forma de aprobarlo desde la muestra.\n'
        '\n'
        '### Qué se ejecuta aquí, y qué no\n'
        '\n'
        'Solo se codifica el **ganador de D09b**, leído de `ordenadas` y no escrito a mano: si la regla cambiara de ganador, la sección lo sigue. El motivo es de coste, medido y no estimado a ojo:\n'
        '\n'
        '| configuración | 1.500 docs | 15.000 (×10) |\n'
        '|---|---:|---:|\n'
        '| `gemini-2` | ~50 s | **~8 min** ✅ |\n'
        '| `jina-v3` | ~5.750 s | ~16 h ❌ |\n'
        '| `granite-311m-r2` | ~17.490 s | ~49 h ❌ |\n'
        '\n'
        'La celda calcula esa extrapolación y **se niega a lanzar** cualquier codificación por encima de `LIMITE_HORAS`: con un modelo local imprimiría el coste y saltaría el paso en vez de dejar el kernel bloqueado media semana.\n'
        '\n'
        'Dos cosas abaratan la prueba: **las consultas no se recodifican** —su caché es independiente del corpus (`corpus_id="consultas_desarrollo"`), así que los 8 vectores ya están en disco— y **las tres dimensiones admisibles salen de la misma codificación**, porque truncar es gratis y 768, 1.536 y 3.072 se evalúan sin ninguna llamada extra.\n'
        '\n'
        '### Qué responde y qué no\n'
        '\n'
        'Responde a la pregunta del enunciado —**¿el denso sigue batiendo al léxico con el catálogo de verdad?**— pero no al orden **entre modelos densos** a esa escala: para eso habría que pagar las 65 horas de jina y granite. Queda como límite explícito: con `gemini-2` sacando 0,12 sobre BM25 y 0,24 sobre jina en la muestra, el riesgo de que el orden se invierta es bajo, pero *bajo* no es *cero*.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000) — AQUÍ CAMBIA EL CORPUS: hasta la\n'
        '#            sección H todo iba sobre la muestra de 1.500\n'
        'LIMITE_HORAS = 1.0            # por encima de esto la celda no lanza la codificación\n'
        'CORPUS_COMPLETO = "catalogo_productos"\n'
        '\n'
        '# El ganador se lee de la regla, no se escribe a mano: si D09b cambiara de\n'
        '# resultado, esta sección lo sigue sin tocar una línea.\n'
        'ganador = ordenadas.iloc[0]\n'
        'ALIAS_G = ganador["modelo"]\n'
        'CONTRATO_G = ganador["contrato"]\n'
        'DIM_G = int(ganador["dim"])\n'
        'ETIQUETA_G = f"{ALIAS_G} [{CONTRATO_G}]@{DIM_G}"\n'
        '\n'
        'textos_completo = completo[CAMPO].tolist()\n'
        'ids_completo = completo["product_id"].tolist()\n'
        '\n'
        '# Extrapolación lineal desde el coste ya medido sobre la muestra. Es fiable\n'
        '# porque codificar es proporcional al número de documentos: mismo modelo, mismo\n'
        '# lote, mismo hardware.\n'
        'segundos_muestra = float(\n'
        '    costes.query(\n'
        '        "alias == @ALIAS_G and contrato == @CONTRATO_G and tipo == \'document\'"\n'
        '    )["segundos"].iloc[0]\n'
        ')\n'
        'horas = segundos_muestra * len(textos_completo) / len(corpus_textos) / 3600\n'
        '\n'
        'print(f"Ganador de D09b : {ETIQUETA_G}")\n'
        'print(f"Corpus completo : {len(textos_completo)} documentos")\n'
        'print(f"Coste estimado  : ~{horas:.2f} h  ({segundos_muestra:.0f}s para {len(corpus_textos)} docs)")\n'
        '\n'
        'if horas > LIMITE_HORAS:\n'
        '    vectores_completo = None\n'
        '    print(\n'
        '        f"\\n⏭️  Por encima del límite de {LIMITE_HORAS} h: no se lanza.\\n"\n'
        '        "    P01 queda abierto para esta configuración. Sube LIMITE_HORAS si\\n"\n'
        '        "    quieres pagarlo, pero hazlo sabiendo cuántas horas son."\n'
        '    )\n'
        'else:\n'
        '    encoder = fabricar(ALIAS_G)\n'
        '    try:\n'
        '        resultado = encode_corpus(\n'
        '            encoder, textos_completo, corpus_id=CORPUS_COMPLETO,\n'
        '            kind="document", contract=CONTRATO_G,\n'
        '            batch_size=32, cache_dir=CACHE,\n'
        '        )\n'
        '    finally:\n'
        '        del encoder\n'
        '        gc.collect()\n'
        '    vectores_completo = resultado.vectors\n'
        '    origen = "desde caché" if resultado.stats.desde_cache else f"{resultado.stats.segundos:.0f}s"\n'
        '    print(f"\\n✅ {vectores_completo.shape[0]} vectores de {vectores_completo.shape[1]} dims ({origen})")',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000) · 8 de desarrollo · baseline_lexico.json\n'
        '#            rama "completo" — mismo corpus en los dos lados de la comparación\n'
        'if vectores_completo is None:\n'
        '    print("P01 sin cerrar: la celda anterior no codificó el catálogo completo.")\n'
        'else:\n'
        '    # Las consultas NO se recodifican: su caché es independiente del corpus.\n'
        '    vectores_query = VECTORES[(ALIAS_G, CONTRATO_G)]["query"]\n'
        '\n'
        '    def evaluar_completo(dim):\n'
        '        """Igual que `evaluar`, pero contra los 15.000 IDs del catálogo entero.\n'
        '\n'
        '        No se reutiliza `evaluar` porque aquella cerró sobre `corpus_ids`, que\n'
        '        son los 1.500 de la muestra: pasarle estos vectores devolvería IDs\n'
        '        equivocados sin dar ningún error."""\n'
        '        docs = truncate_dim(vectores_completo, dim)\n'
        '        queries = truncate_dim(vectores_query, dim)\n'
        '        retriever = DenseRetriever(docs, ids_completo, metric="cosine")\n'
        '        rankings = rank_queries_dense(retriever, query_ids, queries, k=TOP_K)\n'
        '        return evaluate_rankings(rankings, qrels, k=TOP_K), rankings\n'
        '\n'
        '    lexico_completo = baseline["completo"]["metricas"]\n'
        '    ndcg_muestra = {n: m["ndcg_at_10"] for n, m in lexico_muestra.items()}\n'
        '\n'
        '    filas = []\n'
        '    RANKINGS_COMPLETO = {}\n'
        '    # Las tres dimensiones admisibles de D09b salen de la misma codificación:\n'
        '    # truncar es gratis, así que verlas todas no cuesta ninguna llamada extra.\n'
        '    for dim in sorted(ordenadas.query("admisible")["dim"].unique(), reverse=True):\n'
        '        informe, rankings = evaluar_completo(int(dim))\n'
        '        RANKINGS_COMPLETO[int(dim)] = rankings\n'
        '        etiqueta = f"{ALIAS_G} [{CONTRATO_G}]@{int(dim)}"\n'
        '        ndcg_muestra[etiqueta] = float(\n'
        '            barrido.query(\n'
        '                "modelo == @ALIAS_G and contrato == @CONTRATO_G and dim == @dim"\n'
        '            )["ndcg_at_10"].iloc[0]\n'
        '        )\n'
        '        filas.append({"sistema": etiqueta, "familia": "denso", **informe.summary})\n'
        '\n'
        '    filas += [\n'
        '        {"sistema": nombre, "familia": "léxico", **metricas}\n'
        '        for nombre, metricas in lexico_completo.items()\n'
        '    ]\n'
        '\n'
        '    completo_vs_lexico = (\n'
        '        pd.DataFrame(filas).sort_values("ndcg_at_10", ascending=False).reset_index(drop=True)\n'
        '    )\n'
        '    completo_vs_lexico["muestra_1500"] = completo_vs_lexico["sistema"].map(ndcg_muestra)\n'
        '    completo_vs_lexico["caida_al_escalar"] = (\n'
        '        completo_vs_lexico["ndcg_at_10"] - completo_vs_lexico["muestra_1500"]\n'
        '    ).round(4)\n'
        '\n'
        '    mejor_lexico_completo = max(m["ndcg_at_10"] for m in lexico_completo.values())\n'
        '    ventaja = (\n'
        '        completo_vs_lexico.query("familia == \'denso\'").iloc[0]["ndcg_at_10"]\n'
        '        - mejor_lexico_completo\n'
        '    )\n'
        '    print(\n'
        '        f"Ventaja del mejor denso sobre el mejor léxico:\\n"\n'
        '        f"  muestra  (1.500) : {barrido.iloc[0][\'ndcg_at_10\'] - max(ndcg_muestra[n] for n in lexico_muestra):+.4f}\\n"\n'
        '        f"  completo (15.000): {ventaja:+.4f}"\n'
        '    )\n'
        '    display(\n'
        '        completo_vs_lexico[\n'
        '            ["sistema", "familia", "muestra_1500", "ndcg_at_10",\n'
        '             "caida_al_escalar", "recall_at_10", "mrr_at_10", "precision_at_10"]\n'
        '        ]\n'
        '    )\n'
        '\n'
        '    plot_metric_comparison(\n'
        '        {\n'
        '            fila["sistema"]: {m: float(fila[m]) for m in METRICAS}\n'
        '            for _, fila in completo_vs_lexico.iterrows()\n'
        '        },\n'
        '        title="Denso frente al léxico — catálogo completo",\n'
        '        subtitle=(\n'
        '            f"{CORPUS_COMPLETO} ({len(textos_completo)} docs) · {len(query_ids)} consultas "\n'
        '            f"· k={TOP_K} · mismos qrels que sobre la muestra"\n'
        '        ),\n'
        '    ).show()',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '# J · Robustez entre formulaciones — el slice que el barrido no ve\n'
        '\n'
        'Las secciones A–I eligen el modelo con **8 consultas de desarrollo**, el único conjunto con juicios pero con un punto ciego: un modelo puede ganar por afinidad léxica con esas ocho. Queda sin responder si devuelve **los mismos productos cuando la misma intención se escribe de otra manera**, que es lo que pasa en producción.\n'
        '\n'
        'Las 12 ciegas son **4 intenciones × 3 formulaciones** y no necesitan juicios para eso: el Jaccard@10 entre las tres formas de pedir lo mismo mide la estabilidad directamente.\n'
        '\n'
        '```\n'
        'EVAL-100455-direct    "taladro 24v batería"\n'
        'EVAL-100455-context   "taladro sin cable de 24 voltios que venga con su batería"\n'
        'EVAL-100455-semantic  "quiero una herramienta inalámbrica potente para perforar sin depender de un enchufe"\n'
        '```\n'
        '\n'
        '> 🔍 **Por qué llega después de R02.** La consistencia estaba medida para los baselines (NB01) y para las plantillas (NB03), pero no **entre modelos**, que es una de las patas del criterio *"media y por slices"*. Se añade como comprobación de una decisión ya tomada: si el ganador por nDCG resultara el más inestable, R02 se revisa en vez de esconderse. Se mide sobre la muestra porque es donde están codificados los modelos que compiten (Regla 2); la cifra del ganador sobre el catálogo completo está en NB03.\n'
        '\n'
        '> ⏭️ **`jina-v3` no entra aquí**, y queda declarado en su ficha del registro en vez de saltado en una celda: codificar sus 12 ciegas obliga a cargar 572M de parámetros —2,3 GB de los 7,9 de la máquina— y el barrido ya lo dejó **último de los tres**, por debajo incluso del baseline léxico. La pregunta es cuál de los dos que de verdad compiten aguanta mejor un cambio de formulación, y añadir al descartado no cambiaría esa respuesta, solo el tiempo de ejecución. Es una limitación del equipo, asumida a la vista de un orden ya medido.',
    ),
    (
        "code",
        '# 📄 DATOS · 🔬 muestra (1.500 candidatos) · consultas_evaluacion.csv (12 ciegas, sin juicios)\n'
        '# Cada modelo se mide en SU mejor configuración —la que ganó su rama en el\n'
        '# barrido—, no todos a la misma dimensión: forzar un valor común mediría el\n'
        '# efecto de la dimensión, no el del modelo.\n'
        'consistencia_modelos = []\n'
        'for _, fila in mejor_por_modelo.iterrows():\n'
        '    vectores = VECTORES[(fila["modelo"], fila["contrato"])]\n'
        '    if "query_ciegas" not in vectores:\n'
        '        motivo = (\n'
        '            "excluido a propósito (REGISTRO: ciegas=False)"\n'
        '            if not REGISTRO[fila["modelo"]].get("ciegas", True)\n'
        '            else "sin las ciegas codificadas: re-ejecuta su celda B.2"\n'
        '        )\n'
        '        print(f"⏭️  {fila[\'sistema\']}: {motivo}")\n'
        '        continue\n'
        '    dim = int(fila["dim"])\n'
        '    retriever = DenseRetriever(\n'
        '        truncate_dim(vectores["document"], dim), corpus_ids, metric="cosine"\n'
        '    )\n'
        '    rankings = rank_queries_dense(\n'
        '        retriever, ciegas["evaluation_id"].tolist(),\n'
        '        truncate_dim(vectores["query_ciegas"], dim), k=TOP_K,\n'
        '    )\n'
        '    tabla = formulation_consistency(rankings, k=TOP_K)\n'
        '    columnas = [c for c in tabla.columns if c.startswith("jaccard_")]\n'
        '    consistencia_modelos.append({\n'
        '        "sistema": fila["sistema"],\n'
        '        "dim": dim,\n'
        '        "ndcg_at_10": fila["ndcg_at_10"],\n'
        '        **{c: round(float(tabla[c].mean()), 4) for c in columnas},\n'
        '        "jaccard_medio": round(float(tabla[columnas].to_numpy().mean()), 4),\n'
        '    })\n'
        '\n'
        'consistencia_modelos = (\n'
        '    pd.DataFrame(consistencia_modelos)\n'
        '    .sort_values("jaccard_medio", ascending=False)\n'
        '    .reset_index(drop=True)\n'
        ')\n'
        'consistencia_modelos',
    ),
    (
        "markdown",
        '### Cómo se lee esta tabla\n'
        '\n'
        'La columna que importa no es solo `jaccard_medio` sino **el par `direct` ↔ `semantic`**: las dos formulaciones más lejanas, una en palabras del catálogo y otra en las de un cliente que describe lo que necesita. Ahí se cae un sistema que en realidad depende del vocabulario.\n'
        '\n'
        'Dos lecturas posibles y qué significa cada una:\n'
        '\n'
        '| Si… | Entonces |\n'
        '|---|---|\n'
        '| el orden coincide con el de nDCG@10 | la ventaja del ganador no era afinidad con las 8 consultas: generaliza |\n'
        '| un modelo gana en nDCG pero pierde aquí | su ventaja **es sobreajuste** a la superficie léxica del conjunto de desarrollo, y R02 hay que revisarla |\n'
        '\n'
        '> ⚠️ **Cuánto pesa esto.** Son 4 intenciones, no 400: sirve para detectar un derrumbe, no para separar dos modelos que queden cerca. Un Jaccard de 0,50 frente a 0,55 no distingue nada; uno de 0,50 frente a 0,05 sí.',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '# K · El coste del camino online — indexar una vez, buscar siempre\n'
        '\n'
        'Todo lo medido hasta aquí es **coste de indexación**: un lote grande, una vez. Falta la otra mitad, la que se paga en cada búsqueda y para siempre.\n'
        '\n'
        '**Los números de B.2d no sirven para esto**: allí las 8 consultas se codificaron en un solo lote (0,43 s las ocho con `gemini-2`), y un lote amortiza el viaje de red entre ellas, que es justo lo que domina el camino online. Una búsqueda real codifica **una** consulta y paga el round-trip entero.\n'
        '\n'
        'Hay además una asimetría entre los dos regímenes que el barrido no ve, porque no es una propiedad del vector:\n'
        '\n'
        '| | Modelo local | Modelo por API |\n'
        '|---|---|---|\n'
        '| Indexar | horas de CPU, 0 $ | minutos, $ por token |\n'
        '| Buscar | ms de CPU, 0 $ | ms de red, $ por token |\n'
        '| Si se cae | no se cae: es un fichero | el buscador entero se queda sin codificar consultas |',
    ),
    (
        "markdown",
        '## K.1 · Latencia de **una** consulta — ⏱️ una celda por modelo\n'
        '\n'
        'Mismo patrón que B.2: cada modelo en su celda, porque los locales cargan pesos en memoria (`jina-v3` ~2,3 GB) y no caben dos a la vez. `gemini-2` es barato —son llamadas a la API— y los dos locales son opcionales: sirven de referencia, no deciden nada.\n'
        '\n'
        'El calentamiento no se contabiliza: la primera llamada paga el TLS o la reserva de memoria, y eso no se repite por consulta.',
    ),
    (
        "code",
        '# 📄 DATOS · consultas_desarrollo.csv (8), aquí solo como textos que codificar\n'
        'LATENCIAS = {}\n'
        '\n'
        '\n'
        'def latencia(alias, contrato="nativo", repeticiones=20):\n'
        '    """Mide y libera. Devuelve None si el modelo no está disponible."""\n'
        '    try:\n'
        '        encoder = fabricar(alias)\n'
        '    except Exception as error:\n'
        '        print(f"⛔ {alias}: {type(error).__name__} — {error}")\n'
        '        return None\n'
        '    try:\n'
        '        informe = measure_encode_latency(\n'
        '            encoder, query_textos, kind="query", contract=contrato,\n'
        '            repeticiones=repeticiones, calentamiento=2,\n'
        '        )\n'
        '    finally:\n'
        '        del encoder\n'
        '        gc.collect()\n'
        '    LATENCIAS[(alias, contrato)] = informe\n'
        '    print(f"✅ {alias} [{contrato}]: p50 {informe[\'ms_p50\']} ms · p95 {informe[\'ms_p95\']} ms")\n'
        '    return informe\n'
        '\n'
        '\n'
        '# El ganador de R02, en la rama de contrato que eligió la regla.\n'
        'latencia("gemini-2", "sin_contrato")',
    ),
    (
        "markdown",
        '### K.1b · Los dos locales — ⏱️ **opcional**, carga los pesos (1-2 min cada uno)\n'
        '\n'
        'No cambian ninguna decisión: R02 ya está tomada. Se miden porque son el contrafactual del régimen de coste — *"¿cuánto costaría no depender de una API?"*— y esa cifra es parte del argumento del informe, no un adorno.',
    ),
    (
        "code",
        '# 📄 DATOS · consultas_desarrollo.csv (8), aquí solo como textos que codificar\n'
        'latencia("granite-311m-r2")   # Apache-2.0, el más ligero de los dos\n'
        '# latencia("jina-v3")        # descoméntalo si hay RAM libre: 572M de parámetros',
    ),
    (
        "code",
        '# 📄 DATOS · ninguno: recoge las latencias ya medidas\n'
        'if LATENCIAS:\n'
        '    display(pd.DataFrame(LATENCIAS.values())[\n'
        '        ["modelo", "contrato", "n_llamadas", "ms_p50", "ms_p95", "ms_min", "ms_max"]\n'
        '    ])\n'
        'else:\n'
        '    print("Sin medidas: ejecuta al menos una celda de K.1.")',
    ),
    (
        "markdown",
        '## K.2 · Cuánto cuesta en dinero\n'
        '\n'
        '`gemini-embedding-2` cobra **0,20 $ por millón de tokens** de entrada en la tarifa estándar (0,10 $ con la Batch API, que sirve para indexar pero no para responder a un usuario). El precio es un dato declarado del proveedor, no calculado aquí: puede cambiar, y esconderlo dentro de una celda lo volvería invisible.\n'
        '\n'
        '> ⚠️ **La cuenta de tokens es una aproximación.** Gemini no publica su tokenizador, así que se usa el de `jina-v3` sobre el mismo texto (sección A): dos vocabularios no parten el texto igual y el total tiene un margen de decenas por ciento. Da igual para lo que se decide aquí — la conclusión son céntimos, y lo seguiría siendo con el doble o la mitad.\n'
        '\n'
        '**Unidades de las dos tablas**, que mezclan dólares, consultas y ratios:\n'
        '\n'
        '| Columna | Unidad | Qué es |\n'
        '|---|---|---|\n'
        '| `indexacion_completa_usd` | **USD** | codificar las 15.000 fichas, una vez |\n'
        '| `por_consulta_usd` | **USD** | codificar **una** consulta de usuario |\n'
        '| `por_1000_consultas_usd` | **USD** | lo mismo × 1.000, para que la cifra se lea |\n'
        '| `consultas_equivalentes_a_reindexar` | **consultas** | cuántas búsquedas cuestan lo que una reindexación completa |\n'
        '| `consultas_mes` | **consultas/mes** | volumen supuesto, no medido |\n'
        '| `gasto_mes_usd` | **USD/mes** | lo que costaría ese volumen |\n'
        '| `veces_el_coste_de_indexar` | **ratio** (adimensional) | ese gasto mensual dividido por el de indexar |\n'
        '\n'
        'USD es la moneda en la que factura el proveedor. La conversión a euros no se hace a propósito: el tipo de cambio del día metería ruido en una cifra que ya es una estimación.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 tokens de catalogo_productos.csv (15.000, medidos en A.4)\n'
        '#            + consultas_desarrollo.csv (8) y consultas_evaluacion.csv (12)\n'
        '# `longitudes` viene de A.4: tokens de cada ficha del catálogo completo con el\n'
        '# tokenizador de referencia. `tokens_por_consulta` sale de las consultas reales,\n'
        '# no de una estimación: las 8 de desarrollo más las 12 ciegas.\n'
        'PRECIO_POR_MILLON = 0.20   # $/1M tokens — config.yaml -> nb02_modelo.coste_api\n'
        '\n'
        'tokens_consulta = token_lengths(\n'
        '    query_textos + ciegas_textos, tokenizadores[alias_referencia][0]\n'
        ')\n'
        '\n'
        'coste = api_cost_report(\n'
        '    tokens_indexacion=float(longitudes.sum()),\n'
        '    tokens_por_consulta=float(tokens_consulta.mean()),\n'
        '    precio_por_millon=PRECIO_POR_MILLON,\n'
        ')\n'
        '\n'
        'print(f"Catálogo completo : {longitudes.sum():>10,.0f} tokens")\n'
        'print(f"Consulta media    : {tokens_consulta.mean():>10,.1f} tokens")\n'
        'pd.DataFrame([coste])',
    ),
    (
        "code",
        '# 📄 DATOS · ninguno: aritmética sobre el informe de coste de la celda anterior\n'
        '# El gasto por consulta suelta es demasiado pequeño para decir nada. Lo que\n'
        '# ordena las dos partidas es a qué volumen de búsquedas el coste de buscar\n'
        '# alcanza al de indexar el catálogo entero.\n'
        'pd.DataFrame([\n'
        '    {\n'
        '        "consultas_mes": volumen,\n'
        '        "gasto_mes_usd": round(coste["por_consulta_usd"] * volumen, 2),\n'
        '        "veces_el_coste_de_indexar": round(\n'
        '            coste["por_consulta_usd"] * volumen / coste["indexacion_completa_usd"], 2\n'
        '        ),\n'
        '    }\n'
        '    for volumen in (10_000, 100_000, 1_000_000, 10_000_000)\n'
        '])',
    ),
    (
        "markdown",
        '### Qué sale de la máquina, y en qué momento\n'
        '\n'
        'El mismo eje que el coste, mirado como exposición en vez de como factura. No es lo mismo lo que se envía al indexar que lo que se envía al buscar, y el enunciado pide pesar la dependencia del proveedor:\n'
        '\n'
        '| Momento | Qué viaja a Google | Cuántas veces |\n'
        '|---|---|---|\n'
        '| Indexación | el texto de las 15.000 fichas del catálogo | una vez, y otra por cada reindexado |\n'
        '| Búsqueda | **la consulta que ha escrito el usuario** | una por búsqueda, para siempre |\n'
        '| Altas y actualizaciones (NB08) | el texto de los productos nuevos o modificados | una por evento |\n'
        '\n'
        'La ficha de producto es información pública: un catálogo de comercio electrónico existe para ser visto. **La consulta no.** Es lo que una persona concreta estaba buscando en un momento concreto, y es el único dato del sistema que no era público antes de entrar en él. La dependencia de la API se paga ahí, no en el catálogo.\n'
        '\n'
        '> ⚖️ **Esto es evidencia, no una decisión.** Qué se concluye de ella —si el riesgo es asumible, si un modelo local valdría la diferencia, qué se le cuenta al usuario— va en el README: **D11** ya aceptó la dependencia de API, pero la aceptó pesando el catálogo, no las consultas.',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '# L · Versionado del encoder y deriva silenciosa\n'
        '\n'
        'El modelo forma parte del contrato del índice: los 15.000 vectores solo son comparables entre sí porque salieron del mismo encoder. Mantener esa condición es fácil con pesos locales y **no está garantizado** con un modelo servido por API.\n'
        '\n'
        '| | Pesos locales | Endpoint gestionado |\n'
        '|---|---|---|\n'
        '| Qué fija la versión | el fichero descargado, con su hash | el identificador `gemini-embedding-2`, que el proveedor puede reapuntar |\n'
        '| Si cambia | no cambia si no se descarga otro | los vectores nuevos dejan de vivir en el mismo espacio que los indexados |\n'
        '| Cómo se entera uno | — | **por ninguna excepción**: los vecinos simplemente empeoran |\n'
        '\n'
        'Ese modo de fallo es el que importa: silencioso, y con un síntoma —resultados peores— que se confunde con una representación mediocre. El artefacto de embeddings guarda `model_id`, dimensión, dtype y el SHA-256 del **corpus**, pero no hay huella del **modelo**: la API no devuelve ninguna versión que anotar.\n'
        '\n'
        'Sí se puede comprobar por sus efectos: recodificar unas fichas ya indexadas y mirar si dan el mismo vector. Con vectores unitarios, coseno ≈ 1 significa que el modelo no ha cambiado.\n'
        '\n'
        '**Cuántas fichas, y por qué no es un muestreo.** Un canario no estima una proporción: si el proveedor cambia el modelo no cambian el 5 % de los vectores, cambian **todos**, y con una sola ficha se detectaría. El tamaño compra variedad —para no pasar por alto un cambio parcial que mueva unas zonas del espacio y no otras— y sale casi gratis: el **5 % del corpus indexado** son 750 fichas del catálogo completo, unos veinte segundos y menos de una décima de dólar. Un guardián que tarda un minuto acaba saltándose justo antes de los eventos, que es cuando hace falta.\n'
        '\n'
        'Se toman **las primeras N**, y eso ya da la variedad: el catálogo no está ordenado por ningún atributo, así que las primeras 750 reproducen el conjunto —color vacío al 40,4 % frente al 37,4 % global, marca vacía al 3,7 % frente al 4,4 %, mediana de 1.012 caracteres frente a 936, 675 marcas distintas—. Y son **deterministas**, que es lo que permite compararlas contra los vectores guardados.',
    ),
    (
        "code",
        '# 📄 DATOS · 🔬 muestra: 75 fichas (el 5 % de 1.500). Sobre el índice real serían\n'
        '#            750, el 5 % de 📚 catalogo_productos.csv\n'
        '# El 5 % del corpus indexado: 75 fichas aquí sobre la muestra, 750 sobre el\n'
        '# catálogo completo, que es donde el canario vigila de verdad (NB08). El suelo\n'
        '# evita que en un corpus pequeño la comprobación se quede en cuatro fichas.\n'
        'CANARIO = max(16, round(0.05 * len(corpus_textos)))\n'
        'TOLERANCIA_DERIVA = 1e-3   # se exige un 99,9 % de parecido con el vector guardado\n'
        '\n'
        'clave_ganador = ("gemini-2", "sin_contrato")\n'
        'if clave_ganador in VECTORES:\n'
        '    encoder = fabricar("gemini-2")\n'
        '    try:\n'
        '        deriva = drift_check(\n'
        '            encoder,\n'
        '            corpus_textos[:CANARIO],\n'
        '            VECTORES[clave_ganador]["document"][:CANARIO],\n'
        '            kind="document", contract="sin_contrato",\n'
        '            tolerancia=TOLERANCIA_DERIVA,\n'
        '        )\n'
        '    finally:\n'
        '        del encoder\n'
        '        gc.collect()\n'
        '    display(pd.DataFrame([deriva]))\n'
        '    print(\n'
        '        "✅ El modelo sigue siendo el mismo que indexó el catálogo."\n'
        '        if deriva["sin_deriva"]\n'
        '        else "🚨 El encoder ha cambiado: reindexa ANTES de escribir nada más en el índice."\n'
        '    )\n'
        'else:\n'
        '    print("Sin vectores de gemini-2: ejecuta B.2c antes de esta celda.")',
    ),
    (
        "markdown",
        '### Qué hereda NB04 de esto\n'
        '\n'
        'La comprobación anterior es un diagnóstico, no una estrategia. La estrategia son dos decisiones del notebook siguiente, al definir el esquema: cuestan cero si se toman entonces y mucho si hay que retrofitarlas.\n'
        '\n'
        '1. **El nombre de la colección lleva el contrato del índice** — modelo, plantilla y dimensión. Una colección llamada `productos` no dice con qué se codificó; una llamada `productos__gemini2__A4__768` hace imposible ingerir en ella vectores de otro encoder por descuido.\n'
        '2. **La migración es un cambio de colección, no un borrado.** Reindexar sobre la colección viva deja el buscador respondiendo con un índice a medias; construir la nueva al lado y cambiar el puntero cuando esté completa no tiene ventana de indisponibilidad, y permite volver atrás si algo sale mal.\n'
        '\n'
        'Y una tercera, de operación: **el canario se ejecuta antes de aplicar los eventos de NB08**, que es el único momento en que se escriben vectores nuevos junto a los viejos. Si el encoder ha cambiado, escribir esas altas contamina el índice en silencio.',
    ),
    (
        "code",
        '# 📄 DATOS · actualiza artifacts/comparativa_modelos.json\n'
        '# Los tres criterios de arquitectura que faltaban, al artefacto. Se añaden sobre\n'
        '# el JSON que dejó la sección G en vez de escribir otro fichero: la comparativa\n'
        '# de modelos es una sola cosa, y partirla obligaría a cruzarlas para leerla.\n'
        'destino = Path("..") / "artifacts" / "comparativa_modelos.json"\n'
        'artefacto = json.loads(destino.read_text(encoding="utf-8"))\n'
        '\n'
        'artefacto["criterios_arquitectura"] = {\n'
        '    "consistencia_ciegas": registros(consistencia_modelos),\n'
        '    "latencia_por_consulta_ms": list(LATENCIAS.values()),\n'
        '    "coste_api": {\n'
        '        "precio_por_millon_usd": PRECIO_POR_MILLON,\n'
        '        "tokenizador_usado": alias_referencia,\n'
        '        "tokens_catalogo_completo": int(longitudes.sum()),\n'
        '        "tokens_por_consulta_medio": round(float(tokens_consulta.mean()), 1),\n'
        '        **coste,\n'
        '    },\n'
        '    "deriva_del_encoder": globals().get("deriva"),\n'
        '}\n'
        '\n'
        'destino.write_text(\n'
        '    json.dumps(artefacto, indent=2, ensure_ascii=False, default=str), encoding="utf-8"\n'
        ')\n'
        'print(f"Actualizado {destino.name} con criterios_arquitectura "\n'
        '      f"({destino.stat().st_size / 1024:.1f} KB)")',
    ),
]

NB03_REPRESENTACION = [
    ("markdown", '# NB03 · ¿Qué texto codifico? — plantillas y representación'),
    (
        "markdown",
        '**El montaje de NB02, al revés.** Allí el texto estaba congelado en A0 y variaba el modelo; aquí el modelo queda congelado en el ganador de R02 y **lo único que cambia es el texto** (Regla 1).\n'
        '\n'
        '> 🔒 **Congelado**: `gemini-embedding-2` · contrato `sin_contrato` · **dim 768** · métrica `cosine` · normalización L2 explícita al truncar · k=10.\n'
        '\n'
        '### La pregunta\n'
        '\n'
        'Un embedding resume el significado de **todo** lo que entra. Si de los ~1.300 caracteres que tiene `text` de media la mayoría es prosa comercial —*"perfecto para regalo, calidad premium, ideal para toda ocasión"*—, el vector se acerca al lenguaje genérico de cualquier producto y se aleja de lo que hace distinto a *este*. **Menos texto puede recuperar mejor**: es una hipótesis, y aquí se mide.\n'
        '\n'
        '### Qué NO entra: la familia C (D07)\n'
        '\n'
        'El chunking queda descartado **por medición, no por falta de tiempo**: NB02·A midió `pct_supera_ventana = 0` sobre los 15.000 registros, con un máximo de 1.972 tokens frente a ventanas de 8.192 y 32.768. Partir en trozos resuelve un problema que aquí no existe.\n'
        '\n'
        'Consecuencia: el punto de la base vectorial sigue siendo `record_id` en relación **1:1** con el producto, así que el esquema de NB04 se mantiene simple, la idempotencia no necesita borrar chunks huérfanos, el top-10 no necesita deduplicar y **D08 queda sin aplicar**.',
    ),
    (
        "code",
        'import gc\n'
        'import json\n'
        'import time\n'
        'from pathlib import Path\n'
        '\n'
        'import pandas as pd\n'
        '\n'
        'import sys\n'
        'sys.path.insert(0, str(Path("..") / "src"))\n'
        '\n'
        'from dotenv import load_dotenv\n'
        'import os\n'
        '\n'
        'from aurum.busqueda import DenseRetriever, rank_queries_dense\n'
        'from aurum.datos import relevant_field_nullity\n'
        'from aurum.embeddings import GeminiEncoder, encode_corpus, truncate_dim, vector_health\n'
        'from aurum.evaluacion import (\n'
        '    apply_tolerance_rule,\n'
        '    evaluate_rankings,\n'
        '    formulation_consistency,\n'
        '    per_query_delta,\n'
        '    qrels_from_judgements,\n'
        ')\n'
        'from aurum.graficas import plot_effect_vs_exposure, plot_metric_comparison\n'
        'from aurum.plantillas import (\n'
        '    CONTROLES,\n'
        '    TEMPLATES,\n'
        '    candidatas,\n'
        '    corpus_context,\n'
        '    render_template,\n'
        '    template_stats,\n'
        ')\n'
        '\n'
        'load_dotenv(Path("..") / ".env")\n'
        '\n'
        'DATA = Path("..") / "data"\n'
        'CACHE = Path("..") / "artifacts" / "embeddings"\n'
        '\n'
        '# Espejo de config.yaml -> nb03_representacion.modelo_congelado\n'
        'MODELO, CONTRATO, DIM, METRICA = "gemini-embedding-2", "sin_contrato", 768, "cosine"\n'
        'CORPUS_ID = "catalogo_muestra"      # condición 3 del plan\n'
        'TOP_K = 10\n'
        'TOLERANCIA_R01 = 0.01               # r01_criterio_desempate.tolerancia\n'
        '\n'
        'muestra = pd.read_csv(DATA / "catalogo_muestra.csv")\n'
        'consultas = pd.read_csv(DATA / "consultas_desarrollo.csv")\n'
        'ciegas = pd.read_csv(DATA / "consultas_evaluacion.csv")\n'
        'relevancias = pd.read_csv(DATA / "relevancias_desarrollo.csv")\n'
        'qrels = qrels_from_judgements(relevancias)\n'
        '\n'
        'corpus_ids = muestra["product_id"].tolist()\n'
        'query_ids = [str(q) for q in consultas["query_id"]]\n'
        'query_textos = consultas["query_text"].tolist()\n'
        '\n'
        'CONTEXTO = corpus_context(muestra)\n'
        '\n'
        'print(f"catálogo : {len(muestra)} productos")\n'
        'print(f"corte A4 : {CONTEXTO.a4_chars} caracteres (derivado del corpus, ver A.2)")\n'
        'print(f"consultas: {len(query_textos)} de desarrollo · {len(ciegas)} ciegas")\n'
        'print(f"congelado: {MODELO} [{CONTRATO}] @{DIM} · {METRICA}")',
    ),
    (
        "markdown",
        '## A · Las siete plantillas (D06)\n'
        '\n'
        'Cada plantilla es una receta para construir la cadena que se codifica. Viven en `aurum.plantillas` y no en el notebook porque son **el objeto de estudio**: tienen que poder probarse sin red y sin modelo (`tests/test_plantillas.py`).\n'
        '\n'
        '| | Texto | Por qué está |\n'
        '|---|---|---|\n'
        '| **A0** | `text` tal cual | La que NB02 tuvo congelada. Sin ella no hay punto de comparación |\n'
        '| **A1** | solo `title` | El extremo opuesto, y la única que nunca podría truncarse |\n'
        '| **A2** | `title` + marca + color, sin etiquetas | Aísla si lo que aporta A3 es la información o la nomenclatura |\n'
        '| **A3** | con etiquetas (`Marca: X. Color: Y.`), omitiendo vacíos | Replica la nomenclatura del `text` de origen, aplicando **D02** |\n'
        '| **A3n** | igual que A3 pero rellenando los vacíos | **El control de D02** — ver abajo |\n'
        '| **A4** | recorte de `text` por la **mediana del corpus**, en frontera de palabra | El único punto intermedio entre A0 y A3. El corte no lo elige nadie: sale de los datos (A.2) |\n'
        '| **A5** | A3 sin `color` | `color` falta en el 36,6 %: ¿aporta o estorba? |\n'
        '\n'
        '### 🔬 Por qué A3n no se llama A6\n'
        '\n'
        'Porque no es otra receta de la secuencia: es **el control de A3**. D02 decidió omitir la sección de un campo vacío en lugar de escribir `"Color: desconocido"`, con el argumento de que un literal compartido por el 36,6 % del catálogo crearía una señal común artificial — productos que se acercan por compartir una palabra, no por parecerse.\n'
        '\n'
        'El razonamiento era sólido pero **no estaba medido**, y §3.1 no da por buena una justificación sin datos. A3 frente a A3n aísla esa política: si contamina, A3n sale peor; si da igual, D02 era una precaución sin coste; y si sale mejor, la decisión estaba equivocada y se descubre a tiempo.',
    ),
    ("code", 'template_stats(muestra)'),
    (
        "markdown",
        '### A.1 · El mismo producto por las siete recetas\n'
        '\n'
        'Ver el texto real evita discutir sobre abstracciones. Lo que importa es la distancia entre A0 y el resto: si A3 gana, la conclusión no será *"las etiquetas ayudan"* sino que **el resto del texto era relleno**.',
    ),
    (
        "code",
        'fila = muestra.head(1)\n'
        'for nombre in TEMPLATES:\n'
        '    texto = render_template(fila, nombre, context=CONTEXTO)[0]\n'
        '    print(f"\\n──── {nombre}  ({len(texto)} chars) " + "─" * 40)\n'
        '    print(texto[:300] + ("…" if len(texto) > 300 else ""))',
    ),
    (
        "markdown",
        '### A.2 · De dónde sale el recorte de A4\n'
        '\n'
        'A4 recorta `text`, pero **el punto de corte no lo elige nadie**: se deriva del corpus. Un número escrito a mano —512, pongamos— sería una decisión de diseño disfrazada de detalle de implementación, imposible de justificar frente a 400 o 600 y sin sentido en cuanto cambiara el catálogo.\n'
        '\n'
        'Se usa la **mediana** de `text` y no la media: la distribución está sesgada a la derecha y topada en 3.000 caracteres, así que la media queda por encima de lo típico y apenas tocaría a cuatro de cada diez fichas. La mediana parte el catálogo en dos mitades exactas y convierte A4 en una pregunta nítida: **¿sobra la mitad más larga de cada ficha?**',
    ),
    (
        "code",
        'longitudes = muestra["text"].fillna("").str.len()\n'
        '\n'
        'pd.DataFrame([{\n'
        '    "corte_A4": CONTEXTO.a4_chars,\n'
        '    "origen": "mediana de `text`",\n'
        '    "media": round(float(longitudes.mean()), 1),\n'
        '    "mediana": int(longitudes.median()),\n'
        '    "p90": int(longitudes.quantile(0.9)),\n'
        '    "max": int(longitudes.max()),\n'
        '    "pct_productos_recortados": round(100 * float((longitudes > CONTEXTO.a4_chars).mean()), 1),\n'
        '}])',
    ),
    (
        "markdown",
        '### A.3 · Ninguna plantilla se trunca\n'
        '\n'
        'No hace falta volver a medirlo: NB02 comprobó que **A0 —la más larga de las siete— no supera la ventana en ningún registro** del catálogo completo (máximo 1.972 tokens frente a 8.192), y las demás son estrictamente más cortas.\n'
        '\n'
        'Eso vacía de contenido la *medición de truncado*, pero no la pregunta que había detrás, que se reformula: ya no es *"¿se pierde información al truncar?"* sino **"¿el texto largo es señal o es relleno?"** — y esa la responde A4 frente a A0.',
    ),
    (
        "markdown",
        '## B · Codificar las siete variantes\n'
        '\n'
        '⏱️ **~6 minutos**: Gemini codifica 1.500 documentos en ~50 s. Con los modelos locales de NB02 (5 h y 14 h) este barrido habría sido inviable — consecuencia directa de qué modelo ganó R02.\n'
        '\n'
        '⚠️ **Cada plantilla invalida la caché**, y es lo correcto: el nombre del artefacto lleva el SHA-256 del corpus, así que cambiar el texto fuerza a recodificar en vez de comparar en silencio vectores de dos textos distintos.\n'
        '\n'
        '**Las consultas no se recodifican**: las plantillas describen *productos*, y una consulta es lo que escribe la persona. Sus vectores están en caché desde NB02.',
    ),
    (
        "code",
        'VECTORES_PLANTILLA = {}\n'
        'COSTES_PLANTILLA = {}\n'
        'ERRORES_PLANTILLA = {}\n'
        '\n'
        '\n'
        'def encoder_congelado():\n'
        '    """El ganador de R02. Se construye por llamada y se libera después."""\n'
        '    return GeminiEncoder(\n'
        '        api_key=os.environ.get("GEMINI_API_KEY"),\n'
        '        model_id=MODELO,\n'
        '        native_dim=3072,\n'
        '        window=8192,\n'
        '    )\n'
        '\n'
        '\n'
        'def codificar_plantilla(nombre):\n'
        '    """Codifica el catálogo con una plantilla y deja el resultado en memoria."""\n'
        '    textos = render_template(muestra, nombre)\n'
        '    encoder = encoder_congelado()\n'
        '    try:\n'
        '        resultado = encode_corpus(\n'
        '            encoder, textos, corpus_id=f"{CORPUS_ID}__{nombre}",\n'
        '            kind="document", contract=CONTRATO, batch_size=32, cache_dir=CACHE,\n'
        '        )\n'
        '    finally:\n'
        '        del encoder\n'
        '        gc.collect()\n'
        '    VECTORES_PLANTILLA[nombre] = resultado.vectors\n'
        '    COSTES_PLANTILLA[nombre] = {"plantilla": nombre, **resultado.stats.as_row()}\n'
        '    return resultado\n'
        '\n'
        '\n'
        'for nombre in TEMPLATES:\n'
        '    inicio = time.perf_counter()\n'
        '    try:\n'
        '        r = codificar_plantilla(nombre)\n'
        '        origen = "caché" if r.stats.desde_cache else f"{time.perf_counter() - inicio:.1f}s"\n'
        '        print(f"✅ {nombre:<4} {r.vectors.shape} ({origen})")\n'
        '    except Exception as error:\n'
        '        ERRORES_PLANTILLA[nombre] = f"{type(error).__name__}: {error}"\n'
        '        print(f"⛔ {nombre:<4} {ERRORES_PLANTILLA[nombre][:120]}")',
    ),
    (
        "markdown",
        '### B.1 · Consultas y salud de los vectores\n'
        '\n'
        'Las consultas se leen de la caché de NB02 — mismo modelo, mismo contrato, mismo corpus de consultas. Y antes de creerse ninguna métrica, las comprobaciones de siempre: sin `NaN`, normas coherentes y sin filas duplicadas.\n'
        '\n'
        'Un dato a vigilar en la tabla: **`n_filas_duplicadas`**. Si una plantilla produce vectores idénticos para productos distintos, es que está tirando la información que los diferencia — y eso se ve aquí antes que en el nDCG.',
    ),
    (
        "code",
        'encoder = encoder_congelado()\n'
        'try:\n'
        '    vectores_query = encode_corpus(\n'
        '        encoder, query_textos, corpus_id="consultas_desarrollo",\n'
        '        kind="query", contract=CONTRATO, batch_size=32, cache_dir=CACHE,\n'
        '    ).vectors\n'
        '    vectores_ciegas = encode_corpus(\n'
        '        encoder, ciegas["query_text"].tolist(), corpus_id="consultas_evaluacion",\n'
        '        kind="query", contract=CONTRATO, batch_size=32, cache_dir=CACHE,\n'
        '    ).vectors\n'
        'finally:\n'
        '    del encoder\n'
        '    gc.collect()\n'
        '\n'
        'print(f"consultas de desarrollo: {vectores_query.shape}")\n'
        'print(f"consultas ciegas       : {vectores_ciegas.shape}")\n'
        '\n'
        'pd.DataFrame([\n'
        '    {"plantilla": nombre, **vector_health(truncate_dim(matriz, DIM))}\n'
        '    for nombre, matriz in VECTORES_PLANTILLA.items()\n'
        '])',
    ),
    (
        "markdown",
        '## C · El barrido (R01)\n'
        '\n'
        'Cada plantilla se evalúa con el modelo congelado a 768 dimensiones sobre las mismas 8 consultas, los mismos qrels y el mismo `k`. Solo cambia el texto de los documentos.',
    ),
    (
        "code",
        'def evaluar_plantilla(nombre, *, dim=DIM, metric=METRICA):\n'
        '    """Evalúa una plantilla sobre las 8 consultas de desarrollo."""\n'
        '    docs = truncate_dim(VECTORES_PLANTILLA[nombre], dim)\n'
        '    queries = truncate_dim(vectores_query, dim)\n'
        '    retriever = DenseRetriever(docs, corpus_ids, metric=metric)\n'
        '    rankings = rank_queries_dense(retriever, query_ids, queries, k=TOP_K)\n'
        '    return evaluate_rankings(rankings, qrels, k=TOP_K), rankings\n'
        '\n'
        '\n'
        'longitudes = template_stats(muestra).set_index("plantilla")["chars_media"]\n'
        '\n'
        'BARRIDO_PLANTILLAS = []\n'
        'RANKINGS_PLANTILLA = {}\n'
        'for nombre in VECTORES_PLANTILLA:\n'
        '    informe, rankings = evaluar_plantilla(nombre)\n'
        '    RANKINGS_PLANTILLA[nombre] = rankings\n'
        '    BARRIDO_PLANTILLAS.append({\n'
        '        "plantilla": nombre,\n'
        '        **informe.summary,\n'
        '        "chars_media": float(longitudes[nombre]),\n'
        '        "pct_vs_A0": round(100 * float(longitudes[nombre]) / float(longitudes["A0"]), 1),\n'
        '        "segundos": COSTES_PLANTILLA[nombre]["segundos"],\n'
        '    })\n'
        '\n'
        'barrido_plantillas = (\n'
        '    pd.DataFrame(BARRIDO_PLANTILLAS).sort_values("ndcg_at_10", ascending=False).reset_index(drop=True)\n'
        ')\n'
        'barrido_plantillas',
    ),
    ("markdown", '### C.1 · Las cuatro métricas, en una escala común'),
    (
        "code",
        'METRICAS = ["precision_at_10", "recall_at_10", "mrr_at_10", "ndcg_at_10"]\n'
        '\n'
        'plot_metric_comparison(\n'
        '    {\n'
        '        fila["plantilla"]: {m: float(fila[m]) for m in METRICAS}\n'
        '        for _, fila in barrido_plantillas.iterrows()\n'
        '    },\n'
        '    title="Calidad por plantilla",\n'
        '    subtitle=(\n'
        '        f"{MODELO} [{CONTRATO}] @{DIM} · {CORPUS_ID} ({len(muestra)} docs) · "\n'
        '        f"{len(query_ids)} consultas · k={TOP_K}"\n'
        '    ),\n'
        ').show()',
    ),
    (
        "markdown",
        '### C.2 · Tabla por consulta\n'
        '\n'
        'Una plantilla puede ganar de media y hundir dos consultas. Con 8 consultas, la media macro se mueve 0,125 por cada una que cambie de sitio, así que el agregado por sí solo no basta para decidir.',
    ),
    (
        "code",
        'por_consulta_plantilla = pd.concat([\n'
        '    evaluar_plantilla(nombre)[0].per_query_frame().assign(plantilla=nombre)\n'
        '    for nombre in VECTORES_PLANTILLA\n'
        '])\n'
        'por_consulta_plantilla.pivot(index="query_id", columns="plantilla", values="ndcg@10")',
    ),
    (
        "markdown",
        '## D · A3 frente a A3n — el control de D02\n'
        '\n'
        'Las dos plantillas se diferencian en **una sola cosa**: qué hacer cuando un campo está vacío. A3 omite la sección; A3n escribe `"Color: desconocido"`. Son unos pocos caracteres de diferencia de media — exactamente los 549 productos sin `color`.\n'
        '\n'
        '**Cómo se lee la Δ:**\n'
        '\n'
        '- **Δ > 0** (gana A3) → la hipótesis de contaminación se confirma: el literal compartido acerca productos que no se parecen. D02 era necesaria.\n'
        '- **Δ ≈ 0** → D02 era una precaución sin coste. Se mantiene por prudencia, pero ya no como hecho medido.\n'
        '- **Δ < 0** (gana A3n) → la decisión estaba equivocada. Rellenar aporta, y conviene revisar D02 antes de la ingesta final.',
    ),
    (
        "code",
        'if {"A3", "A3n"} <= set(VECTORES_PLANTILLA):\n'
        '    a3 = barrido_plantillas.query("plantilla == \'A3\'").iloc[0]\n'
        '    a3n = barrido_plantillas.query("plantilla == \'A3n\'").iloc[0]\n'
        '    delta_d02 = round(float(a3["ndcg_at_10"]) - float(a3n["ndcg_at_10"]), 4)\n'
        '    veredicto = (\n'
        '        "gana A3 — la contaminación existe, D02 era necesaria" if delta_d02 > TOLERANCIA_R01\n'
        '        else "gana A3n — rellenar aporta, hay que revisar D02" if delta_d02 < -TOLERANCIA_R01\n'
        '        else "indistinguible — D02 era prudente, no imprescindible"\n'
        '    )\n'
        '    display(pd.DataFrame([{\n'
        '        "A3 (omite)": a3["ndcg_at_10"],\n'
        '        "A3n (rellena)": a3n["ndcg_at_10"],\n'
        '        "delta": delta_d02,\n'
        '        "veredicto": veredicto,\n'
        '    }]))\n'
        '    print(f"\\nD02 · {veredicto}")\n'
        'else:\n'
        '    print("A3 o A3n sin codificar: el control de D02 no se puede evaluar.")',
    ),
    (
        "markdown",
        '### D.1 · ¿Es señal o es perturbación?\n'
        '\n'
        'Que la media suba no basta: con 8 consultas, **una sola que cambie de sitio mueve la media macro 0,125**, de sobra para fabricar cualquier diferencia que aparezca aquí. Antes de tocar D02 hay que responder a otra pregunta: *¿la mejora aparece donde el relleno actúa?*\n'
        '\n'
        'El relleno solo toca a las fichas con `color` vacío, así que cada consulta tiene una **exposición** medible: qué porcentaje de sus productos relevantes lleva ese campo en blanco. De ahí sale una predicción falsable — si rellenar aportara información sobre el color, las consultas más expuestas serían las que más se mueven. Si el efecto aparece repartido al azar, y sobre todo si **la consulta con exposición total no se mueve**, lo que se está midiendo es que añadir texto compartido desplaza el espacio vectorial, no que aporte significado.',
    ),
    (
        "code",
        'delta_d02 = per_query_delta(\n'
        '    por_consulta_plantilla, sistema_a="A3n", sistema_b="A3", metrica="ndcg@10"\n'
        ')\n'
        'exposicion = relevant_field_nullity(relevancias, muestra, field="color")\n'
        '\n'
        'control_d02 = (\n'
        '    delta_d02\n'
        '    .merge(exposicion, on="query_id")\n'
        '    .merge(\n'
        '        consultas.assign(query_id=consultas["query_id"].astype(str))[["query_id", "query_text"]],\n'
        '        on="query_id",\n'
        '    )\n'
        '    .sort_values("pct_sin_color", ascending=False)\n'
        ')\n'
        '\n'
        'base_sin_color = round(100 * float(muestra["color"].isna().mean()), 1)\n'
        'print(f"Linea base del catalogo: {base_sin_color}% de productos sin `color`")\n'
        '\n'
        'control_d02[["query_id", "query_text", "A3", "A3n", "delta", "n_relevantes", "pct_sin_color"]]',
    ),
    (
        "code",
        'plot_effect_vs_exposure(\n'
        '    control_d02,\n'
        '    exposure="pct_sin_color",\n'
        '    effect="delta",\n'
        '    tolerance=TOLERANCIA_R01,\n'
        '    title="¿El relleno de nulos aporta información, o solo mueve el espacio?",\n'
        '    subtitle=(\n'
        '        "Un punto por consulta · eje X: % de sus productos relevantes con `color` vacío · "\n'
        '        "eje Y: Δ nDCG@10 (A3n − A3) · la banda gris es la zona indistinguible"\n'
        '    ),\n'
        ').show()',
    ),
    (
        "markdown",
        '### D.2 · Cómo se lee el gráfico\n'
        '\n'
        '- **Tendencia ascendente** → a más exposición, más mejora: el relleno aporta información y **D02 estaba equivocada**.\n'
        '- **Nube plana** → la mejora no tiene que ver con dónde actúa el relleno: es una perturbación del espacio vectorial, y **D02 se mantiene**.\n'
        '- **El punto del extremo derecho es el que más pesa.** Es la consulta cuyos productos relevantes están *todos* sin color: la exposición máxima posible. Si esa no se mueve, ninguna hipótesis basada en el significado del relleno se sostiene.\n'
        '\n'
        'Conviene mirar además el eje X contra la línea base del catálogo que imprime la celda anterior: una consulta por **debajo** de esa referencia está menos expuesta que el producto medio, así que una mejora grande ahí es aún más difícil de explicar por el contenido del relleno.',
    ),
    (
        "markdown",
        '## E · Consistencia sobre las 12 consultas ciegas\n'
        '\n'
        'Las 8 consultas de desarrollo tienen juicios; las 12 de evaluación **no**, así que el nDCG es incalculable sobre ellas. Pero sí se puede medir algo que ninguna métrica con etiquetas captura: son **4 intenciones × 3 formulaciones** de lo mismo.\n'
        '\n'
        '```\n'
        'EVAL-100455-direct    "taladro 24v batería"\n'
        'EVAL-100455-context   "taladro sin cable de 24 voltios que venga con su batería"\n'
        'EVAL-100455-semantic  "quiero una herramienta inalámbrica potente para perforar sin depender de un enchufe"\n'
        '```\n'
        '\n'
        'Un buscador semántico debería devolver **prácticamente los mismos productos** para las tres, y el Jaccard@10 entre formulaciones mide esa estabilidad.\n'
        '\n'
        '**Por qué importa aquí:** el nDCG sobre 8 consultas se puede ganar por afinidad léxica con esas ocho. La consistencia dice qué plantilla **generaliza** a otra superficie léxica, que es la de producción; si una gana en nDCG pero pierde aquí, la ventaja probablemente era sobreajuste.',
    ),
    (
        "code",
        'consistencia = []\n'
        'for nombre in VECTORES_PLANTILLA:\n'
        '    docs = truncate_dim(VECTORES_PLANTILLA[nombre], DIM)\n'
        '    retriever = DenseRetriever(docs, corpus_ids, metric=METRICA)\n'
        '    rankings = rank_queries_dense(\n'
        '        retriever, ciegas["evaluation_id"].tolist(), truncate_dim(vectores_ciegas, DIM), k=TOP_K\n'
        '    )\n'
        '    tabla = formulation_consistency(rankings, k=TOP_K)\n'
        '    columnas = [c for c in tabla.columns if c.startswith("jaccard_")]\n'
        '    consistencia.append({\n'
        '        "plantilla": nombre,\n'
        '        **{c: round(float(tabla[c].mean()), 4) for c in columnas},\n'
        '        "jaccard_medio": round(float(tabla[columnas].to_numpy().mean()), 4),\n'
        '    })\n'
        '\n'
        'consistencia_plantillas = (\n'
        '    pd.DataFrame(consistencia).sort_values("jaccard_medio", ascending=False).reset_index(drop=True)\n'
        ')\n'
        'consistencia_plantillas',
    ),
    (
        "markdown",
        '## F · R01 · Aplicar la regla y dejar el artefacto\n'
        '\n'
        '**El criterio se fijó en `config.yaml` antes de ver la tabla**, igual que D09b en NB02:\n'
        '\n'
        '```yaml\n'
        'r01_criterio_desempate:\n'
        '  forma: mas_corta_dentro_de_tolerancia\n'
        '  metrica_primaria: ndcg_at_10\n'
        '  tolerancia: 0.01\n'
        '  coste: chars_media\n'
        '```\n'
        '\n'
        '1. `B` = mejor nDCG@10 de toda la tabla.\n'
        '2. **Admisibles**: las que están a menos de 0,01 de `B`.\n'
        '3. Entre las admisibles gana **la de menor longitud media**; a igualdad, mayor nDCG.\n'
        '\n'
        '### ⚠️ A3n no entra en la elección\n'
        '\n'
        'Su papel es el de **control de D02**, declarado antes de medir: se codifica y se mide igual que las demás —sin eso no habría con qué contrastar— pero no aspira a ser la elegida.\n'
        '\n'
        'Como excluir a una plantilla **después** de verla puntuar alto es lo que el enunciado penaliza, la declaración previa no basta por sí sola: lo que sostiene la exclusión es el análisis de la sección D, donde su ventaja no aparece donde el relleno actúa sino repartida al azar, y la consulta con exposición total es de las que menos se mueven. No se aparta porque incomode, sino porque se investigó de dónde venía.\n'
        '\n'
        '**Por qué la longitud y no el número de campos.** El criterio buscado era *"que diga más con menos"*, y "representar mejor el producto" es justo lo que mide nDCG@10: el desempate **solo se activa cuando esa métrica ya ha declarado dos plantillas equivalentes**, así que en ese punto maximizar densidad se reduce a minimizar el texto. Contar columnas habla de dependencia de datos, no de significado.\n'
        '\n'
        '> ⚖️ **R01 se ratifica sobre esta tabla.** Si el resultado no convence, el sitio para discutirlo es el criterio, no la tabla.',
    ),
    (
        "code",
        '# La regla se aplica solo a las CANDIDATAS. A3n queda fuera porque su papel es\n'
        '# el de control de D02 —declarado en `aurum.plantillas.CONTROLES`, junto a la\n'
        '# definición de la plantilla y antes de medir nada—, no el de aspirante.\n'
        '#\n'
        '# La exclusión no se sostiene sola: lo que la justifica es el análisis de la\n'
        '# sección D, que mostró que su ventaja no viene de donde el relleno actúa.\n'
        'candidatas_r01 = barrido_plantillas[~barrido_plantillas["plantilla"].isin(CONTROLES)]\n'
        'print(f"candidatas: {sorted(candidatas_r01[\'plantilla\'])}")\n'
        'print(f"controles fuera de la regla: {sorted(CONTROLES)}")\n'
        '\n'
        'ordenadas_plantillas = apply_tolerance_rule(\n'
        '    candidatas_r01, metrica="ndcg_at_10", tolerancia=TOLERANCIA_R01,\n'
        '    coste="chars_media", desempates=(),\n'
        ')\n'
        'ordenadas_plantillas[[\n'
        '    "posicion_regla", "plantilla", "ndcg_at_10", "recall_at_10", "mrr_at_10",\n'
        '    "chars_media", "pct_vs_A0", "admisible",\n'
        ']]',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '# G · El barrido sobre el catálogo completo\n'
        '\n'
        'La sección F aplicó la regla sobre la muestra de desarrollo. Aquí se repite el barrido entero sobre los **15.000 productos**, que es el recorrido que evalúa el enunciado (§6), y **R01 se ratifica con estos números**, no con los de la muestra.\n'
        '\n'
        '### Por qué no basta con la muestra\n'
        '\n'
        'En NB02 el margen entre el modelo ganador y el baseline léxico era holgado, y aun así se estrechó un tercio al pasar de 1.500 a 15.000 candidatos. **Aquí los márgenes son de milésimas**: entre la plantilla más corta y la que venía de NB02 hay menos de 0,003 de nDCG@10. Una diferencia así no sobrevive a ninguna reordenación.\n'
        '\n'
        'Y el mecanismo apunta en contra de las plantillas cortas: con diez veces más productos compitiendo por las mismas diez posiciones, una representación de ~120 caracteres tiene mucho menos con lo que separar vecinos próximos que una de ~1.300, así que cabría esperar que **el texto corto se degrade más**. No es seguro —es la dirección del sentido común—, y por eso se mide en vez de suponerse.\n'
        '\n'
        '### ⚠️ Lo que cuesta\n'
        '\n'
        'Siete plantillas × 15.000 documentos, ~8 minutos cada una: cerca de una hora y unos **1,3 GB** en caché. Es caro, pero es la decisión con la que se ingiere el catálogo definitivo en NB04, y equivocarse obliga a recodificar los 15.000 y reconstruir el índice. La celda estima el coste antes de lanzarlo y se niega a empezar si se dispara por encima del límite declarado.\n'
        '\n'
        '> 📌 **El recorte de A4 cambia de valor, y es correcto.** Se deriva de la mediana del corpus que se codifica, así que sobre 15.000 no es el mismo número que sobre 1.500. A4 no es "la misma plantilla con otro corte": es la misma **regla** aplicada al corpus real.',
    ),
    (
        "code",
        'LIMITE_HORAS_TOTAL = 2.0      # por encima de esto la celda no lanza nada\n'
        'CORPUS_COMPLETO = "catalogo_productos"\n'
        '\n'
        'completo = pd.read_csv(DATA / "catalogo_productos.csv")\n'
        'ids_completo = completo["product_id"].tolist()\n'
        'CONTEXTO_COMPLETO = corpus_context(completo)\n'
        '\n'
        '# Extrapolación desde el coste ya medido sobre la muestra: codificar es\n'
        '# proporcional al número de documentos con el mismo modelo y el mismo lote.\n'
        'seg_muestra = sum(c["segundos"] for c in COSTES_PLANTILLA.values())\n'
        'horas = seg_muestra * len(completo) / len(muestra) / 3600\n'
        '\n'
        'print(f"plantillas      : {len(TEMPLATES)}")\n'
        'print(f"documentos      : {len(completo)} (x{len(completo) // len(muestra)} la muestra)")\n'
        'print(f"corte A4        : {CONTEXTO_COMPLETO.a4_chars} chars (era {CONTEXTO.a4_chars} en la muestra)")\n'
        'print(f"coste estimado  : ~{horas:.1f} h   ·   disco ~{len(TEMPLATES) * len(completo) * 3072 * 4 / 1e9:.1f} GB")\n'
        '\n'
        'VECTORES_COMPLETO = {}\n'
        'if horas > LIMITE_HORAS_TOTAL:\n'
        '    print(f"\\n⏭️  Por encima del límite de {LIMITE_HORAS_TOTAL} h: no se lanza.")\n'
        'else:\n'
        '    for nombre in TEMPLATES:\n'
        '        inicio = time.perf_counter()\n'
        '        textos = render_template(completo, nombre, context=CONTEXTO_COMPLETO)\n'
        '        encoder = encoder_congelado()\n'
        '        try:\n'
        '            resultado = encode_corpus(\n'
        '                encoder, textos, corpus_id=f"{CORPUS_COMPLETO}__{nombre}",\n'
        '                kind="document", contract=CONTRATO, batch_size=32, cache_dir=CACHE,\n'
        '            )\n'
        '        finally:\n'
        '            del encoder\n'
        '            gc.collect()\n'
        '        VECTORES_COMPLETO[nombre] = resultado.vectors\n'
        '        origen = "caché" if resultado.stats.desde_cache else f"{time.perf_counter() - inicio:.0f}s"\n'
        '        print(f"  ✅ {nombre:<4} {resultado.vectors.shape} ({origen})")',
    ),
    (
        "markdown",
        '### G.1 · El barrido, a las dos escalas\n'
        '\n'
        'La columna que decide es `ndcg_completo`. `ndcg_muestra` está al lado solo para ver **cuánto** se mueve cada plantilla al escalar: una caída homogénea no cambia nada, una caída desigual sí reordena.',
    ),
    (
        "code",
        'def evaluar_completo(nombre, *, dim=DIM, metric=METRICA):\n'
        '    """Igual que `evaluar_plantilla`, pero contra los 15.000 IDs del catálogo.\n'
        '\n'
        '    No se reutiliza aquella porque cerró sobre `corpus_ids`, que son los 1.500\n'
        '    de la muestra: pasarle estos vectores devolvería identificadores\n'
        '    equivocados sin lanzar ningún error."""\n'
        '    docs = truncate_dim(VECTORES_COMPLETO[nombre], dim)\n'
        '    queries = truncate_dim(vectores_query, dim)\n'
        '    retriever = DenseRetriever(docs, ids_completo, metric=metric)\n'
        '    rankings = rank_queries_dense(retriever, query_ids, queries, k=TOP_K)\n'
        '    return evaluate_rankings(rankings, qrels, k=TOP_K), rankings\n'
        '\n'
        '\n'
        'if not VECTORES_COMPLETO:\n'
        '    print("Sin codificar sobre el catálogo completo: la celda anterior no llegó a lanzarse.")\n'
        'else:\n'
        '    longitudes_completo = template_stats(completo).set_index("plantilla")["chars_media"]\n'
        '    ndcg_muestra = barrido_plantillas.set_index("plantilla")["ndcg_at_10"]\n'
        '\n'
        '    filas, RANKINGS_COMPLETO = [], {}\n'
        '    for nombre in VECTORES_COMPLETO:\n'
        '        informe, rankings = evaluar_completo(nombre)\n'
        '        RANKINGS_COMPLETO[nombre] = rankings\n'
        '        filas.append({\n'
        '            "plantilla": nombre,\n'
        '            **informe.summary,\n'
        '            "chars_media": float(longitudes_completo[nombre]),\n'
        '            "ndcg_muestra": float(ndcg_muestra[nombre]),\n'
        '        })\n'
        '\n'
        '    barrido_completo = (\n'
        '        pd.DataFrame(filas).sort_values("ndcg_at_10", ascending=False).reset_index(drop=True)\n'
        '    )\n'
        '    barrido_completo["caida_al_escalar"] = (\n'
        '        barrido_completo["ndcg_at_10"] - barrido_completo["ndcg_muestra"]\n'
        '    ).round(4)\n'
        '\n'
        '    display(barrido_completo[[\n'
        '        "plantilla", "ndcg_muestra", "ndcg_at_10", "caida_al_escalar",\n'
        '        "recall_at_10", "chars_media",\n'
        '    ]])',
    ),
    (
        "markdown",
        '### G.2 · R01 sobre el catálogo completo\n'
        '\n'
        'La misma regla y la misma tolerancia declaradas de antemano, aplicadas ahora al corpus que de verdad se evalúa. Los controles siguen fuera del conjunto de candidatas.\n'
        '\n'
        '**Esta tabla es la que ratifica R01.** Si la ganadora coincide con la de la muestra, la decisión llega respaldada a las dos escalas; si no, manda esta — y el informe debe decir que la muestra habría llevado a otra elección, que es la trampa que avisa §6.',
    ),
    (
        "code",
        'if not VECTORES_COMPLETO:\n'
        '    print("Sin barrido completo: R01 se queda con la ratificación sobre la muestra.")\n'
        'else:\n'
        '    candidatas_completo = barrido_completo[~barrido_completo["plantilla"].isin(CONTROLES)]\n'
        '    ordenadas_completo = apply_tolerance_rule(\n'
        '        candidatas_completo, metrica="ndcg_at_10", tolerancia=TOLERANCIA_R01,\n'
        '        coste="chars_media", desempates=(),\n'
        '    )\n'
        '\n'
        '    g_muestra = ordenadas_plantillas.iloc[0]["plantilla"]\n'
        '    g_completo = ordenadas_completo.iloc[0]["plantilla"]\n'
        '    veredicto = (\n'
        '        f"COINCIDEN: {g_completo} gana a las dos escalas"\n'
        '        if g_muestra == g_completo\n'
        '        else f"NO COINCIDEN: la muestra decía {g_muestra}, el catálogo completo dice {g_completo}"\n'
        '    )\n'
        '    print(veredicto)\n'
        '    print(f"admisibles sobre el completo: {sorted(ordenadas_completo.query(\'admisible\')[\'plantilla\'])}")\n'
        '\n'
        '    display(ordenadas_completo[[\n'
        '        "posicion_regla", "plantilla", "ndcg_at_10", "recall_at_10",\n'
        '        "mrr_at_10", "chars_media", "admisible",\n'
        '    ]])',
    ),
    (
        "markdown",
        '### G.3 · Consistencia entre formulaciones, a escala real\n'
        '\n'
        'La consistencia se midió antes sobre la muestra, y la decisión acabó tomándose sobre el catálogo completo: dejarlas a escalas distintas invitaría a citar una comprobación que ya no acompaña a la elección. Repetirla aquí no cuesta nada, porque los vectores ya están codificados.\n'
        '\n'
        '**Qué responde.** El nDCG se mide sobre 8 consultas y una plantilla puede ganarlas por afinidad léxica; el Jaccard entre las tres formulaciones de cada intención mide si el sistema devuelve **los mismos productos cuando le preguntan lo mismo con otras palabras**, que es lo de producción.\n'
        '\n'
        '**Cómo leerlo.** Si la ganadora de la regla también va bien aquí, la elección llega respaldada por dos medidas independientes —una con etiquetas y otra sin ellas—; si va mal, es un matiz que debe acompañarla en el informe.\n'
        '\n'
        '> ⚠️ **Consistencia alta no es calidad.** Un sistema que devuelve los mismos diez productos equivocados para las tres formulaciones puntúa 1,0. Una plantilla con poca información puede ser muy estable justamente porque apenas responde a la consulta. Esta columna se lee **junto** al nDCG, nunca en su lugar.',
    ),
    (
        "code",
        'if not VECTORES_COMPLETO:\n'
        '    print("Sin barrido completo: la consistencia se queda con la medida sobre la muestra.")\n'
        'else:\n'
        '    filas = []\n'
        '    for nombre in VECTORES_COMPLETO:\n'
        '        docs = truncate_dim(VECTORES_COMPLETO[nombre], DIM)\n'
        '        retriever = DenseRetriever(docs, ids_completo, metric=METRICA)\n'
        '        rankings = rank_queries_dense(\n'
        '            retriever, ciegas["evaluation_id"].tolist(),\n'
        '            truncate_dim(vectores_ciegas, DIM), k=TOP_K,\n'
        '        )\n'
        '        tabla = formulation_consistency(rankings, k=TOP_K)\n'
        '        columnas = [c for c in tabla.columns if c.startswith("jaccard_")]\n'
        '        filas.append({\n'
        '            "plantilla": nombre,\n'
        '            "jaccard_completo": round(float(tabla[columnas].to_numpy().mean()), 4),\n'
        '        })\n'
        '\n'
        '    ganadora_regla = ordenadas_completo.iloc[0]["plantilla"]\n'
        '    consistencia_completo = (\n'
        '        pd.DataFrame(filas)\n'
        '        .merge(\n'
        '            consistencia_plantillas[["plantilla", "jaccard_medio"]]\n'
        '            .rename(columns={"jaccard_medio": "jaccard_muestra"}),\n'
        '            on="plantilla",\n'
        '        )\n'
        '        .merge(barrido_completo[["plantilla", "ndcg_at_10"]], on="plantilla")\n'
        '        .sort_values("jaccard_completo", ascending=False)\n'
        '        .reset_index(drop=True)\n'
        '    )\n'
        '    consistencia_completo["gana_r01"] = consistencia_completo["plantilla"] == ganadora_regla\n'
        '\n'
        '    puesto = int(consistencia_completo.index[\n'
        '        consistencia_completo["plantilla"] == ganadora_regla\n'
        '    ][0]) + 1\n'
        '    print(f"{ganadora_regla} (ganadora de R01) queda {puesto}a de {len(consistencia_completo)} "\n'
        '          f"en consistencia entre formulaciones")\n'
        '\n'
        '    display(consistencia_completo[[\n'
        '        "plantilla", "jaccard_muestra", "jaccard_completo", "ndcg_at_10", "gana_r01",\n'
        '    ]])',
    ),
    (
        "code",
        'def registros(frame):\n'
        '    """Filas como tipos JSON nativos: `to_dict` dejaría escalares de numpy."""\n'
        '    return json.loads(frame.to_json(orient="records"))\n'
        '\n'
        '\n'
        '# R01 se ratifica sobre el CATALOGO COMPLETO, no sobre la muestra: el orden\n'
        '# entre plantillas cambia al escalar, y decidir con 1.500 candidatos habria\n'
        '# llevado a otra plantilla. Si la seccion G no llego a lanzarse se cae a la\n'
        '# muestra, pero queda dicho en el artefacto que no esta confirmada.\n'
        'if VECTORES_COMPLETO:\n'
        '    ganadora = ordenadas_completo.iloc[0]\n'
        '    decidida_sobre = CORPUS_COMPLETO\n'
        'else:\n'
        '    ganadora = ordenadas_plantillas.iloc[0]\n'
        '    decidida_sobre = f"{CORPUS_ID} (SIN confirmar a escala real)"\n'
        '\n'
        'artefacto = {\n'
        '    "configuracion": {\n'
        '        "corpus": CORPUS_ID,\n'
        '        "n_docs": len(muestra),\n'
        '        "modelo_congelado": {\n'
        '            "id": MODELO, "contrato": CONTRATO, "dim": DIM, "metrica": METRICA,\n'
        '        },\n'
        '        "top_k": TOP_K,\n'
        '        "r01": {\n'
        '            "metrica": "ndcg_at_10", "tolerancia": TOLERANCIA_R01,\n'
        '            "coste": "chars_media", "decidida_sobre": decidida_sobre,\n'
        '            "ganadora": ganadora["plantilla"],\n'
        '        },\n'
        '        "d07_chunking": False,\n'
        '    },\n'
        '    "plantillas": registros(template_stats(muestra)),\n'
        '    "errores_de_codificacion": ERRORES_PLANTILLA,\n'
        '    "costes_de_codificacion": list(COSTES_PLANTILLA.values()),\n'
        '    "barrido": registros(barrido_plantillas),\n'
        '    "regla_r01": registros(ordenadas_plantillas),\n'
        '    "consistencia_ciegas": registros(consistencia_plantillas),\n'
        '    # El barrido a escala real, que es el que ratifica R01. Puede no existir si\n'
        '    # la seccion G no llego a lanzarse.\n'
        '    "barrido_completo": registros(barrido_completo) if VECTORES_COMPLETO else [],\n'
        '    "regla_r01_completo": registros(ordenadas_completo) if VECTORES_COMPLETO else [],\n'
        '    "consistencia_ciegas_completo": (\n'
        '        registros(consistencia_completo) if VECTORES_COMPLETO else []\n'
        '    ),\n'
        '    "rankings": RANKINGS_PLANTILLA,\n'
        '}\n'
        '\n'
        'destino = Path("..") / "artifacts" / "comparativa_representacion.json"\n'
        'destino.write_text(json.dumps(artefacto, indent=2, ensure_ascii=False, default=str), encoding="utf-8")\n'
        '\n'
        'markdown = Path("..") / "artifacts" / "comparativa_representacion.md"\n'
        'markdown.write_text(\n'
        '    "# Comparativa de representacion (NB03)\\n\\n"\n'
        '    f"Modelo congelado: `{MODELO}` [{CONTRATO}] @{DIM} · {METRICA}\\n"\n'
        '    f"Corpus: `{CORPUS_ID}` ({len(muestra)} docs) · k={TOP_K}\\n\\n"\n'
        '    "## Barrido de plantillas\\n\\n" + barrido_plantillas.to_markdown(index=False) + "\\n\\n"\n'
        '    "## Regla R01 aplicada\\n\\n" + ordenadas_plantillas.to_markdown(index=False) + "\\n\\n"\n'
        '    "## Consistencia entre formulaciones (12 consultas ciegas)\\n\\n"\n'
        '    + consistencia_plantillas.to_markdown(index=False) + "\\n",\n'
        '    encoding="utf-8",\n'
        ')\n'
        'print(f"R01 ratificada sobre {decidida_sobre}")\n'
        'print(f"  ganadora: {ganadora[\'plantilla\']} "\n'
        '      f"(nDCG@10 = {ganadora[\'ndcg_at_10\']}, {ganadora[\'chars_media\']:.0f} chars de media)")\n'
        'print(f"Escrito {destino.name} y {markdown.name}")',
    ),
]

NB04_MOTOR = [
    ("markdown", '# NB04 · Motor vectorial: requisitos, humo, esquema e ingesta'),
    (
        "markdown",
        'Hasta aquí los vectores han vivido en memoria y en ficheros `.npy`. Este notebook los pone en una base de datos, y para eso hay que elegir cuál.\n'
        '\n'
        '### El orden: requisitos → motor → índice\n'
        '\n'
        'Podría parecer que primero se elige el índice y después la base que lo contiene —es el orden en que se estudian—, pero al **elegir** la relación se invierte: el índice no es algo que uno monta, es algo que el motor concede. Unos dejan escoger la familia de algoritmo, otros dan grafo y dejan ajustarlo, y otros no dejan ni verlo.\n'
        '\n'
        'Decidir "quiero tal índice" antes que el motor solo tendría sentido si ese requisito **descartara motores**, y aquí no lo hace: 15.000 vectores de 768 dimensiones son 46 MB, caben en memoria con holgura y hasta la búsqueda exacta es cuestión de milisegundos. Lo que sí los descarta son las responsabilidades operativas —filtrar, no duplicar al reingerir, sobrevivir a un reinicio, aplicar altas y bajas—, y por eso van primero.\n'
        '\n'
        'La sección **A** escribe esos requisitos **antes** de mirar ningún motor: si se eligiera primero y luego se ajustaran al ganador, la elección no demostraría nada.\n'
        '\n'
        '| Marca | Corpus | Fichero |\n'
        '|---|---|---|\n'
        '| 🔬 **MUESTRA** | 1.500 registros | `catalogo_muestra.csv` |\n'
        '| 📚 **COMPLETO** | 15.000 registros | `catalogo_productos.csv` |\n'
        '\n'
        '> 📄 Como en los notebooks anteriores, la primera línea de cada celda de código dice qué datos usa.\n'
        '\n'
        '**Aquí el corpus por defecto es el completo**, al revés que en NB02 y NB03: lo que se decide en este notebook —qué se guarda, cómo se filtra, cuánto ocupa— es sobre el índice que se construye de verdad, que contiene los 15.000. La muestra reaparece en la prueba de humo, que es donde conviene equivocarse barato.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000) · 🔬 catalogo_muestra.csv (1.500)\n'
        '#            · consultas_filtradas.csv (4 consultas con filtro de marca)\n'
        'import sys\n'
        'from pathlib import Path\n'
        '\n'
        'sys.path.insert(0, str(Path("..") / "src"))\n'
        '\n'
        'import pandas as pd\n'
        '\n'
        'from aurum.almacen import (\n'
        '    NULL_POLICIES,\n'
        '    PAYLOAD_SCHEMAS,\n'
        '    add_normalized_key,\n'
        '    batch_footprint,\n'
        '    build_payload,\n'
        '    combined_filter_selectivity,\n'
        '    field_byte_profile,\n'
        '    filter_field_profile,\n'
        '    filter_reach,\n'
        '    filter_writing_robustness,\n'
        '    index_footprint,\n'
        '    payload_budget,\n'
        '    robustness_summary,\n'
        ')\n'
        'from aurum.datos import load_csv, normalize_brand, strip_accents, value_frequency\n'
        '\n'
        'DATA = Path("..") / "data"\n'
        'completo = load_csv(DATA / "catalogo_productos.csv")\n'
        'muestra = load_csv(DATA / "catalogo_muestra.csv")\n'
        'filtradas = load_csv(DATA / "consultas_filtradas.csv")\n'
        '\n'
        'DIM = 768                     # R02: gemini-embedding-2 truncado y renormalizado\n'
        'TOP_K = 10\n'
        'NORMALIZACION = "unaccent"    # D03: minúsculas y sin tildes, al buscar\n'
        'CAMPOS_FILTRABLES = ["brand", "color"]\n'
        '\n'
        'print(f"📚 completo: {len(completo)} · 🔬 muestra: {len(muestra)}")\n'
        'print(f"consultas con filtro: {len(filtradas)} — todas por {filtradas[\'filter_field\'].unique()}")',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '# A · Los requisitos, escritos antes de mirar ningún motor\n'
        '\n'
        'La lista sale de tres sitios: el guion de selección de una base vectorial de la sesión 3, lo que necesitan los notebooks posteriores de este trabajo, y lo que el enunciado exige sin margen.\n'
        '\n'
        'Buena parte no se decide: **se mide**. La celda siguiente calcula esa parte.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000) + consultas_filtradas.csv\n'
        '# Los requisitos que son un hecho del problema, no una preferencia. Se calculan\n'
        '# aquí para que la tabla de abajo no lleve ni un número escrito a mano.\n'
        'huella = index_footprint(len(completo), DIM)\n'
        'marcas_pedidas = filtradas["filter_value"].tolist()\n'
        'alcance_marca = filter_reach(completo, marcas_pedidas, field="brand")\n'
        'selectividad = (\n'
        '    alcance_marca.query("modo == \'raw\'")\n'
        '    .assign(pct_catalogo=lambda d: (100 * d["n_productos"] / len(completo)).round(2))\n'
        '    [["filtro", "n_productos", "pct_catalogo"]]\n'
        ')\n'
        '\n'
        'print(f"vectores            : {huella[\'n_points\']:,} de {huella[\'dim\']} dimensiones")\n'
        'print(f"memoria de vectores : {huella[\'mb_vectores\']} MB")\n'
        'print(f"resultados por consulta: {TOP_K}")\n'
        'print()\n'
        'print("Selectividad de los filtros que exige el enunciado:")\n'
        'selectividad',
    ),
    (
        "markdown",
        '### A.1 · Lo que ya está fijado\n'
        '\n'
        '**De recuperación:**\n'
        '\n'
        '| Requisito | Valor | De dónde sale |\n'
        '|---|---|---|\n'
        '| Volumen y crecimiento | 15.000 vectores; la secuencia de eventos deja otros 15.000 (ocho altas, ocho bajas) | Catálogo y eventos |\n'
        '| Representación | 768 dimensiones, decimales de 4 bytes, norma 1 | R02 y D10 |\n'
        '| Un vector por producto | Sí, nunca varios | D07: ninguna ficha se acerca a la ventana del modelo |\n'
        '| Resultados por consulta | 10 | Enunciado |\n'
        '| Campos por los que se filtra | Marca (obligatorio) y color (añadido) | Consultas filtradas + decisión propia |\n'
        '| Búsqueda dispersa o híbrida | No se le exige al motor | Si se probara el híbrido, se fusionaría por posición fuera del motor: no le pide nada |\n'
        '| Cuánto cambia el corpus | 24 eventos, aplicados dos veces | Enunciado |\n'
        '\n'
        '**De operación:**\n'
        '\n'
        '| Requisito | Valor |\n'
        '|---|---|\n'
        '| Carga | Mínima: un experimento reproducible, no un servicio con tráfico |\n'
        '| Memoria de la máquina | 7,9 GB, compartidos con el notebook. Los motores, de uno en uno |\n'
        '| Presupuesto | Cero recurrente: quien corrija reproduce el trabajo y no puede heredar una factura |\n'
        '| Despliegue | Docker, con los ficheros de la sesión 3 adaptados a este repo |\n'
        '\n'
        '**Y lo que el enunciado impone sin margen:** SDK nativo del motor —ninguna capa de abstracción sustituyendo la configuración—, filtro ejecutado por el motor, ingesta repetible sin duplicar, persistencia real y mutaciones cuya visibilidad se pueda comprobar.\n'
        '\n'
        '### A.2 · Lo que se aparca a propósito\n'
        '\n'
        'El **recall mínimo aceptable** y la **latencia máxima** no se fijan aquí: se declaran justo antes de ver la curva de fidelidad, que es NB06. Anotarlos ahora invitaría a ajustarlos luego a lo que salga.\n'
        '\n'
        'Se descartan por no aplicar a una práctica en una sola máquina: caudal de peticiones concurrentes, alta disponibilidad, tiempo de recuperación ante desastre y reparto entre varias máquinas.',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '# B · El campo extra: filtrar también por color\n'
        '\n'
        'El enunciado solo pide filtrar por marca. Añadir el color es una mejora propia, y se somete a la misma evidencia que el resto en vez de darse por buena porque suene bien.\n'
        '\n'
        'La pregunta no es *"¿se puede filtrar por color?"* sino **"¿es el color la clase de campo sobre la que un filtro de igualdad significa algo?"**. Un campo de vocabulario cerrado —diez colores repetidos miles de veces— se filtra con igualdad y no hay más que hablar; uno de texto libre se comporta de otra manera, y conviene saberlo **antes** de prometer un filtro que devuelve la mitad de lo que debería.\n'
        '\n'
        '### Cómo leer la tabla siguiente\n'
        '\n'
        'Una fila por campo filtrable. Las tres últimas columnas son el veredicto *taxonomía vs. texto libre*, y las tres apuntan en la misma dirección: **cuanto más altas, menos sirve un filtro de igualdad**.\n'
        '\n'
        '| Columna | Qué mide | Alto significa |\n'
        '|---|---|---|\n'
        '| `pct_valores_unicos` | valores distintos que aparecen en un solo producto | cola larga: cada quien escribió lo suyo |\n'
        '| `pct_compuestos` | productos cuyo valor lleva un separador dentro (`Negro/Rojo`) | la igualdad estricta los deja fuera |\n'
        '| `pct_multipalabra` | productos con dos o más palabras (`azul marino`) | pedir una sola palabra no los encuentra |\n'
        '\n'
        'La comparación que importa es **entre las dos filas**: `brand` y `color` deberían comportarse distinto, y si no lo hacen, la mejora del color no está justificada.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000)\n'
        '# ¿Taxonomía o texto libre? Un `pct_valores_unicos` alto significa cola larga:\n'
        '# la mayoría de los valores los escribió una sola persona una sola vez.\n'
        'filter_field_profile(completo, CAMPOS_FILTRABLES)',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000)\n'
        '# Los valores más frecuentes de cada campo filtrable, para ver de qué hablamos.\n'
        'display(value_frequency(completo, "color").head(8))\n'
        'display(value_frequency(completo, "brand").head(5))',
    ),
    (
        "markdown",
        '## B.1 · Igualdad frente a *contiene*\n'
        '\n'
        'Si el campo es texto libre, un valor puede llevar varios dentro —`"Negro/Rojo"`, `"Negro (Black)"`— o ser de dos palabras —`"azul marino"`—: con igualdad estricta, quien pida negro no ve los dos primeros y quien pida azul no ve el tercero.\n'
        '\n'
        'La celda mide las dos políticas sobre los mismos valores y los mismos tres modos de normalización, para no mezclar dos cambios a la vez.\n'
        '\n'
        '### Cómo leer la tabla\n'
        '\n'
        'Cada **fila** es un color pedido bajo un modo de normalización; cada **columna**, una política de comparación. El número son productos alcanzados, así que **más es mejor** — con la reserva que mide B.3.\n'
        '\n'
        '| Modo de normalización | Qué le hace al valor guardado **y** al pedido |\n'
        '|---|---|\n'
        '| `raw` | nada: compara tal cual está en el catálogo |\n'
        '| `casefold` | pasa los dos a minúsculas |\n'
        '| `unaccent` | minúsculas y además sin tildes |\n'
        '\n'
        'La columna `ganancia_x` deja hecha la división: cuántas veces más alcanza *contiene* que *igualdad exacta* en esa misma fila.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000)\n'
        '# Los tres colores más frecuentes salen del propio catálogo, no de una lista\n'
        '# escrita a mano: elegirlos a dedo sería elegir el resultado.\n'
        'colores_frecuentes = (\n'
        '    value_frequency(completo, "color")\n'
        '    .query("color != \'(vacío)\'")\n'
        '    .head(3)["color"].str.lower().tolist()\n'
        ')\n'
        '\n'
        'comparacion = pd.concat([\n'
        '    filter_reach(completo, colores_frecuentes, field="color", match=match)\n'
        '    for match in ("equals", "contains")\n'
        '])\n'
        'print(f"colores medidos: {colores_frecuentes}")\n'
        '\n'
        '# Los nombres de columna del pivote son los de la API (`equals`/`contains`);\n'
        '# se renombran solo para mostrar, que es donde los lee una persona.\n'
        'igualdad_contiene = (\n'
        '    comparacion\n'
        '    .pivot_table(index=["filtro", "modo"], columns="match", values="n_productos")\n'
        '    .rename(columns={"equals": "igualdad_exacta", "contains": "contiene"})\n'
        '    .rename_axis(index={"filtro": "color_pedido", "modo": "normalizacion"})\n'
        ')\n'
        'igualdad_contiene.assign(\n'
        '    ganancia_x=lambda d: (d["contiene"] / d["igualdad_exacta"]).round(1)\n'
        ')',
    ),
    (
        "markdown",
        '## B.2 · Cómo lo escribe el usuario\n'
        '\n'
        'La tabla anterior compara políticas suponiendo **una sola forma** de escribir la consulta, la minúscula: una suposición cómoda y sesgada a favor de normalizar, porque nadie escribe siempre igual.\n'
        '\n'
        'Aquí la suposición se sustituye por una medición: la misma consulta escrita como la escribiría una persona —todo minúsculas, inicial en mayúscula, todo mayúsculas y, si la palabra lleva tilde, las tres formas otra vez con ella— contra los tres modos de normalización.\n'
        '\n'
        '### Cómo leer las tablas\n'
        '\n'
        'Las **filas** son lo que escribe la persona. Las **columnas** son lo que hace el sistema con esa consulta **y con el valor guardado** antes de compararlos:\n'
        '\n'
        '| Modo | Qué hace |\n'
        '|---|---|\n'
        '| `raw` | no toca nada |\n'
        '| `casefold` | pasa ambos a minúsculas |\n'
        '| `unaccent` | minúsculas y además quita tildes a ambos |\n'
        '\n'
        'Lo que se busca es la columna donde **todas las filas dan el mismo número** y ese número es **el más alto** de la tabla.\n'
        '\n'
        'El veredicto tiene **dos ejes que conviene no mezclar**:\n'
        '\n'
        '| Eje | Qué pregunta | Por qué no basta el otro |\n'
        '|---|---|---|\n'
        '| **Consistente** | ¿Encuentran todos lo mismo? | Un filtro cuyo resultado depende de la tecla de mayúsculas no es un filtro |\n'
        '| **Alcanza el máximo** | ¿Y encuentran *todo* lo que hay? | Que todo el mundo encuentre la mitad también es consistente, y está igual de roto |\n'
        '\n'
        'La segunda columna existe porque el primer diseño solo medía la primera, y un test destapó el caso: quitar solo las mayúsculas deja a quien escribe la tilde y a quien no encontrando cada uno su mitad del catálogo, tan consistentes como incompletos.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000)\n'
        '# Se prueban dos consultas: el color más frecuente y el más frecuente que\n'
        '# lleva tilde. El segundo sale del catálogo, no de mi cabeza: sin una palabra\n'
        '# acentuada, el eje de las tildes no se puede medir.\n'
        'frecuencias = value_frequency(completo, "color").query("color != \'(vacío)\'")\n'
        'con_tilde = frecuencias[\n'
        '    frecuencias["color"] != frecuencias["color"].map(strip_accents)\n'
        ']\n'
        'consultas_escritura = [frecuencias["color"].iloc[0]]\n'
        'if len(con_tilde):\n'
        '    consultas_escritura.append(con_tilde["color"].iloc[0])\n'
        '\n'
        'for consulta in consultas_escritura:\n'
        '    tabla = filter_writing_robustness(\n'
        '        completo, consulta, field="color", match="contains"\n'
        '    )\n'
        '    print(f"── {consulta!r} ──")\n'
        '    display(tabla.pivot(index=["variante", "escritura"], columns="modo",\n'
        '                        values="n_productos"))\n'
        '    display(robustness_summary(tabla))',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000)\n'
        '# Lo mismo sobre la marca, que es el filtro obligatorio. Las cuatro marcas del\n'
        '# enunciado se escriben con mayúsculas distintas entre sí -NIKE, Apple,\n'
        '# SAMSUNG, Einhell-, así que aquí el eje de la caja no es hipotético.\n'
        'for marca in marcas_pedidas:\n'
        '    tabla = filter_writing_robustness(completo, marca, field="brand")\n'
        '    resumen = robustness_summary(tabla).assign(marca=marca)\n'
        '    display(resumen[["marca", "modo", "n_variantes", "alcances_distintos",\n'
        '                     "minimo", "maximo", "consistente", "alcanza_el_maximo"]])',
    ),
    (
        "markdown",
        '## B.3 · Lo que cuesta el *contiene*: falsos positivos\n'
        '\n'
        'Buscar un texto dentro de otro alcanza más productos, pero también entra donde no debería: el texto buscado puede aparecer **dentro de otra palabra** en vez de como palabra suelta. Pedir `rosa` y que entre `rosado` es el caso típico.\n'
        '\n'
        '### Cómo leer la tabla\n'
        '\n'
        '| Columna | Qué cuenta | Dirección |\n'
        '|---|---|---|\n'
        '| `n_productos` | todo lo que alcanza el *contiene* | más es mejor |\n'
        '| `n_dentro_de_otra_palabra` | de esos, los que entran **solo** porque el texto está dentro de otra palabra | ⚠️ **más es peor**: son falsos positivos |\n'
        '| `pct_falsos_positivos` | qué parte del alcance es basura | ⚠️ más es peor |\n'
        '\n'
        'La coincidencia exacta y los valores compuestos (`Negro/Rojo`, `Blanco (White)`) **no** cuentan como falso positivo: ahí el color pedido está como palabra suelta, que es lo que el *contiene* viene a rescatar.\n'
        '\n'
        'Es lo que convierte la política en decisión medida: si el porcentaje es despreciable, el *contiene* sale gratis; si no, hay que decidir si el alcance extra compensa la pérdida de precisión.\n'
        '\n'
        '> 📌 Se mide con la clave **normalizada**, no con las tres. El coste hay que verlo en la configuración que se va a desplegar: sobre el valor crudo el alcance es diez veces menor (B.4) y los falsos positivos saldrían artificialmente bajos.\n'
        '\n'
        '### ⚠️ Limitación declarada: el género y el número\n'
        '\n'
        'Hay un hueco que **ninguna política de filtro resuelve**, y conviene dejarlo escrito antes de que lo encuentre otro: `negro` y `negra` son cadenas distintas y **ninguna contiene a la otra**, así que no se encuentran entre sí ni por subcadena, ni por palabras, ni con igualdad.\n'
        '\n'
        '| Filtro | Subcadena | Por palabras |\n'
        '|---|---:|---:|\n'
        '| `negro` | 215 | 212 |\n'
        '| `negra` | 3 | 3 |\n'
        '\n'
        'Los 3 de `negra` **no están** entre los 215 ni entre los 212. Y los 3 de diferencia entre 215 y 212 son otra cosa: valores donde `negro` aparece dentro de otra palabra — los falsos positivos que cuenta esta sección.\n'
        '\n'
        '**La asimetría es lo grave.** Quien busca `negro` pierde 3 productos, un 0,2 %; quien busca `negra` recibe esos 3 y **pierde los otros 212**: la forma menos frecuente devuelve un resultado plausible y esconde el grueso del catálogo sin avisar. Un vacío ruidoso se detecta; una página con tres resultados correctos, no.\n'
        '\n'
        'No depende del motor —la tienen los tres candidatos, así que **no entra en R03**— y las salidas serían *stemming* en el tokenizador o normalizar el género al construir `color_normalized`, que añaden un eje que habría que medir. Se declara como limitación conocida, que es lo que §5 llama *"atribución de errores: datos o filtros"*.\n'
        '\n'
        '### Cuándo dejaría de importar todo esto\n'
        '\n'
        'El requisito duro de esta sección —que el motor sepa buscar texto dentro de un metadato— y la ventaja del nivel 2 **solo pesan porque `color` es texto libre**. Con una taxonomía limpia —un color por registro, vocabulario cerrado y **ningún valor subcadena de otro**—, `like \'%x%\'` no podría dar falsos positivos y los tres motores empatarían en este eje.\n'
        '\n'
        'Y esa condición no hay que juzgarla a ojo: **es la columna de arriba**.\n'
        '\n'
        '> `n_dentro_de_otra_palabra == 0` para todos los colores filtrados ⟺ el `contains` por subcadena es seguro\n'
        '\n'
        'Hoy no se cumple, y por partida doble:\n'
        '\n'
        '- La sección B midió que `color` **no es una taxonomía** sino texto libre. La muestra trae `"2 unidades negra"`, `"Negro, 1 Piezas"`, `"negra + oro rosa"`, `"como se muestra"` — cantidades, recuentos de piezas y varios colores en el mismo campo.\n'
        '- `oro` es subcadena de `incoloro`, que es el caso que los tests fijan.\n'
        '\n'
        'Si el catálogo se limpiara —o si el color viniera de un desplegable en vez de texto libre— **R03 podría decidirse por los otros criterios**: memoria, control del ANN, calidad del error, dependencia del proveedor. Conviene reevaluarlo entonces en vez de arrastrar una restricción que ya no aplica.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000)\n'
        'falsos = filter_reach(\n'
        '    completo, colores_frecuentes, field="color",\n'
        '    match="contains", modes=(NORMALIZACION,),\n'
        ')\n'
        'falsos.assign(\n'
        '    pct_falsos_positivos=lambda d: (\n'
        '        100 * d["n_dentro_de_otra_palabra"] / d["n_productos"]\n'
        '    ).round(2)\n'
        ')[["filtro", "n_productos", "n_dentro_de_otra_palabra",\n'
        '   "pct_falsos_positivos", "n_valores_distintos", "valores"]]',
    ),
    (
        "markdown",
        '## B.4 · La clave derivada, y por qué el filtro no puede vivir sin ella\n'
        '\n'
        '**D03** decidió dos cosas que solo encajan si el punto lleva una clave más: el valor **se guarda tal cual viene** y **se normaliza al buscar**. Un filtro nativo compara lo almacenado con lo pedido byte a byte, así que normalizar solo la consulta no sirve: la normalización tiene que estar materializada en el punto. La misma política se extiende ahora al color.\n'
        '\n'
        '### Cómo leer las dos tablas\n'
        '\n'
        'Son las dos caras de la decisión: la primera dice **qué cuesta** la clave, la segunda **qué compra**.\n'
        '\n'
        '**Coste** — `mb_total` de cada clave derivada, que se suma a lo que ya ocupa el valor crudo (D03 guarda los dos), y se compara con los MB de vectores de la sección A.\n'
        '\n'
        '**Beneficio** — las dos columnas son lo mismo medido de dos formas, renombradas porque el nombre de la API despista:\n'
        '\n'
        '| Columna | Qué es en realidad |\n'
        '|---|---|\n'
        '| `sin_clave_derivada` | el filtro compara contra el valor **tal cual está guardado**; solo la consulta se normalizó. Es lo que pasa si el punto no lleva la clave |\n'
        '| `con_clave_derivada` | los dos lados normalizados, que es lo que D03 exige |\n'
        '| `pct_alcanzado_sin_clave` | qué parte del alcance real consigue la primera. ⚠️ **cuanto más bajo, más se pierde** |\n'
        '\n'
        'Ojo con el sentido: un `pct_alcanzado_sin_clave` del 9 % significa que sin la clave **se pierde el 91 %** de lo que debería encontrarse. Y no es un subconjunto cualquiera: son los productos que alguien tecleó con mayúscula al cargar el catálogo, o sea un filtro que falla en silencio y sin patrón.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000)\n'
        'con_claves = completo.copy()\n'
        'for campo in CAMPOS_FILTRABLES:\n'
        '    con_claves = add_normalized_key(con_claves, field=campo, mode=NORMALIZACION)\n'
        '\n'
        '# 1) Lo que CUESTA la clave: bytes por campo derivado.\n'
        'derivadas = [f"{campo}_normalized" for campo in CAMPOS_FILTRABLES]\n'
        'display(field_byte_profile(con_claves, derivadas))\n'
        '\n'
        '# 2) Lo que COMPRA: el mismo filtro con y sin ella. `raw` no es "no normalizar\n'
        '# nada", es "normalizar solo la consulta" -que es el escenario sin clave.\n'
        'sin_clave = filter_reach(\n'
        '    completo, colores_frecuentes, field="color",\n'
        '    match="contains", modes=("raw", NORMALIZACION),\n'
        ')\n'
        'ganancia = (\n'
        '    sin_clave\n'
        '    .pivot(index="filtro", columns="modo", values="n_productos")\n'
        '    .rename(columns={"raw": "sin_clave_derivada",\n'
        '                     NORMALIZACION: "con_clave_derivada"})\n'
        '    .rename_axis(index="color_pedido", columns=None)\n'
        ')\n'
        'ganancia.assign(\n'
        '    pct_alcanzado_sin_clave=lambda d: (\n'
        '        100 * d["sin_clave_derivada"] / d["con_clave_derivada"]\n'
        '    ).round(1)\n'
        ')',
    ),
    (
        "markdown",
        '## B.5 · Filtros combinados: el escenario donde filtrar después se rompe\n'
        '\n'
        'Una marca sola ya es selectiva. Una marca **con** un color concreto lo es mucho más, y ahí es donde recuperar diez vecinos y descartar los que no cumplen deja de funcionar: puede no quedar ninguno.\n'
        '\n'
        '### Cómo leer la tabla\n'
        '\n'
        'Cada fila es un cruce marca × color. La marca se compara con **igualdad** y el color con **contiene**, cada uno contra su clave normalizada — las mismas políticas que se acaban de decidir, no otras.\n'
        '\n'
        '| Columna | Qué cuenta |\n'
        '|---|---|\n'
        '| `n_brand` | productos de esa marca |\n'
        '| `n_brand_con_color` | de esos, cuántos tienen el color **anotado** |\n'
        '| `n_brand_sin_color` | de esos, cuántos lo tienen vacío |\n'
        '| `n_brand_y_color` | los que pasan **los dos** filtros |\n'
        '| `pct_del_catalogo` | qué parte de los 15.000 sobrevive al cruce |\n'
        '| `cero_por_falta_de_dato` | ⚠️ el cero de esa fila **no dice nada del catálogo** |\n'
        '\n'
        'Las tres columnas del medio están porque **un cero sin ellas es ambiguo**: con un 37 % de colores vacíos, `n_brand_y_color = 0` puede significar que la marca no tiene ningún producto de ese color —un hecho del catálogo, y el argumento se sostiene— o que sus productos no lo tienen anotado —un hecho sobre la cobertura, que cambia en cuanto alguien rellene la columna—. `cero_por_falta_de_dato` marca el segundo caso para que no se lea como el primero.\n'
        '\n'
        'Cuanto más baja sea `pct_del_catalogo`, más se refuerza el argumento: si el filtro deja menos supervivientes que los 10 resultados pedidos, recuperar primero y descartar después no puede llenar la página.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000)\n'
        '# La función recibe las columnas SIN normalizar y normaliza ella los dos lados\n'
        '# -lo pedido y lo almacenado- con la política de D03, igual que filter_reach.\n'
        '# Repetir esa normalización a mano en la celda es como se coló antes un filtro\n'
        '# de marca que comparaba contra el valor crudo.\n'
        'combinados = combined_filter_selectivity(\n'
        '    completo,\n'
        '    primary_values=marcas_pedidas,\n'
        '    secondary_values=colores_frecuentes,\n'
        '    primary_field="brand",\n'
        '    secondary_field="color",\n'
        '    mode=NORMALIZACION,\n'
        ')\n'
        'combinados.sort_values("n_brand_y_color")',
    ),
    (
        "markdown",
        '### Qué se lleva el esquema de todo esto\n'
        '\n'
        'Tres consecuencias, y ninguna es opcional una vez tomada la decisión del campo extra:\n'
        '\n'
        '1. **El payload lleva cuatro claves donde parecía llevar dos:** el valor crudo de marca y color —que es lo que se enseña al usuario— y su versión normalizada —que es contra lo que filtra el motor—. Sin las derivadas, el filtro no puede cumplir D03.\n'
        '2. **El motor tiene que saber buscar texto dentro de un campo de metadatos.** Es un requisito duro: quien no pueda queda descartado sin prueba de humo. Filtra la lista de candidatos y por eso está escrito aquí, antes de mirarlos.\n'
        '3. **Hay que indexar los dos campos derivados**, no solo uno, y ese coste se paga en la ingesta.\n'
        '\n'
        '> ⚠️ Lo obligatorio manda sobre la mejora: las cuatro consultas del enunciado filtran por marca con igualdad, y eso tiene que seguir siendo exacto. El color se añade **al lado**, nunca cambiando cómo se comporta la marca.',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '# C · D13 · Qué se guarda en cada punto\n'
        '\n'
        'El enunciado (§3.2) exige definir *"esquema, dimensión, métrica, IDs, metadatos y política de valores nulos"*. Esta sección cubre los metadatos; la D, los nulos.\n'
        '\n'
        '**Buena parte de esta decisión no se decide: se hereda.** Los notebooks posteriores consumen campos concretos, y quitarlos no es ahorrar sino romper lo que viene después:\n'
        '\n'
        '| Campo | Quién lo necesita |\n'
        '|---|---|\n'
        '| `product_id` | Los CSV de salida del §6 |\n'
        '| `title` | NB05 para mostrar el resultado (§3.3 lo exige) y NB07 para la similitud de título |\n'
        '| `brand`, `color` | Lo que se le enseña al usuario |\n'
        '| `brand_normalized`, `color_normalized` | Contra lo que filtra el motor — sección B |\n'
        '| `catalog_version`, `active` | NB08, para comprobar la versión y la baja |\n'
        '| `text` | **Nadie** |\n'
        '\n'
        'Con eso, `minimo` queda descartado por los requisitos de aguas abajo y no por preferencia. **Lo que D13 decide de verdad es el último escalón:** si el índice guarda además el texto de origen.\n'
        '\n'
        '### La pregunta, en una línea\n'
        '\n'
        '> ¿El índice debe poder reconstruirse **sin el CSV**?\n'
        '\n'
        'Guardar `text` hace el índice autosuficiente. Cuesta lo que cuesta el campo más largo del catálogo, y es el único que ningún notebook posterior abre.\n'
        '\n'
        '### Cómo leer la tabla\n'
        '\n'
        'Una fila por esquema, ordenados de menos a más campos y **anidados** —cada uno contiene al anterior—, para que la comparación mida *qué añade llevar más* y no dos esquemas distintos.\n'
        '\n'
        '| Columna | Qué es | Dirección |\n'
        '|---|---|---|\n'
        '| `bytes_medios` | lo que ocupa un payload típico | menos es mejor |\n'
        '| `bytes_p95` · `bytes_max` | el caso malo y el peor | avisan de si la media engaña |\n'
        '| `mb_total` | los 15.000 puntos juntos | se compara con los MB de vectores de la sección A |\n'
        '\n'
        'El número que decide no es `mb_total` a secas sino **qué fracción de la memoria del índice representa**: un payload que suma un 2 % sobre los vectores es ruido; uno que suma un 30 % es una decisión.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000)\n'
        '# Los tres esquemas de D13 salen de PAYLOAD_SCHEMAS, no de una lista escrita\n'
        '# en la celda: así el notebook no puede inventarse un cuarto a mitad de tabla.\n'
        'print("Esquemas candidatos:")\n'
        'for nombre, campos in PAYLOAD_SCHEMAS.items():\n'
        '    print(f"  {nombre:<18} {len(campos)} campos · {\', \'.join(campos)}")\n'
        '\n'
        '# Se presupuesta sobre el catálogo CON las claves derivadas ya calculadas:\n'
        '# son parte del punto que se va a escribir, no un extra.\n'
        'presupuesto = payload_budget(con_claves, null_policy="omitir_campo")\n'
        'presupuesto.assign(\n'
        '    pct_sobre_vectores=lambda d: (100 * d["mb_total"] / huella["mb_vectores"]).round(1)\n'
        ')',
    ),
    (
        "markdown",
        '### C.1 · De dónde sale cada byte\n'
        '\n'
        'La tabla anterior dice cuánto ocupa cada esquema; esta, **qué campo se lo come**. Si un solo campo explica casi todo el salto, la elección deja de ser "mínimo o completo" y pasa a ser "ese campo, sí o no".',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000)\n'
        'campos_del_maximo = list(PAYLOAD_SCHEMAS["completo_con_text"])\n'
        'por_campo = field_byte_profile(con_claves, campos_del_maximo)\n'
        'por_campo.assign(\n'
        '    pct_del_payload=lambda d: (100 * d["mb_total"] / d["mb_total"].sum()).round(1)\n'
        ').sort_values("mb_total", ascending=False)',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '# D · D14 · Qué se escribe cuando el campo está vacío\n'
        '\n'
        '**La decisión de más consecuencia de este notebook**, y la que menos lo parece. Afecta al **4,39 % de las marcas** y al **37,39 % de los colores** — más de cinco mil productos.\n'
        '\n'
        'El motivo cabe en una frase: **`campo ausente` y `campo == ""` no son lo mismo para un motor**, aunque en Python lo parezcan. De ahí depende que un producto sin marca sea alcanzable o invisible, y eso cae directo sobre los filtros de NB05.\n'
        '\n'
        '| Opción | Qué escribe en el punto | Qué le pasa al producto sin marca |\n'
        '|---|---|---|\n'
        '| `omitir_campo` | la clave no existe | **Invisible** a cualquier filtro sobre ese campo. En varios motores ni siquiera se puede preguntar "¿cuáles no tienen marca?" |\n'
        '| `cadena_vacia` | `brand: ""` | Existe para el filtro; alcanzable con una condición explícita de vacío |\n'
        '| `centinela` | `brand: "(desconocido)"` | Alcanzable y legible, pero **contamina**: un *contiene* que caiga dentro del centinela lo trae sin querer |\n'
        '\n'
        '> 📌 **No confundir con D02**, que era la política de nulos en el *texto codificado*. Datos distintos, decisión distinta — y D02 además quedó sin efecto al ganar A4.\n'
        '\n'
        '### La pregunta de negocio que hay debajo\n'
        '\n'
        'Para las cuatro consultas del enunciado da igual: piden marcas concretas y los tres comportamientos coinciden. Importa para el **control de catálogo**: si nadie puede listar los productos sin marca, nadie los va a arreglar nunca.\n'
        '\n'
        '### Cómo leer la tabla\n'
        '\n'
        'Una fila por política sobre el mismo esquema, para que la única diferencia sea la política. `mb_total` es lo que cuesta; `n_puntos_afectados`, a cuántos les cambia el contenido del punto.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000)\n'
        '# A cuántos puntos afecta la decisión, antes de mirar lo que cuesta: si\n'
        '# fueran cuatro productos, la política sería indiferente.\n'
        'display(field_byte_profile(con_claves, CAMPOS_FILTRABLES + derivadas)\n'
        '        [["campo", "n_vacios", "pct_vacios"]])\n'
        '\n'
        '# Las tres políticas sobre el MISMO esquema: lo único que cambia es qué se\n'
        '# escribe en el hueco.\n'
        'coste_nulos = pd.concat([\n'
        '    payload_budget(con_claves, null_policy=politica,\n'
        '                   schemas={"completo": PAYLOAD_SCHEMAS["completo"]})\n'
        '    .assign(politica=politica)\n'
        '    for politica in NULL_POLICIES\n'
        '])\n'
        'coste_nulos[["politica", "bytes_medios", "bytes_p95", "bytes_max", "mb_total"]]',
    ),
    (
        "markdown",
        '### D.1 · El punto que se escribe, con cada política\n'
        '\n'
        'Los números de arriba ordenan las opciones por coste; esta celda enseña **la consecuencia**: el mismo producto sin color escrito de las tres formas, que es lo que el motor recibe y lo que su filtro compara.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000)\n'
        '# Un producto real sin color, para no razonar sobre un ejemplo inventado.\n'
        'sin_color = con_claves[con_claves["color"].isna()].iloc[0]\n'
        'print(f"producto {sin_color[\'product_id\']} — color ausente en el origen\\n")\n'
        '\n'
        'for politica in NULL_POLICIES:\n'
        '    punto = build_payload(\n'
        '        sin_color, fields=PAYLOAD_SCHEMAS["completo"], null_policy=politica\n'
        '    )\n'
        '    presente = "color" in punto\n'
        '    print(f"{politica:<14} clave presente: {presente!s:<5} "\n'
        '          f"valor: {punto.get(\'color\', \'(sin clave)\')!r}")',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '# E · D15 · Tamaño del lote de ingesta\n'
        '\n'
        '⚠️ **No es la decisión de memoria que parecía.** Se planteó como *"con 8 GB y el modelo cargado, mide RAM antes de subir"*, pero aquí **el modelo no se carga**: los vectores vienen ya calculados de `artifacts/embeddings/`. A 768 dimensiones y 4 bytes por decimal, un vector ocupa 3.072 bytes y el lote más grande ronda el megabyte, así que la RAM no arbitra.\n'
        '\n'
        'Lo que sí decide son tres cosas, y la celda las mide:\n'
        '\n'
        '| Columna | Qué decide | Dirección |\n'
        '|---|---|---|\n'
        '| `pct_del_limite` · `cabe_en_un_mensaje` | El **único techo duro**: pasarse del máximo de mensaje gRPC no ralentiza, corta la petición | ⚠️ acercarse a 100 es fallo, no lentitud |\n'
        '| `n_lotes` | Viajes de red. Menos lotes, menos ida y vuelta | menos es más rápido |\n'
        '| `puntos_reintentados_si_falla` | Trabajo perdido cuando un lote se cae | menos es más seguro |\n'
        '\n'
        'Los dos últimos tiran en direcciones opuestas: **el lote grande ahorra viajes y paga más cuando falla**. Ese es el compromiso real de D15.\n'
        '\n'
        '> 🔗 **D13 y D15 están acopladas.** El techo de mensaje sube con el payload: si D13 se lleva `text`, el lote grande se acerca al límite. Por eso la celda se calcula con el esquema elegido y no con uno fijo.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000)\n'
        '# El payload medio sale del esquema de D13, no de un número redondo: cambiar\n'
        '# de esquema tiene que mover esta tabla, que es de lo que avisa el acople.\n'
        'ESQUEMA_D13 = "completo"      # ← al cambiarlo se ve el efecto sobre el lote\n'
        '\n'
        'bytes_payload = float(\n'
        '    payload_budget(con_claves, null_policy="omitir_campo",\n'
        '                   schemas={ESQUEMA_D13: PAYLOAD_SCHEMAS[ESQUEMA_D13]})\n'
        '    ["bytes_medios"].iloc[0]\n'
        ')\n'
        'print(f"esquema {ESQUEMA_D13}: {bytes_payload:.0f} bytes de payload por punto")\n'
        'print(f"vector           : {DIM * 4} bytes por punto\\n")\n'
        '\n'
        'batch_footprint(DIM, n_points=len(completo), payload_bytes_medios=bytes_payload)',
    ),
    (
        "markdown",
        '### Qué se lleva NB05 de estas tres decisiones\n'
        '\n'
        'Las tres se pagan en el índice que se construye una vez, pero se cobran en cada consulta:\n'
        '\n'
        '1. **D13** fija qué puede devolver `buscar()` sin volver al CSV. Un campo que no esté en el payload no está en el resultado, y §3.3 exige `product_id`, posición, título, metadatos y score.\n'
        '2. **D14** fija si un producto con el metadato vacío es alcanzable. Es la que decide qué contesta el sistema a un filtro que nadie pensó al indexar.\n'
        '3. **D15** solo se nota en el tiempo de ingesta y en qué pasa cuando algo falla a medio camino — que es exactamente lo que el paso 3 del guion de humo comprueba al repetirla.',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '# F · La prueba de humo — el mismo guion contra cada candidato\n'
        '\n'
        '**D12** dejó tres motores en pie: **Qdrant**, **Weaviate** y **Milvus**. Los tres pasan por el mismo guion de diez pasos, escrito **una sola vez** en `aurum.motores.humo` y ejecutado contra un puerto común. Eso es lo que hace que la comparativa compare motores: si cada uno tuviera su guion, las diferencias de la tabla podrían venir del código.\n'
        '\n'
        '### El corpus es la 🔬 muestra, no el catálogo\n'
        '\n'
        'Al revés que el resto de NB04: aquí se prueba el recorrido completo —crear, ingerir, repetir, buscar, filtrar, leer, borrar— tres veces seguidas, y es donde conviene equivocarse barato. El índice bueno se construye después, una sola vez y con el ganador.\n'
        '\n'
        '### Qué automatiza esta sección y qué no\n'
        '\n'
        '| Pasos | Quién |\n'
        '|---|---|\n'
        '| **1–7** · crear · ingerir · repetir · buscar · filtrar · leer · borrar | Las celdas de abajo |\n'
        '| **8–10** · reinicio · motor caído · RAM y volumen | **A mano**, desde la terminal |\n'
        '\n'
        'Los tres últimos salen del proceso de Python, así que aparecen en la tabla como filas `✍️ manual` — presentes y vacías. Si no estuvieran, la persistencia, que es requisito del enunciado, desaparecería del artefacto sin que se note.\n'
        '\n'
        '> ⚠️ **Antes de ejecutar nada, levanta el motor.** Los tres no caben a la vez en esta máquina: uno cada vez, y Milvus el último porque son tres contenedores.\n'
        '>\n'
        '> ```bash\n'
        '> make motor-up MOTOR=qdrant     # espera al healthcheck\n'
        '> ```',
    ),
    (
        "code",
        '# 📄 DATOS · 🔬 catalogo_muestra.csv (1.500) + sus vectores A4 ya codificados\n'
        '# Los vectores salen de la caché de NB03; no se recodifica nada aquí.\n'
        'import os\n'
        '\n'
        'import numpy as np\n'
        'from dotenv import load_dotenv\n'
        '\n'
        'from aurum.embeddings import GeminiEncoder, encode_corpus, truncate_dim, vector_health\n'
        'from aurum.motores import (\n'
        '    MANUAL_STEP_NUMBERS,\n'
        '    MEDICION_ADVERTENCIAS,\n'
        '    FilterCondition,\n'
        '    Point,\n'
        '    FilterProbe,\n'
        '    contains_level,\n'
        '    error_quality,     # paso 9: de quién es la excepción, no solo si la hay\n'
        '    load_smoke,\n'
        '    persistence_check, # paso 8: recuento, conjunto y orden por separado\n'
        '    probe_filters,\n'
        '    read_snapshot,\n'
        '    record_manual,     # los pasos manuales se anotan, no se editan a mano\n'
        '    resource_note,\n'
        '    resource_table,    # paso 10: leído del transcrito de la terminal\n'
        '    run_smoke_test,\n'
        '    save_smoke,\n'
        '    smoke_differences, # lo que la tabla de ✅/❌ esconde\n'
        '    smoke_display,   # para mirar: envuelve el texto en vez de recortarlo\n'
        '    smoke_table,     # para el artefacto: los mismos datos, sin formato\n'
        ')\n'
        'from aurum.plantillas import render_template\n'
        '\n'
        'load_dotenv(Path("..") / ".env")\n'
        'CACHE = Path("..") / "artifacts" / "embeddings"\n'
        'MODELO, CONTRATO = "gemini-embedding-2", "sin_contrato"   # R02\n'
        'PLANTILLA = "A4"                                          # R01\n'
        'LOTE = 128                    # D15\n'
        'ESQUEMA = "completo"          # D13\n'
        'POLITICA_NULOS = "cadena_vacia"   # D14\n'
        '\n'
        'textos = render_template(muestra, PLANTILLA)\n'
        'codificado = encode_corpus(\n'
        '    GeminiEncoder(api_key=os.environ.get("GEMINI_API_KEY"), model_id=MODELO,\n'
        '                  native_dim=3072, window=8192),\n'
        '    textos, corpus_id=f"catalogo_muestra__{PLANTILLA}",\n'
        '    kind="document", contract=CONTRATO, batch_size=32, cache_dir=CACHE,\n'
        ')\n'
        '# 3.072 -> 768 y renormalizar: truncar un vector unitario deja la norma < 1\n'
        '# y el coseno dejaría de ser un coseno.\n'
        'vectores = truncate_dim(codificado.vectors, DIM)\n'
        '# `stats.desde_cache` y no `metadata[...]`: el metadata guarda el valor de la\n'
        '# codificación ORIGINAL -siempre False- y diría que se recodificó cuando no.\n'
        '# Aquí eso no es cosmético: si sale False de verdad, se ha pagado a la API.\n'
        'print(f"vectores: {vectores.shape} · desde caché: {codificado.stats.desde_cache}")\n'
        'print(f"salud   : {vector_health(vectores)}")',
    ),
    (
        "markdown",
        '## F.1 · Los puntos, con el esquema que acaban de decidir C, D y E\n'
        '\n'
        'Aquí se junta todo lo anterior: el vector de R01+R02, el payload de **D13**, los huecos de **D14** y las claves derivadas que la sección B hizo obligatorias. El identificador es `record_id` —UUIDv5, lo impone `README_DATOS`— y es lo que hace la idempotencia del paso 3 gratis: reingerir el mismo punto lo sobrescribe en vez de duplicarlo.',
    ),
    (
        "code",
        '# 📄 DATOS · 🔬 catalogo_muestra.csv (1.500)\n'
        'muestra_con_claves = muestra.copy()\n'
        'for campo in CAMPOS_FILTRABLES:\n'
        '    muestra_con_claves = add_normalized_key(\n'
        '        muestra_con_claves, field=campo, mode=NORMALIZACION\n'
        '    )\n'
        '\n'
        'puntos = [\n'
        '    Point(\n'
        '        record_id=fila["record_id"],\n'
        '        vector=vectores[i],\n'
        '        payload=build_payload(\n'
        '            fila, fields=PAYLOAD_SCHEMAS[ESQUEMA], null_policy=POLITICA_NULOS\n'
        '        ),\n'
        '    )\n'
        '    for i, fila in enumerate(muestra_con_claves.to_dict("records"))\n'
        ']\n'
        '\n'
        'print(f"{len(puntos)} puntos · {len(set(p.record_id for p in puntos))} ids distintos")\n'
        'print(f"payload de ejemplo: {puntos[0].payload}")',
    ),
    (
        "markdown",
        '## F.2 · La consulta del paso 5, y por qué esa y no otra\n'
        '\n'
        'El paso 5 comprueba que **el motor** ejecute el filtro auditando que todo lo devuelto cumpla la condición, y esa auditoría solo destapa a quien lo ignore **si la condición es selectiva**: con una marca que ya domine el top-10, un filtro ignorado pasaría por bueno.\n'
        '\n'
        'De ahí **FILTER-001 del enunciado**: `"herramienta inalámbrica para perforar"` con `brand = Einhell`, el 0,2 % del catálogo. El detalle que lo hace la prueba correcta es que **la palabra "Einhell" no aparece en el texto de la consulta**: la búsqueda vectorial no tiene forma de preferir esa marca, así que si los resultados la cumplen es porque el filtro trabajó.',
    ),
    (
        "code",
        '# 📄 DATOS · consultas_filtradas.csv (4) — se codifican aquí por primera vez\n'
        '# NB02 y NB03 codificaron desarrollo y evaluación, no estas. Son 4 textos\n'
        '# cortos: la caché hace que solo se pague una vez.\n'
        'consulta = filtradas.iloc[0]\n'
        'vec_consultas = encode_corpus(\n'
        '    GeminiEncoder(api_key=os.environ.get("GEMINI_API_KEY"), model_id=MODELO,\n'
        '                  native_dim=3072, window=8192),\n'
        '    filtradas["query_text"].tolist(), corpus_id="consultas_filtradas",\n'
        '    kind="query", contract=CONTRATO, batch_size=8, cache_dir=CACHE,\n'
        ')\n'
        '# Las CUATRO, no solo la primera: el paso 5 del guion usa una, pero la tabla\n'
        '# de filtros las necesita todas -§5 las pide como evidencia mínima-.\n'
        'vectores_consultas = truncate_dim(vec_consultas.vectors, DIM)\n'
        'vector_consulta = vectores_consultas[0]      # FILTER-001, la del paso 5\n'
        '\n'
        '# El filtro se expresa en los términos de la decisión -campo, valor, política-,\n'
        '# no en el lenguaje de ningún motor: cada adaptador lo traduce al suyo, y esa\n'
        '# traducción es justo donde se ve qué sabe hacer cada uno.\n'
        'FILTRO = [FilterCondition(\n'
        '    field="brand",\n'
        '    value=normalize_brand(consulta["filter_value"], NORMALIZACION),\n'
        '    operator="equals",   # D03: vocabulario cerrado, una marca por ficha\n'
        ')]\n'
        'print(f"{consulta[\'workload_id\']}: {consulta[\'query_text\']!r}")\n'
        'print(f"filtro: {FILTRO[0].field} {FILTRO[0].operator} {FILTRO[0].value!r}")\n'
        'print(f"⚠️ la marca NO aparece en el texto: "\n'
        '      f"{consulta[\'filter_value\'].lower() not in consulta[\'query_text\'].lower()}")',
    ),
    (
        "markdown",
        '## F.3 · Ejecutar el guion — ⏱️ **una celda por motor**\n'
        '\n'
        'Cambia `MOTOR` y ejecuta. Antes de cada uno: levanta ese y **baja el anterior**.\n'
        '\n'
        '| Motor | Levantar | Puertos |\n'
        '|---|---|---|\n'
        '| `qdrant` | `make motor-up MOTOR=qdrant` | 6333 · 6334 |\n'
        '| `weaviate` | `make motor-up MOTOR=weaviate` | 8080 · 50051 |\n'
        '| `milvus` | `make motor-up MOTOR=milvus` | 19530 · 9091 |\n'
        '\n'
        '> 🔒 La colección se borra y se recrea en cada pasada, así que necesita `AURUM_ALLOW_RESET=true` en el `.env`. Son dos barreras a propósito —el nombre tiene que empezar por `aurum_humo` **y** el permiso tiene que estar dado—, y conviene devolverlo a `false` al terminar NB04.\n'
        '\n'
        '### Cómo leer la tabla\n'
        '\n'
        'Una fila por paso. `esperado` es lo que el guion exige; `observado`, lo que hizo el motor.\n'
        '\n'
        '| `resultado` | Significa |\n'
        '|---|---|\n'
        '| ✅ pasa | El motor cumple ese paso |\n'
        '| ❌ falla | No cumple — y la fila dice qué pasó, incluida la excepción si la hubo |\n'
        '| ✍️ manual | Se rellena a mano tras ejecutar el comando de F.4 |\n'
        '\n'
        '**Un paso que falla no interrumpe el guion.** Un motor que no sepa filtrar tiene que llegar igualmente al paso de borrado: la tabla vale para comparar porque todas las filas están rellenas, y parar en el primer fallo dejaría al motor peor *descrito*, no peor *valorado*.',
    ),
    (
        "code",
        '# Andamiaje común: se ejecuta una vez y sirve para los tres motores.\n'
        'CANDIDATOS = ("qdrant", "weaviate", "milvus")        # D12\n'
        'HUMO_DIR = Path("..") / "artifacts" / "humo"\n'
        'SDK = {"qdrant": "qdrant_client", "weaviate": "weaviate", "milvus": "pymilvus"}\n'
        '\n'
        '# Reentrante a propósito. `HUMO = {}` borraría una pasada que estuviera en\n'
        '# memoria, y volver a medirla cuesta levantar el motor otra vez. Manda lo que\n'
        '# haya en el kernel; si no hay nada, lo que se guardó en disco; si tampoco,\n'
        '# se empieza vacío.\n'
        '_en_disco, _filtros_en_disco = load_smoke(HUMO_DIR)\n'
        'HUMO = globals().get("HUMO") or _en_disco\n'
        'FILTROS = globals().get("FILTROS") or _filtros_en_disco\n'
        'PERSISTENCIA = globals().get("PERSISTENCIA") or {}\n'
        '\n'
        '\n'
        'def guardar(motor):\n'
        '    """Deja en disco lo medido contra ese motor, en cuanto se mide.\n'
        '\n'
        '    Sin esto el resultado vive solo en la memoria del kernel y reescribir la\n'
        '    comparativa exige volver a levantar los tres motores. El §8 pide lo\n'
        '    contrario: que los artefactos se regeneren sin repetir la medición."""\n'
        '    save_smoke(HUMO_DIR, motor=motor, results=HUMO[motor], filters=FILTROS.get(motor))\n'
        '\n'
        '\n'
        'def anotar(motor, paso, observado, pasa):\n'
        '    """Rellena una de las tres filas manuales y la deja guardada.\n'
        '\n'
        '    La fila ya trae escrito su `esperado` desde antes de medir, así que lo\n'
        '    observado se apunta CONTRA el criterio y no en lugar de él."""\n'
        '    record_manual(HUMO[motor], paso, observed=observado, passed=pasa)\n'
        '    guardar(motor)\n'
        '    print(f"  → fila {paso} de {motor}: {\'✅ pasa\' if pasa else \'❌ falla\'}")\n'
        '\n'
        '\n'
        'def abrir(motor):\n'
        '    """Construye el adaptador del motor. El SDK se importa aquí dentro: probar\n'
        '    uno no obliga a tener instalados los otros dos."""\n'
        '    coleccion = f"aurum_humo_{motor}"   # el prefijo es la primera salvaguarda\n'
        '    if motor == "qdrant":\n'
        '        from aurum.motores.qdrant import QdrantStore\n'
        '        return QdrantStore(\n'
        '            collection=coleccion,\n'
        '            url=os.environ.get("AURUM_QDRANT_URL", "http://localhost:6333"),\n'
        '            api_key=os.environ.get("AURUM_QDRANT_API_KEY"),\n'
        '        )\n'
        '    if motor == "weaviate":\n'
        '        from aurum.motores.weaviate import WeaviateStore\n'
        '        return WeaviateStore(\n'
        '            collection=coleccion,\n'
        '            host=os.environ.get("AURUM_WEAVIATE_HOST", "localhost"),\n'
        '            api_key=os.environ.get("AURUM_WEAVIATE_API_KEY"),\n'
        '        )\n'
        '    if motor == "milvus":\n'
        '        from aurum.motores.milvus import MilvusStore\n'
        '        return MilvusStore(\n'
        '            collection=coleccion,\n'
        '            uri=os.environ.get("AURUM_MILVUS_URI", "http://localhost:19530"),\n'
        '            token=os.environ.get("AURUM_MILVUS_TOKEN"),\n'
        '        )\n'
        '    raise ValueError(f"D12 dejó tres motores; {motor!r} no es uno")\n'
        '\n'
        '\n'
        'def probar(motor):\n'
        '    """Ejecuta los pasos 1-7 contra un motor y guarda el resultado.\n'
        '\n'
        '    Conserva las anotaciones manuales que ya tuviera ese motor: los pasos 9 y\n'
        '    10 son propiedades del motor y del despliegue, no de esta coleccion, y\n'
        '    perderlos por reingerir costaria volver a pararlo y volver a medirlo. El\n'
        '    8 si depende de la coleccion, y por eso se avisa de que hay que rehacerlo.\n'
        '    """\n'
        '    previos = {\n'
        '        r.step: r for r in HUMO.get(motor, [])\n'
        '        if r.step in MANUAL_STEP_NUMBERS and r.passed is not None\n'
        '    }\n'
        '    store = abrir(motor)\n'
        '    try:\n'
        '        print(f"{motor} · versión del servidor: {store.server_version()}")\n'
        '        HUMO[motor] = run_smoke_test(\n'
        '            store, puntos, query_vector=vector_consulta, dim=DIM,\n'
        '            metric="cosine", top_k=TOP_K, batch_size=LOTE, filters=FILTRO,\n'
        '        )\n'
        '        for paso, anterior in previos.items():\n'
        '            record_manual(HUMO[motor], paso,\n'
        '                          observed=anterior.observed, passed=anterior.passed)\n'
        '        if previos:\n'
        '            print(f"anotaciones manuales conservadas: {sorted(previos)}"\n'
        '                  + (" — el paso 8 hay que rehacerlo, la coleccion es otra"\n'
        '                     if 8 in previos else ""))\n'
        '    finally:\n'
        '        # Siempre, incluso si el guion revienta: con 7,9 GB no se puede dejar\n'
        '        # una conexión abierta antes de levantar el siguiente motor.\n'
        '        store.close()\n'
        '    guardar(motor)   # antes de mostrar nada: la pasada ya no depende del kernel\n'
        '    # `smoke_display` y no `smoke_table`: pandas recorta las celdas largas con\n'
        '    # "..." y aquí lo largo es lo que hay que leer -el mensaje de la excepción\n'
        '    # cuando un paso falla-. Los datos son los mismos; cambia cómo se muestran.\n'
        '    return smoke_display(HUMO[motor], motor=motor)\n'
        '\n'
        '\n'
        '# ── Las sondas de filtro: lo que el paso 5 del guion NO llega a probar ───\n'
        '# El guion ejerce UNA consulta con igualdad. Aquí van las cuatro del\n'
        '# enunciado (§5 las pide como evidencia mínima) y el `contains` sobre color,\n'
        '# que la sección B declaró requisito duro y que ningún paso ejerce.\n'
        'COLOR_SONDA = colores_frecuentes[0]          # el más frecuente del catálogo\n'
        'COLOR_FRAGMENTO = COLOR_SONDA[:-1]           # "negro" -> "negr"\n'
        '\n'
        '\n'
        'def oraculo(valor, *, campo, match):\n'
        '    """Cuántos productos cumplen la condición SEGÚN EL CATÁLOGO, en pandas.\n'
        '\n'
        '    Sin esto un cero del motor no dice nada: puede ser que esa marca no tenga\n'
        '    productos en los 1.500 de la muestra, o que el filtro esté roto, y las dos\n'
        '    cosas se leen igual. Es la misma idea que el enunciado pide para la\n'
        '    fidelidad ANN -comparar contra un oráculo exacto-, aplicada al filtro.\n'
        '\n'
        '    Se calcula sobre `muestra`, que es EXACTAMENTE el corpus ingerido: usar el\n'
        '    catálogo completo daría un número que el motor no puede alcanzar.\n'
        '    """\n'
        '    fila = filter_reach(\n'
        '        muestra, [valor], field=campo, match=match, modes=(NORMALIZACION,)\n'
        '    ).iloc[0]\n'
        '    return int(fila["n_productos"])\n'
        '\n'
        '\n'
        'SONDAS = [\n'
        '    FilterProbe(\n'
        '        name=fila["workload_id"],\n'
        '        query_vector=vectores_consultas[i],\n'
        '        conditions=[FilterCondition(\n'
        '            "brand", normalize_brand(fila["filter_value"], NORMALIZACION), "equals"\n'
        '        )],\n'
        '        role="obligatorio",\n'
        '        expected=oraculo(fila["filter_value"], campo="brand", match="equals"),\n'
        '        note=f\'{fila["query_text"]!r} — la marca no aparece en el texto\',\n'
        '    )\n'
        '    for i, fila in enumerate(filtradas.to_dict("records"))\n'
        '] + [\n'
        '    FilterProbe("color · igualdad", vector_consulta,\n'
        '                [FilterCondition("color", COLOR_SONDA, "equals")],\n'
        '                role="referencia",\n'
        '                expected=oraculo(COLOR_SONDA, campo="color", match="equals"),\n'
        '                note="referencia: cuántos casan con el valor entero"),\n'
        '    FilterProbe("color · contiene palabra", vector_consulta,\n'
        '                [FilterCondition("color", COLOR_SONDA, "contains")],\n'
        '                role="palabra",\n'
        '                expected=oraculo(COLOR_SONDA, campo="color", match="contains"),\n'
        '                note="si supera a la igualdad, alcanza los compuestos de B.1"),\n'
        '    FilterProbe("color · contiene fragmento", vector_consulta,\n'
        '                [FilterCondition("color", COLOR_FRAGMENTO, "contains")],\n'
        '                role="fragmento",\n'
        '                expected=oraculo(COLOR_FRAGMENTO, campo="color", match="contains"),\n'
        '                note="0 ⇒ por palabras (nivel 2) · >0 ⇒ subcadena literal (nivel 3)"),\n'
        ']\n'
        '\n'
        'print("oráculo (sobre los 1.500 ingeridos, sin motor de por medio):")\n'
        'for s in SONDAS:\n'
        '    print(f"  {s.name:<28} {s.conditions[0].field} {s.conditions[0].operator}"\n'
        '          f" {s.conditions[0].value!r:<12} → {s.expected} productos")\n'
        '\n'
        '\n'
        'def probar_filtros(motor):\n'
        '    """Las 4 consultas del enunciado + las 3 sondas de `contains`, de una vez."""\n'
        '    store = abrir(motor)\n'
        '    try:\n'
        '        # El paso 7 del guion borró un punto para comprobar el borrado, así\n'
        '        # que la colección tiene 1.499 y el oráculo cuenta sobre 1.500. Si el\n'
        '        # punto borrado casara con alguna sonda, la fila diría "devuelve de\n'
        '        # menos" -o FILTRO ROTO si era el único- siendo el motor correcto.\n'
        '        # Con Einhell a 1 solo producto en la muestra, ese riesgo es real.\n'
        '        # Se repone: el upsert es idempotente por record_id, así que si no\n'
        '        # faltaba no cambia nada.\n'
        '        store.upsert(puntos[:1], batch_size=1)\n'
        '        n = store.count()\n'
        '        if n != len(puntos):\n'
        '            print(f"⚠️ la colección tiene {n} puntos y el oráculo cuenta sobre "\n'
        '                  f"{len(puntos)}: los veredictos van a salir sesgados")\n'
        '        FILTROS[motor] = probe_filters(store, SONDAS, top_k=TOP_K)\n'
        '    finally:\n'
        '        store.close()\n'
        '    if motor in HUMO:\n'
        '        guardar(motor)\n'
        '\n'
        '    tabla = FILTROS[motor]\n'
        '    obligatorias = tabla[tabla["papel"] == "obligatorio"]\n'
        '    # §8: "las consultas filtradas nunca devuelven otra marca". Con cero\n'
        '    # resultados la condición se cumple de forma vacía, así que el oráculo\n'
        '    # decide si ese cero es una ausencia real o un filtro roto.\n'
        '    con_resultados = obligatorias[obligatorias["n_resultados"] > 0]\n'
        '    rotos = obligatorias[obligatorias["veredicto"].str.contains("ROTO")]\n'
        '    print(f"{motor} · marca: {int(con_resultados[\'todos_cumplen\'].sum())}"\n'
        '          f"/{len(con_resultados)} consultas con resultados, todas de la marca pedida")\n'
        '    print(f"{motor} · ceros : {len(obligatorias) - len(con_resultados)} sin resultados"\n'
        '          f" — de ellos {len(rotos)} son FILTRO ROTO según el catálogo")\n'
        '    print(f"{motor} · contains: {contains_level(tabla)}")\n'
        '    return tabla.style.set_properties(**{\n'
        '        "white-space": "pre-wrap", "text-align": "left", "vertical-align": "top",\n'
        '    }).hide(axis="index")\n'
        '\n'
        '\n'
        'def instantanea(motor, etiqueta):\n'
        '    """PASO 8 · Solo lee: no crea, no ingiere, no borra.\n'
        '\n'
        '    Llamar a `probar()` otra vez NO sirve para esto: recrea la colección y\n'
        '    borraría justo los datos cuya supervivencia se quiere comprobar.\n'
        '\n'
        '    Con las dos mitades hechas, `persistence_check` separa las tres preguntas\n'
        '    que un `==` entre listas junta en un solo booleano: recuento, conjunto de\n'
        '    ids y orden. Solo las dos primeras son persistencia.\n'
        '    """\n'
        '    store = abrir(motor)\n'
        '    try:\n'
        '        PERSISTENCIA[(motor, etiqueta)] = read_snapshot(\n'
        '            store, vector_consulta, top_k=TOP_K\n'
        '        )\n'
        '    finally:\n'
        '        store.close()\n'
        '\n'
        '    actual = PERSISTENCIA[(motor, etiqueta)]\n'
        '    print(f"{motor} · {etiqueta}: count={actual.count} · top-{len(actual.ids)} leído")\n'
        '    if etiqueta not in ("antes", "despues"):\n'
        '        return   # lectura suelta de F.3e; se compara a mano con `comparar()`\n'
        '    antes, despues = (PERSISTENCIA.get((motor, e)) for e in ("antes", "despues"))\n'
        '    if not (antes and despues):\n'
        '        print("  (falta la otra mitad: reinicia el motor y llama con la otra etiqueta)")\n'
        '        return\n'
        '\n'
        '    check = persistence_check(antes, despues)\n'
        '    print(f"  recuento : {antes.count} → {despues.count}")\n'
        '    print("  conjunto : " + ("los mismos ids" if check.same_set else\n'
        '                             f"salen {len(check.lost)}, entran {len(check.gained)}"\n'
        '                             f" (solapamiento {check.overlap:.0%})"))\n'
        '    print("  orden    : " + ("igual" if check.same_order else\n'
        '                             f"{check.moved} posiciones cambian"))\n'
        '    if check.max_score_shift is not None:\n'
        '        # La prueba de que un reordenamiento es desempate y no otro índice.\n'
        '        print(f"  score    : se mueve como mucho {check.max_score_shift:.2e}")\n'
        '    print(f"\\n  {check.verdict()}")\n'
        '    if motor in HUMO:\n'
        '        anotar(motor, 8, check.verdict(), check.passed)\n'
        '\n'
        '\n'
        'def comparar(motor, etiqueta_a, etiqueta_b):\n'
        '    """Dos instantáneas cualesquiera, sin anotar nada. Diagnóstico de F.3e.\n'
        '\n'
        '    `instantanea` compara «antes» contra «despues» porque ese par ES el paso\n'
        '    8. Aquí hacen falta tres lecturas para separar dos causas que ese par\n'
        '    confunde: el tiempo que pasa y el reinicio en sí."""\n'
        '    check = persistence_check(PERSISTENCIA[(motor, etiqueta_a)],\n'
        '                              PERSISTENCIA[(motor, etiqueta_b)])\n'
        '    print(f"{motor} · {etiqueta_a} → {etiqueta_b}")\n'
        '    print(f"  {check.verdict()}")\n'
        '    return check\n'
        '\n'
        '\n'
        'def probar_caido(motor):\n'
        '    """PASO 9 · Ejecutar SOLO con el motor parado. Con él vivo no prueba nada.\n'
        '\n'
        '    No mide que el motor se caiga -se para a propósito-, sino qué cuenta el\n'
        '    SDK cuando pasa. Y distingue tres cosas que la palabra "tipada"\n'
        '    junta: una excepción de `builtins` obliga a adivinar por el mensaje, una\n'
        '    del SDK se captura por tipo, y una del transporte (gRPC) es tipada pero\n'
        '    ata el manejo de errores de NB05 a la capa de red del cliente.\n'
        '    """\n'
        '    try:\n'
        '        caido = abrir(motor)\n'
        '        caido.search(vector_consulta, top_k=TOP_K)\n'
        '    except Exception as error:\n'
        '        observado, pasa = error_quality(error, sdk_package=SDK[motor])\n'
        '        print(observado)\n'
        '        if motor in HUMO:\n'
        '            anotar(motor, 9, observado, pasa)\n'
        '    else:\n'
        '        print("El motor sigue respondiendo: no está parado, así que el paso 9 no prueba nada")\n'
        '\n'
        '\n'
        '# Lo que ya estuviera medido, a disco. Aquí y no al final: si esta celda se\n'
        '# ejecuta con una pasada en memoria que nunca se guardó, esa pasada deja de\n'
        '# depender del kernel en el mismo momento.\n'
        'for _motor in list(HUMO):\n'
        '    guardar(_motor)\n'
        'print(f"medido: {sorted(HUMO) or \'nada todavía\'} · guardado en {HUMO_DIR}")\n'
        'print("listo · ejecuta el bloque del motor que tengas levantado")',
    ),
    (
        "markdown",
        '### F.3a · Qdrant\n'
        '\n'
        'El nº 1 del orden de preferencia previo (D12). Panel web en <http://localhost:6333/dashboard>, sin clave.\n'
        '\n'
        'Lo que hay que mirar con atención en su fila: **el paso 5**. Qdrant resuelve el `contains` con `MatchText` sobre un índice de texto declarado al crear la colección, que es coincidencia **por palabras** y no subcadena literal. Si funciona, da el alcance de B.1 **sin** los falsos positivos de B.3 — una ventaja que no estaba prevista.\n'
        '\n'
        '#### ① Levantar\n'
        '\n'
        '```bash\n'
        'make motor-up MOTOR=qdrant\n'
        '```',
    ),
    (
        "markdown",
        '#### ② Pasos 1–7 · automáticos',
    ),
    ("code", 'probar("qdrant")'),
    (
        "markdown",
        '#### ②b Los filtros que el guion no llega a probar\n'
        '\n'
        'El paso 5 ejerce **una** consulta con **igualdad**, que es el mínimo del enunciado. Esta tabla cubre lo que deja fuera, y las dos cosas son exigencias escritas:\n'
        '\n'
        '- **Las cuatro consultas de `consultas_filtradas.csv`.** El §5 las pide como evidencia mínima —*"resultados que cumplan la marca en las cuatro consultas"*— y el §8 lo repite como criterio de corrección: *"las consultas filtradas nunca devuelven otra marca"*.\n'
        '- **El `contains` sobre color.** La sección B lo declaró **requisito duro**, y hasta ahora estaba afirmado desde la documentación de cada SDK en vez de medido.\n'
        '\n'
        '##### Cómo leer la tabla\n'
        '\n'
        '| Columna | Qué es |\n'
        '|---|---|\n'
        '| `papel` | `obligatorio` = las 4 del enunciado · `referencia`/`palabra`/`fragmento` = las sondas de color |\n'
        '| **`n_en_catalogo`** | **el oráculo**: cuántos cumplen según pandas, sobre los mismos 1.500, sin motor |\n'
        '| `n_resultados` | cuántos devolvió el motor, **topado en `top_k`** |\n'
        '| `cumplen` · `todos_cumplen` | cuántos satisfacen de verdad la condición, auditado contra la clave normalizada |\n'
        '| **`veredicto`** | confronta las dos columnas anteriores |\n'
        '| `clave_auditada` | contra qué clave se comprobó (`brand_normalized`, no `brand`) |\n'
        '\n'
        '##### Por qué hace falta el oráculo\n'
        '\n'
        'Un `0` del motor **no dice nada por sí solo**: puede que esa marca no tenga productos entre los 1.500 de la muestra o que el filtro esté roto, y las dos cosas se leen igual. Es la ambigüedad que resolvió `cero_por_falta_de_dato` en B.5, y la misma idea que el enunciado aplica a la fidelidad ANN — *"comparar IDs con un oráculo exacto"*.\n'
        '\n'
        'El oráculo es `filter_reach` sobre **`muestra`**, el corpus ingerido: contra el catálogo completo daría un número que el motor no puede alcanzar.\n'
        '\n'
        '| Veredicto | Qué significa |\n'
        '|---|---|\n'
        '| ✅ ausencia real | El catálogo tampoco tiene ninguno. El cero es correcto |\n'
        '| ✅ coincide con el catálogo | El motor devolvió lo que había, topado en `top_k` |\n'
        '| ❌ **FILTRO ROTO** | El catálogo tiene N y el motor devolvió 0 |\n'
        '| ❌ devuelve de más | El filtro dejó pasar lo que no cumplía |\n'
        '| ⚠️ legítimo si filtra por palabras | Solo en sondas de `contains`: el oráculo cuenta **subcadenas**, así que un motor de nivel 2 devuelve menos **a propósito** |\n'
        '\n'
        '**El truco del `fragmento`:** `negr` es subcadena de `negro` pero **no** una palabra suelta. Un motor de subcadena literal lo encuentra; uno que tokeniza, no. Eso clasifica el nivel sin creerse ninguna documentación:\n'
        '\n'
        '| Nivel | Qué significa |\n'
        '|---|---|\n'
        '| **1 · no soportado** | Descarta al motor por el requisito duro |\n'
        '| **2 · por palabras** | Alcanza los compuestos de B.1 **sin** los falsos positivos de B.3. El mejor |\n'
        '| **3 · subcadena** | Alcanza lo mismo, y además trae los falsos positivos que B.3 cuantificó |\n'
        '\n'
        '> ⚠️ Una consulta con **cero resultados no es un fallo del filtro**: puede que esa marca no tenga productos en la muestra de 1.500. Por eso el recuento de aprobadas se hace solo sobre las que devolvieron algo, y las vacías se cuentan aparte.',
    ),
    ("code", 'probar_filtros("qdrant")'),
    (
        "markdown",
        '#### ③ Paso 10 · recursos — **con el motor vivo**\n'
        '\n'
        '```bash\n'
        'make motor-stats\n'
        '```\n'
        '\n'
        'Va **antes** que el paso 9: con el motor apagado la medición no vale nada. La salida entera —comando incluido— se pega en `artifacts/recursos/qdrant.txt`, y F.3d deriva la fila 10 de ese fichero para que ningún número se recopie a mano.\n'
        '\n'
        '#### ④ Paso 8 · persistencia\n'
        '\n'
        'Ejecuta la celda de abajo, **después** reinicia, y **después** vuelve a ejecutarla con `"despues"`:\n'
        '\n'
        '```bash\n'
        'make motor-down MOTOR=qdrant && make motor-up MOTOR=qdrant\n'
        '```\n'
        '\n'
        '> ⚠️ **No mires el número absoluto, mira que no cambie.** El paso 7 borró un punto y ②b repuso ese mismo punto para poder comparar contra el oráculo, así que la colección debería tener los **1.500**. Lo que el paso 8 comprueba es que ese recuento —sea cual sea— **sobreviva al reinicio**, y que el top-10 traiga los mismos ids. Un cambio entre «antes» y «después» es el fallo, no el valor en sí.',
    ),
    ("code", 'instantanea("qdrant", "antes")'),
    # ── reinicia el motor entre las dos celdas ───────────────────────────────
    ("code", 'instantanea("qdrant", "despues")'),
    (
        "markdown",
        '#### ⑤ Paso 9 · calidad del error — **el último, deja el motor caído**\n'
        '\n'
        '```bash\n'
        'docker compose -f docker/qdrant/compose.yaml stop\n'
        '```',
    ),
    ("code", 'probar_caido("qdrant")'),
    (
        "markdown",
        '#### ⑥ Bajar antes del siguiente\n'
        '\n'
        '```bash\n'
        'make motor-down MOTOR=qdrant\n'
        '```',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '### F.3b · Weaviate\n'
        '\n'
        'Dos cosas propias de este motor que la tabla debería reflejar:\n'
        '\n'
        '- **Devuelve distancia, no similitud** — menor es mejor. El paso 4 lo anota en `observado`; si ahí pusiera `similarity`, el adaptador estaría mintiendo y la comparativa ordenaría al revés.\n'
        '- **La dimensión no se declara:** Weaviate la fija con el primer vector escrito. El paso 1 no puede verificarla, así que la comprobación real es que el paso 4 devuelva algo coherente.\n'
        '\n'
        '#### ① Levantar — con Qdrant ya abajo\n'
        '\n'
        '```bash\n'
        'make motor-down MOTOR=qdrant     # por si acaso\n'
        'make motor-up   MOTOR=weaviate\n'
        '```',
    ),
    (
        "markdown",
        '#### ② Pasos 1–7 · automáticos',
    ),
    ("code", 'probar("weaviate")'),
    (
        "markdown",
        '#### ②b Los filtros — las 4 consultas y el `contains`\n'
        '\n'
        'Weaviate resuelve el `contains` con `like "*valor*"`, que es **subcadena literal**. Si la tabla dice nivel 2 en vez de 3, el adaptador está haciendo otra cosa de la que cree.',
    ),
    ("code", 'probar_filtros("weaviate")'),
    (
        "markdown",
        '#### ③ Paso 10 · recursos · ④ Paso 8 · persistencia\n'
        '\n'
        '```bash\n'
        'make motor-stats                                     # ③ con el motor vivo\n'
        'make motor-down MOTOR=weaviate && make motor-up MOTOR=weaviate   # ④\n'
        '```\n'
        '\n'
        'La salida de ③, entera, a `artifacts/recursos/weaviate.txt`.',
    ),
    ("code", 'instantanea("weaviate", "antes")'),
    ("code", 'instantanea("weaviate", "despues")'),
    (
        "markdown",
        '#### ⑤ Paso 9 · calidad del error\n'
        '\n'
        '```bash\n'
        'docker compose -f docker/weaviate/compose.yaml stop\n'
        '```',
    ),
    ("code", 'probar_caido("weaviate")'),
    (
        "markdown",
        '#### ⑥ Bajar antes del siguiente\n'
        '\n'
        '```bash\n'
        'make motor-down MOTOR=weaviate\n'
        '```',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '### F.3c · Milvus\n'
        '\n'
        '> ⚠️ **El último, y con los otros dos abajo.** Son tres contenedores (etcd + minio + standalone) y es el candidato pesado para 7,9 GB.\n'
        '\n'
        'Dos puntos donde espero que falle o se comporte distinto:\n'
        '\n'
        '- **`pymilvus` instalado es 3.x** y el adaptador se escribió contra la API 2.x documentada. Es el sospechoso principal de la primera pasada.\n'
        '- **El `like "%valor%"` con comodín por delante.** De eso depende que cumpla el requisito duro: las versiones antiguas de Milvus solo resolvían prefijos. Aquí solo se usa `equals`, así que **el paso 5 pasando no demuestra que el `contains` funcione** — eso hay que probarlo aparte antes de darlo por bueno para el color.\n'
        '\n'
        '#### ① Levantar — con los otros dos abajo\n'
        '\n'
        '```bash\n'
        'make all-down                    # apaga los tres, por si queda alguno\n'
        'make motor-up MOTOR=milvus\n'
        '```',
    ),
    (
        "markdown",
        '#### ② Pasos 1–7 · automáticos',
    ),
    ("code", 'probar("milvus")'),
    (
        "markdown",
        '#### ②b Los filtros — **aquí se juega el requisito duro**\n'
        '\n'
        'Esta es la celda que decide si Milvus sirve. El adaptador usa `like "%valor%"` con **comodín por delante**, y las versiones antiguas de Milvus solo resolvían prefijos: si esa sonda devuelve `NO SOPORTADO` o cero, Milvus queda descartado por el requisito duro de la sección B, sin importar cómo le haya ido en los pasos 1–7.',
    ),
    ("code", 'probar_filtros("milvus")'),
    (
        "markdown",
        '#### ③ Paso 10 · recursos — **suma los tres contenedores**\n'
        '\n'
        '```bash\n'
        'make motor-stats\n'
        '```\n'
        '\n'
        'La salida entera va a `artifacts/recursos/milvus.txt`, y F.3d la suma sola.\n'
        '\n'
        '> ⚠️ Tienen que salir **etcd + minio + standalone**, no solo `standalone`: es lo que de verdad cuesta tener Milvus en marcha, y es el criterio por el que el plan lo marca como pesado.\n'
        '>\n'
        '> **Attu queda fuera.** Es una herramienta de inspección, no parte del motor: sumarla haría que Milvus pareciera más caro frente a Qdrant, que sirve su panel desde el mismo proceso. Levántala después de anotar, con `docker compose -f docker/milvus/compose.yaml up -d attu` (<http://localhost:8000>).\n'
        '\n'
        '#### ④ Paso 8 · persistencia\n'
        '\n'
        '```bash\n'
        'make motor-down MOTOR=milvus && make motor-up MOTOR=milvus\n'
        '```',
    ),
    ("code", 'instantanea("milvus", "antes")'),
    ("code", 'instantanea("milvus", "despues")'),
    (
        "markdown",
        '#### ⑤ Paso 9 · calidad del error\n'
        '\n'
        '```bash\n'
        'docker compose -f docker/milvus/compose.yaml stop\n'
        '```',
    ),
    ("code", 'probar_caido("milvus")'),
    (
        "markdown",
        '#### ⑥ Apagar y devolver el permiso\n'
        '\n'
        '```bash\n'
        'make all-down\n'
        '```\n'
        '\n'
        'Y pon `AURUM_ALLOW_RESET=false` en el `.env`: el permiso de borrar solo hacía falta mientras durase esta sección.',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '## F.3d · El paso 10, desde los transcritos de la terminal\n'
        '\n'
        '`docker stats` y `docker system df` se ejecutan en la terminal, así que lo que entra al repo es el texto pegado tal cual en `artifacts/recursos/{motor}.txt`. Esta celda lo parsea: la tabla se **deriva** del transcrito en vez de recopiarse a mano, que es donde se cuela un número que ya no corresponde a lo que se midió.\n'
        '\n'
        '### Lo que la tabla dice, y lo que no\n'
        '\n'
        '| Columna | Cómo se lee |\n'
        '|---|---|\n'
        '| `ram_mib` | Suma de los contenedores del motor. Milvus son tres; `attu` queda fuera por la regla escrita en su `compose.yaml` |\n'
        '| `dentro_del_limite` | El **único criterio del paso 10 que estaba declarado antes de medir**: el `mem_limit` de cada compose. Inventar ahora un umbral en MiB sería ponerle la vara al ganador |\n'
        '| `mas_apretado` | Qué contenedor va más justo dentro de su propio límite |\n'
        '| `volumen_mb` | ⚠️ **No comparable entre motores** — ver abajo |\n'
        '| `en_marcha` | Cuánto llevaba vivo el contenedor al tomar la foto |\n'
        '\n'
        '**Por qué el volumen no compara.** Los volúmenes con nombre sobreviven a `down`, así que cada uno arrastra todo lo que ese motor escribió mientras se desarrollaba su adaptador, más la preasignación de WAL —los 144,9 MB de `etcd` son fichero reservado, no catálogo—. Los mismos 1.500 × 768 son 4,6 MB en los tres: esa columna mide edad del volumen, no eficiencia. El número que sí valdrá es el de la sección G, con colección nueva y un solo motor.\n'
        '\n'
        '**Y la RAM es una foto en reposo**, no un pico bajo carga. Con 4,6 MB de dato dentro, lo que se está midiendo es el coste base del proceso — que es justamente lo que separa a un contenedor de tres.',
    ),
    (
        "code",
        '# 📄 DATOS · artifacts/recursos/{motor}.txt — lo que imprimió `make motor-stats`\n'
        'RECURSOS = Path("..") / "artifacts" / "recursos"\n'
        '# Attu es panel de inspección, no motor: sumarlo haría a Milvus más caro\n'
        '# frente a Qdrant, que sirve el suyo desde el mismo proceso.\n'
        'recursos = resource_table(RECURSOS, CANDIDATOS, exclude=("aurum-market-milvus-attu",))\n'
        '\n'
        'for _, fila in recursos.iterrows():\n'
        '    if fila["motor"] in HUMO:\n'
        '        anotar(fila["motor"], 10, resource_note(fila), bool(fila["dentro_del_limite"]))\n'
        '\n'
        'sin_medir = [m for m in CANDIDATOS if m not in set(recursos["motor"])]\n'
        'print(f"sin transcrito: {sin_medir or \'ninguno\'}")\n'
        'recursos',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '## F.3e · La anomalía de Milvus, aislada\n'
        '\n'
        'La primera vez que se midió el paso 8, Milvus mantuvo el recuento pero **devolvió otro top-10**; al repetirlo sobre la misma colección, horas después y sin reingerir, salió idéntico. La anomalía existió y no se reprodujo, y hay que explicar las dos cosas.\n'
        '\n'
        '### El problema del experimento original\n'
        '\n'
        'Aquel par de lecturas mezclaba **dos variables**: entre «antes» y «después» pasó el reinicio, pero también pasó el tiempo — y la primera lectura se tomó justo tras ingerir 1.500 puntos y el `upsert` de reposición de ②b. Con dos lecturas no se sabe cuál causó el cambio.\n'
        '\n'
        'La hipótesis es el estado interno de segmentos —Milvus sirve desde segmentos en crecimiento y desde otros sellados e indexados, y una consulta puede ver un reparto distinto según cuándo llegue—, pero es **hipótesis, no causa medida**.\n'
        '\n'
        '### Tres lecturas en vez de dos\n'
        '\n'
        '| Lectura | Cuándo |\n'
        '|---|---|\n'
        '| `recien_ingerido` | inmediatamente después de `probar("milvus")`, sin esperar |\n'
        '| `asentado` | dos o tres minutos después, **sin reiniciar nada** |\n'
        '| `tras_reinicio` | después de `down` + `up` |\n'
        '\n'
        'Cada par aísla una cosa:\n'
        '\n'
        '| Par | Qué aísla | Si cambia ahí |\n'
        '|---|---|---|\n'
        '| `recien_ingerido` → `asentado` | **el tiempo**, sin reinicio de por medio | El índice se estaba asentando. El reordenamiento no lo causa el reinicio, sino consultar antes de que el índice esté al día |\n'
        '| `asentado` → `tras_reinicio` | **el reinicio**, con el índice ya asentado | Es del reinicio: Milvus recarga o reconstruye de otra forma. Este par **es** el paso 8 |\n'
        '| `recien_ingerido` → `tras_reinicio` | las dos a la vez | Es lo que se midió la primera vez, y por eso no distinguía |\n'
        '\n'
        '> ⚡ **Por qué esto importa más allá de Milvus.** Si el cambio aparece en el primer par, conecta directamente con el paso 3: Milvus es de los que responden `índice: no lo reporta el motor`, y §3.2 pide *"verificad el recuento final y el estado de indexación antes de aceptar consultas"*. Un motor que no informa de cuándo su índice está al día, y que mientras tanto devuelve otro top-10 para la misma consulta, es exactamente el riesgo que ese requisito cubre. Qdrant lo reporta y no divergió ninguna de las dos veces.\n'
        '\n'
        'Si no cambia en ningún par, la conclusión también vale: la anomalía no es reproducible y se declara como observación única, sin atribuirle una causa que no se ha medido.\n'
        '\n'
        '#### ① Reingerir en limpio\n'
        '\n'
        '```bash\n'
        'make motor-up MOTOR=milvus\n'
        '```\n'
        '\n'
        '> 🔒 Recrea la colección, así que necesita `AURUM_ALLOW_RESET=true` en el `.env`.\n'
        '>\n'
        '> Las anotaciones manuales de Milvus **se conservan** al reejecutar `probar`, pero la del paso 8 quedará pendiente de rehacer: la colección es otra.',
    ),
    ("code", 'probar("milvus")'),
    (
        "markdown",
        '#### ② La lectura inmediata — **sin esperar, es la mitad del experimento**',
    ),
    ("code", 'instantanea("milvus", "recien_ingerido")'),
    (
        "markdown",
        '#### ③ La misma lectura, dos o tres minutos después y **sin tocar el motor**',
    ),
    ("code", 'instantanea("milvus", "asentado")'),
    (
        "markdown",
        '#### ④ Ahora sí, el reinicio\n'
        '\n'
        '```bash\n'
        'make motor-down MOTOR=milvus && make motor-up MOTOR=milvus\n'
        '```',
    ),
    ("code", 'instantanea("milvus", "tras_reinicio")'),
    (
        "markdown",
        '#### ⑤ Los tres pares, y la fila 8 definitiva\n'
        '\n'
        'La fila 8 se anota con el par `asentado → tras_reinicio`, que es el que responde a lo que el paso 8 pregunta: si el estado sobrevive a un reinicio. El otro par es diagnóstico y no entra en la tabla — explica *por qué*, no *si*.',
    ),
    (
        "code",
        'tiempo = comparar("milvus", "recien_ingerido", "asentado")\n'
        'reinicio = comparar("milvus", "asentado", "tras_reinicio")\n'
        'mezclado = comparar("milvus", "recien_ingerido", "tras_reinicio")\n'
        '\n'
        'print()\n'
        'if not tiempo.same_order:\n'
        '    print("→ el índice se estaba asentando: el reinicio no era la causa.")\n'
        '    print("  Conecta con el paso 3: Milvus no reporta el estado de indexación.")\n'
        'elif not reinicio.same_order:\n'
        '    print("→ es el reinicio, con el índice ya asentado.")\n'
        'else:\n'
        '    print("→ no se reproduce por ninguna vía: queda como observación única.")\n'
        '\n'
        '# El par que ES el paso 8. Sustituye a la anotación anterior de Milvus.\n'
        'anotar("milvus", 8, reinicio.verdict(), reinicio.passed)',
    ),
    (
        "markdown",
        '```bash\n'
        'make motor-down MOTOR=milvus\n'
        '```',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '## F.4 · La comparativa, y el artefacto\n'
        '\n'
        'Cuando los tres motores hayan pasado por F.3, esta celda junta las tres tablas y escribe `artifacts/comparativa_motores.md`, el artefacto de **R03**.\n'
        '\n'
        '### La tabla de ✅/❌ no elige motor\n'
        '\n'
        'Si los tres pasan los diez pasos —y los pasan—, esa tabla demuestra que los tres **sirven**, que era la pregunta eliminatoria, pero no separa a ninguno. Lo que separa está en la columna `observado`, y por eso el artefacto lleva además:\n'
        '\n'
        '| Sección | Qué aporta a R03 |\n'
        '|---|---|\n'
        '| **Dónde no se comportaron igual** | Solo los pasos con `observado` distinto entre motores. Las filas donde coinciden no deciden nada y estorban |\n'
        '| **Segundos por paso** | El coste de cada operación. Ojo: son de una sola pasada sobre 1.500 puntos, no un banco de pruebas |\n'
        '| **Nivel de `contains`** | El requisito duro de la sección B |\n'
        '| **Paso 10 · recursos** | Con sus condiciones de medición declaradas |\n'
        '\n'
        '> **R03 no se elige aquí, se lee.** El motor sale de esta tabla contrastada con los criterios de la sección A y con los seis de la sesión 1 —memoria del índice y dependencia del proveedor son los que más se mueven al elegir motor—. Escribir la conclusión antes de tener las tres filas sería elegir primero y justificar después.',
    ),
    (
        "code",
        '# Requiere haber ejecutado F.3 con cada motor. Con uno solo, la tabla se\n'
        '# genera igual y deja claro cuáles faltan: un artefacto a medias que dice\n'
        '# que lo está es más útil que ninguno.\n'
        '# No hace falta tener el kernel de F.3 vivo: `HUMO` se rellena desde\n'
        '# artifacts/humo/ si la memoria está vacía (§8, artefactos regenerables).\n'
        'faltan = [m for m in CANDIDATOS if m not in HUMO]\n'
        'print(f"motores medidos: {list(HUMO) or \'ninguno todavía\'}")\n'
        'print(f"faltan         : {faltan or \'ninguno\'}")\n'
        '\n'
        'if HUMO:\n'
        '    comparativa = pd.concat(\n'
        '        [smoke_table(pasos, motor=nombre) for nombre, pasos in HUMO.items()]\n'
        '    )\n'
        '    resumen = comparativa.pivot_table(\n'
        '        index=["paso", "comprobacion"], columns="motor",\n'
        '        values="resultado", aggfunc="first",\n'
        '    )\n'
        '    display(resumen)\n'
        '\n'
        '    destino = Path("..") / "artifacts" / "comparativa_motores.md"\n'
        '    cabecera = [\n'
        '        "# Prueba de humo · comparativa de motores (D12 → R03)",\n'
        '        "",\n'
        '        f"Corpus: `catalogo_muestra.csv` ({len(puntos)} puntos) · "\n'
        '        f"dim {DIM} · métrica cosine · lote {LOTE} (D15)",\n'
        '        f"Payload: `{ESQUEMA}` (D13) · nulos: `{POLITICA_NULOS}` (D14)",\n'
        '        f"Filtro del paso 5: `{FILTRO[0].field} {FILTRO[0].operator} "\n'
        '        f"{FILTRO[0].value!r}` ({consulta[\'workload_id\']})",\n'
        '        "",\n'
        '    ]\n'
        '    if faltan:\n'
        '        cabecera += [f"> ⚠️ Tabla incompleta: falta medir {\', \'.join(faltan)}.", ""]\n'
        '\n'
        '    # El nivel de `contains` de cada motor: es el requisito duro de la\n'
        '    # sección B y no sale de la tabla de pasos, así que se añade aparte.\n'
        '    niveles = [\n'
        '        "", "## Nivel de `contains` sobre metadatos (requisito duro · sección B)", "",\n'
        '        *(f"- **{m}** — {contains_level(t)}" for m, t in FILTROS.items()),\n'
        '        "",\n'
        '        "## Las cuatro consultas filtradas (§5)", "",\n'
        '    ]\n'
        '    if FILTROS:\n'
        '        obligatorias = pd.concat([\n'
        '            t[t["papel"] == "obligatorio"].assign(motor=m) for m, t in FILTROS.items()\n'
        '        ])\n'
        '        niveles.append(\n'
        '            obligatorias.pivot_table(\n'
        '                index=["caso", "filtro"], columns="motor",\n'
        '                values="todos_cumplen", aggfunc="first",\n'
        '            ).to_markdown()\n'
        '        )\n'
        '\n'
        '    # Lo que la tabla de ✅/❌ esconde. Con diez aprobados en los tres, estas\n'
        '    # son las unicas filas que pueden decidir R03.\n'
        '    diferencias = smoke_differences(HUMO)\n'
        '    if not diferencias.empty:\n'
        '        niveles += [\n'
        '            "", "## Donde los motores NO se comportaron igual", "",\n'
        '            "Solo los pasos con `observado` distinto entre motores: los que "\n'
        '            "coinciden no separan a nadie.", "",\n'
        '            diferencias.to_markdown(), "",\n'
        '        ]\n'
        '    tiempos = comparativa.pivot_table(\n'
        '        index=["paso", "comprobacion"], columns="motor",\n'
        '        values="segundos", aggfunc="first",\n'
        '    ).dropna(how="all")\n'
        '    if not tiempos.empty:\n'
        '        niveles += [\n'
        '            "", "## Segundos por paso", "",\n'
        '            "Una sola pasada sobre 1.500 puntos, no un banco de pruebas: "\n'
        '            "sirven para ver ordenes de magnitud, no para afinar.", "",\n'
        '            tiempos.round(3).to_markdown(), "",\n'
        '        ]\n'
        '\n'
        '    # El paso 10 y las condiciones en que se midió. Van al artefacto y no\n'
        '    # solo a la celda: quien lea la comparativa tiene que poder ver que el\n'
        '    # volumen no compara ANTES de usarlo para elegir motor.\n'
        '    recursos = resource_table(\n'
        '        Path("..") / "artifacts" / "recursos", CANDIDATOS,\n'
        '        exclude=("aurum-market-milvus-attu",),\n'
        '    )\n'
        '    if not recursos.empty:\n'
        '        niveles += [\n'
        '            "", "## Paso 10 · recursos, y en qué condiciones se midieron", "",\n'
        '            recursos.to_markdown(index=False), "",\n'
        '            *(f"- {aviso}" for aviso in MEDICION_ADVERTENCIAS),\n'
        '            "",\n'
        '            "Transcritos íntegros de `docker stats` y `docker system df -v` en "\n'
        '            "`artifacts/recursos/`; la tabla de arriba se deriva de ellos.",\n'
        '        ]\n'
        '\n'
        '    destino.write_text(\n'
        '        "\\n".join(cabecera) + resumen.to_markdown() + "\\n"\n'
        '        + "\\n".join(niveles) + "\\n",\n'
        '        encoding="utf-8",\n'
        '    )\n'
        '    print(f"\\nEscrito {destino}")',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '# G · El índice definitivo: los 15.000 sobre Qdrant\n'
        '\n'
        'R03 eligió Qdrant. Aquí se construye la colección que van a usar NB05, NB06 y NB08, y se comprueba que **es la que se cree que es**, que no es lo mismo que comprobar que se creó.\n'
        '\n'
        '### Qué cambia respecto a la prueba de humo\n'
        '\n'
        '| | F · humo | G · índice |\n'
        '|---|---|---|\n'
        '| Corpus | 🔬 1.500 desechables | 📚 **15.000, los de verdad** |\n'
        '| Colección | `aurum_humo_*`, se recrea en cada pasada | `aurum_catalogo__*`, **se conserva** |\n'
        '| La pregunta | ¿sirve este motor? | ¿es este índice el que creo? |\n'
        '| `recreate` | `True` por defecto | **`False` por defecto** |\n'
        '| Pasos manuales | 8, 9 y 10 | **7 y 8** — el error con el motor caído no se repite: es del SDK, no del índice |\n'
        '\n'
        '### Los dos prefijos no son cosmética\n'
        '\n'
        'El guion de humo **borra y recrea** su colección cada vez que se ejecuta. Si el índice bueno viviera bajo el mismo prefijo, una errata en un nombre bastaría para llevarse por delante los 15.000 puntos. Con `aurum_catalogo` y `aurum_humo` separados en el guardián, el guion de humo no puede alcanzarlo **ni equivocándose**.\n'
        '\n'
        '### El nombre lleva el contrato dentro\n'
        '\n'
        '`aurum_catalogo__gemini_embedding_2__A4__768` — modelo, plantilla y dimensión. Los tres invalidan los vectores guardados si cambian, así que el nombre los declara en vez de dejarlos en la memoria de quien lo creó.\n'
        '\n'
        'Es además la mitad del control de versionado al que `config.yaml` se comprometió en el criterio 6: **migrar es construir la colección nueva al lado y cambiar el puntero**, nunca reindexar sobre la viva. Con el contrato en el nombre, las dos pueden convivir mientras dure la migración.\n'
        '\n'
        '### El índice se construye con la configuración ANN por defecto, a propósito\n'
        '\n'
        'Lo que se prueba en G es **el motor contra el catálogo completo**: que ingiere los 15.000 sin duplicarlos, que el esquema es el declarado, que sobrevive a un reinicio y que los canarios vuelven donde deben. Para eso los parámetros del ANN no hacen falta, y tocarlos aquí sería peor: **D16 —el recall mínimo y el p95 máximo— se fija antes de ver ninguna curva**, y unos valores elegidos ahora se acabarían escribiendo a la medida de lo que saliera.\n'
        '\n'
        'El estudio del índice —familia, parámetros, fidelidad y latencia— es **NB06**. Allí se barre `ef`, que se ajusta por consulta y no obliga a reconstruir; si llegara a tocar `m` o `ef_construct`, que se fijan al construir, haría falta un índice nuevo al lado — la operación que el criterio 6 ya contempla.\n'
        '\n'
        '### Antes de empezar: el volumen limpio\n'
        '\n'
        'El volumen de Qdrant arrastra 293,8 MB de historia del desarrollo del adaptador. Para que el tamaño medido en el paso 8 sea el **del índice** y no el de esa historia, hay que borrarlo antes de construir:\n'
        '\n'
        '```bash\n'
        'docker compose -f docker/qdrant/compose.yaml down -v   # -v borra el volumen\n'
        'make motor-up MOTOR=qdrant\n'
        '```\n'
        '\n'
        'Se lleva también la colección de humo, y no pasa nada: sus resultados están en `artifacts/humo/*.json` y la comparativa se regenera sin motor.',
    ),
    (
        "markdown",
        '## G.1 · Los 15.000 puntos\n'
        '\n'
        'Mismo montaje que F.1 con dos diferencias: el corpus es el completo y el recorte de A4 se calcula sobre él, así que el corte es la mediana de los 15.000 —**936 caracteres**— y no la de la muestra.\n'
        '\n'
        '> 💸 **El freno de la celda.** Los vectores tienen que salir de la caché de NB03. Si la clave no casara —una plantilla distinta, otro contrato—, `encode_corpus` se pondría a codificar 15.000 documentos contra la API de pago sin preguntar. La celda comprueba que el `.npy` existe **antes** de llamar, y para si no está.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000) + sus vectores A4 de la caché\n'
        'from aurum.embeddings import cache_key, corpus_fingerprint\n'
        'from aurum.motores import (\n'
        '    ACCEPTANCE_MANUAL_NUMBERS,\n'
        '    CATALOG_PREFIX,\n'
        '    catalog_collection_name,\n'
        '    run_acceptance,\n'
        '    self_retrieval_canaries,\n'
        ')\n'
        '\n'
        'COLECCION = catalog_collection_name(model=MODELO, template=PLANTILLA, dim=DIM)\n'
        'CORPUS_ID = f"catalogo_productos__{PLANTILLA}"\n'
        '\n'
        '# A4 sobre el catalogo completo: el corte es la mediana de los 15.000.\n'
        'textos_completo = render_template(completo, PLANTILLA)\n'
        '\n'
        '# El freno: mirar la cache ANTES de llamar al codificador. `encode_corpus`\n'
        '# codificaria sin preguntar, y son 15.000 documentos contra una API de pago.\n'
        'clave = cache_key(\n'
        '    model_id=MODELO, kind="document", contract=CONTRATO,\n'
        '    corpus_id=CORPUS_ID, fingerprint=corpus_fingerprint(textos_completo),\n'
        ')\n'
        'if not (CACHE / f"{clave}.npy").exists():\n'
        '    raise RuntimeError(\n'
        '        f"Los vectores de {CORPUS_ID} no estan en cache ({clave}).\\n"\n'
        '        f"Codificarlos son 15.000 llamadas de pago, asi que la celda para "\n'
        '        f"aqui en vez de pagarlas sin avisar. Comprueba que MODELO, "\n'
        '        f"CONTRATO y PLANTILLA son los de R01/R02 antes de forzar nada."\n'
        '    )\n'
        '\n'
        'codificado_completo = encode_corpus(\n'
        '    GeminiEncoder(api_key=os.environ.get("GEMINI_API_KEY"), model_id=MODELO,\n'
        '                  native_dim=3072, window=8192),\n'
        '    textos_completo, corpus_id=CORPUS_ID,\n'
        '    kind="document", contract=CONTRATO, batch_size=32, cache_dir=CACHE,\n'
        ')\n'
        'vectores_completo = truncate_dim(codificado_completo.vectors, DIM)\n'
        '\n'
        'print(f"coleccion : {COLECCION}")\n'
        'print(f"vectores  : {vectores_completo.shape} · desde cache: "\n'
        '      f"{codificado_completo.stats.desde_cache}")\n'
        '# Normas ~1 y sin NaN/inf: dos de las comprobaciones que pide el plan, y se\n'
        '# hacen sobre la matriz ANTES de subirla. Un NaN dentro del motor ya no se\n'
        '# distingue de un vector legitimo.\n'
        'print(f"salud     : {vector_health(vectores_completo)}")',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000)\n'
        '# Las claves derivadas de D03 y el payload de D13/D14, igual que en F.1.\n'
        'completo_con_claves = completo.copy()\n'
        'for campo in CAMPOS_FILTRABLES:\n'
        '    completo_con_claves = add_normalized_key(\n'
        '        completo_con_claves, field=campo, mode=NORMALIZACION\n'
        '    )\n'
        '\n'
        'puntos_completo = [\n'
        '    Point(\n'
        '        record_id=fila["record_id"],\n'
        '        vector=vectores_completo[i],\n'
        '        payload=build_payload(\n'
        '            fila, fields=PAYLOAD_SCHEMAS[ESQUEMA], null_policy=POLITICA_NULOS\n'
        '        ),\n'
        '    )\n'
        '    for i, fila in enumerate(completo_con_claves.to_dict("records"))\n'
        ']\n'
        'canarios = self_retrieval_canaries(puntos_completo, n=3)\n'
        '\n'
        'print(f"{len(puntos_completo)} puntos · "\n'
        '      f"{len(set(p.record_id for p in puntos_completo))} ids distintos")\n'
        'print(f"canarios: {[c.record_id for c in canarios]}")',
    ),
    (
        "markdown",
        '## G.2 · Construir y aceptar\n'
        '\n'
        'Seis comprobaciones automáticas. Las cuatro primeras son de contrato y las dos últimas son las que de verdad cuestan encontrar de otra forma:\n'
        '\n'
        '| Paso | Qué responde |\n'
        '|---|---|\n'
        '| 1 · colección | Dimensión y métrica explícitas, con el prefijo del catálogo |\n'
        '| 2 · ingesta | `count() == 15.000`, y **cuántos vectores por segundo** — el número que va al README |\n'
        '| 3 · índice al día | §3.2 lo pide **antes** de aceptar consultas. Dos datos: si estaba listo al terminar la ingesta y, si no, **cuánto tardó** |\n'
        '| 4 · dimensión real | Preguntándosela a la colección, no repitiendo la variable del notebook |\n'
        '| 5 · idempotencia | Reingerir los mismos 15.000 no puede sumar ni uno |\n'
        '| **6 · canarios** | Tres puntos buscados **con su propio vector** deben volver los primeros |\n'
        '\n'
        '### Por qué el canario es la búsqueda de sí mismo\n'
        '\n'
        'Lo tentador sería usar una consulta de desarrollo con su producto relevante, pero sería un mal canario: que un relevante entre en el top-10 es una pregunta de **calidad**, y con un Recall@10 de 0,26 fallaría a menudo con el índice perfectamente sano. Un canario que salta por lo que no vigila no vigila nada.\n'
        '\n'
        'Buscar un punto con su propio vector tiene una respuesta que no depende del modelo: **debe volver él, el primero**. Y detecta la avería que ningún recuento ve —vectores y payloads desalineados durante la ingesta por lotes—, que es el fallo silencioso clásico: con recuento, dimensión e idempotencia correctos, un índice desalineado pasa todo lo demás.\n'
        '\n'
        '### El paso 3 mide dos cosas, y la segunda solo se puede medir aquí\n'
        '\n'
        'Con 1.500 puntos el índice se construía en un suspiro y el paso salía `listo` siempre. Con 15.000 no: el `count()` ya dice 15.000 mientras el grafo HNSW sigue construyéndose por detrás. En Qdrant eso **no da resultados incorrectos** —busca en exacto sobre los segmentos sin indexar, y por eso los canarios pasan igual—, pero es el momento en que no hay que aceptar tráfico ni medir latencia.\n'
        '\n'
        'Por eso la fila anota las dos mitades: **si estaba listo al terminar la ingesta** y **cuántos segundos tardó en estarlo**. Sin ese segundo número la alternativa es dormir un rato a ojo, y el enunciado pide lo contrario —*saber esperar, fallar o informar*—: la espera sondea hasta un tope y se rinde con un mensaje claro en vez de colgarse.\n'
        '\n'
        '> ⏱️ La ingesta de 15.000 ronda los 40 s a ~375 vectores/s. La celda la hace **dos veces** —la segunda es el paso 5— y además puede quedarse esperando al índice, así que cuenta un par de minutos en total.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 los 15.000 puntos de G.1 · ⚠️ requiere `make motor-up MOTOR=qdrant`\n'
        'from aurum.motores.qdrant import QdrantStore\n'
        '\n'
        'indice = QdrantStore(\n'
        '    collection=COLECCION,\n'
        '    url=os.environ.get("AURUM_QDRANT_URL", "http://localhost:6333"),\n'
        '    api_key=os.environ.get("AURUM_QDRANT_API_KEY"),\n'
        '    prefix=CATALOG_PREFIX,   # el guardian del indice bueno, no el de humo\n'
        ')\n'
        'try:\n'
        '    print(f"qdrant · version del servidor: {indice.server_version()}")\n'
        '    ACEPTACION = run_acceptance(\n'
        '        indice, puntos_completo, dim=DIM, metric="cosine",\n'
        '        batch_size=LOTE, top_k=TOP_K, n_canarios=3,\n'
        '    )\n'
        'finally:\n'
        '    indice.close()\n'
        '\n'
        'INDICE_DIR = Path("..") / "artifacts" / "indice"\n'
        'save_smoke(INDICE_DIR, motor="qdrant", results=ACEPTACION)\n'
        'smoke_display(ACEPTACION, motor="qdrant")',
    ),
    (
        "markdown",
        '## G.3 · Paso 7 · persistencia, con los 15.000 dentro\n'
        '\n'
        'La misma comprobación que el paso 8 de la prueba de humo, pero sobre el índice que se entrega. Ejecuta la celda, reinicia, y vuelve a ejecutarla con la otra etiqueta:\n'
        '\n'
        '```bash\n'
        'make motor-down MOTOR=qdrant && make motor-up MOTOR=qdrant\n'
        '```\n'
        '\n'
        'Se lee con el vector del primer canario, así que comprueba dos cosas a la vez: que el recuento sobrevive y que ese punto sigue encontrándose a sí mismo donde estaba.',
    ),
    (
        "code",
        'INDICE_PERSISTENCIA = globals().get("INDICE_PERSISTENCIA") or {}\n'
        '\n'
        '\n'
        'def relectura(etiqueta):\n'
        '    """PASO 7 · Solo lee. No crea, no ingiere y no borra."""\n'
        '    store = QdrantStore(\n'
        '        collection=COLECCION,\n'
        '        url=os.environ.get("AURUM_QDRANT_URL", "http://localhost:6333"),\n'
        '        api_key=os.environ.get("AURUM_QDRANT_API_KEY"),\n'
        '        prefix=CATALOG_PREFIX,\n'
        '    )\n'
        '    try:\n'
        '        INDICE_PERSISTENCIA[etiqueta] = read_snapshot(\n'
        '            store, canarios[0].vector, top_k=TOP_K\n'
        '        )\n'
        '    finally:\n'
        '        store.close()\n'
        '\n'
        '    actual = INDICE_PERSISTENCIA[etiqueta]\n'
        '    print(f"{etiqueta}: count={actual.count} · top-1 = {actual.ids[0]}")\n'
        '    print(f"  ¿el canario se encuentra a si mismo? "\n'
        '          f"{actual.ids[0] == canarios[0].record_id}")\n'
        '    antes, despues = (INDICE_PERSISTENCIA.get(e) for e in ("antes", "despues"))\n'
        '    if not (antes and despues):\n'
        '        print("  (falta la otra mitad: reinicia y llama con la otra etiqueta)")\n'
        '        return\n'
        '    check = persistence_check(antes, despues)\n'
        '    print(f"\\n  {check.verdict()}")\n'
        '    record_manual(ACEPTACION, 7, observed=check.verdict(), passed=check.passed,\n'
        '                  manual_steps=ACCEPTANCE_MANUAL_NUMBERS)\n'
        '    save_smoke(INDICE_DIR, motor="qdrant", results=ACEPTACION)\n'
        '    print(f"  → fila 7 anotada: {\'✅ pasa\' if check.passed else \'❌ falla\'}")\n'
        '\n'
        '\n'
        'relectura("antes")',
    ),
    ("code", 'relectura("despues")'),
    (
        "markdown",
        '## G.4 · Paso 8 · recursos, esta vez los que van al README\n'
        '\n'
        'Con el motor vivo y los 15.000 dentro:\n'
        '\n'
        '```bash\n'
        'make motor-stats\n'
        '```\n'
        '\n'
        'La salida entera a `artifacts/recursos/qdrant_indice.txt`. Con el volumen borrado antes de construir, el tamaño que salga es el del índice y de nada más.\n'
        '\n'
        '> ⚠️ **Se toma justo después de G.2, antes del reinicio de G.3.** Un contenedor recién reiniciado y sin tráfico no ha tocado sus datos, y su cifra no es la del índice sirviendo: la primera medición salió a 28,7 MiB con **6,72 kB** de `NET I/O` —menos memoria que con 1.500 puntos, que no puede ser—. Si esa columna está a cero, la foto no vale.\n'
        '\n'
        'Y aun así es RAM **en reposo tras ingerir**, no bajo carga: la huella sirviendo consultas sale gratis en NB06, con el bucle de latencia corriendo.',
    ),
    (
        "code",
        '# 📄 DATOS · artifacts/recursos/qdrant_indice.txt\n'
        'from aurum.motores import resource_row\n'
        '\n'
        'transcrito = RECURSOS / "qdrant_indice.txt"\n'
        'if not transcrito.exists():\n'
        '    print(f"Falta {transcrito}: pega ahi la salida de `make motor-stats`.")\n'
        'else:\n'
        '    fila_recursos = resource_row(\n'
        '        transcrito.read_text(encoding="utf-8"), motor="qdrant"\n'
        '    ).iloc[0]\n'
        '    record_manual(ACEPTACION, 8, observed=resource_note(fila_recursos),\n'
        '                  passed=bool(fila_recursos["dentro_del_limite"]),\n'
        '                  manual_steps=ACCEPTANCE_MANUAL_NUMBERS)\n'
        '    save_smoke(INDICE_DIR, motor="qdrant", results=ACEPTACION)\n'
        '    print(resource_note(fila_recursos))',
    ),
    (
        "markdown",
        '## G.5 · El artefacto del índice\n'
        '\n'
        'La tabla de aceptación con el contrato de la colección delante. Es lo que un corrector necesita para saber qué se construyó y con qué se comprobó, sin abrir el notebook.',
    ),
    (
        "code",
        'destino_indice = Path("..") / "artifacts" / "indice_catalogo.md"\n'
        'tabla_aceptacion = smoke_table(ACEPTACION, motor="qdrant").drop(columns="motor")\n'
        'pendientes = [r.step for r in ACEPTACION if r.passed is None]\n'
        '\n'
        'contrato = [\n'
        '    "# El índice del catálogo · aceptación (NB04 § G)",\n'
        '    "",\n'
        '    f"Colección: `{COLECCION}`  ·  motor: **qdrant** (R03)",\n'
        '    "",\n'
        '    "| Elemento | Valor | De dónde sale |",\n'
        '    "|---|---|---|",\n'
        '    f"| Id del punto | `record_id` (UUIDv5) | README_DATOS · idempotencia por id |",\n'
        '    f"| Modelo | `{MODELO}` [{CONTRATO}] | R02 |",\n'
        '    f"| Plantilla | `{PLANTILLA}` | R01 |",\n'
        '    f"| Dimensión | {DIM} (truncada de 3.072 y renormalizada) | D09b |",\n'
        '    f"| Métrica | cosine | D10 |",\n'
        '    f"| Payload | `{ESQUEMA}` · nulos `{POLITICA_NULOS}` | D13 · D14 |",\n'
        '    f"| Lote de ingesta | {LOTE} | D15 |",\n'
        '    f"| Índices de payload | brand_normalized (keyword) · color_normalized (texto) | Sección B |",\n'
        '    f"| Puntos | {len(puntos_completo):,} | catalogo_productos.csv |".replace(",", "."),\n'
        '    "",\n'
        '    "## Cómo leer los números del paso 8",\n'
        '    "",\n'
        '    "- La **RAM** es en reposo tras ingerir, no bajo carga. Vale solo si el "\n'
        '    "`NET I/O` del transcrito no está a cero: un contenedor recién "\n'
        '    "reiniciado no ha tocado sus datos y da una cifra que no es la del "\n'
        '    "índice. La huella sirviendo consultas se mide en NB06.",\n'
        '    "- El **volumen** no lo llena el índice. Con los 1.500 de la prueba de "\n'
        '    "humo eran 293,8 MB y con los 15.000 son 289,5: diez veces más puntos y "\n'
        '    "no crece. Los vectores son 46 MB; el resto es asignación del motor "\n'
        '    "—WAL— y por eso escribir esos 46 MB costó 665 MB de `BLOCK I/O`.",\n'
        '    "",\n'
        '    "## Las ocho comprobaciones",\n'
        '    "",\n'
        ']\n'
        'if pendientes:\n'
        '    contrato += [f"> ⚠️ Pasos sin medir todavía: {pendientes}.", ""]\n'
        '\n'
        'destino_indice.write_text(\n'
        '    "\\n".join(contrato) + "\\n" + tabla_aceptacion.to_markdown(index=False) + "\\n",\n'
        '    encoding="utf-8",\n'
        ')\n'
        'print(f"Escrito {destino_indice}")\n'
        'tabla_aceptacion',
    ),
]

NB05_RECUPERACION = [
    ("markdown", '# NB05 · Recuperación: una sola puerta de entrada'),
    (
        "markdown",
        'NB04 dejó un índice de 15.000 puntos en Qdrant. Este notebook le pone la puerta: **una función que recibe una consulta en texto y devuelve resultados normalizados**, que es lo que §3.3 pide con esas palabras.\n'
        '\n'
        'Lo que se demuestra no es que el buscador encuentre cosas buenas —eso es calidad, y se mide en NB09—, sino que **el sistema se comporta como dice**: que el filtro lo ejecuta la base y no Python, que una respuesta vacía y un fallo salen por canales distintos, y que los cuatro casos borde del enunciado están tratados a propósito.\n'
        '\n'
        '> 📋 **Se demuestra enseñando lo que sale**: cada sección trae los `product_id`, los títulos y los scores que devolvió el motor, no un veredicto sobre ellos. Un ✅ es una afirmación del código sobre sí mismo; las filas son la prueba con la que cualquiera puede contradecirlo.\n'
        '>\n'
        '> Y sobre **varias consultas**: seis de demostración —tres formulaciones de la misma necesidad y tres de cliente— más las cuatro filtradas del §5. Con una sola frase, ninguna tabla distingue un caso borde bien tratado de una consulta que no casaba con nada.\n'
        '\n'
        '### Lo que se hereda y no se vuelve a decidir\n'
        '\n'
        '| | |\n'
        '|---|---|\n'
        '| Motor | Qdrant (R03) |\n'
        '| Colección | `aurum_catalogo__gemini_embedding_2__A4__768` (NB04 § G) |\n'
        '| Modelo · plantilla · dim | `gemini-embedding-2` [sin_contrato] · A4 · 768 |\n'
        '| Normalización del filtro | `casefold_unaccent` al buscar, el dato crudo guardado (D03) |\n'
        '\n'
        '**Nada de ANN se toca aquí.** Los parámetros del índice son los de por defecto y su estudio es NB06, con D16 fijada antes de ver la curva.\n'
        '\n'
        '### Las cuatro decisiones de este notebook\n'
        '\n'
        '| | Decidido | Por qué |\n'
        '|---|---|---|\n'
        '| Forma del resultado | `Resultado` **hereda** de `SearchResult` | Un `Resultado` *es* un `SearchResult`, así que NB09 lo mezcla con el baseline léxico sin traducir |\n'
        '| Colección vacía | Lista vacía | *"A nivel de usuario, cuando busque, no le aparecerá nada si no se encuentra nada"* |\n'
        '| Timeout | **30 s** y excepción controlada | *"No es el mismo caso que no haber encontrado resultados"* |\n'
        '| Color en la interfaz | **No** | Queda como capacidad del almacén; el enunciado solo pide marca |',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000) · las tres familias de consultas\n'
        'import os\n'
        'import sys\n'
        'from functools import lru_cache, partial\n'
        'from pathlib import Path\n'
        '\n'
        'sys.path.insert(0, str(Path("..") / "src"))\n'
        '\n'
        'from dotenv import load_dotenv\n'
        '\n'
        'from aurum.almacen import filter_reach\n'
        'from aurum.busqueda import (\n'
        '    BuscadorVectorial,\n'
        '    auditar_casos_borde,\n'
        '    auditar_filtro_de_marca,\n'
        '    auditar_forma_de_los_resultados,\n'
        '    auditar_post_filtro,\n'
        '    auditar_variantes_de_marca,\n'
        '    solapamiento_entre_consultas,\n'
        '    tabla_de_resultados,\n'
        ')\n'
        'from aurum.datos import load_csv\n'
        'from aurum.embeddings import GeminiEncoder, encode_corpus, truncate_dim\n'
        'from aurum.motores import CATALOG_PREFIX, Point, catalog_collection_name\n'
        'from aurum.motores.qdrant import QdrantStore\n'
        '\n'
        'load_dotenv(Path("..") / ".env")\n'
        'DATA, CACHE = Path("..") / "data", Path("..") / "artifacts" / "embeddings"\n'
        'completo = load_csv(DATA / "catalogo_productos.csv")\n'
        'filtradas = load_csv(DATA / "consultas_filtradas.csv")\n'
        'evaluacion = load_csv(DATA / "consultas_evaluacion.csv")\n'
        'desarrollo = load_csv(DATA / "consultas_desarrollo.csv")\n'
        '\n'
        'MODELO, CONTRATO, PLANTILLA = "gemini-embedding-2", "sin_contrato", "A4"\n'
        'DIM, TOP_K = 768, 10\n'
        'TIMEOUT_S = 30            # decision de NB05\n'
        'COLECCION = catalog_collection_name(model=MODELO, template=PLANTILLA, dim=DIM)\n'
        '\n'
        '# Las seis consultas con las que se ejercita la interfaz. No son inventadas:\n'
        '# salen de los CSV del enunciado. Las tres primeras son la MISMA necesidad\n'
        '# escrita de tres formas (§5 pide consultas de distinto tipo) y las tres\n'
        '# últimas son consultas de cliente, con sus faltas y sus negaciones.\n'
        'ORDEN_TIPO = {"direct": 0, "semantic": 1, "context": 2}\n'
        'TRES_FORMULACIONES = [\n'
        '    (f["evaluation_id"], f["query_text"])\n'
        '    for f in sorted(\n'
        '        (f for f in evaluacion.to_dict("records")\n'
        '         if f["evaluation_id"].startswith("EVAL-100455")),\n'
        '        key=lambda f: ORDEN_TIPO[f["query_type"]],\n'
        '    )\n'
        ']\n'
        'DE_CLIENTE = [\n'
        '    (f["workload_id"], f["query_text"])\n'
        '    for f in desarrollo.to_dict("records")\n'
        '    if f["workload_id"] in ("DEV-38249", "DEV-43240", "DEV-61533")\n'
        ']\n'
        'DEMO = TRES_FORMULACIONES + DE_CLIENTE\n'
        'POR_CASO = dict(DEMO)\n'
        '\n'
        'print(f"coleccion: {COLECCION}")\n'
        'print(f"catalogo : {len(completo)} productos · {len(filtradas)} consultas filtradas")\n'
        'for caso, texto in DEMO:\n'
        '    print(f"  {caso:22s} {texto}")',
    ),
    (
        "markdown",
        '## A · El buscador\n'
        '\n'
        'Tres piezas y una decisión de diseño en cada una.\n'
        '\n'
        '**El codificador se inyecta.** `aurum.busqueda` no sabe de Gemini: recibe una función que convierte texto en vector. Eso es lo que permite probar el buscador entero sin red —los 41 tests del módulo no tocan la API ni Docker— y lo que hará que cambiar de modelo no toque este fichero.\n'
        '\n'
        '**Y se memoiza.** El contrato de §3.3 es *recibe una consulta en texto*, así que `buscar()` codifica; pero cada consulta se lanza muchas veces —la sección C repite las filtradas seis veces con distintos tamaños de recuperación— y pagar la misma llamada de API una y otra vez sería tirar dinero por un detalle de implementación. `lru_cache` lo evita sin cambiar el contrato, y la caché en disco hace que la segunda ejecución no pague ninguna.\n'
        '\n'
        '**El timeout lo aplica el cliente, no el buscador.** Quien tiene el socket es el SDK, así que los 30 s se declaran al construir `QdrantStore`. El buscador los conoce solo para poder decir en el mensaje de error contra qué límite se agotó.',
    ),
    (
        "code",
        '# 📄 DATOS · las 4 consultas ya codificadas en NB04 (caché de artifacts/embeddings)\n'
        '_encoder = GeminiEncoder(\n'
        '    api_key=os.environ.get("GEMINI_API_KEY"), model_id=MODELO,\n'
        '    native_dim=3072, window=8192,\n'
        ')\n'
        '\n'
        '\n'
        '@lru_cache(maxsize=256)\n'
        'def codificar_consulta(texto: str):\n'
        '    """Texto -> vector de 768, pagando la API una sola vez por consulta."""\n'
        '    # `corpus_id` constante a propósito: la clave de caché incluye el\n'
        '    # SHA del texto, así que ya distingue una consulta de otra. Meter el\n'
        '    # texto en el nombre del fichero lo rompería con la primera consulta\n'
        '    # que llevara una barra o un acento.\n'
        '    codificado = encode_corpus(\n'
        '        _encoder, [texto], corpus_id="consulta_suelta",\n'
        '        kind="query", contract=CONTRATO, batch_size=1, cache_dir=CACHE,\n'
        '    )\n'
        '    return truncate_dim(codificado.vectors, DIM)[0]\n'
        '\n'
        '\n'
        'almacen = QdrantStore(\n'
        '    collection=COLECCION,\n'
        '    url=os.environ.get("AURUM_QDRANT_URL", "http://localhost:6333"),\n'
        '    api_key=os.environ.get("AURUM_QDRANT_API_KEY"),\n'
        '    prefix=CATALOG_PREFIX,\n'
        '    timeout=TIMEOUT_S,        # el limite lo aplica el cliente\n'
        ')\n'
        'buscador = BuscadorVectorial(\n'
        '    almacen, codificar_consulta, top_k=TOP_K, timeout_s=TIMEOUT_S,\n'
        ')\n'
        '\n'
        'print(f"puntos en la coleccion: {almacen.count():,}".replace(",", "."))\n'
        'print(f"indice al dia         : {almacen.index_ready()}")',
    ),
    (
        "markdown",
        '### Seis consultas, y lo que devuelve cada una\n'
        '\n'
        'Antes de medir nada, **verlo**. Las seis salen de los CSV del enunciado y están elegidas para que no se parezcan entre sí:\n'
        '\n'
        '| Consultas | De dónde | Qué ponen a prueba |\n'
        '|---|---|---|\n'
        '| `EVAL-100455` ×3 | `consultas_evaluacion.csv` | La **misma necesidad** escrita como palabras clave (`direct`), como frase natural (`semantic`) y como situación (`context`) |\n'
        '| `DEV-38249` · `DEV-43240` · `DEV-61533` | `consultas_desarrollo.csv` | Consultas **de cliente**: sin acentos, con negación (*sin taladro*, *sin tapa*) y una de categoría lejana (*lentejas sin gluten*) |\n'
        '\n'
        'Una fila por producto recuperado, con los **5 primeros de cada consulta** para que quepa; el top-10 completo se recupera igual y es el que usan las secciones siguientes.\n'
        '\n'
        'Qué mirar aquí, que no es la calidad —eso es NB09—:\n'
        '\n'
        '- Que **las seis devuelven diez**, incluida `lentejas sin gluten`: un buscador denso siempre devuelve `k` vecinos, no existe el "no hay nada parecido". Que lo devuelto sea relevante es otra pregunta.\n'
        '- Que la columna del score se llama `score_mayor_mejor`. El nombre lo pone la propia tabla a partir de lo que declara el motor: si esto fuera Weaviate saldría `score_menor_mejor` y el orden se leería al revés.\n'
        '\n'
        '> 💸 La primera ejecución paga seis llamadas de codificación (una por consulta) y las guarda en `artifacts/embeddings`. Las siguientes no pagan ninguna.',
    ),
    (
        "code",
        '# 📄 DATOS · las 6 consultas de DEMO, del catálogo indexado en NB04\n'
        'demo = [(caso, texto, buscador.buscar(texto, top_k=TOP_K)) for caso, texto in DEMO]\n'
        '\n'
        'tabla_de_resultados(demo, top=5).style.hide(axis="index").set_properties(\n'
        '    **{"white-space": "pre-wrap", "text-align": "left", "vertical-align": "top"}\n'
        ')',
    ),
    (
        "markdown",
        '### Un resultado, por dentro\n'
        '\n'
        'La tabla de arriba aplana; el objeto tiene más. Los campos son los que exige §3.3 —`product_id`, posición, título, metadatos y score— y dos detalles que conviene mirar:\n'
        '\n'
        '- **Los dos identificadores.** `document_id` es el `product_id`, que es lo que juzgan los qrels y lo que piden los CSV de entrega; `record_id` es el id del punto en Qdrant. No son intercambiables.\n'
        '- **`score_es_similitud`** viene del motor, no de una suposición. Qdrant devuelve similitud; Weaviate habría devuelto distancia y el orden se leería al revés.',
    ),
    (
        "code",
        'caso_ejemplo, texto_ejemplo, resultados_ejemplo = demo[1]   # la formulación semántica\n'
        'print(f"{caso_ejemplo}: {texto_ejemplo!r}\\n")\n'
        'for r in resultados_ejemplo[:3]:\n'
        '    print(f"{r.rank}. {r.document_id}  score={r.score:.4f}  "\n'
        '          f"similitud={r.score_es_similitud}")\n'
        '    print(f"   {r.titulo[:80]}")\n'
        '    print(f"   record_id={r.record_id} · marca={r.metadatos.get(\'brand\') or \'(vacía)\'}")\n'
        '    print(f"   payload: {sorted(r.metadatos)}")',
    ),
    (
        "markdown",
        '### La misma necesidad, escrita de tres formas\n'
        '\n'
        'Las tres primeras piden lo mismo —un taladro inalámbrico de 24 V con batería— con palabras muy distintas: si la interfaz fuera léxica, `taladro 24v batería` y *"quiero una herramienta inalámbrica potente para perforar sin depender de un enchufe"* no compartirían casi nada, porque **no comparten casi ninguna palabra**.\n'
        '\n'
        'La tabla cuenta cuántos `product_id` comparten sus top-10 y si coinciden en el primero: cuanto más alto el solapamiento, **menos depende el resultado de cómo se escriba la consulta**. No es calidad —tres formulaciones podrían coincidir en diez resultados igual de malos—, es que la puerta responda igual ante las tres.',
    ),
    (
        "code",
        'solapamiento_entre_consultas(demo[:3]).style.hide(axis="index").set_properties(\n'
        '    **{"white-space": "pre-wrap", "text-align": "left", "vertical-align": "top"}\n'
        ')',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '## B · Las cuatro consultas filtradas (§5)\n'
        '\n'
        'El §8 lo pide como criterio de corrección: *"las consultas filtradas nunca devuelven otra marca"*. Medir solo la pureza no basta, por el motivo que ya apareció en NB04: **una respuesta vacía la cumple de forma vacía**, y un filtro roto que no devuelve nada saca el mismo 100 % que uno perfecto.\n'
        '\n'
        'Por eso la tabla lleva el oráculo al lado —cuántos productos de esa marca hay en el catálogo, contados con pandas y sin motor—:\n'
        '\n'
        '| Columna | Qué dice |\n'
        '|---|---|\n'
        '| `n_en_catalogo` | El oráculo. Distingue un cero legítimo de un filtro roto |\n'
        '| `de_la_marca` | Cuántos de los devueltos son de verdad de esa marca, auditado contra `brand_normalized` |\n'
        '| `pureza` | El criterio del §8 |\n'
        '| `veredicto` | Cruza las dos cosas |\n'
        '\n'
        'Hay un tercer veredicto que interesa especialmente: **cobertura corta**. Las cuatro marcas tienen más de 10 productos, así que las cuatro deben devolver 10; si alguna devuelve menos, lo más probable es que se esté filtrando después en Python en vez de en la base — justo lo que mide la sección C.\n'
        '\n'
        'Debajo del resumen van **los 40 resultados**, uno por fila: la tabla de arriba dice *"pureza 100 %"* y la de abajo permite comprobarlo sin fiarse.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv — el oráculo, contado sin motor\n'
        '# `unaccent` y no `raw`: el motor filtra contra `brand_normalized`, así que\n'
        '# el oráculo tiene que contar sobre la misma clave o compararíamos cosas\n'
        '# distintas.\n'
        'alcance = filter_reach(\n'
        '    completo, filtradas["filter_value"].tolist(), field="brand",\n'
        '    modes=("unaccent",),\n'
        ')\n'
        'ORACULO = dict(zip(alcance["filtro"], alcance["n_productos"]))\n'
        'print("productos por marca en el catálogo:", ORACULO)\n'
        '\n'
        'pureza = auditar_filtro_de_marca(\n'
        '    buscador, filtradas.to_dict("records"), alcance=ORACULO, top_k=TOP_K\n'
        ')\n'
        'pureza.style.hide(axis="index").set_properties(\n'
        '    **{"white-space": "pre-wrap", "text-align": "left", "vertical-align": "top"}\n'
        ')',
    ),
    (
        "markdown",
        '#### Los 40 resultados, para poder contradecir la tabla anterior\n'
        '\n'
        'Una fila por producto devuelto en las cuatro consultas filtradas. `marca` es el valor **crudo** que se guardó en el payload y `marca_normalizada` es la clave contra la que filtra el motor (D03: se guarda tal cual y se normaliza al buscar).\n'
        '\n'
        'Juntas hacen visible el filtro: `NIKE` y `Nike` son marcas distintas para un `equals` sobre el valor crudo y **la misma** para el que se ejecuta de verdad. Si en `marca_normalizada` apareciera un solo valor distinto del pedido, el 100 % de arriba sería mentira.\n'
        '\n'
        '**Y las cuatro piden su marca escrita a propósito de otra forma**: el catálogo guarda `NIKE` y `SAMSUNG` en mayúsculas, pero se piden `einhell`, `  APPLE `, `Nike` y `Sámsung`. Si el filtro solo sirviera para quien escribe la marca igual que el CSV, estas cuarenta filas estarían vacías — y la pureza **seguiría dando 100 %**.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 consultas_filtradas.csv — las 4 consultas del §5, con filtro nativo\n'
        '# Cada una pide su marca escrita como la escribiría un usuario, no como\n'
        '# está en el catálogo: minúsculas, mayúsculas con espacios, capitalizada y\n'
        '# con un acento colado. Las cuatro tienen que devolver lo mismo.\n'
        'ESCRITURAS = {\n'
        '    "Einhell": "einhell",\n'
        '    "Apple": "  APPLE ",\n'
        '    "NIKE": "Nike",\n'
        '    "SAMSUNG": "Sámsung",\n'
        '}\n'
        'resultados_filtrados = [\n'
        '    (\n'
        '        f["workload_id"],\n'
        '        f"{f[\'query_text\']}  ·  marca pedida={ESCRITURAS[f[\'filter_value\']]!r}",\n'
        '        buscador.buscar(\n'
        '            f["query_text"], top_k=TOP_K, marca=ESCRITURAS[f["filter_value"]]\n'
        '        ),\n'
        '    )\n'
        '    for f in filtradas.to_dict("records")\n'
        ']\n'
        '\n'
        'detalle_filtradas = tabla_de_resultados(\n'
        '    resultados_filtrados,\n'
        '    metadatos={"brand": "marca", "brand_normalized": "marca_normalizada"},\n'
        ')\n'
        'detalle_filtradas.style.hide(axis="index").set_properties(\n'
        '    **{"white-space": "pre-wrap", "text-align": "left", "vertical-align": "top"}\n'
        ')',
    ),
    (
        "markdown",
        '#### La misma marca, escrita de todas las formas en que se escribe\n'
        '\n'
        'La celda anterior usa una escritura distinta por consulta; esta las barre todas. Cada fila es **la misma consulta filtrada por la misma marca**, cambiando solo cómo se teclea, y se compara con lo que devolvió la escritura del CSV.\n'
        '\n'
        '| Columna | Qué dice |\n'
        '|---|---|\n'
        '| `marca_pedida` | Lo que se teclea, entre comillas para que se vean los espacios |\n'
        '| `viaja_al_motor` | Lo que sale de `normalize_brand` y llega a la base. **Aquí es donde ocurre D03** |\n'
        '| `iguales_a_la_canonica` | Cuántos de los diez coinciden con los de la escritura del CSV, y si además en el mismo orden |\n'
        '| `esperado` | `los mismos 10` para las seis primeras; `0` para las dos últimas |\n'
        '\n'
        'Las dos últimas variantes de cada marca son **faltas de verdad** —un espacio de más dentro y una letra de menos— y tienen que devolver cero. Están para marcar el límite: normalizar iguala la caja y los acentos, **no corrige faltas**. Sin ellas, una tabla en la que todo devuelve diez no distinguiría un filtro que normaliza de uno que casa con cualquier cosa.\n'
        '\n'
        '> 🎯 Lo que esta tabla caza es un fallo caro y silencioso. Si la normalización fuera `casefold` en vez de `unaccent`, `Sámsung` viajaría como `sámsung`, no casaría con nada, y **la pureza del §8 seguiría saliendo al 100 %** mientras el usuario ve una lista vacía. Ese es el caso que ningún resumen de veredictos enseña.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 consultas_filtradas.csv — 4 marcas × sus escrituras\n'
        'variantes = auditar_variantes_de_marca(\n'
        '    buscador, filtradas.to_dict("records"), top_k=TOP_K\n'
        ')\n'
        'variantes.style.hide(axis="index").set_properties(\n'
        '    **{"white-space": "pre-wrap", "text-align": "left", "vertical-align": "top"}\n'
        ')',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '## C · Filtro nativo contra post-filtro\n'
        '\n'
        'La decisión de filtrar en la base y no en Python estaba tomada desde NB04 —es lo que el enunciado exige—, pero hasta ahora se sostenía en un argumento aritmético: **`Einhell` son 30 productos de 15.000, el 0,2 %**, así que para esperar diez suyos habría que recuperar del orden de 5.000 candidatos y descartar 4.990.\n'
        '\n'
        'La celda lo mide en vez de suponerlo, y **en las cuatro marcas, no solo en `Einhell`**: con una sola la conclusión dependería de cuál se eligiera, porque la del enunciado es la más rara y no llega a diez ni con mil candidatos, mientras que una marca frecuente sí llega y haría parecer que el post-filtro *funciona con un poco de sobre-recuperación*.\n'
        '\n'
        'Cada bloque de seis filas es una marca: la primera es el filtro nativo y las cinco siguientes recuperan sin filtrar `10 × factor` candidatos y se quedan con los de la marca.\n'
        '\n'
        '| Columna | Cómo se lee |\n'
        '|---|---|\n'
        '| `n_en_catalogo` · `pct_del_catalogo` | Cuántos productos de esa marca hay. **Cuanto más bajo, antes falla el post-filtro** |\n'
        '| `candidatos` | Cuántos productos trajo el motor en esa estrategia |\n'
        '| `de_la_marca` | Cuántos de esos candidatos eran de la marca pedida |\n'
        '| `descartados` | Los que Python tendría que tirar. Es el coste, con la resta hecha |\n'
        '| `llega_a_10` | Si la estrategia consigue el top-10 que pidió el usuario |\n'
        '| `ms` | Lo que tardó. El filtro nativo trae 10 y ya; el post-filtro paga por traer basura |\n'
        '\n'
        '> Lo que hay que buscar no es que el post-filtro sea más lento —que lo es—, sino **dónde falla**: si `llega_a_10` sigue en `False` con ×100 para la marca rara mientras las frecuentes lo consiguen, no es una alternativa peor, es que **falla justo donde el filtro hace falta**.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 consultas_filtradas.csv — las 4 marcas × 6 estrategias\n'
        'for caso in filtradas.to_dict("records"):\n'
        '    n_marca = ORACULO[caso["filter_value"]]\n'
        '    print(f"{caso[\'workload_id\']}: marca {caso[\'filter_value\']!r} · "\n'
        '          f"{n_marca} productos de {len(completo)} "\n'
        '          f"({100 * n_marca / len(completo):.2f} % del catálogo)")\n'
        '\n'
        'comparativa = auditar_post_filtro(\n'
        '    buscador, filtradas.to_dict("records"), alcance=ORACULO,\n'
        '    top_k=TOP_K, factores=(1, 5, 10, 50, 100), n_catalogo=len(completo),\n'
        ')\n'
        'comparativa.style.hide(axis="index").set_properties(\n'
        '    **{"white-space": "pre-wrap", "text-align": "left", "vertical-align": "top"}\n'
        ')',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '## D · Los cuatro casos borde\n'
        '\n'
        'El enunciado los exige por escrito: *"tratamiento explícito de colecciones vacías, filtros sin resultados y proveedores no disponibles"*, más el `top_k` mayor que el número de puntos que añade el plan.\n'
        '\n'
        '| Caso | Qué debe pasar | Por qué se decidió así |\n'
        '|---|---|---|\n'
        '| Colección vacía | Lista vacía | Decisión de NB05: el usuario no ve nada, y no ver nada no es un error |\n'
        '| Filtro sin resultados | Lista vacía | Lo impone el enunciado, no se decide |\n'
        '| Motor no disponible | `MotorNoDisponible` | Es un fallo, no una respuesta: mezclarlo con la lista vacía haría creer que el catálogo no tiene nada |\n'
        '| `top_k` > nº de puntos | Devuelve lo que haya | — |\n'
        '| Consulta en blanco | `ValueError` | No es *"no encontré nada"*: es que no hay nada que buscar. Codificar `"   "` y devolver diez vecinos sería peor que fallar |\n'
        '| Marca en blanco | `ValueError` | D14 dejó los huecos como cadena vacía, así que `equals ""` no dejaría de filtrar: devolvería justo los productos **sin marca** |\n'
        '\n'
        '**Cada caso se ejecuta con tres frases distintas** —palabras clave, la misma necesidad en prosa y una consulta de categoría lejana—, y las dos últimas filas con tres variantes de entrada en blanco: con una sola frase, *"la colección vacía devuelve lista vacía"* sería indistinguible de *"esa consulta no casaba con nada"*.\n'
        '\n'
        'Los dos primeros y el cuarto necesitan una colección aparte, así que se crea una **vacía y desechable** bajo el prefijo `aurum_humo`, que existe para eso. El tercero se prueba apuntando a un puerto donde no hay nadie: la misma condición que apagar el contenedor, sin apagarlo.\n'
        '\n'
        '> 🧹 Al terminar se puede borrar desde el panel de Qdrant (<http://localhost:6333/dashboard>): queda vacía y no ocupa nada, pero no es del índice.',
    ),
    (
        "code",
        '# Colección desechable: prefijo `aurum_humo`, el que existe para esto.\n'
        '# `recreate=False`, así que no borra nada ni necesita AURUM_ALLOW_RESET.\n'
        'vacia = QdrantStore(\n'
        '    collection="aurum_humo_casos_borde",\n'
        '    url=os.environ.get("AURUM_QDRANT_URL", "http://localhost:6333"),\n'
        '    api_key=os.environ.get("AURUM_QDRANT_API_KEY"),\n'
        '    timeout=TIMEOUT_S,\n'
        ')\n'
        'vacia.create_collection(dim=DIM, metric="cosine", recreate=False)\n'
        '\n'
        '# La colección vive en el volumen y sobrevive al notebook: si esto ya se\n'
        '# ejecutó una vez, dentro están los 3 puntos del último caso borde y el\n'
        '# primero dejaría de medir una colección vacía —diría "3 resultados" y\n'
        '# parecería un fallo del sistema en vez de un residuo de la ejecución\n'
        '# anterior—. Se borran por id, que no necesita AURUM_ALLOW_RESET.\n'
        'IDS_DESECHABLES = [f"00000000-0000-0000-0000-00000000000{i}" for i in (1, 2, 3)]\n'
        'for record_id in IDS_DESECHABLES:\n'
        '    vacia.delete(record_id)\n'
        '\n'
        'buscador_vacio = BuscadorVectorial(vacia, codificar_consulta, timeout_s=TIMEOUT_S)\n'
        '\n'
        '# Un motor en un puerto donde no hay nadie: la misma condición que tenerlo\n'
        '# apagado, sin tener que apagarlo.\n'
        '# `prefer_grpc=True` es el valor por defecto de QdrantStore, y el cliente\n'
        '# de gRPC no deriva su puerto de `url` —usa `grpc_port`, que por defecto es\n'
        '# el 6334 real—. Sin desactivarlo aquí, `.buscar()` ignora el 6399 apagado\n'
        '# y sale por gRPC contra el Qdrant que sí está levantado: el caso borde no\n'
        '# mediría nada. `prefer_grpc=False` fuerza a que todo vaya por el `url`\n'
        '# REST, que es el puerto que de verdad se apagó.\n'
        'ausente = QdrantStore(\n'
        '    collection=COLECCION, url="http://localhost:6399",\n'
        '    prefix=CATALOG_PREFIX, timeout=5, prefer_grpc=False,\n'
        ')\n'
        'buscador_ausente = BuscadorVectorial(ausente, codificar_consulta, timeout_s=5)\n'
        '\n'
        'print(f"colección desechable: {vacia.count()} puntos")',
    ),
    (
        "code",
        '# Tres frases por caso: palabras clave, la misma necesidad en prosa, y una\n'
        '# consulta de categoría lejana. Si el comportamiento dependiera de la frase,\n'
        '# se vería aquí.\n'
        'FRASES_BORDE = [\n'
        '    POR_CASO[caso]\n'
        '    for caso in ("EVAL-100455-direct", "EVAL-100455-context", "DEV-61533")\n'
        ']\n'
        '\n'
        '\n'
        'def poblar_y_pedir_de_mas(consulta):\n'
        '    """Mete 3 puntos en la desechable y pide 50. Debe devolver 3.\n'
        '\n'
        '    Va después del caso de la colección vacía y **el orden importa**: si\n'
        '    esto corriera antes, aquel caso ya no mediría una colección vacía."""\n'
        '    vacia.upsert([\n'
        '        Point(\n'
        '            record_id=record_id,\n'
        '            vector=codificar_consulta(consulta),\n'
        '            payload={"product_id": f"TEST-{i}", "title": f"punto de prueba {i}"},\n'
        '        )\n'
        '        for i, record_id in enumerate(IDS_DESECHABLES, start=1)\n'
        '    ], batch_size=3)\n'
        '    return buscador_vacio.buscar(consulta, top_k=50)\n'
        '\n'
        '\n'
        '# `partial` y no `lambda`: en un bucle, el lambda capturaría la variable y\n'
        '# las tres filas acabarían ejecutando la última frase.\n'
        'casos_borde = []\n'
        'for frase in FRASES_BORDE:\n'
        '    casos_borde.append((\n'
        '        "colección vacía", frase, "lista vacía",\n'
        '        partial(buscador_vacio.buscar, frase),\n'
        '    ))\n'
        'for frase in FRASES_BORDE:\n'
        '    casos_borde.append((\n'
        '        "filtro sin resultados", f"{frase} · marca=MarcaQueNoExiste", "lista vacía",\n'
        '        partial(buscador.buscar, frase, marca="MarcaQueNoExiste"),\n'
        '    ))\n'
        'for frase in FRASES_BORDE:\n'
        '    casos_borde.append((\n'
        '        "motor no disponible", frase, "MotorNoDisponible, no lista vacía",\n'
        '        partial(buscador_ausente.buscar, frase),\n'
        '    ))\n'
        '# Este puebla la colección desechable, así que va después del caso que la\n'
        '# necesita vacía.\n'
        'for frase in FRASES_BORDE:\n'
        '    casos_borde.append((\n'
        '        "top_k=50 sobre 3 puntos", frase, "devuelve 3, sin reventar",\n'
        '        partial(poblar_y_pedir_de_mas, frase),\n'
        '    ))\n'
        '# En estos dos la entrada inválida ES el caso, así que la variedad va ahí.\n'
        'for blanca in ("", "   ", "\\n"):\n'
        '    casos_borde.append((\n'
        '        "consulta en blanco", repr(blanca), "ValueError: es entrada inválida",\n'
        '        partial(buscador.buscar, blanca),\n'
        '    ))\n'
        'for marca_blanca in ("", "   "):\n'
        '    casos_borde.append((\n'
        '        "marca en blanco", f"{FRASES_BORDE[0]} · marca={marca_blanca!r}",\n'
        '        "ValueError: un filtro vacío dejaría de filtrar",\n'
        '        partial(buscador.buscar, FRASES_BORDE[0], marca=marca_blanca),\n'
        '    ))\n'
        '\n'
        'bordes = auditar_casos_borde(casos_borde)\n'
        'bordes.style.hide(axis="index").set_properties(\n'
        '    **{"white-space": "pre-wrap", "text-align": "left", "vertical-align": "top"}\n'
        ')',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '## E · Las comprobaciones de forma\n'
        '\n'
        'Las cuatro que no dependen del filtro: que `k` se respeta, que no hay `product_id` repetidos dentro de una consulta, que el orden es monótono en la dirección que declara `score_es_similitud`, y que todo lo devuelto existe en el catálogo.\n'
        '\n'
        'Se ejecutan sobre **las diez consultas del notebook** —las cuatro filtradas y las seis de demostración—, y la tabla enseña el número medido en vez de un `True`:\n'
        '\n'
        '| Columna | Qué número lleva y cómo se lee |\n'
        '|---|---|\n'
        '| `devueltos` | `10 de 10 posibles (pedidos 10)`. Los "posibles" son el oráculo: los productos de esa marca, o los 15.000 puntos si no hay filtro |\n'
        '| `posiciones` | El rango de `rank`. Tiene que ser `1→n`, sin huecos ni ceros |\n'
        '| `ids_distintos` | `10 de 10` significa ninguno repetido. Menos a la izquierda es un duplicado dentro de la misma consulta |\n'
        '| `score_primero_ultimo` | El score del primero y el del último. Enseña de paso **cuánto se aplana** el ranking, que un booleano escondería |\n'
        '| `orden` | Qué dirección lleva y cuál debería llevar según el motor |\n'
        '| `fuera_del_catalogo` | Ids recuperados que no existen en el CSV. Cualquier cosa distinta de `0` es un desajuste entre lo indexado y los datos |\n'
        '\n'
        'La diferencia no es estética: un `k_respetado = True` con `top_k=10` y una marca de 3 productos es correcto y **parece un fallo**, mientras que `3 de 3 posibles (pedidos 10)` se entiende sin mirar el código. Son comprobaciones baratas y aburridas, y son las que impiden que un fallo tonto llegue a `resultados_busqueda.csv` en NB09.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv + las 10 consultas de este notebook\n'
        'EN_CATALOGO = set(completo["product_id"])\n'
        'casos_forma = filtradas.to_dict("records") + [\n'
        '    {"workload_id": caso, "query_text": texto} for caso, texto in DEMO\n'
        ']\n'
        '\n'
        'forma = auditar_forma_de_los_resultados(\n'
        '    buscador, casos_forma,\n'
        '    ids_del_catalogo=EN_CATALOGO, n_puntos=almacen.count(),\n'
        '    alcance=ORACULO, top_k=TOP_K,\n'
        ')\n'
        'forma.style.hide(axis="index").set_properties(\n'
        '    **{"white-space": "pre-wrap", "text-align": "left", "vertical-align": "top"}\n'
        ')',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '## F · El artefacto\n'
        '\n'
        'Las seis tablas juntas en `artifacts/recuperacion.md`, **con los resultados dentro**. Es lo que un corrector necesita para ver que la interfaz cumple §3.3 sin abrir el notebook, y sobre todo para poder discutirla: un resumen de veredictos no se puede contradecir, una lista de `product_id` con su marca y su score sí.',
    ),
    (
        "code",
        'destino = Path("..") / "artifacts" / "recuperacion.md"\n'
        'bloques = [\n'
        '    "# Recuperación · la interfaz común (NB05)",\n'
        '    "",\n'
        '    f"Colección: `{COLECCION}` · motor **qdrant** (R03) · `top_k` {TOP_K} · "\n'
        '    f"timeout {TIMEOUT_S} s · {almacen.count()} puntos",\n'
        '    "",\n'
        '    "## Seis consultas y lo que devolvieron",\n'
        '    "",\n'
        '    "Una fila por producto recuperado; los 5 primeros de cada consulta.",\n'
        '    "",\n'
        '    tabla_de_resultados(demo, top=5).to_markdown(index=False),\n'
        '    "",\n'
        '    "### Solapamiento entre las tres formulaciones de la misma necesidad",\n'
        '    "",\n'
        '    solapamiento_entre_consultas(demo[:3]).to_markdown(index=False),\n'
        '    "",\n'
        '    "## Las cuatro consultas filtradas (§5)",\n'
        '    "",\n'
        '    pureza.to_markdown(index=False),\n'
        '    "",\n'
        '    "### Los 40 resultados, uno por fila",\n'
        '    "",\n'
        '    "Cada consulta pide su marca escrita de otra forma: "\n'
        '    + ", ".join(f"`{v}`" for v in ESCRITURAS.values()) + ".",\n'
        '    "",\n'
        '    detalle_filtradas.to_markdown(index=False),\n'
        '    "",\n'
        '    "### La misma marca, escrita de todas las formas en que se escribe",\n'
        '    "",\n'
        '    variantes.to_markdown(index=False),\n'
        '    "",\n'
        '    "## Filtro nativo contra post-filtro, en las cuatro marcas",\n'
        '    "",\n'
        '    "Las marcas van de "\n'
        '    f"{min(ORACULO.values())} a {max(ORACULO.values())} productos sobre "\n'
        '    f"{len(completo)}: el post-filtro falla donde la marca es rara.",\n'
        '    "",\n'
        '    comparativa.to_markdown(index=False),\n'
        '    "",\n'
        '    "## Casos borde (§3.3), cada uno con varias frases",\n'
        '    "",\n'
        '    bordes.to_markdown(index=False),\n'
        '    "",\n'
        '    "## Comprobaciones de forma",\n'
        '    "",\n'
        '    forma.to_markdown(index=False),\n'
        '    "",\n'
        ']\n'
        'destino.write_text("\\n".join(bloques), encoding="utf-8")\n'
        'print(f"Escrito {destino} · {destino.stat().st_size / 1024:.1f} KB")',
    ),
]

NB06_ANN = [
    ("markdown", '# NB06 · ANN: parámetros, fidelidad y latencia'),
    (
        "markdown",
        'NB05 dejó una puerta de entrada que ya funciona; este notebook mide **qué paga y qué gana** al buscar de forma aproximada en vez de exacta: separar el error del índice del error del modelo, y elegir el punto de operación con una restricción declarada **antes** de ver la curva.\n'
        '\n'
        '> 🚨 **La distinción que más nota da.**\n'
        '>\n'
        '> ```\n'
        '> recall ANN@10  =  |IDs_ANN ∩ IDs_exactos| / 10   → ¿el ÍNDICE es fiel al espacio?\n'
        '> Recall@10      =  |relevantes ∩ top10| / |rel|   → ¿la REPRESENTACIÓN es buena?\n'
        '> ```\n'
        '>\n'
        '> Un ANN puede tener recall 1,0 y una relevancia pésima: reproduce fielmente un espacio mediocre. Y puede perder un vecino exacto sin bajar nDCG si el sustituto es igual de relevante. Por eso `aurum.ann` no reutiliza `evaluacion.recall_at_k` -mezclarlas sería aplanar justo esto-.\n'
        '\n'
        '### Lo que se hereda y no se vuelve a decidir\n'
        '\n'
        '| | |\n'
        '|---|---|\n'
        '| Motor · colección | Qdrant (R03) · `aurum_catalogo__gemini_embedding_2__A4__768` (NB04) |\n'
        '| Interfaz | `BuscadorVectorial` de NB05, con un `ef` nuevo por instancia |\n'
        '| Oráculo exacto | `busqueda.DenseRetriever` (ya construido y probado en NB02), no FAISS ni `SearchParams(exact=True)` |\n'
        '\n'
        '### Las decisiones de este notebook\n'
        '\n'
        '| | Decidido | Por qué |\n'
        '|---|---|---|\n'
        '| **D16** 🚨 | `recall ANN@10 ≥ 0,90` ∧ `p95 ≤ 20 ms` | Fijada **antes** de correr el barrido — si se fija después de ver la curva, deja de ser una restricción y pasa a describir el resultado. Opción "permisivo": prioriza velocidad, confiando en que la comparación nDCG lo confirme |\n'
        '| **D17** | No entra el laboratorio FAISS opcional | NB06 se limita al motor de entrega; comparar HNSW/IVF/IVF-PQ queda fuera de alcance |\n'
        '| `m` / `ef_construct` | Por defecto de Qdrant (**16** / **100**) | Tocarlos exige una colección nueva al lado (`HnswConfigDiff` se fija al crear). Queda anotado como mejora futura, no como decisión por omisión |\n'
        '| Barrido | `ef ∈ {16, 32, 64, 128, 256}` | La tabla de familias HNSW del plan. `ef` es de **consulta**: se barre sin reconstruir nada |\n'
        '\n'
        '> 💡 **Referencia:** `sesion_02` §7.2 aplica exactamente este procedimiento -fijar umbral, descartar lo que no llega, elegir lo más barato entre lo que sobra- y es explícito en que el umbral "no es una recomendación universal, es una decisión de producto y de riesgo".',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000) + consultas de desarrollo y evaluación\n'
        'import os\n'
        'import sys\n'
        'from pathlib import Path\n'
        '\n'
        'import numpy as np\n'
        'import pandas as pd\n'
        '\n'
        'sys.path.insert(0, str(Path("..") / "src"))\n'
        '\n'
        'from dotenv import load_dotenv\n'
        '\n'
        'from aurum.ann import (\n'
        '    aplicar_restriccion,\n'
        '    barrido_ef,\n'
        '    comparar_ndcg_con_oraculo,\n'
        '    comparar_ndcg_por_consulta,\n'
        '    tabla_recall_por_consulta,\n'
        ')\n'
        'from aurum.busqueda import BuscadorVectorial, DenseRetriever, rank_queries_dense\n'
        'from aurum.datos import load_csv\n'
        'from aurum.embeddings import GeminiEncoder, cache_key, corpus_fingerprint, encode_corpus, truncate_dim\n'
        'from aurum.evaluacion import qrels_from_judgements\n'
        'from aurum.graficas import plot_ann_pareto\n'
        'from aurum.motores import CATALOG_PREFIX, catalog_collection_name\n'
        'from aurum.motores.qdrant import QdrantStore\n'
        'from aurum.plantillas import render_template\n'
        '\n'
        'load_dotenv(Path("..") / ".env")\n'
        'DATA, CACHE = Path("..") / "data", Path("..") / "artifacts" / "embeddings"\n'
        'completo = load_csv(DATA / "catalogo_productos.csv")\n'
        'desarrollo = load_csv(DATA / "consultas_desarrollo.csv")\n'
        'evaluacion = load_csv(DATA / "consultas_evaluacion.csv")\n'
        'relevancias = load_csv(DATA / "relevancias_desarrollo.csv")\n'
        '\n'
        'MODELO, CONTRATO, PLANTILLA = "gemini-embedding-2", "sin_contrato", "A4"\n'
        'DIM, TOP_K = 768, 10\n'
        'COLECCION = catalog_collection_name(model=MODELO, template=PLANTILLA, dim=DIM)\n'
        '\n'
        '# D16, fijada en config.yaml -> nb06_ann ANTES de correr el barrido.\n'
        'RECALL_MINIMO = 0.90\n'
        'P95_MAXIMO_MS = 20.0\n'
        'VALORES_EF = [16, 32, 64, 128, 256]\n'
        '\n'
        'print(f"coleccion : {COLECCION}")\n'
        'print(f"D16       : recall >= {RECALL_MINIMO} y p95 <= {P95_MAXIMO_MS} ms")\n'
        'print(f"barrido   : ef en {VALORES_EF}")',
    ),
    (
        "markdown",
        '## A · El oráculo exacto, desde la caché\n'
        '\n'
        'El oráculo son los vectores del catálogo completo con la plantilla A4 -los mismos que se ingirieron en NB04-, ya en `artifacts/embeddings/` desde entonces. Igual que en G.1 de NB04, la celda **comprueba la caché antes de llamar**: sin eso, un fallo de clave lanzaría 15.000 documentos contra la API de pago sin preguntar.\n'
        '\n'
        '`DenseRetriever` con métrica `cosine` reproduce exactamente el ranking de Qdrant sin aproximar nada -es el mismo buscador que ya validó NB02-, así que no hace falta traer FAISS ni pedirle a Qdrant `exact=True`.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000) — vectores desde la caché de NB04\n'
        'CORPUS_ID = f"catalogo_productos__{PLANTILLA}"\n'
        'textos_completo = render_template(completo, PLANTILLA)\n'
        'clave = cache_key(\n'
        '    model_id=MODELO, kind="document", contract=CONTRATO,\n'
        '    corpus_id=CORPUS_ID, fingerprint=corpus_fingerprint(textos_completo),\n'
        ')\n'
        'if not (CACHE / f"{clave}.npy").exists():\n'
        '    raise RuntimeError(\n'
        '        f"Los vectores de {CORPUS_ID} no estan en cache ({clave}).\\n"\n'
        '        f"Codificarlos son 15.000 llamadas de pago: esta celda para en vez de "\n'
        '        f"pagarlas sin avisar. Deberian estar desde NB04."\n'
        '    )\n'
        '\n'
        '_encoder = GeminiEncoder(\n'
        '    api_key=os.environ.get("GEMINI_API_KEY"), model_id=MODELO,\n'
        '    native_dim=3072, window=8192,\n'
        ')\n'
        'codificado_completo = encode_corpus(\n'
        '    _encoder, textos_completo, corpus_id=CORPUS_ID,\n'
        '    kind="document", contract=CONTRATO, batch_size=32, cache_dir=CACHE,\n'
        ')\n'
        'vectores_completo = truncate_dim(codificado_completo.vectors, DIM)\n'
        'ids_completo = completo["product_id"].tolist()\n'
        '\n'
        'retriever_exacto = DenseRetriever(vectores_completo, ids_completo, metric="cosine")\n'
        'print(f"oraculo : {vectores_completo.shape} · desde cache: {codificado_completo.stats.desde_cache}")',
    ),
    (
        "markdown",
        '## B · Las consultas del barrido\n'
        '\n'
        'Las 8 de desarrollo (`consultas_desarrollo.csv`, con juicios de relevancia) más las 12 de evaluación (`consultas_evaluacion.csv`, sin juicios) — **20 en total**, no una sola. El recall ANN no necesita etiquetas, así que entran las 20; el nDCG de la sección F solo puede medirse donde hay juicios, así que ahí solo entran las 8.\n'
        '\n'
        'Las claves de las 8 de desarrollo son su `query_id` numérico (`"13357"`, no `"DEV-13357"`): es la clave que usan los qrels, y así el mismo diccionario sirve para el recall ANN y para el nDCG sin traducir nada.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 consultas_desarrollo.csv (8) + consultas_evaluacion.csv (12)\n'
        'QUERY_IDS_DESARROLLO = [str(q) for q in desarrollo["query_id"]]\n'
        'QUERY_IDS_EVALUACION = list(evaluacion["evaluation_id"])\n'
        'QUERY_IDS_BARRIDO = QUERY_IDS_DESARROLLO + QUERY_IDS_EVALUACION\n'
        '\n'
        'CONSULTAS_BARRIDO = dict(zip(QUERY_IDS_DESARROLLO, desarrollo["query_text"])) | dict(\n'
        '    zip(QUERY_IDS_EVALUACION, evaluacion["query_text"])\n'
        ')\n'
        'QRELS = qrels_from_judgements(relevancias)\n'
        '\n'
        '# Vectores de consulta: de la cache de NB02/NB03 (mismos corpus_id, batch)\n'
        'vectores_desarrollo = encode_corpus(\n'
        '    _encoder, desarrollo["query_text"].tolist(), corpus_id="consultas_desarrollo",\n'
        '    kind="query", contract=CONTRATO, batch_size=32, cache_dir=CACHE,\n'
        ').vectors\n'
        'vectores_evaluacion = encode_corpus(\n'
        '    _encoder, evaluacion["query_text"].tolist(), corpus_id="consultas_evaluacion",\n'
        '    kind="query", contract=CONTRATO, batch_size=32, cache_dir=CACHE,\n'
        ').vectors\n'
        'vectores_query_barrido = truncate_dim(\n'
        '    np.vstack([vectores_desarrollo, vectores_evaluacion]), DIM\n'
        ')\n'
        '\n'
        'ORACULO = rank_queries_dense(\n'
        '    retriever_exacto, QUERY_IDS_BARRIDO, vectores_query_barrido, k=TOP_K\n'
        ')\n'
        'print(f"consultas del barrido: {len(CONSULTAS_BARRIDO)} "\n'
        '      f"({len(QUERY_IDS_DESARROLLO)} desarrollo + {len(QUERY_IDS_EVALUACION)} evaluación)")',
    ),
    (
        "markdown",
        '## C · El barrido de `ef`\n'
        '\n'
        'Para cada `ef`: un `BuscadorVectorial` nuevo contra la misma colección, su recall ANN@10 frente al oráculo (media, mínimo y p5 — la media sola esconde si la pérdida se reparte o se concentra), y su latencia (calentamiento aparte, 30 repeticiones cíclicas sobre las 20 consultas).\n'
        '\n'
        '> 💸 La primera ejecución paga la codificación de las consultas que NB05 no usó en su demo (hasta 14 llamadas nuevas, una por consulta — nunca una por `ef`, porque `codificar_consulta` cachea). Las siguientes ejecuciones no pagan ninguna.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000) — la colección definitiva de NB04\n'
        'from functools import lru_cache\n'
        '\n'
        '\n'
        '@lru_cache(maxsize=256)\n'
        'def codificar_consulta(texto: str):\n'
        '    codificado = encode_corpus(\n'
        '        _encoder, [texto], corpus_id="consulta_suelta",\n'
        '        kind="query", contract=CONTRATO, batch_size=1, cache_dir=CACHE,\n'
        '    )\n'
        '    return truncate_dim(codificado.vectors, DIM)[0]\n'
        '\n'
        '\n'
        'almacen = QdrantStore(\n'
        '    collection=COLECCION,\n'
        '    url=os.environ.get("AURUM_QDRANT_URL", "http://localhost:6333"),\n'
        '    api_key=os.environ.get("AURUM_QDRANT_API_KEY"),\n'
        '    prefix=CATALOG_PREFIX,\n'
        '    timeout=30,\n'
        ')\n'
        'print(f"puntos en la coleccion: {almacen.count():,}".replace(",", ".") + \n'
        '      f" · indice al dia: {almacen.index_ready()}")\n'
        '\n'
        '\n'
        'def construir_buscador(ef):\n'
        '    return BuscadorVectorial(almacen, codificar_consulta, top_k=TOP_K, ef=ef)\n'
        '\n'
        '\n'
        'barrido = barrido_ef(\n'
        '    construir_buscador, CONSULTAS_BARRIDO, ORACULO,\n'
        '    valores_ef=VALORES_EF, top_k=TOP_K, repeticiones=30, calentamiento=5,\n'
        ')\n'
        'barrido.style.hide(axis="index")',
    ),
    (
        "markdown",
        'PRUEBA: Después de ver la gráfica, me extrañó que los tiempos de ef = 64 fueran tan altos con respecto a un ef de 128 o 256. Por eso, la siguiente celda ejecutará aisladamente las consultas con un ef de 64 y verificar si los tiempos se mantienen o es un outlier producido por la ejecución secuencial de consultas con ef inferiores que pudieran "pervertir" los tiempos',
    ),
    (
        "code",
        '# Aislado: mismo barrido_ef, solo ef=64. Separa un problema real de ef=64\n'
        '# -sus tiempos salieron muy por encima de los de ef=128 y 256- de un efecto\n'
        '# de arrastre por ejecutar 16→32→64→128→256 seguidos. Variable propia -no\n'
        '# `barrido`- para no pisar el barrido completo que usa la sección D.\n'
        'barrido_ef64_aislado = barrido_ef(\n'
        '    construir_buscador, CONSULTAS_BARRIDO, ORACULO,\n'
        '    valores_ef=[64], top_k=TOP_K, repeticiones=30, calentamiento=5,\n'
        ')\n'
        'barrido_ef64_aislado.style.hide(axis="index")\n'
        '\n'
        '# RESULTADO: era ruido de aquella ejecución. Aislado, ef=64 da tiempos\n'
        '# coherentes con lo que explora.',
    ),
    (
        "markdown",
        '## D · D16 aplicada: qué configuraciones sobreviven, y cuál gana\n'
        '\n'
        'Se descarta lo que no llega al recall mínimo o se pasa del p95 máximo; entre lo que sobra, gana el `ef` de **menor p95 medido** -el coste real, no un supuesto de que más `ef` siempre tarda más-. La tabla se enseña completa, con las dos columnas nuevas (`cumple_d16`, `elegido_r04`): así se ve también lo que no ganó, no solo el veredicto final.',
    ),
    (
        "code",
        'barrido_anotado = aplicar_restriccion(\n'
        '    barrido, recall_minimo=RECALL_MINIMO, p95_maximo_ms=P95_MAXIMO_MS,\n'
        '    columna_recall=f"recall_ann_at_{TOP_K}",\n'
        ')\n'
        '\n'
        'elegidas = barrido_anotado[barrido_anotado["elegido_r04"]]\n'
        'if elegidas.empty:\n'
        '    EF_ELEGIDO = None\n'
        '    print("⚠️ Ninguna configuración cumple D16 a la vez. R04 queda sin fijar "\n'
        '          "-revisar la tabla y decidir si D16 se relaja o si hace falta subir ef.")\n'
        'else:\n'
        '    EF_ELEGIDO = int(elegidas.iloc[0]["ef"])\n'
        '    print(f"R04: ef = {EF_ELEGIDO}")\n'
        '\n'
        'barrido_anotado.style.hide(axis="index")',
    ),
    (
        "markdown",
        '### D.1 · Qué consulta hay detrás de un `_min` bajo\n'
        '\n'
        'La tabla de arriba resume, pero no dice **cuál** de las 20 consultas hunde el `_min` de una fila. Se inspeccionan **todos** los `ef` empatados en el peor `_min` del barrido -quedarse con uno solo por `idxmin()` escondería a los demás-; para mirar otro que no esté en el empate, se añade a `EFS_A_INSPECCIONAR`.\n'
        '\n'
        'Cada bloque sale ordenado de **peor a mejor recall**, así que la consulta problemática siempre cae arriba, con su texto y los `product_id` que se perdió -no un número suelto que hay que creerse-.',
    ),
    (
        "code",
        '# Todos los ef empatados en el peor _min del barrido, no solo el primero.\n'
        'MINIMO_GLOBAL = barrido_anotado[f"recall_ann_at_{TOP_K}_min"].min()\n'
        'EFS_A_INSPECCIONAR = sorted(\n'
        '    int(ef) for ef in barrido_anotado.loc[\n'
        '        barrido_anotado[f"recall_ann_at_{TOP_K}_min"] == MINIMO_GLOBAL, "ef"\n'
        '    ]\n'
        ')\n'
        'print(f"peor _min del barrido: {MINIMO_GLOBAL:.2f} · empatan: {EFS_A_INSPECCIONAR}\\n")\n'
        '\n'
        'bloques = []\n'
        'for ef in EFS_A_INSPECCIONAR:\n'
        '    buscador_inspeccion = construir_buscador(ef)\n'
        '    ann_inspeccion = {\n'
        '        query_id: [r.document_id for r in buscador_inspeccion.buscar(texto, top_k=TOP_K)]\n'
        '        for query_id, texto in CONSULTAS_BARRIDO.items()\n'
        '    }\n'
        '    bloque = tabla_recall_por_consulta(\n'
        '        ORACULO, ann_inspeccion, consultas=CONSULTAS_BARRIDO, k=TOP_K\n'
        '    ).sort_values("recall_ann")\n'
        '    bloque.insert(0, "ef", ef)\n'
        '    bloques.append(bloque)\n'
        '\n'
        'detalle_inspeccion = pd.concat(bloques, ignore_index=True)\n'
        'detalle_inspeccion.style.hide(axis="index")',
    ),
    (
        "markdown",
        'PRUEBA: las tres configuraciones de `ef` con el `recall_ann_at_10_min` más bajo tienen el mismo problema, así que hay que verificar si es un error '
        'estructural o de densidad en las posibles respuestas — es decir, que los resultados puedan ser buenos aunque los que el oráculo daría por aptos '
        'no entren en el top 10.',
    ),
    (
        "code",
        '# ef=128 en concreto -ya no está en el empate de _min-, para ver si la\n'
        '# misma consulta problemática se recupera o si aparece otra distinta.\n'
        '# Variables con sufijo _128 para no pisar las de la celda D.1 de arriba.\n'
        'EFS_INSPECCION_128 = [128]\n'
        '\n'
        'bloques_128 = []\n'
        'for ef in EFS_INSPECCION_128:\n'
        '    buscador_inspeccion = construir_buscador(ef)\n'
        '    ann_inspeccion = {\n'
        '        query_id: [r.document_id for r in buscador_inspeccion.buscar(texto, top_k=TOP_K)]\n'
        '        for query_id, texto in CONSULTAS_BARRIDO.items()\n'
        '    }\n'
        '    bloque = tabla_recall_por_consulta(\n'
        '        ORACULO, ann_inspeccion, consultas=CONSULTAS_BARRIDO, k=TOP_K\n'
        '    ).sort_values("recall_ann")\n'
        '    bloque.insert(0, "ef", ef)\n'
        '    bloques_128.append(bloque)\n'
        '\n'
        'detalle_128 = pd.concat(bloques_128, ignore_index=True)\n'
        'detalle_128.style.hide(axis="index")\n'
        '\n'
        '# CONCLUSIÓN: es densidad de la región, no un fallo del índice. Con ef=128\n'
        '# la consulta problemática recupera 10 de 10. Que ef=32 no lo haga no\n'
        '# significa que sus resultados sean malos para quien busca: eso lo mide la\n'
        '# comparación de nDCG contra el oráculo, más abajo.',
    ),
    (
        "markdown",
        '## E · La curva recall-latencia\n'
        '\n'
        'La región sombreada es la intersección de D16 -`p95 ≤ 20 ms` **y** `recall ≥ 0,90`-, no dos bandas cruzadas: un punto dentro de ella cumple las dos condiciones a la vez. El punto en otro color es R04.',
    ),
    (
        "code",
        'plot_ann_pareto(\n'
        '    barrido_anotado, recall_minimo=RECALL_MINIMO, p95_maximo_ms=P95_MAXIMO_MS,\n'
        '    recall_column=f"recall_ann_at_{TOP_K}",\n'
        '    subtitle=f"{COLECCION} · {len(CONSULTAS_BARRIDO)} consultas · top_k={TOP_K}",\n'
        ')',
    ),
    (
        "markdown",
        '## F · Recall por consulta del `ef` elegido\n'
        '\n'
        'El resumen (media/mínimo/p5) puede esconder si el error se reparte o se concentra; esta tabla enseña, consulta a consulta, qué `product_id` se perdió -no un booleano-.',
    ),
    (
        "code",
        'if EF_ELEGIDO is None:\n'
        '    print("Sin ef elegido por D16: no hay configuración que auditar aquí.")\n'
        'else:\n'
        '    buscador_elegido = construir_buscador(EF_ELEGIDO)\n'
        '    ann_elegido = {\n'
        '        query_id: [r.document_id for r in buscador_elegido.buscar(texto, top_k=TOP_K)]\n'
        '        for query_id, texto in CONSULTAS_BARRIDO.items()\n'
        '    }\n'
        '    detalle = tabla_recall_por_consulta(\n'
        '        ORACULO, ann_elegido, consultas=CONSULTAS_BARRIDO, k=TOP_K\n'
        '    )\n'
        '    display(detalle.style.hide(axis="index"))',
    ),
    (
        "markdown",
        '## G · La métrica clave: nDCG con el oráculo frente al ANN elegido\n'
        '\n'
        'Si el nDCG no baja al pasar del oráculo exacto al `ef` elegido, la fidelidad perdida -si la hay- no le costó nada al negocio: es la comprobación de que optimizar por D16 (velocidad) no se hizo a costa de la calidad real. Solo entran las 8 consultas de desarrollo, las únicas con juicios de relevancia.',
    ),
    (
        "code",
        'if EF_ELEGIDO is None:\n'
        '    print("Sin ef elegido por D16: no hay nDCG que comparar.")\n'
        'else:\n'
        '    ann_desarrollo = {qid: ann_elegido[qid] for qid in QUERY_IDS_DESARROLLO}\n'
        '    oraculo_desarrollo = {qid: ORACULO[qid] for qid in QUERY_IDS_DESARROLLO}\n'
        '    tabla_ndcg = comparar_ndcg_con_oraculo(\n'
        '        ann_desarrollo, oraculo_desarrollo, QRELS, k=TOP_K\n'
        '    )\n'
        '    columnas_pct = [c for c in tabla_ndcg.columns if c != "sistema"]\n'
        '    display(\n'
        '        tabla_ndcg.style.hide(axis="index")\n'
        '        .format({columna: "{:.1%}" for columna in columnas_pct})\n'
        '    )',
    ),
    (
        "markdown",
        '### G.1 · nDCG consulta a consulta\n'
        '\n'
        'El agregado de arriba puede esconder dos historias muy distintas: que la pérdida se reparta un poco entre las 8, o que se concentre en la misma consulta que ya delató un recall ANN bajo -la 18868, "botines marrones mujer tacon medio", si sigue siendo la peor con este `ef`-. Si su `delta` sale ~0 pese al recall ANN bajo, es la prueba de que el sustituto que trajo el ANN era igual de relevante para el negocio aunque no fuera el vecino exacto. Ordenada por `delta`: la consulta que más pierde queda arriba.',
    ),
    (
        "code",
        'if EF_ELEGIDO is None:\n'
        '    print("Sin ef elegido por D16: no hay nDCG que comparar.")\n'
        'else:\n'
        '    ndcg_por_consulta = comparar_ndcg_por_consulta(\n'
        '        ann_desarrollo, oraculo_desarrollo, QRELS,\n'
        '        consultas=CONSULTAS_BARRIDO, k=TOP_K,\n'
        '    )\n'
        '    columnas_pct = [c for c in ndcg_por_consulta.columns if c not in ("query_id", "consulta")]\n'
        '    display(\n'
        '        ndcg_por_consulta.style.hide(axis="index")\n'
        '        .format({columna: "{:+.1%}" if columna == "delta" else "{:.1%}" for columna in columnas_pct})\n'
        '    )',
    ),
    (
        "markdown",
        '## H · Lo que este notebook no mide, y por qué\n'
        '\n'
        '| Métrica del plan | Por qué no aparece aquí |\n'
        '|---|---|\n'
        '| Tiempo de construcción | `m`/`ef_construct` se quedan en los valores por defecto (D16); no se reconstruye ninguna colección en este barrido, así que no hay tiempo de construcción que comparar entre configuraciones |\n'
        '| Tamaño del índice | `ef` es un parámetro de **consulta**: no cambia el grafo ni el volumen en disco. El tamaño ya está medido en NB04 (289,5 MB) y es el mismo para las cinco filas de la tabla |\n'
        '| Nº de distancias/consulta | El cliente de Qdrant no expone esta estadística por consulta -a diferencia de `hnsw_stats.ndis` en FAISS-. Es un caso real de "si el proveedor oculta esa decisión, explicad qué control se pierde" (§3.2): se pierde la explicación mecánica de *por qué* sube el recall al subir `ef`, aunque la curva de la sección E sí muestra *que* sube |\n'
        '\n'
        '---\n'
        '\n'
        '## I · El artefacto',
    ),
    (
        "code",
        'destino = Path("..") / "artifacts" / "benchmark_ann.csv"\n'
        'barrido_anotado.to_csv(destino, index=False)\n'
        'print(f"Escrito {destino} · {destino.stat().st_size / 1024:.1f} KB · ef elegido (R04): {EF_ELEGIDO}")',
    ),
]

NB08_MUTACIONES = [
    ("markdown", '# NB08 · Mutaciones: los 24 eventos'),
    (
        "markdown",
        'Objetivo: demostrar **idempotencia y visibilidad**. No se cronometra qué motor es más rápido -eso ya lo hizo NB04/NB06-.\n'
        '\n'
        '### Lo que se hereda y no se vuelve a decidir\n'
        '\n'
        '| | |\n'
        '|---|---|\n'
        '| Motor · colección | Qdrant (R03) · `aurum_catalogo__gemini_embedding_2__A4__768` (NB04), 15.000 puntos |\n'
        '| Escritura síncrona | `QdrantStore.upsert(..., wait=True)` — ya fijado desde NB04, no es D18 |\n'
        '| Esquema de payload | `PAYLOAD_SCHEMAS["completo"]` (D13), nulos como cadena vacía (D14) |\n'
        '\n'
        '### Las decisiones de este notebook (`config.yaml` → `nb08_mutaciones`)\n'
        '\n'
        '| | Decidido | Por qué |\n'
        '|---|---|---|\n'
        '| **D18** | Espera activa con timeout, en la **lectura** (la escritura ya es síncrona) | El enunciado (§4.1) exige "saber esperar, fallar o informar" también al leer, no solo al escribir: `wait=True` no garantiza que el HNSW ya lo refleje |\n'
        '| **D19** | Este notebook se ejecuta **después** de NB07 | NB07 necesitaba calibrar su regla de duplicados contra el catálogo sin mutar — ver `notebooks/07_duplicados.ipynb` |\n'
        '\n'
        '`eventos_catalogo.csv` trae 24 operaciones ordenadas por `sequence`: 8 actualizaciones (`UPSERT`, el registro ya existía → `catalog_version` 1→2), 8 bajas (`DELETE`) y 8 altas (`UPSERT`, `product_id` nuevo, `catalog_version=1`). La regla de clasificación ya está verificada contra el dato (`README_DATOS.md`), así que `clasificar_eventos` la aplica, no la redescubre.\n'
        '\n'
        '### 🗺️ Mapa de datos: qué crea cada sección y quién lo usa después\n'
        '\n'
        'Una variable creada en una celda sigue viva en las de abajo, así que el orden de ejecución importa. La tabla es el hilo conductor: qué produce cada sección y dónde se usa después. Las tres filas en negrita son las que **escriben de verdad en Qdrant**.\n'
        '\n'
        '| Sección | Crea | Se usa después en |\n'
        '|---|---|---|\n'
        '| Setup | `eventos` (24 filas + columna `tipo`) · `completo` (15.000 filas) | B, C, D, E, F |\n'
        '| A | `indice` (la conexión) · `RECUENTO_ANTES` | `indice`: B→H · `RECUENTO_ANTES`: H |\n'
        '| B | `eventos_upsert`/`eventos_baja` (el reparto en dos grupos) · `vectores_upsert_por_id` (16 vectores nuevos) · `vectores_baja_por_id` (8 vectores originales, de la cache) | C (construye los puntos) · D (verifica) · F (elige la muestra) |\n'
        '| **C** | `puntos_upsert`, `ids_baja` · **1ª escritura real en Qdrant** (`indice.upsert`/`indice.delete`, dentro de `aplicar_secuencia`) · `RECUENTO_DESPUES_1` | D, E, F, G, H |\n'
        '| D | `trazas`, `tabla_trazas` (visibilidad de los 24 eventos) | H |\n'
        '| E | `tabla_titulos` (título antes/después de las 8 actualizaciones) | H |\n'
        '| **F** | **2ª escritura**, repite exactamente la de C con los mismos `puntos_upsert`/`ids_baja` · `RECUENTO_DESPUES_2`, `tabla_muestra_idempotencia`, `IDEMPOTENTE` | H |\n'
        '| **G** | **3ª escritura**, repite la de C con el mismo contenido pero en otro orden · `RECUENTO_BARAJADO` | H |\n'
        '| H | Junta todo lo anterior en `artifacts/mutaciones.json` | — (es la última celda) |',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 eventos_catalogo.csv (24) + catalogo_productos.csv (15.000, desde cache)\n'
        'import json\n'
        'import os\n'
        'import sys\n'
        'from pathlib import Path\n'
        '\n'
        'import numpy as np\n'
        'import pandas as pd\n'
        '\n'
        'sys.path.insert(0, str(Path("..") / "src"))\n'
        '\n'
        'from dotenv import load_dotenv\n'
        '\n'
        'from aurum.almacen import PAYLOAD_SCHEMAS, add_normalized_key, build_payload\n'
        'from aurum.datos import load_csv\n'
        'from aurum.embeddings import (\n'
        '    GeminiEncoder, cache_key, corpus_fingerprint, encode_corpus, truncate_dim,\n'
        ')\n'
        'from aurum.motores import CATALOG_PREFIX, catalog_collection_name\n'
        'from aurum.motores.base import Point\n'
        'from aurum.motores.qdrant import QdrantStore\n'
        'from aurum.mutaciones import aplicar_secuencia, clasificar_eventos, verificar_evento\n'
        '\n'
        'load_dotenv(Path("..") / ".env")\n'
        'DATA = Path("..") / "data"\n'
        'CACHE = Path("..") / "artifacts" / "embeddings"\n'
        'eventos = clasificar_eventos(load_csv(DATA / "eventos_catalogo.csv"))\n'
        'completo = load_csv(DATA / "catalogo_productos.csv")\n'
        '\n'
        'MODELO, CONTRATO, PLANTILLA = "gemini-embedding-2", "sin_contrato", "A4"\n'
        'DIM = 768\n'
        'COLECCION = catalog_collection_name(model=MODELO, template=PLANTILLA, dim=DIM)\n'
        'LOTE = 128                    # D15, mismo lote que la ingesta de NB04\n'
        'ESQUEMA = "completo"          # D13\n'
        'POLITICA_NULOS = "cadena_vacia"   # D14\n'
        'NORMALIZACION = "unaccent"    # D03\n'
        'CAMPOS_FILTRABLES = ["brand", "color"]\n'
        'TIMEOUT_S, INTERVALO_S = 15.0, 0.5   # D18\n'
        '\n'
        'print(eventos["tipo"].value_counts().to_string())\n'
        'print(f"\\ncoleccion: {COLECCION}")',
    ),
    (
        "markdown",
        '## A · Conexión y recuento inicial\n'
        '\n'
        '**Entrada:** `COLECCION` (del setup). **Salida:** `indice` -la conexión que usan **todas** las celdas de aquí en adelante- y `RECUENTO_ANTES`.\n'
        '\n'
        '`RECUENTO_ANTES` no sale de ningún CSV -es lo que Qdrant reporta ahora mismo, en vivo- y es el número contra el que se comparan todos los recuentos posteriores (secciones C, F y G).',
    ),
    (
        "code",
        '# ⚠️ requiere `make motor-up MOTOR=qdrant`\n'
        'indice = QdrantStore(\n'
        '    collection=COLECCION,\n'
        '    url=os.environ.get("AURUM_QDRANT_URL", "http://localhost:6333"),\n'
        '    api_key=os.environ.get("AURUM_QDRANT_API_KEY"),\n'
        '    prefix=CATALOG_PREFIX,\n'
        '    timeout=30,\n'
        ')\n'
        'RECUENTO_ANTES = indice.count()\n'
        'print(f"puntos antes de los eventos: {RECUENTO_ANTES:,}".replace(",", "."))',
    ),
    (
        "markdown",
        '## B · Vectores para altas y actualizaciones\n'
        '\n'
        '**Entrada:** `eventos` (del setup). **Salida:** `eventos_upsert`/`eventos_baja` (el mismo `eventos`, repartido en dos grupos) y los dos diccionarios de vectores, `vectores_upsert_por_id` y `vectores_baja_por_id` -las claves son `record_id`, así que en las secciones D y F basta con `vectores_..._por_id[record_id]` para encontrar el vector de un evento concreto-.\n'
        '\n'
        'Las bajas no se reencodean -traen `title`/`brand` vacíos-: solo hace falta su `record_id` para borrarlas, y su vector **original** (de la cache de NB04/NB06) para poder comprobar después que ya no aparecen en una búsqueda. Por eso son dos celdas de código separadas: la primera pide vectores **nuevos** a la API (altas+actualizaciones), la segunda solo **lee de la cache** (bajas) y no gasta ninguna llamada.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 16 textos nuevos (8 altas + 8 actualizaciones) — 💸 primera vez: 16 llamadas de pago\n'
        'eventos_upsert = eventos[eventos["tipo"] != "baja"].reset_index(drop=True)\n'
        'eventos_baja = eventos[eventos["tipo"] == "baja"].reset_index(drop=True)\n'
        '\n'
        '_encoder = GeminiEncoder(\n'
        '    api_key=os.environ.get("GEMINI_API_KEY"), model_id=MODELO,\n'
        '    native_dim=3072, window=8192,\n'
        ')\n'
        'codificado_mutados = encode_corpus(\n'
        '    _encoder, eventos_upsert["text"].tolist(), corpus_id="eventos_catalogo_upsert",\n'
        '    kind="document", contract=CONTRATO, batch_size=16, cache_dir=CACHE,\n'
        ')\n'
        'vectores_upsert_por_id = dict(zip(\n'
        '    eventos_upsert["record_id"], truncate_dim(codificado_mutados.vectors, DIM)\n'
        '))\n'
        'print(f"vectores nuevos: {len(vectores_upsert_por_id)} · desde cache: "\n'
        '      f"{codificado_mutados.stats.desde_cache}")',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000) — vectores desde la cache de NB04, solo para las 8 bajas\n'
        'CORPUS_ID_COMPLETO = f"catalogo_productos__{PLANTILLA}"\n'
        'from aurum.plantillas import render_template\n'
        '\n'
        'textos_completo = render_template(completo, PLANTILLA)\n'
        'clave = cache_key(\n'
        '    model_id=MODELO, kind="document", contract=CONTRATO,\n'
        '    corpus_id=CORPUS_ID_COMPLETO, fingerprint=corpus_fingerprint(textos_completo),\n'
        ')\n'
        'if not (CACHE / f"{clave}.npy").exists():\n'
        '    raise RuntimeError(\n'
        '        f"Los vectores de {CORPUS_ID_COMPLETO} no estan en cache ({clave}).\\n"\n'
        '        f"Deberian estar desde NB04/NB06 -esta celda no paga 15.000 llamadas nuevas."\n'
        '    )\n'
        'codificado_completo = encode_corpus(\n'
        '    _encoder, textos_completo, corpus_id=CORPUS_ID_COMPLETO,\n'
        '    kind="document", contract=CONTRATO, batch_size=32, cache_dir=CACHE,\n'
        ')\n'
        'vectores_completo = truncate_dim(codificado_completo.vectors, DIM)\n'
        'indice_por_record_id = {rid: i for i, rid in enumerate(completo["record_id"])}\n'
        'vectores_baja_por_id = {\n'
        '    rid: vectores_completo[indice_por_record_id[rid]] for rid in eventos_baja["record_id"]\n'
        '}\n'
        'print(f"vectores originales recuperados para las bajas: {len(vectores_baja_por_id)}")',
    ),
    (
        "markdown",
        '## C · Construir los puntos y aplicar la secuencia (1ª pasada)\n'
        '\n'
        '**Entrada:** `eventos_upsert`/`eventos_baja` y los vectores, de la sección B. **Salida:** `puntos_upsert`, `ids_baja` -se reutilizan tal cual en F y G, sin recalcular nada- y `RECUENTO_DESPUES_1`.\n'
        '\n'
        'El payload de cada punto sale de las columnas de `eventos_catalogo.csv` -`product_id`, `title`, `brand`, `color`, `catalog_version`, `active`, más las claves normalizadas- filtradas por `PAYLOAD_SCHEMAS["completo"]` (D13); el vector es el de la sección B, ya emparejado por `record_id`.\n'
        '\n'
        '⚠️ **Aquí se escribe de verdad en Qdrant, la primera de tres veces.** `aplicar_secuencia` (`src/aurum/mutaciones.py`) no hace nada propio: envuelve dos llamadas -`indice.upsert(puntos_upsert, ...)` para las 16 altas y actualizaciones, y `indice.delete(record_id)` por cada una de las 8 bajas- para no repetirlas en las tres secciones (C, F, G) que aplican la secuencia completa.',
    ),
    (
        "code",
        'eventos_upsert_claves = eventos_upsert.copy()\n'
        'for campo in CAMPOS_FILTRABLES:\n'
        '    eventos_upsert_claves = add_normalized_key(\n'
        '        eventos_upsert_claves, field=campo, mode=NORMALIZACION\n'
        '    )\n'
        '\n'
        'puntos_upsert = [\n'
        '    Point(\n'
        '        record_id=fila["record_id"],\n'
        '        vector=vectores_upsert_por_id[fila["record_id"]],\n'
        '        payload=build_payload(\n'
        '            fila, fields=PAYLOAD_SCHEMAS[ESQUEMA], null_policy=POLITICA_NULOS\n'
        '        ),\n'
        '    )\n'
        '    for fila in eventos_upsert_claves.to_dict("records")\n'
        ']\n'
        'ids_baja = list(eventos_baja["record_id"])\n'
        '\n'
        '# aplicar_secuencia hace, por debajo, exactamente esto:\n'
        '#   indice.upsert(puntos_upsert, batch_size=LOTE)      -> las 16 altas + actualizaciones\n'
        '#   for record_id in ids_baja: indice.delete(record_id) -> las 8 bajas\n'
        'resultado_1 = aplicar_secuencia(indice, puntos_upsert, ids_baja, batch_size=LOTE)\n'
        'RECUENTO_DESPUES_1 = indice.count()\n'
        'print(f"aplicados: {resultado_1[\'n_upsert\']} upsert + {resultado_1[\'n_delete\']} delete "\n'
        '      f"en {resultado_1[\'segundos\']:.2f} s")\n'
        'print(f"recuento: {RECUENTO_ANTES:,} -> {RECUENTO_DESPUES_1:,} "\n'
        '      f"(esperado {RECUENTO_ANTES - len(ids_baja) + len(eventos_upsert[eventos_upsert[\'tipo\']==\'alta\'])})"\n'
        '      .replace(",", "."))',
    ),
    (
        "markdown",
        '## D · Verificación de visibilidad, evento a evento (D18)\n'
        '\n'
        '**Entrada:** `eventos` (setup) + los vectores de B -esta sección no escribe nada nuevo en Qdrant, solo **lee** lo que la sección C acaba de escribir-. **Salida:** `tabla_trazas`, que va al artefacto en H.\n'
        '\n'
        'Las dos rutas que pide el plan -lectura por ID y búsqueda vectorial- para los 24 eventos, no solo una muestra: es barato (24×2 comprobaciones, con reintento hasta 15 s cada una) y es justo lo que va al artefacto.',
    ),
    (
        "code",
        'trazas = []\n'
        'for fila in eventos.to_dict("records"):\n'
        '    tipo = fila["tipo"]\n'
        '    if tipo == "baja":\n'
        '        vector = vectores_baja_por_id[fila["record_id"]]\n'
        '        version_esperada = None\n'
        '    else:\n'
        '        vector = vectores_upsert_por_id[fila["record_id"]]\n'
        '        version_esperada = 2 if tipo == "actualizacion" else None\n'
        '    traza = verificar_evento(\n'
        '        indice, tipo, record_id=fila["record_id"], vector=vector,\n'
        '        catalog_version_esperado=version_esperada,\n'
        '        timeout_s=TIMEOUT_S, intervalo_s=INTERVALO_S,\n'
        '    )\n'
        '    traza["event_id"], traza["product_id"] = fila["event_id"], fila["product_id"]\n'
        '    trazas.append(traza)\n'
        '\n'
        'tabla_trazas = pd.DataFrame([\n'
        '    {\n'
        '        "event_id": t["event_id"], "tipo": t["tipo"], "product_id": t["product_id"],\n'
        '        "visible_por_id (lectura directa, get)": t["por_id"]["visible"],\n'
        '        "seg_hasta_visible_por_id": round(t["por_id"]["segundos"], 3),\n'
        '        "visible_por_busqueda (indice vectorial, search)": t["por_busqueda"]["visible"],\n'
        '        "seg_hasta_visible_por_busqueda": round(t["por_busqueda"]["segundos"], 3),\n'
        '    }\n'
        '    for t in trazas\n'
        '])\n'
        'TODO_VISIBLE = bool(\n'
        '    tabla_trazas["visible_por_id (lectura directa, get)"].all()\n'
        '    and tabla_trazas["visible_por_busqueda (indice vectorial, search)"].all()\n'
        ')\n'
        'print(f"24/24 eventos visibles por las dos rutas: {TODO_VISIBLE}")\n'
        'tabla_trazas.style.hide(axis="index")',
    ),
    (
        "markdown",
        '## E · Título nuevo en las actualizaciones\n'
        '\n'
        '**Entrada:** `completo` (el catálogo original, del setup) y `eventos` -tampoco escribe nada, solo compara tres fuentes de la misma columna `title`-. **Salida:** `tabla_titulos`, al artefacto en H.\n'
        '\n'
        'D18 ya comprobó `catalog_version=2` evento a evento; falta el segundo requisito del plan: que el `title` leído sea el nuevo, no un punto que exista con el dato viejo. Para verlo de verdad hace falta el **antes** -el título que tenía el producto en `catalogo_productos.csv`, antes de aplicar nada- junto al **después** -lo que devuelve `indice.get()` ahora-, no solo un `True`/`False` agregado.',
    ),
    (
        "code",
        'titulo_antes_por_id = dict(zip(completo["record_id"], completo["title"]))\n'
        '\n'
        'tabla_titulos = pd.DataFrame([\n'
        '    {\n'
        '        "event_id": fila["event_id"],\n'
        '        "product_id": fila["product_id"],\n'
        '        "titulo_antes (catalogo_productos.csv)": titulo_antes_por_id.get(fila["record_id"]),\n'
        '        "titulo_despues (indice.get)": indice.get(fila["record_id"]).payload.get("title"),\n'
        '        "titulo_esperado (eventos_catalogo.csv)": fila["title"],\n'
        '    }\n'
        '    for fila in eventos[eventos["tipo"] == "actualizacion"].to_dict("records")\n'
        '])\n'
        'tabla_titulos["actualizado_ok"] = (\n'
        '    tabla_titulos["titulo_despues (indice.get)"]\n'
        '    == tabla_titulos["titulo_esperado (eventos_catalogo.csv)"]\n'
        ')\n'
        'print(f"actualizaciones con titulo nuevo: {tabla_titulos[\'actualizado_ok\'].sum()}/8")\n'
        'tabla_titulos.style.hide(axis="index")',
    ),
    (
        "markdown",
        '## F · Repetir la secuencia completa (idempotencia)\n'
        '\n'
        '**Entrada:** `puntos_upsert`/`ids_baja` de la sección **C** -exactamente los mismos objetos, no se reconstruyen-. **Salida:** `RECUENTO_DESPUES_2`, `tabla_muestra_idempotencia`, `IDEMPOTENTE`, al artefacto en H.\n'
        '\n'
        'Es la **misma llamada a `aplicar_secuencia` de la sección C**, con los mismos puntos y los mismos IDs a borrar. `upsert` sobrescribe por `record_id` y borrar un id ya ausente no es error en Qdrant, así que el recuento debe quedar igual — pero el recuento total podría cuadrar por casualidad si algo se borrara y otra cosa se creara a la vez. Por eso se captura un punto de cada tipo **antes** y **después** de repetir, y se enseña que ni cambia de contenido ni se duplica.',
    ),
    (
        "code",
        'muestra_idempotencia = {\n'
        '    "alta": eventos_upsert.loc[eventos_upsert["tipo"] == "alta", "record_id"].iloc[0],\n'
        '    "actualizacion": eventos_upsert.loc[\n'
        '        eventos_upsert["tipo"] == "actualizacion", "record_id"\n'
        '    ].iloc[0],\n'
        '    "baja": eventos_baja["record_id"].iloc[0],\n'
        '}\n'
        '\n'
        '\n'
        'def _resumen_punto(punto):\n'
        '    if punto is None:\n'
        '        return {"existe": False, "catalog_version": None, "title": None}\n'
        '    return {\n'
        '        "existe": True,\n'
        '        "catalog_version": punto.payload.get("catalog_version"),\n'
        '        "title": punto.payload.get("title"),\n'
        '    }\n'
        '\n'
        '\n'
        'antes_de_repetir = {tipo: indice.get(rid) for tipo, rid in muestra_idempotencia.items()}\n'
        '\n'
        'resultado_2 = aplicar_secuencia(indice, puntos_upsert, ids_baja, batch_size=LOTE)\n'
        'RECUENTO_DESPUES_2 = indice.count()\n'
        '\n'
        'despues_de_repetir = {tipo: indice.get(rid) for tipo, rid in muestra_idempotencia.items()}\n'
        '\n'
        'tabla_muestra_idempotencia = pd.DataFrame([\n'
        '    {\n'
        '        "tipo": tipo,\n'
        '        "record_id": rid,\n'
        '        **{f"{k}_antes": v for k, v in _resumen_punto(antes_de_repetir[tipo]).items()},\n'
        '        **{f"{k}_despues": v for k, v in _resumen_punto(despues_de_repetir[tipo]).items()},\n'
        '    }\n'
        '    for tipo, rid in muestra_idempotencia.items()\n'
        '])\n'
        'def _campos_iguales(a, b):\n'
        '    # None/NaN no es igual a si mismo con `==` (semantica estandar de NaN):\n'
        '    # sin este caso, la fila "baja" -None en ambos lados, porque el punto no\n'
        '    # existe ni antes ni despues- se marcaria como "distinta" por error.\n'
        '    if pd.isna(a) and pd.isna(b):\n'
        '        return True\n'
        '    return a == b\n'
        '\n'
        '\n'
        'tabla_muestra_idempotencia["identico"] = tabla_muestra_idempotencia.apply(\n'
        '    lambda f: all(\n'
        '        _campos_iguales(f[f"{campo}_antes"], f[f"{campo}_despues"])\n'
        '        for campo in ("existe", "catalog_version", "title")\n'
        '    ),\n'
        '    axis=1,\n'
        ')\n'
        '\n'
        'IDEMPOTENTE = (\n'
        '    RECUENTO_DESPUES_1 == RECUENTO_DESPUES_2\n'
        '    and bool(tabla_muestra_idempotencia["identico"].all())\n'
        ')\n'
        'print(f"recuento tras repetir: {RECUENTO_DESPUES_2:,} "\n'
        '      f"(1a pasada: {RECUENTO_DESPUES_1:,}) · idempotente: {IDEMPOTENTE}".replace(",", "."))\n'
        'tabla_muestra_idempotencia.style.hide(axis="index")',
    ),
    (
        "markdown",
        '## G · [Opcional] Eventos fuera de orden\n'
        '\n'
        '**Entrada:** `puntos_upsert`/`ids_baja` de C otra vez, pero barajados. **Salida:** `RECUENTO_BARAJADO`, al artefacto en H.\n'
        '\n'
        'Tercera y última escritura -misma `aplicar_secuencia`, mismo contenido, solo cambia el orden de envío-. Los 24 eventos tocan 24 `record_id` distintos entre sí, así que el orden no debería alterar el resultado: se baraja y se reaplica, y si el recuento final cambiara, el proceso no sería correcto.',
    ),
    (
        "code",
        'import random\n'
        '\n'
        'puntos_barajados = list(puntos_upsert)\n'
        'random.Random(0).shuffle(puntos_barajados)\n'
        'ids_baja_barajados = list(ids_baja)\n'
        'random.Random(1).shuffle(ids_baja_barajados)\n'
        '\n'
        'aplicar_secuencia(indice, puntos_barajados, ids_baja_barajados, batch_size=LOTE)\n'
        'RECUENTO_BARAJADO = indice.count()\n'
        'print(f"recuento tras aplicar fuera de orden: {RECUENTO_BARAJADO:,} · "\n'
        '      f"igual que antes: {RECUENTO_BARAJADO == RECUENTO_DESPUES_2}".replace(",", "."))',
    ),
    (
        "markdown",
        '## H · El artefacto\n'
        '\n'
        '**Entrada:** todo lo de A-G (`RECUENTO_ANTES`, `RECUENTO_DESPUES_1/2`, `RECUENTO_BARAJADO`, `IDEMPOTENTE`, `tabla_muestra_idempotencia`, `tabla_titulos`, `trazas`) — esta celda no calcula nada nuevo, solo empaqueta. **Salida:** `artifacts/mutaciones.json` en disco, fin del notebook.',
    ),
    (
        "code",
        'artefacto = {\n'
        '    "coleccion": COLECCION,\n'
        '    "recuento": {\n'
        '        "antes": RECUENTO_ANTES,\n'
        '        "despues_1a_pasada": RECUENTO_DESPUES_1,\n'
        '        "despues_2a_pasada": RECUENTO_DESPUES_2,\n'
        '        "despues_fuera_de_orden": RECUENTO_BARAJADO,\n'
        '        "idempotente": IDEMPOTENTE,\n'
        '        # via to_json: to_dict() deja bool_/int64 de numpy, que json.dumps no acepta\n'
        '        "muestra_idempotencia": json.loads(\n'
        '            tabla_muestra_idempotencia.drop(columns=["record_id"]).to_json(orient="records")\n'
        '        ),\n'
        '    },\n'
        '    "titulos_actualizados_ok": int(tabla_titulos["actualizado_ok"].sum()),\n'
        '    "eventos": [\n'
        '        {\n'
        '            "event_id": t["event_id"], "product_id": t["product_id"], "tipo": t["tipo"],\n'
        '            "por_id": t["por_id"], "por_busqueda": t["por_busqueda"],\n'
        '        }\n'
        '        for t in trazas\n'
        '    ],\n'
        '}\n'
        'destino = Path("..") / "artifacts" / "mutaciones.json"\n'
        'destino.write_text(json.dumps(artefacto, indent=2, ensure_ascii=False), encoding="utf-8")\n'
        'print(f"Escrito {destino} · {destino.stat().st_size / 1024:.1f} KB")',
    ),
]

NB07_DUPLICADOS = [
    ("markdown", '# NB07 · Altas potencialmente duplicadas'),
    (
        "markdown",
        'Una regla reproducible para decidir si una alta nueva es un duplicado de algo que ya está en el catálogo, calibrada **solo** con las 14 altas de desarrollo y congelada antes de abrir el conjunto de evaluación.\n'
        '\n'
        '### Lo que se hereda y no se vuelve a decidir\n'
        '\n'
        '| | |\n'
        '|---|---|\n'
        '| Motor · colección | Qdrant (R03) · `aurum_catalogo__gemini_embedding_2__A4__768` (NB04) |\n'
        '| `ef` de consulta | **32** (R04, NB06) — la misma colección de producción, no el oráculo exacto |\n'
        '| Interfaz | `BuscadorVectorial`, igual que NB05/NB06 |\n'
        '\n'
        '### Las decisiones de este notebook (`config.yaml` → `nb07_duplicados`)\n'
        '\n'
        '| | Decidido | Por qué |\n'
        '|---|---|---|\n'
        '| **D19** | Los candidatos salen de Qdrant tal como está *ahora*, antes de que NB08 aplique ningún evento | La base vectorial debe seguir siendo el mecanismo de generación de candidatos (enunciado §4.2); recalcular en local no lo prueba |\n'
        '| **D20** | Señales: `score_top1`, margen (top1−top2), marca, color | Léxico de título descartado -el embedding ya debería cubrir títulos reordenados-; marca y color usan el payload, no el `text` codificado |\n'
        '| **D21** | Dos caminos en OR: `score ≥ umbral_texto_solo ∧ margen ≥ margen_minimo` **o** `score ≥ umbral_texto_corroborado ∧ (marca ∨ color)` | El color nunca bloquea un duplicado por sí solo -"mismo producto, otra talla o color" puede contar como duplicado-; el OR entre marca y color evita que el 37,4%/4,4% de productos sin esos campos bloqueen el segundo camino |\n'
        '| **D22** 🚨 | Maximizar recall con como mucho **2 falsos positivos** de los 7 negativos de desarrollo | Fijado **antes** de ver la curva P/R -si se fija después, deja de ser una restricción y pasa a describir el resultado, mismo patrón que D16-. Los falsos negativos degradan el top-10 de búsqueda con variantes redundantes sin que nada vuelva a auditar el catálogo publicado |\n'
        '\n'
        '`umbral_texto_solo` (antes "τ_alto") es el listón para fiarse del texto **solo**, sin corroboración; `umbral_texto_corroborado` (antes "τ_bajo") es el listón, más bajo, para cuando marca o color respaldan la similitud; `margen_minimo` (antes "δ") es cuánto tiene que ganarle el top-1 al top-2 para no ser una foto-finish.\n'
        '\n'
        '> ⚠️ El Camino 2 (corroborado) no tiene ni un ejemplo que lo active en solitario dentro de las 14 altas de desarrollo -los 7 positivos son reenvíos casi literales que ya caen en el Camino 1-. `umbral_texto_corroborado` se fija por criterio razonado, no por barrido: es una limitación declarada, no una carencia oculta.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 altas_desarrollo.csv (14) + altas_evaluacion.csv (14)\n'
        'import os\n'
        'import sys\n'
        'from pathlib import Path\n'
        '\n'
        'import numpy as np\n'
        'import pandas as pd\n'
        '\n'
        'sys.path.insert(0, str(Path("..") / "src"))\n'
        '\n'
        'from dotenv import load_dotenv\n'
        '\n'
        'from aurum.busqueda import BuscadorVectorial\n'
        'from aurum.datos import load_csv\n'
        'from aurum.duplicados import (\n'
        '    barrido_umbrales,\n'
        '    calcular_senales,\n'
        '    elegir_punto_operacion,\n'
        '    resultados_duplicados,\n'
        ')\n'
        'from aurum.embeddings import GeminiEncoder, encode_corpus, truncate_dim\n'
        'from aurum.graficas import plot_duplicate_threshold_sweep\n'
        'from aurum.motores import CATALOG_PREFIX, catalog_collection_name\n'
        'from aurum.motores.qdrant import QdrantStore\n'
        '\n'
        'load_dotenv(Path("..") / ".env")\n'
        'DATA = Path("..") / "data"\n'
        'CACHE = Path("..") / "artifacts" / "embeddings"\n'
        'altas_desarrollo = load_csv(DATA / "altas_desarrollo.csv")\n'
        'altas_evaluacion = load_csv(DATA / "altas_evaluacion.csv")\n'
        '\n'
        'MODELO, CONTRATO, PLANTILLA = "gemini-embedding-2", "sin_contrato", "A4"\n'
        'DIM = 768\n'
        'COLECCION = catalog_collection_name(model=MODELO, template=PLANTILLA, dim=DIM)\n'
        'EF = 32  # R04 (NB06): el punto de operación de producción, no el oráculo exacto\n'
        'MAX_FP = 2  # D22\n'
        '\n'
        'print(f"desarrollo : {len(altas_desarrollo)} altas ({altas_desarrollo[\'is_duplicate\'].sum()} duplicados)")\n'
        'print(f"evaluacion : {len(altas_evaluacion)} altas, sin etiqueta")\n'
        'print(f"coleccion  : {COLECCION} · ef={EF}")',
    ),
    (
        "markdown",
        '## A · Conexión a Qdrant\n'
        '\n'
        'La misma colección de NB04-NB06, todavía sin las mutaciones de NB08 (D19). `codificar_consulta` no lleva instrucción de tarea -`CONTRATO="sin_contrato"`, D10- así que `kind="document"` aquí es solo trazabilidad de caché, no cambia el vector.',
    ),
    (
        "code",
        '_encoder = GeminiEncoder(\n'
        '    api_key=os.environ.get("GEMINI_API_KEY"), model_id=MODELO,\n'
        '    native_dim=3072, window=8192,\n'
        ')\n'
        '\n'
        '\n'
        'def codificar_consulta(texto: str):\n'
        '    codificado = encode_corpus(\n'
        '        _encoder, [texto], corpus_id="altas_duplicados",\n'
        '        kind="document", contract=CONTRATO, batch_size=1, cache_dir=CACHE,\n'
        '    )\n'
        '    return truncate_dim(codificado.vectors, DIM)[0]\n'
        '\n'
        '\n'
        'almacen = QdrantStore(\n'
        '    collection=COLECCION,\n'
        '    url=os.environ.get("AURUM_QDRANT_URL", "http://localhost:6333"),\n'
        '    api_key=os.environ.get("AURUM_QDRANT_API_KEY"),\n'
        '    prefix=CATALOG_PREFIX,\n'
        '    timeout=30,\n'
        ')\n'
        'buscador = BuscadorVectorial(almacen, codificar_consulta, top_k=2, ef=EF)\n'
        'print(f"puntos en la coleccion: {almacen.count():,}".replace(",", ".") +\n'
        '      f" · indice al dia: {almacen.index_ready()}")',
    ),
    (
        "markdown",
        '## B · Señales de DESARROLLO (D20)\n'
        '\n'
        'Para cada una de las 14 altas: su top-2 en la colección, y las cuatro señales frente al top-1. `margen = +∞` señalaría que no hubo segundo candidato -no debería pasar con 15.000 puntos, pero `SenalesDuplicado.margen` lo trata como la máxima confianza posible, no como cero-.',
    ),
    (
        "code",
        'senales_desarrollo = calcular_senales(buscador, altas_desarrollo)\n'
        '\n'
        'tabla_senales = pd.DataFrame([\n'
        '    {\n'
        '        "incoming_id": s.incoming_id,\n'
        '        "matched_product_id": s.matched_product_id,\n'
        '        "score_top1": round(s.score_top1, 4),\n'
        '        "score_top2": round(s.score_top2, 4),\n'
        '        "margen": round(s.margen, 4),\n'
        '        "marca_coincide": s.marca_coincide,\n'
        '        "color_coincide": s.color_coincide,\n'
        '        "is_duplicate (real)": s.is_duplicate,\n'
        '    }\n'
        '    for s in senales_desarrollo\n'
        '])\n'
        'tabla_senales.style.hide(axis="index")',
    ),
    (
        "markdown",
        '## C · El barrido (D21)\n'
        '\n'
        'La rejilla de `umbral_texto_solo`/`umbral_texto_corroborado` sale del rango de `score_top1` **observado** en estas 14 altas, no de un valor de libro -no hay forma de saber de antemano en qué rango cae la similitud de `gemini-embedding-2` para este catálogo-. `margen_minimo` barre desde 0 hasta el margen máximo observado.',
    ),
    (
        "code",
        'scores_observados = [s.score_top1 for s in senales_desarrollo]\n'
        'margenes_observados = [s.margen for s in senales_desarrollo if s.margen != float("inf")]\n'
        'SCORE_MIN, SCORE_MAX = min(scores_observados), max(scores_observados)\n'
        'MARGEN_MAX = max(margenes_observados) if margenes_observados else 0.1\n'
        '\n'
        'VALORES_UMBRAL_TEXTO_SOLO = sorted(set(round(v, 4) for v in np.linspace(SCORE_MIN, SCORE_MAX, 9)))\n'
        'VALORES_UMBRAL_TEXTO_CORROBORADO = sorted(set(round(v, 4) for v in np.linspace(SCORE_MIN, SCORE_MAX, 9)))\n'
        'VALORES_MARGEN_MINIMO = sorted(set(round(v, 4) for v in np.linspace(0.0, MARGEN_MAX, 6)))\n'
        '\n'
        'barrido = barrido_umbrales(\n'
        '    senales_desarrollo,\n'
        '    valores_umbral_texto_solo=VALORES_UMBRAL_TEXTO_SOLO,\n'
        '    valores_margen_minimo=VALORES_MARGEN_MINIMO,\n'
        '    valores_umbral_texto_corroborado=VALORES_UMBRAL_TEXTO_CORROBORADO,\n'
        ')\n'
        'print(f"combinaciones evaluadas: {len(barrido)} "\n'
        '      f"(umbral_texto_solo/umbral_texto_corroborado en [{SCORE_MIN:.3f}, {SCORE_MAX:.3f}], "\n'
        '      f"margen_minimo en [0, {MARGEN_MAX:.3f}])")\n'
        '# recall: de los 7 duplicados reales, cuántos atrapa la regla (1.0 = todos)\n'
        '# precision: de lo que la regla marca duplicado, cuánto lo era de verdad\n'
        'barrido.sort_values(["recall", "fp"], ascending=[False, True]).head(10).style.hide(axis="index")',
    ),
    (
        "markdown",
        '## D · El punto de operación (D22)\n'
        '\n'
        'De las combinaciones con como mucho 2 falsos positivos, la de mayor recall -empate a recall y fp, gana el F1 más alto-. La tabla sale completa, con `cumple_d22` y `elegido_r05`, para poder contradecir la elección mirando también lo que no ganó.',
    ),
    (
        "code",
        'tabla_operacion = elegir_punto_operacion(barrido, max_fp=MAX_FP)\n'
        '\n'
        'figura_barrido = plot_duplicate_threshold_sweep(\n'
        '    tabla_operacion, max_fp=MAX_FP,\n'
        '    subtitle=f"{len(tabla_operacion)} combinaciones · 7 positivos + 7 negativos de desarrollo",\n'
        ')\n'
        'figura_barrido.show()',
    ),
    (
        "code",
        'ganadora = tabla_operacion[tabla_operacion["elegido_r05"]]\n'
        'if ganadora.empty:\n'
        '    raise RuntimeError(\n'
        '        "Ninguna combinacion del barrido cumple D22 (fp <= "\n'
        '        f"{MAX_FP}): amplia VALORES_UMBRAL_TEXTO_SOLO/VALORES_UMBRAL_TEXTO_CORROBORADO/"\n'
        '        "VALORES_MARGEN_MINIMO."\n'
        '    )\n'
        'fila = ganadora.iloc[0]\n'
        'UMBRAL_TEXTO_SOLO = float(fila["umbral_texto_solo"])\n'
        'MARGEN_MINIMO_BARRIDO = float(fila["margen_minimo"])\n'
        'UMBRAL_TEXTO_CORROBORADO = float(fila["umbral_texto_corroborado"])\n'
        '\n'
        'print("R05 -lo que devuelve el barrido-:")\n'
        'print(f"  umbral_texto_solo        = {UMBRAL_TEXTO_SOLO}")\n'
        'print(f"  margen_minimo (barrido)  = {MARGEN_MINIMO_BARRIDO}")\n'
        'print(f"  umbral_texto_corroborado = {UMBRAL_TEXTO_CORROBORADO}")\n'
        'print(f"  medido   : precision={fila[\'precision\']:.3f} · recall={fila[\'recall\']:.3f} "\n'
        '      f"· f1={fila[\'f1\']:.3f} · fp={int(fila[\'fp\'])}/7")',
    ),
    (
        "markdown",
        '## D.1 · Override manual de `margen_minimo`\n'
        '\n'
        'El barrido devolvió `margen_minimo=0.0` no porque sea el mejor valor, sino porque **empata** con cualquier otro candidato -las 7 altas positivas tienen `marca_coincide=True` y ya superan `umbral_texto_corroborado` por sí solas, así que el Camino 2 las atrapa sin necesitar el margen (ver `aviso_margen_minimo_cero` en `config.yaml`)-. Subir `margen_minimo` no cuesta nada medible aquí, y añade la protección real que el margen fue pensado para dar.\n'
        '\n'
        'El único dato que tenemos para elegir un valor es el margen más alto visto entre los **negativos** (0,0577, `DEV-NEW-001`) -el "ruido" que existe aunque no haya ningún duplicado real-. `margen_minimo=0.1` queda por encima de ese ruido sin exigir tanto como para que el Camino 1 dependa casi siempre del Camino 2 (con 0,2 solo `DEV-DUP-005` lo pasaría por sí sola).',
    ),
    (
        "code",
        'MARGEN_MINIMO = 0.1  # override manual (D21): por encima del ruido de\n'
        '# negativos (max observado 0.0577, DEV-NEW-001), decisión razonada, no\n'
        '# resultado del barrido -que era indiferente a este valor-.\n'
        '\n'
        'print(f"margen_minimo: barrido={MARGEN_MINIMO_BARRIDO} · usado={MARGEN_MINIMO} (override manual)")',
    ),
    (
        "markdown",
        '## E · 🚨 CONGELA — a partir de aquí, `altas_evaluacion.csv`\n'
        '\n'
        'El protocolo es explícito: barrer, elegir, dejarlo registrado, **congelar**, y solo entonces abrir el conjunto de evaluación. `UMBRAL_TEXTO_SOLO`/`MARGEN_MINIMO`/`UMBRAL_TEXTO_CORROBORADO` no se vuelven a tocar de aquí en adelante.',
    ),
    (
        "code",
        'senales_evaluacion = calcular_senales(buscador, altas_evaluacion, etiquetas_col=None)\n'
        '\n'
        'resultados = resultados_duplicados(\n'
        '    senales_evaluacion,\n'
        '    umbral_texto_solo=UMBRAL_TEXTO_SOLO,\n'
        '    margen_minimo=MARGEN_MINIMO,\n'
        '    umbral_texto_corroborado=UMBRAL_TEXTO_CORROBORADO,\n'
        ')\n'
        '\n'
        'destino = Path("..") / "resultados" / "resultados_duplicados.csv"\n'
        'resultados.to_csv(destino, index=False)\n'
        'print(f"Escrito {destino} · {len(resultados)} filas · "\n'
        '      f"{int(resultados[\'predicted_duplicate\'].sum())} duplicados detectados")\n'
        'resultados.style.hide(axis="index")',
    ),
    (
        "markdown",
        '## F · Lo que este notebook no puede afirmar, y por qué\n'
        '\n'
        '| Limitación | Por qué |\n'
        '|---|---|\n'
        '| El Camino 2 (marca/color) no está calibrado empíricamente | Ningún ejemplo de las 14 altas de desarrollo lo activa en solitario -los 7 positivos ya caen en el Camino 1-. `τ_bajo` es una decisión razonada, no medida |\n'
        '| El corroborador de marca no se activa en el 4,4% del catálogo | `brand` vacío en 658/15.000 productos (verificado sobre `catalogo_productos.csv`) |\n'
        '| El corroborador de color no se activa en el 37,4% del catálogo | Mismo motivo, sobre `color` |\n'
        '| La rejilla de umbrales/margen se calibra sobre 14 ejemplos | Cualquier cifra de precisión con más de un decimal es más precisa de lo que 7 negativos pueden sostener -por eso D22 se expresa en cuenta de falsos positivos, no en un umbral de precisión- |',
    ),
]

NB09_EVALUACION = [
    ("markdown", '# NB09 · Evaluación consolidada, atribución de errores y artefactos'),
    (
        "markdown",
        'Objetivo: la tabla que decide, los tres artefactos de entrega, y la atribución de ≥3 fallos. No hay decisiones D en este notebook -las 22 ya están cerradas-; lo que hace falta es reunir lo que NB01-NB08 ya midieron, generar lo único que falta (el top-10 de las 12 consultas de evaluación) y razonar sobre los fallos con evidencia real.\n'
        '\n'
        '### 🗺️ Mapa de datos: de dónde sale cada número de la tabla comparativa\n'
        '\n'
        'Ningún notebook anterior dejó la fila del ANN elegido en un artefacto -NB06 solo la mostró en pantalla-, así que es la única fila que este notebook **recalcula** en vez de leer. Todo lo demás es lectura de artefactos ya escritos.\n'
        '\n'
        '| Fila de la tabla | Viene de | Cómo |\n'
        '|---|---|---|\n'
        '| TF-IDF, BM25 | `artifacts/baseline_lexico.json` (NB01) | lectura directa |\n'
        '| Denso, oráculo exacto (R01, plantilla A4) | `artifacts/comparativa_representacion.json` (NB03) | lectura directa, fila `posicion_regla == 1` |\n'
        '| Denso, modelo ganador sobre la muestra (R02) | `artifacts/comparativa_modelos.json` (NB02) | lectura directa -⚠️ sobre `catalogo_muestra`, no comparable en crudo con el resto- |\n'
        '| **Denso, ANN elegido (R04, `ef=32`)** | recalculado aquí (sección B), latencia de `artifacts/benchmark_ann.csv` | única fila que no viene de un JSON ya escrito |\n'
        '\n'
        'El resto del notebook (`resultados_busqueda.csv`, consistencia entre formulaciones, consultas filtradas, atribución de errores) no depende de esta tabla: son piezas independientes que comparten la misma conexión a Qdrant.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000, desde cache) + consultas_desarrollo.csv (8) + consultas_evaluacion.csv (12) + consultas_filtradas.csv (4)\n'
        'import json\n'
        'import os\n'
        'import sys\n'
        'from functools import lru_cache\n'
        'from pathlib import Path\n'
        '\n'
        'import numpy as np\n'
        'import pandas as pd\n'
        '\n'
        'sys.path.insert(0, str(Path("..") / "src"))\n'
        '\n'
        'from dotenv import load_dotenv\n'
        '\n'
        'from aurum.ann import comparar_ndcg_con_oraculo\n'
        'from aurum.busqueda import BuscadorVectorial, DenseRetriever, auditar_filtro_de_marca, rank_queries_dense\n'
        'from aurum.consolidacion import diagnosticar_consulta, fila_comparativa, tabla_comparativa\n'
        'from aurum.datos import load_csv\n'
        'from aurum.embeddings import (\n'
        '    GeminiEncoder, cache_key, corpus_fingerprint, encode_corpus, truncate_dim,\n'
        ')\n'
        'from aurum.evaluacion import formulation_consistency, qrels_from_judgements\n'
        'from aurum.motores import CATALOG_PREFIX, catalog_collection_name\n'
        'from aurum.motores.qdrant import QdrantStore\n'
        'from aurum.plantillas import render_template\n'
        '\n'
        'load_dotenv(Path("..") / ".env")\n'
        'DATA = Path("..") / "data"\n'
        'CACHE = Path("..") / "artifacts" / "embeddings"\n'
        'completo = load_csv(DATA / "catalogo_productos.csv")\n'
        'desarrollo = load_csv(DATA / "consultas_desarrollo.csv")\n'
        'evaluacion = load_csv(DATA / "consultas_evaluacion.csv")\n'
        'relevancias = load_csv(DATA / "relevancias_desarrollo.csv")\n'
        'filtradas = load_csv(DATA / "consultas_filtradas.csv")\n'
        'QRELS = qrels_from_judgements(relevancias)\n'
        '\n'
        'MODELO, CONTRATO, PLANTILLA = "gemini-embedding-2", "sin_contrato", "A4"\n'
        'DIM, TOP_K = 768, 10\n'
        'COLECCION = catalog_collection_name(model=MODELO, template=PLANTILLA, dim=DIM)\n'
        'EF_ELEGIDO = 32   # R04 (NB06)\n'
        '\n'
        'print(f"coleccion : {COLECCION} · ef={EF_ELEGIDO}")\n'
        'print(f"consultas : {len(desarrollo)} desarrollo · {len(evaluacion)} evaluacion · {len(filtradas)} filtradas")',
    ),
    (
        "markdown",
        '## A · Conexión, oráculo exacto y buscador ANN\n'
        '\n'
        '**Entrada:** las constantes del setup. **Salida:** `oraculo` (`DenseRetriever` sobre los 15.000, exacto -igual que en NB02/NB06-) y `buscador` (`BuscadorVectorial` contra Qdrant con `ef=32`, el mismo R04 que ya usaron NB07 y NB08). Los vectores del catálogo salen de la cache de NB04 -si no están, la celda para en vez de pagar 15.000 llamadas-.',
    ),
    (
        "code",
        '# ⚠️ requiere `make motor-up MOTOR=qdrant`\n'
        'CORPUS_ID = f"catalogo_productos__{PLANTILLA}"\n'
        'textos_completo = render_template(completo, PLANTILLA)\n'
        'clave = cache_key(\n'
        '    model_id=MODELO, kind="document", contract=CONTRATO,\n'
        '    corpus_id=CORPUS_ID, fingerprint=corpus_fingerprint(textos_completo),\n'
        ')\n'
        'if not (CACHE / f"{clave}.npy").exists():\n'
        '    raise RuntimeError(\n'
        '        f"Los vectores de {CORPUS_ID} no estan en cache ({clave}).\\n"\n'
        '        f"Deberian estar desde NB04/NB06 -esta celda no paga 15.000 llamadas nuevas."\n'
        '    )\n'
        '\n'
        '_encoder = GeminiEncoder(\n'
        '    api_key=os.environ.get("GEMINI_API_KEY"), model_id=MODELO,\n'
        '    native_dim=3072, window=8192,\n'
        ')\n'
        'codificado_completo = encode_corpus(\n'
        '    _encoder, textos_completo, corpus_id=CORPUS_ID,\n'
        '    kind="document", contract=CONTRATO, batch_size=32, cache_dir=CACHE,\n'
        ')\n'
        'vectores_completo = truncate_dim(codificado_completo.vectors, DIM)\n'
        'ids_completo = completo["product_id"].tolist()\n'
        'oraculo_retriever = DenseRetriever(vectores_completo, ids_completo, metric="cosine")\n'
        '\n'
        '\n'
        '@lru_cache(maxsize=256)\n'
        'def codificar_consulta(texto: str):\n'
        '    codificado = encode_corpus(\n'
        '        _encoder, [texto], corpus_id="consulta_suelta",\n'
        '        kind="query", contract=CONTRATO, batch_size=1, cache_dir=CACHE,\n'
        '    )\n'
        '    return truncate_dim(codificado.vectors, DIM)[0]\n'
        '\n'
        '\n'
        'almacen = QdrantStore(\n'
        '    collection=COLECCION,\n'
        '    url=os.environ.get("AURUM_QDRANT_URL", "http://localhost:6333"),\n'
        '    api_key=os.environ.get("AURUM_QDRANT_API_KEY"),\n'
        '    prefix=CATALOG_PREFIX,\n'
        '    timeout=30,\n'
        ')\n'
        'buscador = BuscadorVectorial(almacen, codificar_consulta, top_k=TOP_K, ef=EF_ELEGIDO)\n'
        'print(f"puntos en la coleccion: {almacen.count():,}".replace(",", ".") +\n'
        '      f" · indice al dia: {almacen.index_ready()}")',
    ),
    (
        "markdown",
        '## B · La fila que falta: ANN elegido (R04) sobre las 8 de desarrollo\n'
        '\n'
        '**Entrada:** `oraculo_retriever`, `buscador`, `desarrollo`, `QRELS`. **Salida:** `metricas_ann_elegido` -un dict con `ndcg_at_10`/`recall_at_10`/`mrr_at_10`-, usado en la sección C. Mismo cálculo que la sección G de `06_ann.ipynb`, reejecutado porque esa tabla nunca se guardó en un artefacto.',
    ),
    (
        "code",
        'QUERY_IDS_DEV = [str(q) for q in desarrollo["query_id"]]\n'
        'vectores_dev = encode_corpus(\n'
        '    _encoder, desarrollo["query_text"].tolist(), corpus_id="consultas_desarrollo",\n'
        '    kind="query", contract=CONTRATO, batch_size=32, cache_dir=CACHE,\n'
        ').vectors\n'
        'vectores_dev = truncate_dim(vectores_dev, DIM)\n'
        '\n'
        'oraculo_dev = rank_queries_dense(oraculo_retriever, QUERY_IDS_DEV, vectores_dev, k=TOP_K)\n'
        'ann_dev = {\n'
        '    qid: [r.document_id for r in buscador.buscar(texto, top_k=TOP_K)]\n'
        '    for qid, texto in zip(QUERY_IDS_DEV, desarrollo["query_text"])\n'
        '}\n'
        '\n'
        'tabla_ndcg = comparar_ndcg_con_oraculo(ann_dev, oraculo_dev, QRELS, k=TOP_K)\n'
        'metricas_ann_elegido = (\n'
        '    tabla_ndcg[tabla_ndcg["sistema"] == "ANN elegido (R04)"].iloc[0].to_dict()\n'
        ')\n'
        'tabla_ndcg.style.hide(axis="index").format({\n'
        '    columna: "{:.1%}" for columna in tabla_ndcg.columns if columna != "sistema"\n'
        '})',
    ),
    (
        "markdown",
        '## C · La tabla comparativa\n'
        '\n'
        '**Entrada:** los tres artefactos ya escritos (mapa de datos, arriba) + `metricas_ann_elegido` de B + la fila `ef=32` de `benchmark_ann.csv` para la latencia. **Salida:** `tabla_comparativa_final`, escrita como `artifacts/tabla_comparativa.md` en la sección H.\n'
        '\n'
        'La fila del modelo ganador (R02) queda marcada como no comparable en crudo -se decidió sobre `catalogo_muestra` (1.500), no sobre el catálogo completo- para no enseñar cuatro números en la misma columna que en realidad miden corpus distintos.',
    ),
    (
        "code",
        'baseline = json.loads((Path("..") / "artifacts" / "baseline_lexico.json").read_text(encoding="utf-8"))\n'
        'representacion = json.loads((Path("..") / "artifacts" / "comparativa_representacion.json").read_text(encoding="utf-8"))\n'
        'modelos = json.loads((Path("..") / "artifacts" / "comparativa_modelos.json").read_text(encoding="utf-8"))\n'
        'benchmark_ann = pd.read_csv(Path("..") / "artifacts" / "benchmark_ann.csv")\n'
        '\n'
        'def _ganadora(regla, nombre):\n'
        '    # Misma exigencia que consolidar_metricas._ganadora: la posicion 1\n'
        '    # sin "admisible" no es una ganadora, es la mejor de un barrido sin\n'
        '    # ganador -devolverla igual escondería que la regla no eligió nada-.\n'
        '    primera = next(f for f in regla if f["posicion_regla"] == 1)\n'
        '    if not primera.get("admisible"):\n'
        '        raise RuntimeError(f"{nombre} no dejo ninguna configuracion admisible.")\n'
        '    return primera\n'
        '\n'
        '\n'
        'ganadora_r01 = _ganadora(representacion["regla_r01_completo"], "R01")\n'
        'ganadora_r02 = _ganadora(modelos["regla_d09b"], "R02")\n'
        'fila_ann = benchmark_ann[benchmark_ann["ef"] == EF_ELEGIDO].iloc[0]\n'
        '\n'
        'filas = [\n'
        '    fila_comparativa(\n'
        '        "C0a · TF-IDF", modelo="TF-IDF", metrica="coseno TF-IDF", ann="exacto",\n'
        '        **{k: baseline["completo"]["metricas"]["tfidf"][k] for k in\n'
        '           ("ndcg_at_10", "recall_at_10", "mrr_at_10")},\n'
        '    ),\n'
        '    fila_comparativa(\n'
        '        "C0b · BM25", modelo="BM25", metrica="BM25", ann="exacto",\n'
        '        **{k: baseline["completo"]["metricas"]["bm25"][k] for k in\n'
        '           ("ndcg_at_10", "recall_at_10", "mrr_at_10")},\n'
        '    ),\n'
        '    fila_comparativa(\n'
        '        "C1 · denso, muestra (R02)", modelo=ganadora_r02["modelo"], dim=ganadora_r02["dim"],\n'
        '        metrica="coseno", ann="exacto",\n'
        '        ndcg_at_10=ganadora_r02["ndcg_at_10"], recall_at_10=ganadora_r02["recall_at_10"],\n'
        '        mrr_at_10=ganadora_r02["mrr_at_10"],\n'
        '        nota="sobre catalogo_muestra (1.500) -no comparable en crudo con el resto, sobre 15.000-",\n'
        '    ),\n'
        '    fila_comparativa(\n'
        '        "C2 · denso, oraculo exacto (R01)", modelo=MODELO, plantilla=ganadora_r01["plantilla"],\n'
        '        dim=DIM, metrica="coseno", ann="exacto",\n'
        '        ndcg_at_10=ganadora_r01["ndcg_at_10"], recall_at_10=ganadora_r01["recall_at_10"],\n'
        '        mrr_at_10=ganadora_r01["mrr_at_10"],\n'
        '    ),\n'
        '    fila_comparativa(\n'
        '        "C3 · denso, ANN elegido (R04) — el sistema real", modelo=MODELO, plantilla=PLANTILLA,\n'
        '        dim=DIM, metrica="coseno", ann=f"Qdrant HNSW ef={EF_ELEGIDO}",\n'
        '        ndcg_at_10=metricas_ann_elegido["ndcg_at_10"],\n'
        '        recall_at_10=metricas_ann_elegido["recall_at_10"],\n'
        '        mrr_at_10=metricas_ann_elegido["mrr_at_10"],\n'
        '        p50_ms=float(fila_ann["ms_p50"]), p95_ms=float(fila_ann["ms_p95"]),\n'
        '        nota="la distancia con C2 se lee con la nota al pie -el oraculo de C2 y '
        'este indice no comparan el mismo corpus, ver G.1.c-",\n'
        '    ),\n'
        ']\n'
        'tabla_comparativa_final = tabla_comparativa(filas)\n'
        'tabla_comparativa_final.style.hide(axis="index")',
    ),
    (
        "markdown",
        '## D · `resultados_busqueda.csv` — top-10 de las 12 consultas de evaluación\n'
        '\n'
        '**Entrada:** `evaluacion` (setup) + `buscador` (A) — la primera vez que este notebook escribe un CSV de entrega, no solo lee. **Salida:** `resultados/resultados_busqueda.csv`, 120 filas (12 × 10).\n'
        '\n'
        'Van con el `buscador` real -el mismo `ef=32` de producción, no el oráculo-, porque es lo que el sistema entregaría de verdad ante estas 12 consultas.',
    ),
    (
        "code",
        'filas_resultados = [\n'
        '    {\n'
        '        "evaluation_id": evaluation_id,\n'
        '        "rank": resultado.rank,\n'
        '        "product_id": resultado.document_id,\n'
        '        "score": resultado.score,\n'
        '    }\n'
        '    for evaluation_id, texto in zip(evaluacion["evaluation_id"], evaluacion["query_text"])\n'
        '    for resultado in buscador.buscar(texto, top_k=TOP_K)\n'
        ']\n'
        'resultados_busqueda = pd.DataFrame(filas_resultados)\n'
        '\n'
        'destino_busqueda = Path("..") / "resultados" / "resultados_busqueda.csv"\n'
        'resultados_busqueda.to_csv(destino_busqueda, index=False)\n'
        'filas_por_consulta = resultados_busqueda.groupby("evaluation_id").size()\n'
        'print(f"Escrito {destino_busqueda} · {len(resultados_busqueda)} filas\\n"\n'
        '      f"consultas con menos de {TOP_K} resultados: "\n'
        '      f"{(filas_por_consulta < TOP_K).sum()}/{len(filas_por_consulta)}\\n"\n'
        '      f"product_id repetido dentro de alguna consulta: "\n'
        '      f"{(resultados_busqueda.groupby(\'evaluation_id\')[\'product_id\'].nunique() < filas_por_consulta).sum()}")\n'
        'resultados_busqueda.head(10).style.hide(axis="index")',
    ),
    (
        "markdown",
        '## E · Consistencia entre formulaciones (Jaccard, sin etiquetas)\n'
        '\n'
        '**Entrada:** los mismos rankings de D, agrupados por intención. **Salida:** `tabla_consistencia`, 4 filas -una por intención-, al artefacto en H.\n'
        '\n'
        'Sin juicios para las 12 de evaluación, esta es la única evidencia de calidad que no depende de etiquetas: si `direct`/`context`/`semantic` de la misma intención devuelven catálogos parecidos, el sistema entiende la intención y no solo la superficie léxica.',
    ),
    (
        "code",
        'rankings_evaluacion = {\n'
        '    evaluation_id: grupo.sort_values("rank")["product_id"].tolist()\n'
        '    for evaluation_id, grupo in resultados_busqueda.groupby("evaluation_id")\n'
        '}\n'
        'tabla_consistencia = formulation_consistency(rankings_evaluacion, k=TOP_K)\n'
        'tabla_consistencia.style.hide(axis="index")',
    ),
    (
        "markdown",
        '## F · Consultas filtradas: pureza y cobertura\n'
        '\n'
        '**Entrada:** `filtradas` (setup) + `buscador`. **Salida:** `tabla_filtros`, al artefacto en H.\n'
        '\n'
        '`auditar_filtro_de_marca` (NB05) ya resuelve esto: compara contra `alcance` -cuántos productos de esa marca hay realmente en el catálogo- para distinguir un cero legítimo de un filtro roto, y una cobertura corta de una marca con pocos productos.',
    ),
    (
        "code",
        'alcance_por_marca = completo["brand"].value_counts().to_dict()\n'
        'tabla_filtros = auditar_filtro_de_marca(\n'
        '    buscador, filtradas.to_dict("records"), alcance=alcance_por_marca, top_k=TOP_K,\n'
        ')\n'
        '# "veredicto" ya distingue una respuesta vacia legitima (la marca no\n'
        '# tiene productos) de un filtro roto (la marca SI tiene y devolvio 0):\n'
        '# comparar solo de_la_marca == n_resultados confundiria las dos.\n'
        'FILTROS_OK = int(tabla_filtros["veredicto"].str.startswith("\N{WHITE HEAVY CHECK MARK}").sum())\n'
        'print(f"consultas filtradas puras: {FILTROS_OK}/{len(tabla_filtros)}")\n'
        'tabla_filtros.style.hide(axis="index")',
    ),
    (
        "markdown",
        '## G · Atribución de ≥3 fallos\n'
        '\n'
        'Procedimiento, en orden: (1) ¿ya es malo en el oráculo exacto? → **representación**; (2) ¿el oráculo lo recupera y el ANN no? → **índice**; (3) ¿falta el producto o lo excluye el filtro? → **datos/filtros**; (4) ¿el estado leído no coincide con la traza de NB08? → **persistencia**.\n'
        '\n'
        '**Entrada:** `oraculo_dev`/`ann_dev` (B), `QRELS` (setup), `tabla_consistencia` (E). **Salida:** evidencia impresa -no un veredicto-. Las celdas de abajo solo *reúnen* la evidencia de las capas 1 y 2 para las dos candidatas de desarrollo señaladas de antemano (`diagnosticar_consulta`, `src/aurum/consolidacion.py`) y la divergencia Jaccard de la peor intención entre formulaciones; la conclusión -qué capa falló y por qué, con el `product_id` que lo sostiene- se redacta en la celda de markdown de después, mirando estos números. Falta a propósito: precodificarla sin haber visto los datos sería inventar el hallazgo.',
    ),
    (
        "code",
        'def _evidencia_capas_1_2(query_id_int):\n'
        '    qid = str(query_id_int)\n'
        '    texto = desarrollo.loc[desarrollo["query_id"] == query_id_int, "query_text"].iloc[0]\n'
        '    diagnostico = diagnosticar_consulta(\n'
        '        qid, ranking_oraculo=oraculo_dev[qid], ranking_ann=ann_dev[qid],\n'
        '        qrels=QRELS[qid], k=TOP_K,\n'
        '    )\n'
        '    print(f"consulta {qid} \\"{texto}\\"")\n'
        '    print(f"  relevantes en el oraculo (top-{TOP_K}): "\n'
        '          f"{diagnostico[\'n_relevantes_en_oraculo\']} -> {diagnostico[\'relevantes_en_oraculo\']}")\n'
        '    print(f"  relevantes en el ANN (top-{TOP_K})    : "\n'
        '          f"{diagnostico[\'n_relevantes_en_ann\']} -> {diagnostico[\'relevantes_en_ann\']}")\n'
        '    print(f"  relevantes que el ANN perdio          : {diagnostico[\'perdidos_por_el_ann\']}")\n'
        '    return diagnostico\n'
        '\n'
        '\n'
        'diagnostico_13357 = _evidencia_capas_1_2(13357)   # "base tapizada 160x200 sin patas"\n'
        'print()\n'
        'diagnostico_33633 = _evidencia_capas_1_2(33633)   # "disfraz halloween talla grande hombre"',
    ),
    (
        "markdown",
        '### G.1 · Capa 3 (datos/filtros), solo si hace falta\n'
        '\n'
        'Si `perdidos_por_el_ann` de alguna de las dos de arriba está vacío pero `n_relevantes_en_oraculo` también es bajo, antes de concluir "representación" hay que comprobar si el producto relevante ni siquiera está indexado o le falta el metadato que necesitaría un filtro -`indice.get(record_id)` y mirar su payload, mismo patrón que NB08 sección D-.',
    ),
    (
        "markdown",
        '### G.1.b · PRUEBAS POST-RESULTADOS · NO DECISIONES — ¿es `ef` la palanca?\n'
        '\n'
        '**Problema.** Cinco relevantes que el oráculo trae y el ANN (`ef=32`) no. Si fuera pérdida del índice, aflojar la aproximación debería recuperarlos.\n'
        '\n'
        '**Se comprueba.** El mismo top-10 con `ef` ∈ {32, 64, 128, 256} sobre las 8 de desarrollo. Buscador aparte: no toca `buscador`, `ann_dev` ni ningún artefacto — B-F siguen siendo `ef=32`, y R04 no se reabre (la latencia de cada `ef` está en `benchmark_ann.csv`).\n'
        '\n'
        '**Lectura.** `relevantes_top10_*`: más alto es mejor, el oráculo es el techo. `perdidos_*`: más corto es mejor, `-` = ninguno. Se lee en horizontal, por fila.',
    ),
    (
        "code",
        '# PRUEBAS POST-RESULTADOS - NO DECISIONES: R04 (ef=32) no se reabre aqui.\n'
        '# Buscadores aparte a proposito -no reasignan `buscador` ni `ann_dev`-.\n'
        'EFS_PRUEBA = (32, 64, 128, 256)\n'
        '\n'
        'ann_por_ef = {\n'
        '    ef: {\n'
        '        qid: [\n'
        '            r.document_id\n'
        '            for r in BuscadorVectorial(\n'
        '                almacen, codificar_consulta, top_k=TOP_K, ef=ef,\n'
        '            ).buscar(texto, top_k=TOP_K)\n'
        '        ]\n'
        '        for qid, texto in zip(QUERY_IDS_DEV, desarrollo["query_text"])\n'
        '    }\n'
        '    for ef in EFS_PRUEBA\n'
        '}\n'
        '\n'
        '\n'
        'def _fila_barrido(query_id_int):\n'
        '    qid = str(query_id_int)\n'
        '    fila = {"consulta": qid}\n'
        '    for ef in EFS_PRUEBA:\n'
        '        d = diagnosticar_consulta(\n'
        '            qid, ranking_oraculo=oraculo_dev[qid], ranking_ann=ann_por_ef[ef][qid],\n'
        '            qrels=QRELS[qid], k=TOP_K,\n'
        '        )\n'
        '        fila["relevantes_top10_oraculo_exacto"] = d["n_relevantes_en_oraculo"]\n'
        '        fila[f"relevantes_top10_ann_ef{ef}"] = d["n_relevantes_en_ann"]\n'
        '        fila[f"perdidos_ef{ef}"] = ", ".join(d["perdidos_por_el_ann"]) or "-"\n'
        '    return fila\n'
        '\n'
        '\n'
        '# Las 8 de desarrollo, no solo la 13357: una consulta suelta no distingue\n'
        '# "subir ef arregla este fallo" de "subir ef mueve resultados en todas partes".\n'
        'tabla_verificacion_ef = pd.DataFrame(\n'
        '    [_fila_barrido(q) for q in desarrollo["query_id"]]\n'
        ')\n'
        'ann_verificacion = ann_por_ef[64]   # lo usa G.1.c\n'
        'tabla_verificacion_ef.style.hide(axis="index")',
    ),
    (
        "markdown",
        '### G.1.c · PRUEBAS POST-RESULTADOS · NO DECISIONES — ¿qué pierde de verdad el ANN?\n'
        '\n'
        '**Problema.** El oráculo se construye desde `catalogo_productos.csv` (foto previa a NB08) y el índice lleva los 24 eventos aplicados. Ambos tienen 15.000 puntos, pero no los mismos: 8 bajas y 8 altas. Y `AURUM-NEW-008` se titula *"Base tapizada 160 x 200 sin patas"* — literalmente la consulta 13357. Si el ANN lo coloca en su top-10, desplaza a un relevante **sin haberse equivocado**: conoce un producto que el oráculo no puede ver. Con ese sesgo dentro, `perdidos_por_el_ann` no mide pérdida del ANN.\n'
        '\n'
        '**Se comprueba.** Un oráculo exacto sobre el mismo corpus que el índice, sin coste de API: los 16 vectores de los `UPSERT` están cacheados desde NB08. Se replica la semántica de Qdrant, que indexa por `record_id`: −8 bajas, 8 reescrituras, +8 altas = 15.000, que debe cuadrar con `almacen.count()`.\n'
        '\n'
        '⚠️ NB08 codificó los upserts con `text` **crudo**, sin pasar por A4. Para las 8 altas da igual (66-96 caracteres, bajo el corte de 936); **4 de las 8 actualizaciones sí lo superan** —hasta 2.676— y quedaron indexadas sin recortar, a diferencia de los otros 14.996 puntos. El oráculo corregido lo reproduce a propósito: replica lo que el índice contiene, no lo que debería contener.\n'
        '\n'
        '**Lectura.** `lo_pierde_vs_oraculo_post_nb08 = False` es una pérdida que nunca existió. `puesto_en_oraculo_post_nb08` = 11 significa efecto de frontera, no fallo de recuperación.\n'
        '\n'
        '⚠️ `puesto_en_ann_ef_efectivo_200` **no** es la búsqueda de producción: Qdrant exige `hnsw_ef >= limit`, así que pedir 200 resultados sube el `ef` efectivo a 200. La columna dice dónde cae el producto en una búsqueda casi exacta — sirve para separar "el ANN lo entierra" de "el ANN ni lo encuentra", no para leer el comportamiento con `ef=32`.',
    ),
    (
        "code",
        '# PRUEBAS POST-RESULTADOS - NO DECISIONES. Oraculo sobre el corpus POST-NB08,\n'
        '# para que oraculo y ANN comparen por fin el mismo catalogo.\n'
        'from aurum.mutaciones import clasificar_eventos\n'
        'from aurum.plantillas import corpus_context\n'
        '\n'
        'eventos = clasificar_eventos(load_csv(DATA / "eventos_catalogo.csv"))\n'
        'eventos_upsert = eventos[eventos["tipo"] != "baja"].reset_index(drop=True)\n'
        'eventos_baja = eventos[eventos["tipo"] == "baja"].reset_index(drop=True)\n'
        '\n'
        '# Mismo corpus_id, mismo texto y mismo orden que NB08 -> la huella casa y\n'
        '# sale de cache. Si no casara, serian 16 llamadas de pago: se comprueba.\n'
        'textos_upsert = eventos_upsert["text"].tolist()\n'
        'clave_upsert = cache_key(\n'
        '    model_id=MODELO, kind="document", contract=CONTRATO,\n'
        '    corpus_id="eventos_catalogo_upsert", fingerprint=corpus_fingerprint(textos_upsert),\n'
        ')\n'
        'if not (CACHE / f"{clave_upsert}.npy").exists():\n'
        '    raise RuntimeError(\n'
        '        f"Los vectores de eventos_catalogo_upsert no estan en cache ({clave_upsert}).\\n"\n'
        '        f"Deberian estar desde NB08 -esta celda no paga llamadas nuevas."\n'
        '    )\n'
        'vectores_upsert = truncate_dim(\n'
        '    encode_corpus(\n'
        '        _encoder, textos_upsert, corpus_id="eventos_catalogo_upsert",\n'
        '        kind="document", contract=CONTRATO, batch_size=16, cache_dir=CACHE,\n'
        '    ).vectors,\n'
        '    DIM,\n'
        ')\n'
        '\n'
        '# Qdrant indexa por record_id: una baja lo borra y un upsert lo reescribe.\n'
        '# Se conserva de la base todo lo que ningun evento toca, y se anaden los 16.\n'
        'tocados = set(eventos_baja["record_id"]) | set(eventos_upsert["record_id"])\n'
        'conserva = ~completo["record_id"].isin(tocados)\n'
        'vectores_post = np.vstack([vectores_completo[conserva.to_numpy()], vectores_upsert])\n'
        'ids_post = (\n'
        '    completo.loc[conserva, "product_id"].tolist()\n'
        '    + eventos_upsert["product_id"].tolist()\n'
        ')\n'
        'oraculo_post = DenseRetriever(vectores_post, ids_post, metric="cosine")\n'
        'oraculo_dev_post = rank_queries_dense(oraculo_post, QUERY_IDS_DEV, vectores_dev, k=TOP_K)\n'
        '\n'
        'largos_upsert = eventos_upsert["text"].fillna("").str.len()\n'
        'CORTE_A4 = corpus_context(completo).a4_chars\n'
        'print(f"productos en el oraculo corregido : {len(ids_post)}")\n'
        'print(f"puntos en el indice Qdrant        : {almacen.count()}")\n'
        'print(f"  reparto: -{len(eventos_baja)} bajas · "\n'
        '      f"{int((eventos[\'tipo\'] == \'actualizacion\').sum())} reescritas · "\n'
        '      f"+{int((eventos[\'tipo\'] == \'alta\').sum())} altas")\n'
        'print(f"upserts codificados sin A4 que superan el corte de {CORTE_A4}: "\n'
        '      f"{int((largos_upsert > CORTE_A4).sum())} de {len(eventos_upsert)}")',
    ),
    (
        "code",
        '# PRUEBAS POST-RESULTADOS - NO DECISIONES: sigue sin reabrirse R04.\n'
        'PROFUNDIDAD = 200\n'
        '# OJO: Qdrant exige hnsw_ef >= limit, asi que pedir 200 resultados sube el\n'
        '# ef efectivo a 200. Esto NO es el buscador de produccion: es una busqueda\n'
        '# casi exacta, util solo para saber si el producto esta o no esta.\n'
        'buscador_profundo = BuscadorVectorial(\n'
        '    almacen, codificar_consulta, top_k=PROFUNDIDAD, ef=EF_ELEGIDO,\n'
        ')\n'
        '\n'
        '# El oraculo se construye desde el CSV -la foto ANTES de NB08-, el indice\n'
        '# lleva los eventos aplicados: no contienen el mismo corpus.\n'
        'ids_del_oraculo = set(ids_completo)\n'
        '\n'
        'filas_frontera = []\n'
        'for qid, vector, texto in zip(QUERY_IDS_DEV, vectores_dev, desarrollo["query_text"]):\n'
        '    comun = {"qrels": QRELS[qid], "k": TOP_K, "ranking_ann": ann_dev[qid]}\n'
        '    con_pre = diagnosticar_consulta(qid, ranking_oraculo=oraculo_dev[qid], **comun)\n'
        '    con_post = diagnosticar_consulta(qid, ranking_oraculo=oraculo_dev_post[qid], **comun)\n'
        '    if not con_pre["perdidos_por_el_ann"] and not con_post["perdidos_por_el_ann"]:\n'
        '        continue\n'
        '    # El ranking completo del oraculo da el puesto real, no "fuera del top-10".\n'
        '    orden_oraculo = [\n'
        '        r.document_id\n'
        '        for r in oraculo_post.search_vector(vector, k=len(ids_post))\n'
        '    ]\n'
        '    puesto_oraculo = {pid: i + 1 for i, pid in enumerate(orden_oraculo)}\n'
        '    orden_ann = [r.document_id for r in buscador_profundo.buscar(texto, top_k=PROFUNDIDAD)]\n'
        '    # Quien ocupa el sitio: lo que el ANN trae y el oraculo PRE no puede ver.\n'
        '    intrusos = [p for p in ann_dev[qid] if p not in ids_del_oraculo]\n'
        '    for pid in sorted(set(con_pre["perdidos_por_el_ann"]) | set(con_post["perdidos_por_el_ann"])):\n'
        '        filas_frontera.append({\n'
        '            "consulta": qid,\n'
        '            "producto_perdido": pid,\n'
        '            "lo_pierde_vs_oraculo_pre_nb08": pid in con_pre["perdidos_por_el_ann"],\n'
        '            "lo_pierde_vs_oraculo_post_nb08": pid in con_post["perdidos_por_el_ann"],\n'
        '            "puesto_en_oraculo_post_nb08": puesto_oraculo.get(pid, "no esta"),\n'
        '            "puesto_en_ann_ef_efectivo_200": (\n'
        '                orden_ann.index(pid) + 1 if pid in orden_ann else f">{PROFUNDIDAD}"\n'
        '            ),\n'
        '            "altas_de_nb08_en_el_top10_del_ann": ", ".join(intrusos) or "-",\n'
        '        })\n'
        '\n'
        '# Segunda comprobacion: cuanto se separan los dos corpus, y por donde.\n'
        'eventos = load_csv(DATA / "eventos_catalogo.csv")\n'
        'solo_en_el_indice = sorted(\n'
        '    set(eventos.loc[eventos["operation"] == "UPSERT", "product_id"]) - ids_del_oraculo\n'
        ')\n'
        'solo_en_el_oraculo = sorted(\n'
        '    set(eventos.loc[eventos["operation"] == "DELETE", "product_id"]) & ids_del_oraculo\n'
        ')\n'
        'perdidos_unicos = {fila["producto_perdido"] for fila in filas_frontera}\n'
        'perdidos_que_son_baja = perdidos_unicos & set(solo_en_el_oraculo)\n'
        'print(f"altas de NB08 que el indice tiene y el oraculo no: {len(solo_en_el_indice)}")\n'
        'print(f"   {solo_en_el_indice}")\n'
        'print(f"bajas de NB08 que el oraculo aun cree vivas      : {len(solo_en_el_oraculo)}")\n'
        'print(f"   {solo_en_el_oraculo}")\n'
        'print(f"perdidos por el ANN que son una baja de NB08     : "\n'
        '      f"{len(perdidos_que_son_baja)} de {len(perdidos_unicos)}")\n'
        '\n'
        '# Y lo que el confound le cuesta a la fila C3 de la tabla comparativa: el\n'
        '# mismo nDCG del ANN medido contra un oraculo que si ve las altas de NB08.\n'
        'ndcg_pre = comparar_ndcg_con_oraculo(ann_dev, oraculo_dev, QRELS, k=TOP_K)\n'
        'ndcg_post = comparar_ndcg_con_oraculo(ann_dev, oraculo_dev_post, QRELS, k=TOP_K)\n'
        'tabla_dos_oraculos = pd.concat([\n'
        '    ndcg_pre.assign(corpus_del_oraculo="pre-NB08 (el de la seccion B)"),\n'
        '    ndcg_post.assign(corpus_del_oraculo="post-NB08 (mismo que el indice)"),\n'
        '])\n'
        'print()\n'
        'print(tabla_dos_oraculos.to_string(index=False))\n'
        'tabla_frontera = pd.DataFrame(filas_frontera)\n'
        'tabla_frontera.style.hide(axis="index")',
    ),
    (
        "markdown",
        '### G.2 · La formulación `-semantic` con menos consistencia',
    ),
    (
        "code",
        'columnas_semantic = [c for c in tabla_consistencia.columns if "semantic" in c]\n'
        'tabla_consistencia_ordenada = tabla_consistencia.sort_values(columnas_semantic)\n'
        'peor_intencion = str(tabla_consistencia_ordenada.iloc[0]["intencion"])\n'
        '\n'
        'print(f"intencion con menor consistencia semantic-vs-resto: {peor_intencion}")\n'
        'for formulacion in ("direct", "context", "semantic"):\n'
        '    eid = f"EVAL-{peor_intencion}-{formulacion}"\n'
        '    texto = evaluacion.loc[evaluacion["evaluation_id"] == eid, "query_text"].iloc[0]\n'
        '    print(f"  {formulacion:8s} \\"{texto}\\"\\n"\n'
        '          f"           top-5: {rankings_evaluacion[eid][:5]}")\n'
        'tabla_consistencia_ordenada.style.hide(axis="index")',
    ),
    (
        "markdown",
        '### G.2.b · PRUEBAS POST-RESULTADOS · NO DECISIONES — ¿de qué depende la robustez a la paráfrasis?\n'
        '\n'
        '**Problema.** La intención 93437 da `jaccard_direct_semantic = 0,053`: reformular sin las palabras clave cambia casi todo el catálogo devuelto. Es capa 1 por descarte —las tres formulaciones comparten índice, `ef` y corpus—, pero "representación" tiene dos palancas: la **dimensión** (se usan 768 de las 3.072 nativas, y la robustez a paráfrasis es de lo primero que se degrada al truncar) y la **plantilla** (A4 codifica `text` comercial recortado, y R01 la eligió midiendo consultas de tipo `direct`; con paráfrasis nunca se midió).\n'
        '\n'
        '**Se comprueba.** {A4, A3, A0} × {768, 1536, 3072} en búsqueda exacta. Cero llamadas: los vectores de las 7 plantillas y de las 12 consultas ya están en caché a 3.072 dimensiones.\n'
        '\n'
        '⚠️ No comparar con la sección E: aquí no hay ANN ni eventos de NB08, así que `AURUM-NEW-002` no existe. `A4 · 768` es el control interno y la comparación válida es entre filas de esta tabla.\n'
        '\n'
        '**Lectura.** `jaccard_*` va de 0 a 1, más alto es mejor. `jaccard_par_peor_de_las_4` delata si una configuración arregla la media hundiendo otra intención.',
    ),
    (
        "code",
        '# PRUEBAS POST-RESULTADOS - NO DECISIONES: R01 (plantilla A4) y DIM=768 no se\n'
        '# reabren. Todo sale de la cache -si algo faltara, encode_corpus pagaria\n'
        '# 15.000 llamadas, asi que la celda comprueba la clave antes de entrar-.\n'
        'PLANTILLAS_PRUEBA = ("A4", "A3", "A0")\n'
        'DIMS_PRUEBA = (768, 1536, 3072)\n'
        '\n'
        'for nombre in PLANTILLAS_PRUEBA:\n'
        '    corpus_id = f"catalogo_productos__{nombre}"\n'
        '    clave_plantilla = cache_key(\n'
        '        model_id=MODELO, kind="document", contract=CONTRATO, corpus_id=corpus_id,\n'
        '        fingerprint=corpus_fingerprint(render_template(completo, nombre)),\n'
        '    )\n'
        '    if not (CACHE / f"{clave_plantilla}.npy").exists():\n'
        '        raise RuntimeError(\n'
        '            f"{corpus_id} no esta en cache ({clave_plantilla}): esta celda no "\n'
        '            f"paga 15.000 llamadas nuevas. Deberia estar desde NB03."\n'
        '        )\n'
        '\n'
        'vectores_eval_nativos = encode_corpus(\n'
        '    _encoder, evaluacion["query_text"].tolist(), corpus_id="consultas_evaluacion",\n'
        '    kind="query", contract=CONTRATO, batch_size=32, cache_dir=CACHE,\n'
        ').vectors\n'
        'IDS_EVAL = evaluacion["evaluation_id"].tolist()\n'
        'titulo_de = dict(zip(completo["product_id"], completo["title"]))\n'
        '\n'
        'filas_robustez, rankings_por_config = [], {}\n'
        'for nombre in PLANTILLAS_PRUEBA:\n'
        '    docs_nativos = encode_corpus(\n'
        '        _encoder, render_template(completo, nombre),\n'
        '        corpus_id=f"catalogo_productos__{nombre}",\n'
        '        kind="document", contract=CONTRATO, batch_size=32, cache_dir=CACHE,\n'
        '    ).vectors\n'
        '    for dim in DIMS_PRUEBA:\n'
        '        retriever = DenseRetriever(\n'
        '            truncate_dim(docs_nativos, dim), ids_completo, metric="cosine",\n'
        '        )\n'
        '        rankings = rank_queries_dense(\n'
        '            retriever, IDS_EVAL, truncate_dim(vectores_eval_nativos, dim), k=TOP_K,\n'
        '        )\n'
        '        tabla = formulation_consistency(rankings, k=TOP_K)\n'
        '        columnas_j = [c for c in tabla.columns if c.startswith("jaccard_")]\n'
        '        fila_93437 = tabla.loc[tabla["intencion"] == "93437"].iloc[0]\n'
        '        rankings_por_config[(nombre, dim)] = rankings\n'
        '        filas_robustez.append({\n'
        '            "plantilla": nombre,\n'
        '            "dim": dim,\n'
        '            "jaccard_medio_4_intenciones": round(float(tabla[columnas_j].values.mean()), 4),\n'
        '            "jaccard_93437_direct_vs_semantic": float(fila_93437["jaccard_direct_semantic"]),\n'
        '            "jaccard_par_peor_de_las_4": round(float(tabla[columnas_j].values.min()), 4),\n'
        '        })\n'
        '        del retriever\n'
        '    del docs_nativos\n'
        '\n'
        'tabla_robustez = pd.DataFrame(filas_robustez).sort_values(\n'
        '    "jaccard_93437_direct_vs_semantic", ascending=False\n'
        ')\n'
        '\n'
        '# Lo que la metrica no ensena: QUE productos devuelve la formulacion rota.\n'
        'CONSULTA_ROTA = "EVAL-93437-semantic"\n'
        'print(f\'{CONSULTA_ROTA}: "\'\n'
        '      f\'{evaluacion.loc[evaluacion["evaluation_id"] == CONSULTA_ROTA, "query_text"].iloc[0]}"\')\n'
        'for config in ((PLANTILLA, DIM), (PLANTILLA, 3072), ("A3", 3072), ("A0", 3072)):\n'
        '    print(f"  {config[0]} · {config[1]}d")\n'
        '    for pid in rankings_por_config[config][CONSULTA_ROTA][:5]:\n'
        '        print(f"      {pid} · {str(titulo_de.get(pid, \'?\'))[:58]}")\n'
        'tabla_robustez.style.hide(axis="index")',
    ),
    (
        "markdown",
        '### G.3 · Conclusión\n'
        '\n'
        '#### Antes de atribuir: una corrección de la propia medición\n'
        '\n'
        'Al verificar los fallos se detectó que **el oráculo y el índice no contenían el mismo catálogo**. NB08 aplicó 24 eventos sobre Qdrant (8 bajas, 8 actualizaciones, 8 altas), pero el oráculo se seguía construyendo desde `catalogo_productos.csv`, la foto anterior. Ambos suman 15.000 puntos y no son los mismos. Como varias altas responden literalmente a consultas de desarrollo —`AURUM-NEW-008` se titula *"Base tapizada 160 x 200 sin patas"*, que es la consulta 13357—, el ANN las devolvía y `perdidos_por_el_ann` lo apuntaba como pérdida suya.\n'
        '\n'
        'No es una decisión de diseño ni de negocio: es un **defecto de la medición**, detectado al comprobar por qué subir `ef` no recuperaba nada. Se recalculó el oráculo sobre el corpus post-NB08 para equipararlo al índice (G.1.c) y se repitió la comparación, que lo corroboró: la brecha oráculo→ANN cae de **0,1219 a 0,0475 de nDCG@10**. El 61 % de la pérdida atribuida al ANN no existía.\n'
        '\n'
        'La cifra corregida se valida sola: **0,0475 es exactamente la brecha que midió NB06** (`config.yaml` → `nb06_ann.r04_ef_elegido.ndcg_vs_oraculo`: 60,06 % del oráculo frente a 55,31 % de R04, −4,75 puntos), cuando aún no existía ninguna mutación. Las altas de NB08 hunden por igual al oráculo y al ANN —no tienen juicio y por D04 puntúan 0—, así que la brecha se conserva. Dos mediciones independientes, separadas por tres notebooks y 24 eventos de escritura, dan el mismo 4,75: eso es lo que confirma que 0,1219 era el artefacto.\n'
        '\n'
        '#### Los tres fallos, uno por capa\n'
        '\n'
        '| Consulta | Capa | Motivo | Evidencia |\n'
        '|---|---|---|---|\n'
        '| **18868** "botines marrones mujer tacon medio" | **Índice** | densidad de la región (causa ya establecida en NB06): explorando solo 32 candidatos, el recorrido HNSW los consume en vecinos próximos pero no óptimos | `B07H97VGBP` es el **vecino nº 1** del oráculo corregido y no aparece en el top-10 con `ef=32` ni `ef=64` (0 de 3 relevantes). Con `ef=128` vuelven los tres (3 de 3) |\n'
        '| **93437** "sillas oficina ergonómicas" | **Representación** | pesa la **plantilla**, no la dimensión: A4 codifica el `text` comercial, saturado de palabras clave | Jaccard `direct`-vs-`semantic`: **0,333** (A4·768) → **0,818** (A3·3072). Con A4 el top-5 de la formulación parafraseada trae una silla de ducha y *"¡Como en casa!"*; con A3·3072 los cinco son sillas de oficina |\n'
        '| **33633** "disfraz halloween talla grande hombre" | **Datos** | *pooling bias*: el juicio no cubre el catálogo | 22 productos con "disfraz"+"hombre" en el catálogo, **0** en el pool de 16 juzgados. El único `E` es una blusa de mujer (`B07GSVQG2R`) |\n'
        '\n'
        '#### Descartadas tras la corrección: 13357 y 43240\n'
        '\n'
        'El ANN sitúa `B00YMSZDZS` y `B08MQ42Z6P` en el **puesto 11**, exactamente donde los pone el oráculo corregido, y los pierde igual con `ef` 32, 64, 128 y 256 — si fuera aproximación, `ef=256` los recuperaría. Lo que ocupó su asiento fue un producto insertado por NB08 (`AURUM-NEW-008` y `AURUM-NEW-005`). No son fallos de ninguna capa.\n'
        '\n'
        '#### Mejora medida, no aplicada\n'
        '\n'
        '`ef=128` recupera los 3 relevantes de la 18868 y su p95 —**12,36 ms** según `benchmark_ann.csv`— cabe dentro del presupuesto de 20 ms que fijó D16. **R04 se mantiene en `ef=32`**: la decisión se tomó antes de ver la curva y así se deja.\n'
        '\n'
        'Conviene precisar el crédito: **NB06 ya sabía esto**. `nb06_ann.r04_ef_elegido.caso_duro_18868` documenta que esta consulta tiene recall 0 con `ef=32` y se recupera entera con `ef=128`, y R04 eligió `ef=32` sabiéndolo, por ser la de menor p95 entre las cuatro que cumplían D16. Lo que aporta NB09 es repetir el barrido sobre el **índice ya mutado** —el estado real del sistema entregado— y confirmar que el diagnóstico se mantiene tras los 24 eventos. Lo que queda anotado para una iteración futura no es "subir `ef`" sin más, sino **cambiar el criterio de desempate** de R04: menor p95 entre las admisibles premia a una configuración que sacrifica una consulta entera; un criterio que mirase el recall mínimo por consulta, y no solo el agregado, habría elegido `ef=128` sin salirse de D16.\n'
        '\n'
        'Lo mismo aplica a A3 frente a A4: R01 eligió midiendo nDCG@10 sobre consultas de tipo `direct`, y la robustez a la paráfrasis no entró en esa medición.\n'
        '\n'
        '#### Defecto colateral detectado\n'
        '\n'
        'NB08 codificó los 16 `UPSERT` con `text` **crudo**, sin pasar por la plantilla A4. Irrelevante para las 8 altas (66-96 caracteres, por debajo del corte de 936), pero **4 de las 8 actualizaciones lo superan** —hasta 2.676— y quedaron indexadas sin recortar frente a los otros 14.996 puntos de la colección.',
    ),
    (
        "markdown",
        '## H · Los artefactos',
    ),
    (
        "code",
        'destino_tabla = Path("..") / "artifacts" / "tabla_comparativa.md"\n'
        '\n'
        '# La brecha C2->C3 leida en crudo sobrevalora la perdida del ANN: el oraculo\n'
        '# de C2 no comparte corpus con el indice que mide C3 (G.1.c). La tabla no se\n'
        '# toca -son las cifras que midio cada notebook-, se le anade el contexto.\n'
        'if "tabla_dos_oraculos" not in globals():\n'
        '    raise RuntimeError(\n'
        '        "Ejecuta G.1.c antes que H: la nota al pie de C3 sale de ahi."\n'
        '    )\n'
        '\n'
        '\n'
        'def _ndcg_de(sistema, corpus):\n'
        '    fila = tabla_dos_oraculos[\n'
        '        (tabla_dos_oraculos["sistema"] == sistema)\n'
        '        & (tabla_dos_oraculos["corpus_del_oraculo"].str.startswith(corpus))\n'
        '    ]\n'
        '    return float(fila.iloc[0]["ndcg_at_10"])\n'
        '\n'
        '\n'
        'BRECHA_PRE = _ndcg_de("oráculo exacto (DenseRetriever)", "pre") - _ndcg_de(\n'
        '    "ANN elegido (R04)", "pre"\n'
        ')\n'
        'BRECHA_POST = _ndcg_de("oráculo exacto (DenseRetriever)", "post") - _ndcg_de(\n'
        '    "ANN elegido (R04)", "post"\n'
        ')\n'
        'NOTA_C3 = (\n'
        '    "\\n\\n> **Nota sobre la fila C3.** La distancia C2→C3 leída en crudo "\n'
        '    f"({BRECHA_PRE:.4f} de nDCG@10) **sobrevalora la pérdida del ANN**. El "\n'
        '    "oráculo de C2 se construye sobre `catalogo_productos.csv` —el estado "\n'
        '    "previo a NB08—, mientras que el índice que mide C3 lleva aplicados los 24 "\n'
        '    "eventos del ciclo de vida: 8 bajas y 8 altas de diferencia. Varias de esas "\n'
        '    "altas responden literalmente a consultas de desarrollo (`AURUM-NEW-008` = "\n'
        '    "\\"Base tapizada 160 x 200 sin patas\\" para la consulta 13357), así que el "\n'
        '    "ANN las devuelve y no recibe crédito por ellas —no tienen juicio en los "\n'
        '    "qrels, y por D04 puntúan 0— mientras desplazan a relevantes juzgados. "\n'
        '    f"Recalculado el oráculo sobre el mismo corpus del índice, la brecha real "\n'
        '    f"es **{BRECHA_POST:.4f}**: el {1 - BRECHA_POST / BRECHA_PRE:.0%} de la "\n'
        '    "pérdida aparente era diferencia de corpus, no aproximación ANN. "\n'
        '    "Evidencia en las secciones G.1.c y G.3."\n'
        ')\n'
        'destino_tabla.write_text(\n'
        '    tabla_comparativa_final.to_markdown(index=False) + NOTA_C3, encoding="utf-8"\n'
        ')\n'
        '\n'
        'print(f"Escrito {destino_tabla} (con nota al pie de C3: brecha "\n'
        '      f"{BRECHA_PRE:.4f} -> {BRECHA_POST:.4f})")\n'
        'print(f"Escrito {destino_busqueda} · {len(resultados_busqueda)} filas (seccion D)")\n'
        'print(f"Consistencia entre formulaciones: {len(tabla_consistencia)} intenciones (seccion E)")\n'
        'print(f"Consultas filtradas puras: {FILTROS_OK}/{len(tabla_filtros)} (seccion F)")',
    ),
]

NB10_ENTREGA = [
    (
        "markdown",
        '# NB10 · El recorrido de entrega, de principio a fin\n'
        '\n'
        'El sistema completo con la configuración ya decidida: **preparar → ingerir → consultar → mutar → volver a consultar → evaluar → limpiar**. No se decide nada aquí ni se compara nada nuevo; es el recorrido que debe poder ejecutarse en un entorno limpio siguiendo el README.\n'
        '\n'
        '**Antes de empezar:** `make motor-up MOTOR=qdrant`.\n'
        '\n'
        'Trabaja sobre su **propia colección** (sufijo `__e2e`), así que arranca siempre desde cero y no toca el índice de NB04-NB09. La sección F la borra.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000) · consultas_evaluacion.csv (12)\n'
        '#            · consultas_filtradas.csv (4) · eventos_catalogo.csv (24)\n'
        '#            · control: consultas_desarrollo.csv (8) + relevancias_desarrollo.csv\n'
        'import os\n'
        'import sys\n'
        'from functools import lru_cache\n'
        'from pathlib import Path\n'
        '\n'
        'import pandas as pd\n'
        '\n'
        'sys.path.insert(0, str(Path("..") / "src"))\n'
        '\n'
        'from dotenv import load_dotenv\n'
        '\n'
        'from aurum.almacen import PAYLOAD_SCHEMAS, add_normalized_key, build_payload\n'
        'from aurum.busqueda import BuscadorVectorial, DenseRetriever, auditar_filtro_de_marca\n'
        'from aurum.datos import load_csv\n'
        'from aurum.embeddings import (\n'
        '    GeminiEncoder, cache_key, corpus_fingerprint, encode_corpus, truncate_dim,\n'
        ')\n'
        'from aurum.evaluacion import evaluate_rankings, qrels_from_judgements\n'
        'from aurum.motores import CATALOG_PREFIX, catalog_collection_name\n'
        'from aurum.motores.aceptacion import self_retrieval_canaries\n'
        'from aurum.motores.base import Point\n'
        'from aurum.motores.qdrant import QdrantStore\n'
        'from aurum.mutaciones import (\n'
        '    aplicar_secuencia, clasificar_eventos, esperar_visibilidad, verificar_evento,\n'
        ')\n'
        'from aurum.plantillas import render_template\n'
        '\n'
        'load_dotenv(Path("..") / ".env")\n'
        'DATA = Path("..") / "data"\n'
        'CACHE = Path("..") / "artifacts" / "embeddings"\n'
        '\n'
        '# La configuracion ya decidida -config/config.yaml-, no se toca nada aqui.\n'
        'MODELO, CONTRATO, PLANTILLA = "gemini-embedding-2", "sin_contrato", "A4"  # R02 · R01\n'
        'DIM, TOP_K = 768, 10                  # D09b\n'
        'EF = 32                               # R04\n'
        'LOTE = 128                            # D15\n'
        'ESQUEMA = "completo"                  # D13\n'
        'POLITICA_NULOS = "cadena_vacia"       # D14\n'
        'NORMALIZACION = "unaccent"            # D03\n'
        'HNSW_M, HNSW_EF_CONSTRUCT = 16, 100   # D17\n'
        'CAMPOS_FILTRABLES = ["brand", "color"]\n'
        '\n'
        '# Coleccion propia: el recorrido se ejecuta entero sin tocar el indice de\n'
        '# NB04-NB09, del que dependen los artefactos ya escritos.\n'
        'COLECCION = catalog_collection_name(\n'
        '    model=MODELO, template=PLANTILLA, dim=DIM,\n'
        ') + "__e2e"\n'
        '\n'
        'completo = load_csv(DATA / "catalogo_productos.csv")\n'
        'evaluacion = load_csv(DATA / "consultas_evaluacion.csv")\n'
        'filtradas = load_csv(DATA / "consultas_filtradas.csv")\n'
        'eventos = clasificar_eventos(load_csv(DATA / "eventos_catalogo.csv"))\n'
        '\n'
        '# Solo para la celda de control (seccion B): son las que calibraron el\n'
        '# sistema, asi que no pueden ser la ejecucion principal.\n'
        'desarrollo = load_csv(DATA / "consultas_desarrollo.csv")\n'
        'QRELS = qrels_from_judgements(load_csv(DATA / "relevancias_desarrollo.csv"))\n'
        '\n'
        '# Consultas escritas para esta demostracion: no salen de ningun CSV ni\n'
        '# han intervenido en ninguna decision. Una por registro linguistico.\n'
        'CONSULTAS_PROPIAS = {\n'
        '    "PROPIA-1-literal": "auriculares inalambricos con cancelacion de ruido",\n'
        '    "PROPIA-2-necesidad": "algo para tapar la ventana y que no entre luz por la manana",\n'
        '    "PROPIA-3-vaga": "un regalo original para el amigo invisible",\n'
        '}\n'
        '\n'
        'print(f"coleccion : {COLECCION}")\n'
        'print(f"catalogo  : {len(completo)} productos")\n'
        'print(f"consultas : {len(evaluacion)} de evaluacion · {len(filtradas)} filtradas "\n'
        '      f"· {len(CONSULTAS_PROPIAS)} propias")\n'
        'print(f"control   : {len(desarrollo)} de desarrollo (seccion B)")\n'
        'print(eventos["tipo"].value_counts().to_string())',
    ),
    (
        "markdown",
        '## A · Ingerir el catálogo\n'
        '\n'
        'Codifica los 15.000 productos de `catalogo_productos.csv` y los ingiere en Qdrant. La ingesta se lanza **dos veces** con los mismos puntos: el recuento debe ser el mismo.\n'
        '\n'
        'Después espera —reintentando y contando el tiempo— a que el índice HNSW refleje la ingesta, antes de dejar que nadie consulte.\n'
        '\n'
        'Los vectores salen de la caché; si no están, la celda para en vez de pagar 15.000 llamadas.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv (15.000) · ⚠️ requiere `make motor-up MOTOR=qdrant`\n'
        'CORPUS_ID = f"catalogo_productos__{PLANTILLA}"\n'
        'textos = render_template(completo, PLANTILLA)\n'
        'clave = cache_key(\n'
        '    model_id=MODELO, kind="document", contract=CONTRATO,\n'
        '    corpus_id=CORPUS_ID, fingerprint=corpus_fingerprint(textos),\n'
        ')\n'
        'if not (CACHE / f"{clave}.npy").exists():\n'
        '    raise RuntimeError(\n'
        '        f"Los vectores de {CORPUS_ID} no estan en cache ({clave}).\\n"\n'
        '        f"Esta celda no paga 15.000 llamadas nuevas."\n'
        '    )\n'
        '\n'
        'encoder = GeminiEncoder(\n'
        '    api_key=os.environ.get("GEMINI_API_KEY"), model_id=MODELO,\n'
        '    native_dim=3072, window=8192,\n'
        ')\n'
        'vectores = truncate_dim(\n'
        '    encode_corpus(\n'
        '        encoder, textos, corpus_id=CORPUS_ID, kind="document",\n'
        '        contract=CONTRATO, batch_size=32, cache_dir=CACHE,\n'
        '    ).vectors,\n'
        '    DIM,\n'
        ')\n'
        '\n'
        'con_claves = completo.copy()\n'
        'for campo in CAMPOS_FILTRABLES:\n'
        '    con_claves = add_normalized_key(con_claves, field=campo, mode=NORMALIZACION)\n'
        '\n'
        'puntos = [\n'
        '    Point(\n'
        '        record_id=fila["record_id"],\n'
        '        vector=vectores[i],\n'
        '        payload=build_payload(\n'
        '            fila, fields=PAYLOAD_SCHEMAS[ESQUEMA], null_policy=POLITICA_NULOS\n'
        '        ),\n'
        '    )\n'
        '    for i, fila in enumerate(con_claves.to_dict("records"))\n'
        ']\n'
        '\n'
        'indice = QdrantStore(\n'
        '    collection=COLECCION,\n'
        '    url=os.environ.get("AURUM_QDRANT_URL", "http://localhost:6333"),\n'
        '    api_key=os.environ.get("AURUM_QDRANT_API_KEY"),\n'
        '    prefix=CATALOG_PREFIX,\n'
        '    timeout=60,\n'
        ')\n'
        'indice.create_collection(\n'
        '    dim=DIM, metric="cosine",\n'
        '    hnsw_m=HNSW_M, hnsw_ef_construct=HNSW_EF_CONSTRUCT,\n'
        ')\n'
        '\n'
        'indice.upsert(puntos, batch_size=LOTE)\n'
        'RECUENTO_1 = indice.count()\n'
        'indice.upsert(puntos, batch_size=LOTE)          # la misma ingesta, otra vez\n'
        'RECUENTO_2 = indice.count()\n'
        '\n'
        '# D18: espera activa, no un sleep a ciegas. Y no basta con mirar el estado:\n'
        '# `upsert(wait=True)` garantiza que el punto esta escrito, no que la\n'
        '# coleccion entera este servible. Si aqui se deja pasar una ingesta a\n'
        '# medias, las secciones de abajo miden sobre un catalogo incompleto y el\n'
        '# sintoma es sutil -devuelven algo, solo que peor y con menos similitud-.\n'
        '# Por eso la condicion son tres cosas a la vez.\n'
        'canarios = self_retrieval_canaries(puntos, n=3)\n'
        'ESPERADOS = len(puntos)\n'
        '\n'
        '\n'
        'def indice_servible():\n'
        '    """1) estan todos los puntos, 2) Qdrant dice verde y 3) responde.\n'
        '\n'
        '    El canario se busca con su PROPIO vector, asi que la respuesta\n'
        '    correcta no depende de la calidad del modelo: debe volver el, el\n'
        '    primero. Si vuelve otro, lo roto es el indice."""\n'
        '    if indice.count() != ESPERADOS or not indice.index_ready():\n'
        '        return False\n'
        '    return all(\n'
        '        (lambda hits: bool(hits) and hits[0].record_id == punto.record_id)(\n'
        '            indice.search(punto.vector, top_k=1, ef=EF)\n'
        '        )\n'
        '        for punto in canarios\n'
        '    )\n'
        '\n'
        '\n'
        'indexado = esperar_visibilidad(indice_servible, timeout_s=180.0, intervalo_s=1.0)\n'
        '\n'
        'if not indexado["visible"]:\n'
        '    raise RuntimeError(\n'
        '        f"El indice no quedo servible en {indexado[\'segundos\']:.0f} s: "\n'
        '        f"{indice.count()} de {ESPERADOS} puntos · verde={indice.index_ready()}.\\n"\n'
        '        f"Seguir mediria sobre un catalogo incompleto."\n'
        '    )\n'
        '\n'
        'tabla_ingesta = pd.DataFrame([\n'
        '    {"comprobacion": "puntos tras la 1a ingesta", "valor": RECUENTO_1},\n'
        '    {"comprobacion": "puntos tras la 2a ingesta", "valor": RECUENTO_2},\n'
        '    {"comprobacion": "puntos esperados", "valor": ESPERADOS},\n'
        '    {"comprobacion": "segundos hasta servible", "valor": round(indexado["segundos"], 1)},\n'
        '    {"comprobacion": "comprobaciones hasta servible", "valor": indexado["intentos"]},\n'
        '])\n'
        'tabla_ingesta.style.hide(axis="index")',
    ),
    (
        "markdown",
        '## B · Control: ¿responde el motor?\n'
        '\n'
        'Comprobación previa, no la ejecución principal. Lanza las 8 consultas de `consultas_desarrollo.csv` —las únicas con juicios de relevancia— y las 4 de `consultas_filtradas.csv`, que llevan filtro de marca.\n'
        '\n'
        'Son las consultas con las que se calibró el sistema, así que sus métricas confirman que el índice está bien construido, no que el sistema sea bueno.',
    ),
    (
        "code",
        '# 📄 DATOS · consultas_desarrollo.csv (8) + consultas_filtradas.csv (4)\n'
        '@lru_cache(maxsize=256)\n'
        'def codificar_consulta(texto: str):\n'
        '    codificado = encode_corpus(\n'
        '        encoder, [texto], corpus_id="consulta_suelta", kind="query",\n'
        '        contract=CONTRATO, batch_size=1, cache_dir=CACHE,\n'
        '    )\n'
        '    return truncate_dim(codificado.vectors, DIM)[0]\n'
        '\n'
        '\n'
        'buscador = BuscadorVectorial(indice, codificar_consulta, top_k=TOP_K, ef=EF)\n'
        '\n'
        'control = {\n'
        '    str(qid): [r.document_id for r in buscador.buscar(texto, top_k=TOP_K)]\n'
        '    for qid, texto in zip(desarrollo["query_id"], desarrollo["query_text"])\n'
        '}\n'
        'METRICAS_CONTROL = evaluate_rankings(control, QRELS, k=TOP_K).summary\n'
        '\n'
        'alcance_por_marca = completo["brand"].value_counts().to_dict()\n'
        'tabla_filtros = auditar_filtro_de_marca(\n'
        '    buscador, filtradas.to_dict("records"), alcance=alcance_por_marca, top_k=TOP_K,\n'
        ')\n'
        'FILTROS_OK = int(\n'
        '    tabla_filtros["veredicto"].str.startswith("\N{WHITE HEAVY CHECK MARK}").sum()\n'
        ')\n'
        'print(f"consultas filtradas puras: {FILTROS_OK}/{len(tabla_filtros)}")\n'
        'display(\n'
        '    pd.DataFrame([METRICAS_CONTROL])\n'
        '    .style.hide(axis="index")\n'
        '    .format("{:.4f}")\n'
        ')\n'
        'tabla_filtros.style.hide(axis="index")',
    ),
    (
        "markdown",
        '## C · La ejecución real: consultas de evaluación\n'
        '\n'
        'Las 12 de `consultas_evaluacion.csv` —4 intenciones escritas de tres formas cada una— más 3 consultas escritas para esta demostración, que no salen de ningún fichero.\n'
        '\n'
        'Ninguna tiene juicios de relevancia, así que aquí no hay nDCG: se mira lo que devuelve y, en la sección F, cuánto se mueve al cambiar el catálogo.\n'
        '\n'
        'Este es el camino **sin modificaciones**.',
    ),
    (
        "code",
        '# 📄 DATOS · consultas_evaluacion.csv (12) + 3 consultas propias\n'
        'CONSULTAS = {\n'
        '    **dict(zip(evaluacion["evaluation_id"], evaluacion["query_text"])),\n'
        '    **CONSULTAS_PROPIAS,\n'
        '}\n'
        '\n'
        'RESULTADOS_ANTES = {\n'
        '    cid: buscador.buscar(texto, top_k=TOP_K) for cid, texto in CONSULTAS.items()\n'
        '}\n'
        'ANTES = {cid: [r.document_id for r in res] for cid, res in RESULTADOS_ANTES.items()}\n'
        '\n'
        '\n'
        'def mostrar(resultados, consulta, *, etiqueta="", n=3, campo=None):\n'
        '    """Lo que se envia y lo que se recibe, uno encima del otro.\n'
        '\n'
        '    Sin juicios de relevancia no hay metrica que calcular aqui: lo unico\n'
        '    que se puede hacer es leer los titulos y ver si tienen sentido."""\n'
        '    print(f"┌─ {etiqueta}" if etiqueta else "┌─")\n'
        '    print(f\'│  ENVIO   "{consulta}"\')\n'
        '    print("│  RECIBO")\n'
        '    if not resultados:\n'
        '        print("│    (sin resultados)")\n'
        '    for r in resultados[:n]:\n'
        '        extra = f" · {campo}={r.metadatos.get(campo, \'\')!r}" if campo else ""\n'
        '        print(f"│    {r.rank}. [{r.score:.3f}] {r.titulo[:58]}")\n'
        '        print(f"│       {r.document_id}{extra}")\n'
        '    print("└─\\n")\n'
        '\n'
        '\n'
        '# Las tres propias: tres registros distintos de la misma clase de cliente.\n'
        'for cid, texto in CONSULTAS_PROPIAS.items():\n'
        '    mostrar(RESULTADOS_ANTES[cid], texto, etiqueta=cid)\n'
        '\n'
        '# Y una intencion del CSV en sus tres formulaciones: misma necesidad,\n'
        '# tres maneras de escribirla.\n'
        'INTENCION_MUESTRA = str(evaluacion["evaluation_id"].iloc[0]).split("-")[1]\n'
        'for cid in [c for c in evaluacion["evaluation_id"] if INTENCION_MUESTRA in c]:\n'
        '    mostrar(RESULTADOS_ANTES[cid], CONSULTAS[cid], etiqueta=cid)\n'
        '\n'
        'print(f"consultas lanzadas: {len(ANTES)} · "\n'
        '      f"resultados por consulta: {min(len(v) for v in ANTES.values())}"\n'
        '      f"-{max(len(v) for v in ANTES.values())}")',
    ),
    (
        "markdown",
        '### C.1 · El filtro por color\n'
        '\n'
        'La misma consulta, sin filtro y filtrando por color, sobre el catálogo recién ingerido. El filtro lo ejecuta Qdrant contra el índice de texto de `color_normalized`; no se descarta nada en Python.\n'
        '\n'
        'Va por el almacén, no por `buscar()`: NB05 decidió no exponer el color en la interfaz pública, así que esto enseña la capacidad del índice sin abrir una puerta nueva.\n'
        '\n'
        'El filtro casa **palabras** y no subcadenas: `rosa` no arrastra `rosado`. Fue una de las razones de elegir este motor.',
    ),
    (
        "code",
        '# 📄 DATOS · 📚 catalogo_productos.csv · el color va en el payload (D13/D14)\n'
        'from aurum.datos import normalize_brand\n'
        'from aurum.motores.base import FilterCondition\n'
        '\n'
        'CONSULTA_COLOR = "vestido de fiesta para una boda"\n'
        'COLORES = ["negro", "rojo", "azul", "rosa"]\n'
        'vector_color = codificar_consulta(CONSULTA_COLOR)\n'
        '\n'
        '\n'
        'def buscar_por_color(color=None, k=TOP_K):\n'
        '    """`contains` sobre color_normalized. D03: se filtra por la clave\n'
        '    derivada, y el valor pedido se normaliza igual que el almacenado.\n'
        '\n'
        '    Va por el almacen y no por `buscador.buscar()` a proposito: NB05\n'
        '    decidio no exponer el color en la interfaz publica, asi que esto es\n'
        '    la capacidad del indice, no una puerta de entrada nueva."""\n'
        '    filtros = ()\n'
        '    if color is not None:\n'
        '        pedido = normalize_brand(color, NORMALIZACION)\n'
        '        # D14: con los nulos como cadena vacia, un `contains ""` casa con\n'
        '        # TODO y el filtro deja de filtrar sin avisar. Un color en blanco\n'
        '        # es entrada invalida, no un filtro; para no filtrar, color=None.\n'
        '        if not pedido:\n'
        '            raise ValueError(\n'
        '                f"{color!r} se normaliza a vacio: seria un filtro que no filtra. "\n'
        '                f"Para buscar sin filtrar, pasa color=None."\n'
        '            )\n'
        '        filtros = (FilterCondition(\n'
        '            field="color", value=pedido, operator="contains",\n'
        '        ),)\n'
        '    return indice.search(vector_color, top_k=k, filters=filtros, ef=EF)\n'
        '\n'
        '\n'
        'def como_resultados(hits):\n'
        '    """Los SearchHit del motor, con la forma que imprime `mostrar`."""\n'
        '    from aurum.busqueda import Resultado\n'
        '    return [\n'
        '        Resultado(\n'
        '            document_id=str(h.payload.get("product_id", "")),\n'
        '            rank=h.rank, score=h.score,\n'
        '            record_id=h.record_id,\n'
        '            titulo=str(h.payload.get("title", "")),\n'
        '            metadatos=h.payload,\n'
        '        )\n'
        '        for h in hits\n'
        '    ]\n'
        '\n'
        '\n'
        'mostrar(como_resultados(buscar_por_color()), CONSULTA_COLOR,\n'
        '        etiqueta="sin filtro", campo="color")\n'
        'for color in COLORES:\n'
        '    mostrar(como_resultados(buscar_por_color(color)), CONSULTA_COLOR,\n'
        '            etiqueta=f"filtrando color={color!r}", campo="color")\n'
        '\n'
        '# Pureza: de lo devuelto, cuanto lleva de verdad el color pedido.\n'
        '# Se tokeniza con \\w+ y no con .split(): el indice de texto de Qdrant usa\n'
        '# el tokenizador WORD, que parte tambien por `/`, `-` y `.` -asi\n'
        '# "blanco/rosa." es rosa para el motor-. Con .split() saldrian coladas\n'
        '# que no lo son.\n'
        'import re\n'
        '\n'
        'colores_catalogo = con_claves["color_normalized"].fillna("")\n'
        '\n'
        '\n'
        'def lleva_el_color(texto, pedido):\n'
        '    return pedido in re.findall(r"\\w+", str(texto))\n'
        '\n'
        '\n'
        'filas_color = []\n'
        'for color in COLORES:\n'
        '    pedido = normalize_brand(color, NORMALIZACION)\n'
        '    hits = buscar_por_color(color)\n'
        '    filas_color.append({\n'
        '        "color_pedido": color,\n'
        '        "en_el_catalogo": int(\n'
        '            colores_catalogo.apply(lleva_el_color, pedido=pedido).sum()\n'
        '        ),\n'
        '        "devueltos": len(hits),\n'
        '        "de_ese_color": sum(\n'
        '            lleva_el_color(h.payload.get("color_normalized", ""), pedido)\n'
        '            for h in hits\n'
        '        ),\n'
        '    })\n'
        'pd.DataFrame(filas_color).style.hide(axis="index")',
    ),
    (
        "markdown",
        '### C.2 · Qué se pierde por aproximar\n'
        '\n'
        'Las mismas tres consultas propias contra el índice con distintos valores de `ef`, comparadas con la búsqueda exacta sobre los mismos vectores. El sistema entregado usa `ef=32` (R04).\n'
        '\n'
        '`recall_ann@10` es la fracción de los 10 vecinos exactos que el índice recupera: **más alto es mejor**, 1,0 es fidelidad total. No mide si el resultado es bueno, mide si el índice reproduce lo que el modelo dice — son dos preguntas distintas.',
    ),
    (
        "code",
        '# 📄 DATOS · las 3 consultas propias · el oraculo son los mismos vectores\n'
        '# de la seccion A, buscados sin aproximar (recorre los 15.000, ~46 MB).\n'
        'oraculo = DenseRetriever(vectores, completo["product_id"].tolist(), metric="cosine")\n'
        'EFS = (32, 64, 128, 256)\n'
        '\n'
        'filas_fidelidad, top1 = [], {}\n'
        'for cid, texto in CONSULTAS_PROPIAS.items():\n'
        '    qv = codificar_consulta(texto)\n'
        '    exactos = [r.document_id for r in oraculo.search_vector(qv, k=TOP_K)]\n'
        '    fila = {"consulta": cid}\n'
        '    for ef in EFS:\n'
        '        hits = indice.search(qv, top_k=TOP_K, ef=ef)\n'
        '        ids = [str(h.payload.get("product_id")) for h in hits]\n'
        '        fila[f"recall_ann@10_ef{ef}"] = len(set(ids) & set(exactos)) / TOP_K\n'
        '        top1[(cid, ef)] = hits[0]\n'
        '    filas_fidelidad.append(fila)\n'
        '\n'
        'tabla_fidelidad = pd.DataFrame(filas_fidelidad)\n'
        '\n'
        '# La consulta que mas se degrada, con su top-1 en cada extremo: la cifra\n'
        '# sola no ensena que producto se pierde por el camino.\n'
        'columna_baja, columna_alta = f"recall_ann@10_ef{EFS[0]}", f"recall_ann@10_ef{EFS[-1]}"\n'
        'peor = tabla_fidelidad.loc[tabla_fidelidad[columna_baja].idxmin(), "consulta"]\n'
        'print(f\'{peor} · "{CONSULTAS_PROPIAS[peor]}"\')\n'
        'for ef in (EFS[0], EFS[-1]):\n'
        '    h = top1[(peor, ef)]\n'
        '    print(f"  ef={ef:<4} [{h.score:.4f}] {str(h.payload.get(\'title\'))[:56]}")\n'
        'tabla_fidelidad.style.hide(axis="index").format(\n'
        '    {c: "{:.1f}" for c in tabla_fidelidad.columns if c != "consulta"}\n'
        ')',
    ),
    (
        "markdown",
        '### Cómo leer esta tabla\n'
        '\n'
        'No es un fallo nuevo: es el coste de R04 hecho visible. NB06 ya registró `recall_ann_at_10_min = 0,0` y documentó la consulta 18868 con el mismo umbral de recuperación, `ef=128`.\n'
        '\n'
        'Lo que añaden estas tres es que aquel caso duro **no era una rareza del conjunto de desarrollo**: son consultas escritas a mano, que no intervinieron en ninguna decisión, y aun así se degradan con `ef=32` y convergen en el mismo `ef=128`.\n'
        '\n'
        'El problema no es el valor de `ef`, sino el criterio que lo eligió: *"menor p95 entre las admisibles"* optimiza la latencia agregada mientras el recall mínimo por consulta cae. Un criterio sobre el peor caso habría elegido `ef=128`, cuyo p95 —12,36 ms en `benchmark_ann.csv`— cabe en el presupuesto de 20 ms de D16. **R04 no se reabre aquí**; queda registrado en `config.yaml` como mejora medida.',
    ),
    (
        "markdown",
        '## D · Aplicar el ciclo de vida del catálogo\n'
        '\n'
        'Aplica los 24 eventos de `eventos_catalogo.csv` —8 altas, 8 modificaciones y 8 bajas— sobre la colección, y comprueba uno de cada tipo por lectura directa y por búsqueda.',
    ),
    (
        "code",
        '# 📄 DATOS · eventos_catalogo.csv (24) · ⚠️ escribe en Qdrant\n'
        'eventos_upsert = eventos[eventos["tipo"] != "baja"].reset_index(drop=True)\n'
        'eventos_baja = eventos[eventos["tipo"] == "baja"].reset_index(drop=True)\n'
        '\n'
        'vectores_upsert = truncate_dim(\n'
        '    encode_corpus(\n'
        '        encoder, eventos_upsert["text"].tolist(),\n'
        '        corpus_id="eventos_catalogo_upsert", kind="document",\n'
        '        contract=CONTRATO, batch_size=16, cache_dir=CACHE,\n'
        '    ).vectors,\n'
        '    DIM,\n'
        ')\n'
        'vector_por_record = dict(zip(eventos_upsert["record_id"], vectores_upsert))\n'
        '\n'
        'upsert_con_claves = eventos_upsert.copy()\n'
        'for campo in CAMPOS_FILTRABLES:\n'
        '    upsert_con_claves = add_normalized_key(\n'
        '        upsert_con_claves, field=campo, mode=NORMALIZACION\n'
        '    )\n'
        'puntos_upsert = [\n'
        '    Point(\n'
        '        record_id=fila["record_id"],\n'
        '        vector=vector_por_record[fila["record_id"]],\n'
        '        payload=build_payload(\n'
        '            fila, fields=PAYLOAD_SCHEMAS[ESQUEMA], null_policy=POLITICA_NULOS\n'
        '        ),\n'
        '    )\n'
        '    for fila in upsert_con_claves.to_dict("records")\n'
        ']\n'
        '\n'
        'aplicar_secuencia(\n'
        '    indice, puntos_upsert, eventos_baja["record_id"].tolist(), batch_size=LOTE,\n'
        ')\n'
        'RECUENTO_MUTADO = indice.count()\n'
        '\n'
        '# Uno de cada tipo, por lectura directa y por busqueda vectorial (D18).\n'
        'filas_visibilidad = []\n'
        'for tipo in ("alta", "actualizacion", "baja"):\n'
        '    fila = eventos[eventos["tipo"] == tipo].iloc[0]\n'
        '    rid = fila["record_id"]\n'
        '    traza = verificar_evento(\n'
        '        indice, tipo, record_id=rid,\n'
        '        vector=vector_por_record.get(rid),\n'
        '        catalog_version_esperado=2 if tipo == "actualizacion" else None,\n'
        '        top_k=5,\n'
        '    )\n'
        '    filas_visibilidad.append({\n'
        '        "tipo": tipo,\n'
        '        "product_id": fila["product_id"],\n'
        '        "visible_por_id": traza["por_id"]["visible"],\n'
        '        "visible_por_busqueda": traza["por_busqueda"]["visible"],\n'
        '    })\n'
        '\n'
        'print(f"puntos antes de los eventos  : {RECUENTO_2}")\n'
        'print(f"puntos despues de los eventos: {RECUENTO_MUTADO}"\n'
        '      f"  (-{len(eventos_baja)} bajas +{int((eventos[\'tipo\'] == \'alta\').sum())} altas)")\n'
        'pd.DataFrame(filas_visibilidad).style.hide(axis="index")',
    ),
    (
        "markdown",
        '## E · Volver a consultar, ya con el catálogo mutado\n'
        '\n'
        'Repite exactamente las 15 consultas de la sección C sobre la colección modificada.\n'
        '\n'
        'Este es el camino **con modificaciones**.',
    ),
    (
        "code",
        '# 📄 DATOS · las mismas 15 consultas de la seccion C\n'
        'DESPUES = {\n'
        '    cid: [r.document_id for r in buscador.buscar(texto, top_k=TOP_K)]\n'
        '    for cid, texto in CONSULTAS.items()\n'
        '}\n'
        '\n'
        'tabla_cambios = pd.DataFrame([\n'
        '    {\n'
        '        "consulta": cid,\n'
        '        "cambia_el_top10": ANTES[cid] != DESPUES[cid],\n'
        '        "productos_que_entran": len([p for p in DESPUES[cid] if p not in ANTES[cid]]),\n'
        '    }\n'
        '    for cid in CONSULTAS\n'
        '])\n'
        'print(f"consultas cuyo top-{TOP_K} cambia: "\n'
        '      f"{int(tabla_cambios[\'cambia_el_top10\'].sum())}/{len(tabla_cambios)}")\n'
        'tabla_cambios.style.hide(axis="index")',
    ),
    (
        "markdown",
        '## F · Los dos caminos, comparados\n'
        '\n'
        'Sin juicios de relevancia no hay nDCG, así que se mide **Jaccard@10** entre lo que devolvió cada consulta antes y después de los eventos: 1,0 significa que el catálogo cambió pero esa consulta no se enteró; 0,0, que devuelve productos completamente distintos.\n'
        '\n'
        'La segunda tabla mira otra cosa: si las tres formulaciones de una misma intención siguen coincidiendo entre sí después de mutar el catálogo.',
    ),
    (
        "code",
        'from aurum.evaluacion import formulation_consistency, jaccard_at_k\n'
        '\n'
        'filas = []\n'
        'for cid, texto in CONSULTAS.items():\n'
        '    entran = [p for p in DESPUES[cid] if p not in ANTES[cid]]\n'
        '    salen = [p for p in ANTES[cid] if p not in DESPUES[cid]]\n'
        '    filas.append({\n'
        '        "consulta": cid,\n'
        '        "texto": texto[:44],\n'
        '        "jaccard_antes_vs_despues": round(\n'
        '            jaccard_at_k(ANTES[cid], DESPUES[cid], k=TOP_K), 4\n'
        '        ),\n'
        '        "entran_en_el_top10": ", ".join(entran) or "-",\n'
        '        "salen_del_top10": ", ".join(salen) or "-",\n'
        '    })\n'
        'movimiento = pd.DataFrame(filas)\n'
        '\n'
        'print(f"jaccard medio antes-vs-despues: "\n'
        '      f"{movimiento[\'jaccard_antes_vs_despues\'].mean():.4f}  "\n'
        '      f"(1,0 = el catalogo cambio pero la consulta no se entero)")\n'
        'print(f"consultas intactas: "\n'
        '      f"{int((movimiento[\'jaccard_antes_vs_despues\'] == 1.0).sum())}"\n'
        '      f"/{len(movimiento)}")\n'
        'movimiento.style.hide(axis="index")',
    ),
    (
        "code",
        '# Consistencia entre las tres formulaciones de cada intencion, antes y\n'
        '# despues. Solo las 12 de evaluacion: las propias no van por intenciones.\n'
        'ids_evaluacion = list(evaluacion["evaluation_id"])\n'
        'consistencia = (\n'
        '    formulation_consistency({c: ANTES[c] for c in ids_evaluacion}, k=TOP_K)\n'
        '    .merge(\n'
        '        formulation_consistency({c: DESPUES[c] for c in ids_evaluacion}, k=TOP_K),\n'
        '        on="intencion", suffixes=("_antes", "_despues"),\n'
        '    )\n'
        ')\n'
        'consistencia.style.hide(axis="index")',
    ),
    (
        "markdown",
        '### Cómo leer estos números\n'
        '\n'
        'Un Jaccard alto entre antes y después es lo esperable: 8 altas y 8 bajas sobre 15.000 productos solo deberían mover las consultas de su categoría. Una consulta que cambia entera señala que alguna alta compite directamente con ella.\n'
        '\n'
        'La consistencia entre formulaciones no debería moverse apenas: mide si el sistema entiende la intención, y eso no depende de que el catálogo tenga ocho productos más.',
    ),
    (
        "markdown",
        '## G · Limpiar\n'
        '\n'
        'Borra la colección creada en la sección A. **No se ejecuta sola**: hay que poner `LIMPIAR = True`.\n'
        '\n'
        'El índice de NB04-NB09 no se toca en ningún caso. Para parar el motor, `make motor-down MOTOR=qdrant`.',
    ),
    (
        "code",
        'LIMPIAR = False   # ponlo a True para borrar la coleccion de este notebook\n'
        '\n'
        'if LIMPIAR:\n'
        '    # Doble cerrojo: `recreate=True` mas AURUM_ALLOW_RESET en el entorno.\n'
        '    indice.create_collection(dim=DIM, metric="cosine", recreate=True)\n'
        '    print(f"{COLECCION} recreada vacia: {indice.count()} puntos")\n'
        'else:\n'
        '    print(f"{COLECCION} conservada con {indice.count()} puntos.")\n'
        '    print("Pon LIMPIAR = True para borrarla · `make motor-down MOTOR=qdrant` para parar el motor.")',
    ),
]

NOTEBOOKS = {
    "00_datos.ipynb": NB00_DATOS,
    "01_baseline.ipynb": NB01_BASELINE,
    "02_modelo.ipynb": NB02_MODELO,
    "03_representacion.ipynb": NB03_REPRESENTACION,
    "04_motor.ipynb": NB04_MOTOR,
    "05_recuperacion.ipynb": NB05_RECUPERACION,
    "06_ann.ipynb": NB06_ANN,
    "07_duplicados.ipynb": NB07_DUPLICADOS,
    "08_mutaciones.ipynb": NB08_MUTACIONES,
    "09_evaluacion.ipynb": NB09_EVALUACION,
    "10_entrega.ipynb": NB10_ENTREGA,
}
