"""Contenido fuente de los notebooks, en Python plano y versionable.

`build_notebook.py` convierte estas listas de celdas en los .ipynb de
`notebooks/`. Se edita aquí; el JSON del notebook nunca se toca a mano.
"""

NB00_DATOS = [
    ("markdown", "# NB00 · Datos: contrato, perfilado y decisiones de negocio"),
    (
        "markdown",
        "Evidencia real para las decisiones **D01-D04** de `docs/PLAN.md`. La validación "
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
        '**D09 ya está decidida** (`config/config.yaml`): compiten `gemini-embedding-2`, `jinaai/jina-embeddings-v3` e `ibm-granite/granite-embedding-311m-multilingual-r2`, con la **plantilla congelada en A0** (la columna `text` del dataset, tal cual).\n'
        '\n'
        'Este notebook empieza por el **paso 0**: antes de codificar nada, medir **cuánto texto ve realmente cada candidato**. Es la evidencia que decide **D07** (¿hace falta chunking?) y la que verifica que la ventana de contexto de los tres modelos cubre el catálogo.\n'
        '\n'
        '---\n'
        '\n'
        '### ⚠️ Por qué las longitudes de NB01 no sirven aquí\n'
        '\n'
        'NB01 midió `text` en **palabras**, con la expresión regular de `aurum.datos.tokenize` (p50 = 150). El límite de contexto de un modelo se mide en **piezas de su vocabulario de subpalabras**, que es otra unidad: una palabra larga o un código como `160x200` se lleva varias piezas. La única cuenta válida es la del tokenizador de cada modelo, y por eso se descarga aquí.\n'
        '\n'
        '| Marca | Corpus | Fichero |\n'
        '|---|---|---|\n'
        '| 🔬 **MUESTRA** | 1.500 registros | `catalogo_muestra.csv` |\n'
        '| 📚 **COMPLETO** | 15.000 registros | `catalogo_productos.csv` |',
    ),
    (
        "code",
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
    ("code", 'token_length_report(muestra["text"], tokenizadores)'),
    (
        "markdown",
        '## A.3 · 📚 La misma medición sobre el catálogo completo\n'
        '\n'
        '⏱️ Tarda ~30 s: tokeniza 15.000 registros.',
    ),
    (
        "code",
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
        '⚠️ **Es una petición de red por registro del catálogo**, así que se mide un subconjunto en vez de las 15.000: las **50 registros más largos** en caracteres —que son las únicas que podrían acercarse a la ventana— más **100 al azar** para el ratio `chars_por_token`. Con un máximo local de 1.972 tokens contra una ventana de 8.192, el margen es de 4×: no hace falta más precisión para responder a D07.\n'
        '\n'
        'Si no hay `GEMINI_API_KEY`, la celda se salta sin romper el notebook — el corrector puede ejecutar el resto sin clave.',
    ),
    (
        "code",
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
        'if clave:\n'
        '    todos = {MODELO_GEMINI: (gemini, VENTANA_GEMINI), **tokenizadores}\n'
        '    display(token_length_report(subconjunto, todos))',
    ),
    (
        "markdown",
        '## A.6 · Cómo se lee esto para D07\n'
        '\n'
        'El chunking (familia C del plan) solo tiene sentido si el modelo elegido **no puede leer el registro del catálogo entero**. Con `pct_supera_ventana = 0` en los tres candidatos, no hay información que se pierda por truncado y la familia C queda descartada **por medición**, no por falta de tiempo.\n'
        '\n'
        'Consecuencia directa en la base vectorial: el punto sigue siendo `record_id` (relación 1:1 producto↔vector), el esquema de NB04 se mantiene simple y la idempotencia no necesita borrar chunks huérfanos.',
    ),
    (
        "markdown",
        '---\n'
        '\n'
        '# B · Codificación de los tres candidatos\n'
        '\n'
        'Hasta aquí se ha medido **cuánto texto ve** cada modelo. Ahora se codifica de verdad, sobre `catalogo_muestra.csv` (condición 3 del plan: la muestra existe justo para esto; el catálogo completo solo se ingiere en la ejecución final).\n'
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
        'Los tres últimos ejes son **gratis**, y por eso el barrido de dimensión de la sección C cubre 15 configuraciones sin pagar 15 codificaciones.\n'
        '\n'
        '### ⏱️ Lo que esto cuesta en esta máquina\n'
        '\n'
        '4 núcleos, sin GPU. `jina-embeddings-v3` son 572M de parámetros: cargarlo en `float32` ya ocupa ~2,3 GB de los 7,9 GB disponibles, así que **los modelos se cargan y se liberan de uno en uno**. Cuenta con decenas de minutos la primera vez.\n'
        '\n'
        'La segunda vez es instantánea: `encode_corpus` guarda los vectores en `artifacts/embeddings/` junto a un `.json` con el `model_id`, la dimensión, el dtype y el **SHA-256 del corpus**. Si el texto cambia (por ejemplo al pasar de la plantilla A0 a otra en NB03), la huella cambia y la caché se invalida sola — que es lo que impide comparar en silencio vectores de dos textos distintos.',
    ),
    (
        "code",
        'import gc\n'
        'import time\n'
        '\n'
        'import torch\n'
        '\n'
        'from aurum.busqueda import DenseRetriever, rank_queries_dense\n'
        'from aurum.embeddings import (\n'
        '    GeminiEncoder,\n'
        '    SentenceTransformerEncoder,\n'
        '    encode_corpus,\n'
        '    safe_l2_normalize,\n'
        '    truncate_dim,\n'
        '    vector_health,\n'
        ')\n'
        'from aurum.evaluacion import apply_tolerance_rule, evaluate_rankings, qrels_from_judgements\n'
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
        'qrels = qrels_from_judgements(relevancias)\n'
        '\n'
        'corpus_textos = muestra[CAMPO].tolist()\n'
        'corpus_ids = muestra["product_id"].tolist()\n'
        'query_ids = [str(q) for q in consultas["query_id"]]\n'
        'query_textos = consultas["query_text"].tolist()\n'
        '\n'
        'print(f"corpus   : {len(corpus_textos)} documentos (plantilla {PLANTILLA})")\n'
        'print(f"consultas: {len(query_textos)} de desarrollo, {len(qrels)} juzgadas")',
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
        '> 🔎 **Por qué granite no entra en ese eje, y por qué es provisional.** Su `config_sentence_transformers.json` declara literalmente `"prompts": {"query": "", "document": ""}`: los dos prompts son la cadena vacía. Codificar "con contrato" y "sin contrato" daría **exactamente los mismos vectores**, así que la Δ sería 0 por construcción y gastar dos codificaciones en demostrarlo no es evidencia, es tiempo de CPU. Inventarle un prefijo sería peor: estaríamos midiendo un modelo que nadie entrenó.\n'
        '>\n'
        '> Ese argumento se apoyaba en un fichero del repositorio, y §3.1 avisa de que *"no basta con citar la documentación del modelo"*: el fichero prueba que **la librería** no antepone nada, no que **IBM** entrenara el modelo sin instrucción. Quedó registrado como **P02** y **está cerrado**. Revisada la model card completa —todos los backends documentados (`sentence-transformers`, Transformers, ONNX, OpenVINO, vLLM, GGUF) más las secciones *Usage* y *When to Use This Model*—, **no hay ninguna instrucción ni prefijo en ningún sitio**: en el ejemplo de retrieval de IBM, consultas y documentos se pasan por igual a `model.encode()`. El contrato real es **texto plano simétrico**, confirmado por la fuente primaria y no por inferencia de un JSON. `granite` no compitió en desventaja.\n'
        '\n'
        '> ⚠️ `jina-v3` exige `trust_remote_code=True`: se ejecuta código del repositorio de Jina. Es una consecuencia que hereda el corrector y queda anotada en el README. Su licencia **`cc-by-nc-4.0`** es además una de las *"restricciones del caso"* que §3.1 obliga a pesar en la elección: prohíbe el uso comercial, que es exactamente el escenario de un marketplace.',
    ),
    (
        "code",
        'REGISTRO = {\n'
        '    "jina-v3": {\n'
        '        "repo": "jinaai/jina-embeddings-v3",\n'
        '        "ventana": 8192,\n'
        '        "dim_nativa": 1024,\n'
        '        "tasks": {"document": "retrieval.passage", "query": "retrieval.query"},\n'
        '        "trust_remote_code": True,\n'
        '        "dims": [1024, 768, 512, 256, 128],\n'
        '        "licencia": "cc-by-nc-4.0",\n'
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
        'Los tres modelos **no** se codifican en un bucle: cada uno tiene su celda y se puede lanzar por su cuenta, en el orden que quieras y en sesiones distintas. Hay tres razones y ninguna es estética:\n'
        '\n'
        '1. **RAM.** `jina-v3` en `float32` ocupa ~2,3 GB de los 7,9 GB de la máquina. Cada celda construye el modelo, codifica y lo libera con `gc.collect()` antes de devolver el control. Dos modelos vivos a la vez no caben.\n'
        '2. **Tiempo.** Son decenas de minutos por modelo. Un bucle único obliga a esperar a los tres para ver el primer número.\n'
        '3. **Aislamiento de fallos.** Si `jina-v3` revienta por su `trust_remote_code` o Gemini se queda sin cuota, los demás ya están medidos. El enunciado pide *"al menos dos configuraciones relevantes"*: con dos de tres sigues teniendo comparación.\n'
        '\n'
        '### 🔑 Cada celda codifica las DOS variantes del modelo\n'
        '\n'
        '`nativo` (aplicando el contrato de entrada que el modelo declara) y `sin_contrato` (omitiéndolo). Las dos llamadas están juntas a propósito, y no es una comodidad: es lo que hace que **el notebook dé el mismo resultado se ejecute como se ejecute**.\n'
        '\n'
        'El barrido de la sección **C** evalúa todo lo que encuentre en `VECTORES`. Si `sin_contrato` se codificara más abajo —en la sección D, que es donde se analiza—, quien ejecutara el notebook de principio a fin llegaría a C con solo la mitad de las configuraciones, y **C.1, C.2, F y G decidirían sobre media tabla sin mostrar ningún aviso**. Codificando aquí las dos ramas, el orden de ejecución deja de importar.\n'
        '\n'
        'La sección D, por tanto, **no codifica nada**: solo mide la diferencia entre las dos ramas que ya existen.\n'
        '\n'
        '> `granite-311m-r2` no tiene contrato que retirar, así que su segunda llamada no codifica nada y lo dice por pantalla. Es documentación ejecutable: en el informe se ve que se saltó a propósito y no por olvido (**P02**, cerrado con la model card).\n'
        '\n'
        '### Cómo se combinan después\n'
        '\n'
        'Cada celda deposita sus vectores en `VECTORES`, indexado por `(modelo, contrato)`. **Todo lo que viene detrás (barrido C, contrato D, métrica E, comparación F, regla G) lee ese diccionario y trabaja con lo que encuentre**, sea uno, dos o tres modelos. No hace falta que estén los tres para obtener métricas: las tablas saldrán con las filas que haya.\n'
        '\n'
        '> 🔁 **Tras reiniciar el kernel** vuelve a ejecutar las tres celdas: como los vectores están en `artifacts/embeddings/`, la segunda vez son segundos, no minutos. Re-ejecutar una celda tampoco duplica nada — `COSTES` está indexado por `(modelo, contrato, tipo)`, así que sobrescribe en vez de acumular.',
    ),
    (
        "code",
        'VECTORES = {}   # (alias, contrato) -> {"document": ndarray, "query": ndarray}\n'
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
        '        for kind, textos, corpus_id in (\n'
        '            ("document", corpus_textos, CORPUS_ID),\n'
        '            ("query", query_textos, "consultas_desarrollo"),\n'
        '        ):\n'
        '            resultado = encode_corpus(\n'
        '                encoder, textos, corpus_id=corpus_id, kind=kind,\n'
        '                contract=contrato, batch_size=lotes, cache_dir=CACHE,\n'
        '            )\n'
        '            salida[kind] = resultado.vectors\n'
        '            COSTES[(alias, contrato, kind)] = {"alias": alias, **resultado.stats.as_row()}\n'
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
        'ejecutar("jina-v3")                            # con contrato: adaptador LoRA por tarea\n'
        'ejecutar("jina-v3", contrato="sin_contrato")   # sin él: mismos textos, otros pesos',
    ),
    (
        "markdown",
        '### B.2b · `granite-311m-r2` — Apache-2.0, sin código remoto\n'
        '\n'
        '**Una sola codificación.** La segunda llamada está puesta pero no codifica: granite declara sus dos prompts como cadena vacía, así que `nativo` y `sin_contrato` producirían vectores idénticos. La celda imprime el motivo del salto en lugar de gastar otra pasada en demostrar una Δ que es 0 por construcción.\n'
        '\n'
        'Esa exclusión se registró como **P02** y está **cerrada**: la model card completa de IBM no documenta ninguna instrucción ni prefijo en ningún backend, así que el contrato real es texto plano simétrico. La conclusión se sostiene ahora en la fuente primaria, no en un JSON de configuración.',
    ),
    (
        "code",
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
        '**Dos codificaciones, pero baratas.** En Gemini el contrato sí es texto: una instrucción de tarea antepuesta al contenido. Retirarla no cambia los pesos, solo lo que se envía, y como el trabajo lo hace la API el eje cuesta llamadas, no horas de CPU.\n'
        '\n'
        'Necesita `GEMINI_API_KEY` en `.env`. Si no está, las dos llamadas fallan de forma controlada y el notebook continúa con los modelos locales.',
    ),
    (
        "code",
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
        'display(estado())\n'
        'if ERRORES:\n'
        '    display(pd.DataFrame([{"modelo": k, "error": v} for k, v in ERRORES.items()]))\n'
        'pd.DataFrame(COSTES.values())',
    ),
    (
        "markdown",
        '## B.3 · Salud de los vectores — antes de creerse ninguna métrica\n'
        '\n'
        'Un `NaN` o una matriz con filas repetidas producen métricas perfectamente presentables y perfectamente falsas. Estas comprobaciones son las que el plan exige en las *Métricas de verificación de NB02*: finitud, normas y duplicados.\n'
        '\n'
        '**Atención a la columna `normalizado`, que resultó no ser un dato anecdótico.** `SentenceTransformerEncoder` pide `normalize_embeddings=False` a los dos modelos locales, así que cabría esperar que ninguno llegara unitario. No es lo que ocurre: `jina-v3` y `gemini-2` salen con norma exactamente 1 y solo `granite-311m-r2` entrega la salida cruda.\n'
        '\n'
        'El motivo es que ese flag **añade** normalización cuando vale `True`; no **retira** un módulo `Normalize` que forme parte del modelo. El pipeline de `jina-v3` lleva el suyo, y Gemini devuelve vectores unitarios por API.\n'
        '\n'
        'Eso convierte a `granite` en el único candidato con el que la sección E puede medir algo. Conviene tenerlo presente antes de leer aquella tabla.',
    ),
    (
        "code",
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
        'Los tres candidatos están entrenados con **Matryoshka Representation Learning**: las primeras componentes concentran la mayor parte de la información, así que quedarse con un prefijo del vector es una reducción de dimensión válida y sin recodificar.\n'
        '\n'
        'Truncar **obliga a renormalizar**: el prefijo de un vector unitario tiene norma < 1, y sin renormalizar el coseno deja de ser un coseno. `truncate_dim` lo hace siempre.\n'
        '\n'
        'Lo que se busca en la tabla no es solo el máximo, sino **dónde se cae la curva**: si 256 dimensiones pierden menos de 0,02 de nDCG@10 frente a 1.024, el ahorro es de 4× en memoria del motor y en ancho de banda por consulta. Ese es exactamente el compromiso que D09b declaró de antemano.',
    ),
    (
        "code",
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
        'Los mismos números del barrido, en la forma que responde a la pregunta real: **¿dónde se cae la curva?** Una tabla obliga a restar de cabeza para verlo; la línea lo enseña de un vistazo — si la caída es suave, MRL está funcionando; si hay un escalón, esa dimensión ya no basta para este catálogo.\n'
        '\n'
        'Tres elementos del gráfico que no son decoración:\n'
        '\n'
        '- **Color = modelo · trazo = contrato.** Son dos ejes cruzados. Metiéndolos los dos en el color saldrían cinco tonos sin relación aparente entre sí; separándolos, el ojo agrupa primero por modelo y luego compara la línea continua con la discontinua **dentro** de cada uno. Esa comparación —el mismo modelo consigo mismo— es la que importa, y es la que la sección D cuantifica.\n'
        '- **La banda gris es la tolerancia τ = 0,02 de D09b.** Todo punto que cae dentro es *admisible* por la regla, y entre los admisibles gana el de menor dimensión: es decir, **el punto admisible situado más a la izquierda**. La sección G lo calcula formalmente con `apply_tolerance_rule`; aquí se ve venir antes de aplicarlo.\n'
        '- **El eje X está en escala logarítmica** porque las dimensiones se barren dividiendo por dos. En escala lineal, 128 y 256 se amontonarían contra el margen izquierdo y la parte interesante de la curva —justo donde se decide el ahorro— quedaría ilegible.\n'
        '\n'
        'Toda la lógica vive en `aurum.graficas`, cubierta por `tests/test_graficas.py`. El notebook solo declara *qué* quiere ver, no *cómo* se dibuja.',
    ),
    (
        "code",
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
        '> 📌 **Esta sección no codifica nada.** Las dos variantes de cada modelo se generaron en **B.2**, junto al resto de codificaciones. Aquí solo se comparan. Si has ejecutado B.2 completo, esta sección corre en segundos.\n'
        '\n'
        '§3.1 del enunciado pide elegir cuatro cosas y justificarlas juntas:\n'
        '\n'
        '> *"Después elegid **la representación textual, el modelo de embeddings, los prefijos que requiera y la normalización**. No basta con citar la documentación del modelo: la elección debe apoyarse en los resultados de desarrollo y en las restricciones del caso."*\n'
        '\n'
        'El sujeto de "la elección" es esa enumeración entera, así que la exigencia se reparte por todo NB02: la **representación textual** está congelada en A0 y documentada en NB01/NB03, el **modelo** se decide con el barrido de C y la regla de G, la **normalización** se mide en E — y los **prefijos** son esta sección.\n'
        '\n'
        'Lo que se hace aquí es lo que convierte una cita en evidencia: comparar cada modelo consigo mismo, con y sin su contrato.\n'
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
        '> ✅ **P02, cerrado.** Esa última exclusión era justo la que §3.1 no admite tal cual: se apoyaba en el `config_sentence_transformers.json` de granite, es decir, en **la documentación del modelo**. Se cerró leyendo la fuente primaria: la model card completa de IBM, con todos sus ejemplos de uso y las secciones *Usage* y *When to Use This Model*. **No documenta ninguna instrucción ni prefijo en ningún backend** — en su ejemplo de retrieval cross-lingual, `input_queries` e `input_passages` van directos a `model.encode()` sin nada antepuesto, a diferencia de `e5-instruct` o los BGE con instrucciones. El contrato de entrada real es **texto plano simétrico**, así que granite no compitió en desventaja y su exclusión de este eje queda sostenida por evidencia más fuerte que la que la motivó.',
    ),
    ("markdown", '### D.2 · Δ nDCG@10 al retirar el contrato'),
    (
        "code",
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
        'El enunciado (§3.2) pide *"conservar la semántica del score nativo"* y §3.1 *"explicar la relación entre la métrica configurada, la normalización y el significado del score"*. Estas dos celdas son esa explicación, medida:\n'
        '\n'
        '- Con vectores **L2-normalizados**, `cosine` y `dot` producen el **mismo ranking** (el producto escalar de dos unitarios *es* el coseno), y `l2` también, porque `‖a−b‖² = 2 − 2·a·b` es una función monótona decreciente del producto escalar.\n'
        '- **Sin normalizar**, `dot` premia los vectores de norma grande y el ranking cambia. Ese es el fallo silencioso que esta comprobación caza.\n'
        '\n'
        'Si las tres filas normalizadas no coinciden, la normalización no se está aplicando y **todas las métricas del notebook están en duda**.\n'
        '\n'
        '### ⚠️ Cómo leer la tabla: solo un modelo demuestra algo\n'
        '\n'
        'Como se vio en B.3, `jina-v3` y `gemini-2` ya entregan vectores unitarios. Su fila *"sin normalizar"* recibe vectores unitarios igualmente, así que sale **idéntica** a la normalizada: no demuestra nada, solo confirma que normalizar dos veces es idempotente.\n'
        '\n'
        '**Toda la evidencia de esta sección la aporta `granite-311m-r2`**, precisamente por ser el único que llega crudo. Ahí sí se ve el fenómeno: `cosine` no se mueve —normaliza internamente, es inmune— mientras `dot` baja y `l2` sube, y las tres métricas dejan de coincidir.\n'
        '\n'
        'Y lo llamativo es **cuán poco hace falta para romperlo**: las normas de granite se desvían apenas unas milésimas de 1, y eso ya basta para reordenar resultados y mover el nDCG. No hace falta una anomalía grande para que `dot` deje de ser una medida de similitud.\n'
        '\n'
        '> Si los tres modelos normalizaran en origen, esta comprobación pasaría sin detectar nada. Es exactamente el modo en que este tipo de verificación falla en silencio.',
    ),
    (
        "code",
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
        'NB01 dejó ese baseline medido en `artifacts/baseline_lexico.json`. Aquí se recupera **sobre el mismo corpus** (la muestra de 1.500), con las mismas 8 consultas, el mismo `k`, los mismos qrels y el mismo contrato de relevancia. Sin esa igualdad no se compararían métodos, sino entornos (Regla 2).\n'
        '\n'
        'La pregunta que responde la tabla no es *"¿gana el denso?"* sino **"¿cuánto gana y a cambio de qué coste?"**: BM25 se construye en segundos sobre CPU y no necesita ni modelo ni GPU ni base vectorial. Si la mejora del denso fuera marginal, el argumento de negocio para montar toda esta infraestructura sería flojo — y decirlo con un número es mejor informe que esconderlo.',
    ),
    (
        "code",
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
        '> 🗳️ **Paso 6 del bucle — te toca a ti.** La celda produce la ordenación; **R02 la ratificas tú** y la escribes en `config/config.yaml`. Si el resultado te parece equivocado, el sitio para discutirlo es el criterio, no la tabla: cambiar la regla después de ver los números es exactamente lo que el enunciado penaliza.',
    ),
    (
        "code",
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
        'La sección D midió la Δ del contrato **solo en la dimensión nativa**, y con ese único punto la conclusión parecía limpia: retirarlo mejora en los dos modelos que lo tienen. El barrido completo dice algo más incómodo — y más interesante.\n'
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
        'La instrucción de tarea es **texto idéntico en los 1.500 documentos**. En un modelo entrenado con MRL, las primeras componentes concentran la estructura más gruesa y compartida del corpus, y ahí es donde esa señal común pesa más.\n'
        '\n'
        '- **A dimensión completa**, ese prefijo compartido es sobre todo lastre: ocupa norma sin aportar nada que distinga un producto de otro, así que quitarlo mejora.\n'
        '- **Al truncar fuerte**, te quedas casi solo con esas primeras componentes, y ahí la instrucción funciona como condicionamiento de tarea que sí orienta la búsqueda.\n'
        '\n'
        'No está comprobado — habría que mirar la energía por componente en ambas variantes. Queda como hipótesis explícita, no como explicación cerrada.\n'
        '\n'
        '### Consecuencia práctica\n'
        '\n'
        '**"Sin contrato es mejor" no es incondicional: vale a partir de 512.** La configuración que gana D09b (`gemini-2 [sin_contrato] @768`) cae con holgura en esa zona, así que **R02 no se ve afectada**.\n'
        '\n'
        'Pero es una condición que hay que arrastrar. Si más adelante la memoria del índice empujara a bajar de dimensión —15.000 productos a 768 son 46 MB; a 256 serían 15 MB— la decisión sobre el contrato tendría que **revisarse, no heredarse**. Medir ese eje en un solo punto habría dejado la trampa puesta.',
    ),
    (
        "code",
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
        'Todo lo anterior está medido sobre **1.500 documentos**, la muestra de la condición 3 del plan. El enunciado (§6, *Condiciones de comparabilidad*) dice que *"el catálogo completo es el recorrido evaluado; la muestra sirve para desarrollar y depurar"*. Esta sección cierra esa distancia para la configuración que ganó D09b.\n'
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
        'Los qrels no cambian: lo que cambia es que aparecen 13.500 productos más que compiten por las 10 posiciones y que, al no estar juzgados, puntúan 0 (D04). Un sistema que sube documentos no juzgados se desploma; uno que mantiene arriba los juzgados aguanta. **Es un test de precisión bajo distracción**, y no hay forma de aprobarlo desde la muestra.\n'
        '\n'
        '### Qué se ejecuta aquí, y qué no\n'
        '\n'
        'Solo se codifica el **ganador de D09b**, y la celda lo lee de `ordenadas` en vez de escribirlo a mano: si la regla cambiara de ganador, esta sección lo sigue.\n'
        '\n'
        'El motivo es de coste, y está medido, no estimado a ojo:\n'
        '\n'
        '| configuración | 1.500 docs | 15.000 (×10) |\n'
        '|---|---:|---:|\n'
        '| `gemini-2` | ~50 s | **~8 min** ✅ |\n'
        '| `jina-v3` | ~5.750 s | ~16 h ❌ |\n'
        '| `granite-311m-r2` | ~17.490 s | ~49 h ❌ |\n'
        '\n'
        'La celda calcula esa extrapolación y **se niega a lanzar** cualquier codificación por encima de `LIMITE_HORAS`. Si el ganador fuera un modelo local, imprimiría el coste y se saltaría el paso en lugar de dejar el kernel bloqueado media semana.\n'
        '\n'
        'Dos cosas más que abaratan la prueba:\n'
        '\n'
        '- **Las consultas no se recodifican.** Su caché es independiente del corpus (`corpus_id="consultas_desarrollo"`), así que los 8 vectores ya están en disco.\n'
        '- **Las tres dimensiones admisibles salen de la misma codificación.** Truncar es gratis, así que 768, 1.536 y 3.072 se evalúan sin ninguna llamada extra.\n'
        '\n'
        '### Qué responde y qué no\n'
        '\n'
        'Responde a la pregunta del enunciado: **¿el denso sigue batiendo al léxico cuando el catálogo es el de verdad?**\n'
        '\n'
        'No responde al orden **entre modelos densos** a escala completa: para eso habría que pagar las 65 horas de jina y granite. Queda anotado como límite explícito del experimento — con `gemini-2` sacando 0,12 sobre BM25 y 0,24 sobre jina en la muestra, el riesgo de que el orden se invierta es bajo, pero *bajo* no es *cero* y el informe debe decirlo así.',
    ),
    (
        "code",
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
]

NB03_REPRESENTACION = [
    ("markdown", '# NB03 · ¿Qué texto codifico? — plantillas y representación'),
    (
        "markdown",
        '**El montaje de NB02, al revés.** Allí el texto estaba congelado en A0 y variaba el modelo; aquí el modelo queda congelado en el ganador de R02 y **lo único que cambia es el texto** (Regla 1).\n'
        '\n'
        '> 🔒 **Congelado** (`config.yaml` → `nb03_representacion.modelo_congelado`): `gemini-embedding-2` · contrato `sin_contrato` · **dim 768** · métrica `cosine` · normalización L2 explícita al truncar · k=10.\n'
        '\n'
        '### La pregunta\n'
        '\n'
        'Un embedding resume el significado de **todo** lo que entra. Si de los ~1.300 caracteres que tiene `text` de media, la mayoría es prosa comercial —*"perfecto para regalo, calidad premium, ideal para toda ocasión"*—, el vector se acerca al lenguaje genérico de cualquier producto y se aleja de lo que hace distinto a *este*.\n'
        '\n'
        '**Menos texto puede recuperar mejor.** Es una hipótesis, y aquí se mide.\n'
        '\n'
        '### Qué NO entra: la familia C (D07)\n'
        '\n'
        'El chunking queda descartado **por medición, no por falta de tiempo**. NB02·A midió con el tokenizador de cada modelo que `pct_supera_ventana = 0` sobre los 15.000 registros: el máximo son 1.972 tokens frente a ventanas de 8.192 y 32.768, más de 4× de margen. Partir en trozos resuelve un problema que aquí no existe.\n'
        '\n'
        'Consecuencia, ya registrada en `config.yaml`: el punto de la base vectorial sigue siendo `record_id` en relación **1:1** con el producto. El esquema de NB04 se mantiene simple, la idempotencia no necesita borrar chunks huérfanos, el top-10 no necesita deduplicar y **D08 queda sin aplicar**.',
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
        'Porque no es otra receta de la secuencia: es **el control de A3**. D02 decidió omitir la sección de un campo vacío en lugar de escribir `"Color: desconocido"`, y el argumento fue que insertar un literal compartido en el 36,6 % del catálogo crearía una señal común artificial — productos que se acercan por compartir una palabra, no por parecerse.\n'
        '\n'
        'Ese razonamiento era sólido pero **no estaba medido**, y §3.1 no da por buena una justificación sin datos. A3 frente a A3n aísla exactamente esa política: si contamina, A3n saldrá peor; si da igual, D02 era una precaución sin coste; y si sale mejor, la decisión estaba equivocada y se descubre a tiempo.',
    ),
    ("code", 'template_stats(muestra)'),
    (
        "markdown",
        '### A.1 · El mismo producto por las siete recetas\n'
        '\n'
        'Ver el texto real es lo que evita discutir sobre abstracciones. Fíjate en la distancia entre A0 y el resto: si A3 gana, la conclusión no será *"las etiquetas ayudan"* sino que **el resto del texto era relleno**.',
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
        'A4 recorta `text`, pero **el punto de corte no lo elige nadie**: se deriva del propio corpus. Un número escrito a mano —512, pongamos— sería una decisión de diseño disfrazada de detalle de implementación, imposible de justificar frente a 400 o 600 y sin sentido en cuanto cambiara el catálogo.\n'
        '\n'
        'Se usa la **mediana** de `text`, no la media: la distribución está sesgada a la derecha y topada en 3.000 caracteres, así que la media queda por encima de lo típico y apenas tocaría a cuatro de cada diez fichas. La mediana parte el catálogo en dos mitades exactas y convierte A4 en una pregunta nítida: **¿sobra la mitad más larga de cada ficha?**',
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
        'No hace falta volver a medirlo: NB02 comprobó que **A0 —la más larga de las siete— no supera la ventana en ningún registro** del catálogo completo (máximo 1.972 tokens frente a 8.192). Todas las demás son estrictamente más cortas que A0, así que ninguna puede truncarse.\n'
        '\n'
        'Esto vacía de contenido la familia B del plan como *medición de truncado*, pero no la pregunta que había detrás. La reformulamos: ya no es *"¿se pierde información al truncar?"* sino **"¿el texto largo es señal o es relleno?"** — y esa la responde A4 frente a A0.',
    ),
    (
        "markdown",
        '## B · Codificar las siete variantes\n'
        '\n'
        '⏱️ **~6 minutos.** Gemini codifica 1.500 documentos en ~50 s, así que las siete salen por unos 6 min. Con los modelos locales de NB02 (5 h y 14 h) este barrido habría sido inviable — es una consecuencia directa de qué modelo ganó R02.\n'
        '\n'
        '⚠️ **Cada plantilla invalida la caché**, y eso es lo correcto: `encode_corpus` incluye el SHA-256 del corpus en el nombre del artefacto, así que cambiar el texto cambia la huella y fuerza a recodificar. Es justo lo que impide comparar en silencio vectores de dos textos distintos.\n'
        '\n'
        '**Las consultas no se recodifican.** Las plantillas describen *productos*; una consulta es lo que escribe la persona y no tiene plantilla que aplicar. Sus vectores ya están en `artifacts/embeddings/` desde NB02.',
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
        'Que la media suba no basta. Con 8 consultas, **una sola que cambie de sitio mueve la media macro 0,125** — más que de sobra para fabricar cualquier diferencia que veamos aquí. Antes de tocar D02 hay que responder a una pregunta distinta: *¿la mejora aparece donde el relleno actúa?*\n'
        '\n'
        'El relleno solo toca a las fichas con `color` vacío. Así que cada consulta tiene una **exposición** medible: qué porcentaje de sus productos relevantes lleva ese campo en blanco.\n'
        '\n'
        '**Y de ahí sale una predicción que se puede falsar.** Si rellenar aportara información sobre el color, las consultas más expuestas serían las que más se mueven — los puntos dibujarían una tendencia ascendente. Si en cambio el efecto aparece repartido al azar, y sobre todo si **la consulta con exposición total no se mueve**, entonces lo que estamos midiendo es que añadir texto compartido al corpus desplaza el espacio vectorial, no que aporte significado.\n'
        '\n'
        'Es la diferencia entre un hallazgo y una casualidad con buena presencia.',
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
        'Conviene además mirar el eje X contra la línea base del catálogo que imprime la celda anterior: una consulta por **debajo** de esa referencia está menos expuesta que el producto medio, así que una mejora grande ahí es todavía más difícil de explicar por el contenido del relleno.',
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
        'Un buscador semántico debería devolver **prácticamente los mismos productos** para las tres: piden lo mismo con palabras distintas. El Jaccard@10 entre formulaciones mide esa estabilidad.\n'
        '\n'
        '**Por qué importa aquí:** el nDCG sobre 8 consultas se puede ganar por afinidad léxica con esas ocho concretas. La consistencia entre formulaciones dice qué plantilla **generaliza** a otra superficie léxica — y es la que se va a encontrar en producción, donde nadie escribe como el conjunto de desarrollo. Si una plantilla gana en nDCG pero pierde aquí, la ventaja probablemente era sobreajuste.',
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
        'Su papel es el de **control de D02**, y estaba declarado antes de medir: no es otra receta de la secuencia, es la variante que existe para poner a prueba una decisión ya tomada. Se codifica y se mide igual que las demás —sin eso no habría con qué contrastar—, pero no aspira a ser la elegida.\n'
        '\n'
        'Excluir a una plantilla **después** de ver que puntúa alto es exactamente lo que el enunciado penaliza, así que la declaración previa no basta por sí sola. Lo que sostiene la exclusión es el análisis de la sección D: su ventaja no aparece donde el relleno actúa, sino repartida al azar, y la consulta con exposición total al relleno es de las que menos se mueven. No se aparta porque incomode el resultado; se aparta porque se investigó de dónde venía y no venía de lo que la plantilla cambia.\n'
        '\n'
        '**Por qué la longitud y no el número de campos.** El criterio buscado era *"que diga más con menos"*: densidad de significado. Pero "representar mejor el producto" es justo lo que mide nDCG@10, y el desempate **solo se activa cuando esa métrica ya ha declarado dos plantillas equivalentes**. En ese punto el significado está empatado por medición, así que maximizar densidad se reduce exactamente a minimizar el texto. Contar columnas habla de dependencia de datos, no de cuánto significado llevan.\n'
        '\n'
        '> 🗳️ **R01 la ratificas tú** y la escribes en `config/config.yaml`. Si el resultado no te convence, el sitio para discutirlo es el criterio, no la tabla.',
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
        'Y el mecanismo apunta en contra de las plantillas cortas. Con diez veces más productos compitiendo por las mismas diez posiciones, una representación de ~120 caracteres tiene mucho menos con lo que separar vecinos próximos que una de ~1.300. Cabría esperar que **el texto corto se degrade más** al añadir distractores — que es justo lo contrario de lo que necesita para ganar.\n'
        '\n'
        'No es seguro, es la dirección en la que apunta el sentido común. Por eso se mide en vez de suponerse.\n'
        '\n'
        '### ⚠️ Lo que cuesta\n'
        '\n'
        'Siete plantillas × 15.000 documentos, unos **8 minutos cada una**: cerca de una hora. Y unos **1,3 GB** de vectores en la caché.\n'
        '\n'
        'Es caro, pero es la decisión con la que se ingiere el catálogo definitivo en NB04: equivocarse obliga a recodificar los 15.000 y a reconstruir el índice desde cero. La celda estima el coste antes de lanzarlo y se niega a empezar si se dispara por encima del límite declarado.\n'
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
        '**Esta tabla es la que ratifica R01.** Si la ganadora coincide con la de la muestra, la decisión llega respaldada a las dos escalas. Si no coincide, manda esta — y conviene dejar escrito en el informe que la muestra habría llevado a otra elección, porque es exactamente la trampa que el enunciado avisa en §6.',
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
        'La consistencia sobre las 12 consultas ciegas se midió antes con la muestra, y la decisión acabó tomándose sobre el catálogo completo. Dejarlas a escalas distintas invita a citar una comprobación que ya no acompaña a la elección, así que se repite aquí — y no cuesta nada: los vectores ya están codificados.\n'
        '\n'
        '**Qué responde.** El nDCG se mide sobre 8 consultas concretas, y una plantilla puede ganarlas por afinidad léxica con ellas. El Jaccard entre las tres formulaciones de cada intención mide otra cosa: si el sistema devuelve **los mismos productos cuando le preguntan lo mismo con otras palabras**. Eso es lo que se encuentra en producción, donde nadie escribe como el conjunto de desarrollo.\n'
        '\n'
        '**Cómo leerlo.** Si la ganadora de la regla también va bien aquí, la elección llega respaldada por dos medidas independientes —una con etiquetas y otra sin ellas—. Si va mal, hay un matiz que debe acompañar a la decisión en el informe.\n'
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

NOTEBOOKS = {
    "00_datos.ipynb": NB00_DATOS,
    "01_baseline.ipynb": NB01_BASELINE,
    "02_modelo.ipynb": NB02_MODELO,
    "03_representacion.ipynb": NB03_REPRESENTACION,
}
