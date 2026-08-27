"""Carga y perfilado de los ficheros de datos de Aurum Market.

Las funciones de este módulo producen la evidencia numérica que sustenta
las decisiones D01-D04 de NB00. No deciden nada por sí mismas: solo
calculan, y la decisión se toma y se registra aparte.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

ESCI_LABELS = ("E", "S", "C", "I")
BRAND_NORMALIZATION_MODES = ("raw", "casefold", "unaccent")
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

# (patrón ES/EN de la etiqueta, columna estructurada equivalente en el catálogo)
FIELD_LABEL_SPECS: dict[str, tuple[re.Pattern[str], str]] = {
    "marca": (re.compile(r"\b(?:marca|brand)\s*:", re.IGNORECASE), "brand"),
    "color": (re.compile(r"\b(?:color|colou?r)\s*:", re.IGNORECASE), "color"),
}


def load_csv(path: str | Path) -> pd.DataFrame:
    """Lee un CSV del contrato de datos; campos vacíos quedan como NaN."""
    return pd.read_csv(path)


def esci_label_counts_per_query(relevancias: pd.DataFrame) -> pd.DataFrame:
    """Cuenta juicios E/S/C/I por query_id y el denominador de Recall@10
    bajo cada definición de "relevante" (D01: solo E, o E+S)."""
    counts = (
        relevancias.groupby("query_id")["esci_label"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=list(ESCI_LABELS), fill_value=0)
    )
    counts["total_juicios"] = counts[list(ESCI_LABELS)].sum(axis=1)
    counts["relevantes_solo_E"] = counts["E"]
    counts["relevantes_E_mas_S"] = counts["E"] + counts["S"]
    return counts.reset_index()


def null_field_rates(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    """% de valores nulos por campo, sobre el total de filas de df."""
    n_rows = len(df)
    rows = [
        {
            "campo": field,
            "n_nulos": int(df[field].isna().sum()),
            "n_filas": n_rows,
            "pct_nulos": round(100 * df[field].isna().sum() / n_rows, 2),
        }
        for field in fields
    ]
    return pd.DataFrame(rows)


def strip_accents(text: str) -> str:
    """Quita las tildes sin tocar mayúsculas ni minúsculas.

    Vive aparte de `normalize_brand` porque hay un caso que necesita separar los
    dos ejes: generar cómo *escribiría* una persona un valor —con tilde o sin
    ella, en minúscula o en mayúscula— exige poder quitar la tilde conservando
    la caja. Mezclarlo todo en una función haría imposible esa variante."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_brand(value: object, mode: str) -> str | None:
    """Normaliza un valor de marca según D03: raw · casefold · unaccent."""
    if mode not in BRAND_NORMALIZATION_MODES:
        raise ValueError(f"mode debe ser uno de {BRAND_NORMALIZATION_MODES}")
    if pd.isna(value):
        return None
    text = str(value).strip()
    if mode == "raw":
        return text
    text = text.casefold()
    if mode == "casefold":
        return text
    return strip_accents(text)


def brand_normalization_collisions(
    df: pd.DataFrame, mode: str, brand_col: str = "brand"
) -> pd.DataFrame:
    """Marcas distintas (crudas) que colisionan en el mismo valor tras
    normalizar con `mode`. Evidencia real para D03, no una suposición."""
    brands = df[[brand_col]].dropna().drop_duplicates()
    brands["normalizada"] = brands[brand_col].apply(normalize_brand, mode=mode)
    grouped = (
        brands.groupby("normalizada")[brand_col]
        .apply(list)
        .reset_index(name="marcas_crudas")
    )
    grouped["n_marcas_crudas"] = grouped["marcas_crudas"].apply(len)
    return grouped[grouped["n_marcas_crudas"] > 1].reset_index(drop=True)


def text_field_label_summary(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Por cada campo de FIELD_LABEL_SPECS, % de filas cuyo `text_col`
    menciona su etiqueta (ES/EN, vía regex) y, de esas, cuántas tienen el
    campo estructurado equivalente vacío (posible ruido tipo "Marca: .").
    Evidencia real para D02, no una suposición sobre la plantilla."""
    text = df[text_col].fillna("")
    n_rows = len(df)
    rows = []
    for field, (pattern, column) in FIELD_LABEL_SPECS.items():
        has_label = text.str.contains(pattern)
        row = {
            "campo": field,
            "n_filas": n_rows,
            "n_con_etiqueta_en_text": int(has_label.sum()),
            "pct_con_etiqueta_en_text": round(100 * has_label.sum() / n_rows, 2),
        }
        if column in df.columns:
            row["n_etiqueta_con_campo_vacio"] = int((has_label & df[column].isna()).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def value_frequency(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Valores distintos de `column` (los vacíos como "(vacío)") con su
    frecuencia y % sobre el total de filas, de más a menos frecuente."""
    counts = df[column].fillna("(vacío)").value_counts()
    result = counts.rename_axis(column).reset_index(name="n_filas")
    result["pct_filas"] = round(100 * result["n_filas"] / len(df), 2)
    return result


def tokenize(text: object, *, strip_accents: bool = False) -> list[str]:
    """Tokenizador léxico compartido: minúsculas y secuencias alfanuméricas.

    Es el mismo criterio que aplicará el vectorizador del baseline (NB01),
    para que la evidencia de cobertura y el retriever cuenten los mismos
    términos. `strip_accents` replica `TfidfVectorizer(strip_accents=...)`.
    """
    if pd.isna(text):
        return []
    normalized = str(text).casefold()
    if strip_accents:
        decomposed = unicodedata.normalize("NFKD", normalized)
        normalized = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return TOKEN_PATTERN.findall(normalized)


def document_length_stats(
    df: pd.DataFrame, text_cols: list[str] | None = None
) -> pd.DataFrame:
    """Distribución de longitud en tokens por campo de texto.

    Evidencia para D05: BM25 normaliza la longitud explícitamente (parámetro
    `b`) mientras que TF-IDF solo la absorbe en la norma L2 del vector. Cuanto
    mayor es la dispersión (`cv`, `ratio_p95_p50`), más se separan ambos."""
    columns = text_cols or ["title", "text"]
    rows = []
    for column in columns:
        lengths = pd.Series(
            [len(tokenize(value)) for value in df[column]], dtype="float64"
        )
        rows.append(
            {
                "campo": column,
                "n_docs": len(df),
                "tokens_media": round(float(lengths.mean()), 1),
                "tokens_p50": int(lengths.quantile(0.50)),
                "tokens_p90": int(lengths.quantile(0.90)),
                "tokens_p95": int(lengths.quantile(0.95)),
                "tokens_max": int(lengths.max()),
                "cv": round(float(lengths.std() / lengths.mean()), 3),
                "ratio_p95_p50": round(
                    float(lengths.quantile(0.95) / max(lengths.quantile(0.50), 1)), 2
                ),
            }
        )
    return pd.DataFrame(rows)


def corpus_document_frequency(
    df: pd.DataFrame, text_col: str = "text", *, strip_accents: bool = False
) -> dict[str, int]:
    """Nº de documentos en los que aparece cada término del corpus."""
    frequency: dict[str, int] = {}
    for value in df[text_col]:
        for token in set(tokenize(value, strip_accents=strip_accents)):
            frequency[token] = frequency.get(token, 0) + 1
    return frequency


def query_term_coverage(
    consultas: pd.DataFrame,
    catalogo: pd.DataFrame,
    *,
    text_col: str = "text",
    query_col: str = "query_text",
    id_col: str = "query_id",
    strip_accents: bool = False,
) -> pd.DataFrame:
    """Por consulta, cuántos de sus términos existen en el corpus léxico.

    Evidencia para D05: un término que no está en el vocabulario no puede
    aportar señal a TF-IDF ni a BM25. `df_min` (la frecuencia documental del
    término más raro) indica cuánto discrimina la consulta: si es enorme,
    todos sus términos son comunes y el ranking léxico será ruidoso."""
    frequency = corpus_document_frequency(
        catalogo, text_col, strip_accents=strip_accents
    )
    rows = []
    for _, query in consultas.iterrows():
        tokens = tokenize(query[query_col], strip_accents=strip_accents)
        oov = [token for token in tokens if token not in frequency]
        found = [frequency[token] for token in tokens if token in frequency]
        rows.append(
            {
                id_col: query[id_col],
                query_col: query[query_col],
                "n_tokens": len(tokens),
                "n_en_corpus": len(found),
                "n_oov": len(oov),
                "tokens_oov": ", ".join(oov),
                "df_min": min(found) if found else 0,
                "df_max": max(found) if found else 0,
            }
        )
    return pd.DataFrame(rows)


def query_token_frequencies(
    consultas: pd.DataFrame,
    catalogo: pd.DataFrame,
    *,
    text_col: str = "text",
    query_col: str = "query_text",
    id_col: str = "query_id",
) -> pd.DataFrame:
    """Descompone cada consulta en sus términos y da la frecuencia documental
    de cada uno, respetando acentos y quitándolos.

    Es la vista desagregada de `query_term_coverage`: muestra a qué lista de
    documentos apunta cada palabra que escribe el usuario. Un `df` de 0 es un
    término que no existe; un `df` cercano al tamaño del corpus es un término
    que no discrimina; y un `df` muy pequeño recibe mucho peso IDF, para bien
    (término preciso) o para mal (errata o palabra irrelevante)."""
    con_tildes = corpus_document_frequency(catalogo, text_col, strip_accents=False)
    sin_tildes = corpus_document_frequency(catalogo, text_col, strip_accents=True)
    rows = []
    for _, query in consultas.iterrows():
        tokens = tokenize(query[query_col])
        tokens_sin_tildes = tokenize(query[query_col], strip_accents=True)
        for position, (token, token_sin_tildes) in enumerate(
            zip(tokens, tokens_sin_tildes), start=1
        ):
            rows.append(
                {
                    id_col: query[id_col],
                    "posicion": position,
                    "token": token,
                    "df_con_tildes": con_tildes.get(token, 0),
                    "df_sin_tildes": sin_tildes.get(token_sin_tildes, 0),
                }
            )
    return pd.DataFrame(rows)


def literal_match_ceiling(
    consultas: pd.DataFrame,
    relevancias: pd.DataFrame,
    catalogo: pd.DataFrame,
    *,
    relevant_labels: tuple[str, ...] = ("E", "S"),
    text_cols: list[str] | None = None,
    query_col: str = "query_text",
    id_col: str = "query_id",
    strip_accents: bool = False,
) -> pd.DataFrame:
    """% de productos relevantes que contienen **todos** los términos de la
    consulta, por campo.

    Es el techo de un emparejamiento literal (la opción "coincidencia exacta
    de título" de D05) y la medida directa del *vocabulary gap*: lo que ese
    porcentaje deja fuera es lo que ningún método léxico puede recuperar por
    coincidencia exacta."""
    columns = text_cols or ["title", "text"]
    relevantes = relevancias[relevancias["esci_label"].isin(relevant_labels)]
    catalogo_indexado = catalogo.set_index("product_id")
    rows = []
    for _, query in consultas.iterrows():
        tokens = set(tokenize(query[query_col], strip_accents=strip_accents))
        product_ids = relevantes[relevantes[id_col] == query[id_col]]["product_id"]
        presentes = catalogo_indexado.reindex(product_ids.unique()).dropna(how="all")
        row = {
            id_col: query[id_col],
            query_col: query[query_col],
            "n_relevantes": len(product_ids.unique()),
            "n_en_catalogo": len(presentes),
        }
        for column in columns:
            contiene_todos = [
                tokens.issubset(set(tokenize(value, strip_accents=strip_accents)))
                for value in presentes[column]
            ]
            n_contiene = int(sum(contiene_todos))
            row[f"n_todos_en_{column}"] = n_contiene
            row[f"pct_todos_en_{column}"] = (
                round(100 * n_contiene / len(presentes), 1) if len(presentes) else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def qrels_pool_sizes(relevancias: pd.DataFrame) -> pd.DataFrame:
    """Nº de product_id distintos juzgados por cada query_id (D04)."""
    return (
        relevancias.groupby("query_id")["product_id"]
        .nunique()
        .reset_index(name="pool_size")
    )


def qrels_coverage_in_catalog(
    relevancias: pd.DataFrame, catalog: pd.DataFrame, product_id_col: str = "product_id"
) -> pd.DataFrame:
    """Por query_id, cuántos de los product_id juzgados existen en `catalog`.
    Precondición para D04: si el motor busca sobre un catálogo que no
    contiene los productos juzgados, Recall/nDCG de desarrollo no son
    fiables sin importar qué universo de puntuación se elija."""
    catalog_ids = set(catalog[product_id_col])
    en_catalogo = relevancias[product_id_col].isin(catalog_ids)
    coverage = (
        relevancias.assign(en_catalogo=en_catalogo)
        .groupby("query_id")["en_catalogo"]
        .agg(n_juzgados="count", n_presentes="sum")
        .reset_index()
    )
    coverage["pct_presentes"] = round(100 * coverage["n_presentes"] / coverage["n_juzgados"], 2)
    return coverage


def relevant_field_nullity(
    relevancias: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    field: str,
    relevant_labels: Sequence[str] = ("E", "S"),
    product_id_col: str = "product_id",
) -> pd.DataFrame:
    """Porcentaje de productos relevantes de cada consulta con `field` vacío.

    Mide la **exposición** de una consulta a un cambio que solo afecta a las
    fichas donde ese campo falta. Es la variable de control que separa señal de
    perturbación: si rellenar el campo aportara información sobre él, las
    consultas más expuestas tendrían que ser las que más se mueven. Si el efecto
    aparece justo donde la exposición es baja —y no aparece donde es total—, lo
    que se está midiendo es ruido con buena presencia.

    Solo cuenta productos presentes en el catálogo: un juicio sobre un producto
    que no está indexado no puede influir en ningún ranking."""
    if field not in catalog.columns:
        raise ValueError(f"El catálogo no tiene la columna {field!r}.")

    vacio = catalog[field].isna() | (catalog[field].astype("string").str.strip() == "")
    sin_campo = set(catalog.loc[vacio, product_id_col])
    en_catalogo = set(catalog[product_id_col])

    relevantes = relevancias[relevancias["esci_label"].isin(relevant_labels)]
    filas = []
    for query_id, grupo in relevantes.groupby("query_id"):
        productos = [p for p in grupo[product_id_col] if p in en_catalogo]
        if not productos:
            continue
        n_sin = sum(p in sin_campo for p in productos)
        filas.append({
            "query_id": str(query_id),
            "n_relevantes": len(productos),
            f"n_sin_{field}": n_sin,
            f"pct_sin_{field}": round(100 * n_sin / len(productos), 1),
        })
    return pd.DataFrame(filas)
