"""Esquema y presupuesto del almacén vectorial (NB04).

Este módulo **no habla con ningún motor**. Calcula lo que hace falta saber
*antes* de elegirlo: qué ocupa el índice, qué cuesta cada payload y a cuántos
productos llega un filtro de marca según cómo se guarde. Es la evidencia de
D12–D15, no la decisión.

Tres cosas que este módulo deja explícitas porque el esquema las hereda de
decisiones anteriores y es fácil perderlas de vista:

- **El punto es `record_id`** (UUIDv5, lo impone `README_DATOS`). No se decide
  aquí, pero el presupuesto de payload se calcula por punto, no por producto,
  y coinciden porque D07 dejó la relación en 1:1.
- **La política de nulos del payload (D14) no es la del texto (D02).** Son
  decisiones distintas sobre datos distintos: una afecta a lo que se codifica,
  la otra a lo que se filtra. Por eso `build_payload` exige la política de forma
  explícita: sin valor por defecto, nadie la elige por descuido.
- **D03 guarda `brand` cruda y normaliza en búsqueda.** Un filtro nativo compara
  byte a byte, así que normalizar solo la consulta no basta: o el payload lleva
  también la clave normalizada, o el filtro falla con toda variante de escritura
  que no coincida exactamente. `brand_filter_reach` pone el número a ese "toda
  variante".
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

from .datos import BRAND_NORMALIZATION_MODES, normalize_brand, strip_accents
from .plantillas import NULL_PLACEHOLDER

BYTES_PER_FLOAT32 = 4
MIB = 1024 ** 2

# Cómo compara el motor el valor guardado con el que pide el usuario. `equals`
# es lo que basta para un campo con vocabulario cerrado; `contains` es lo que
# hace falta cuando el valor almacenado puede llevar varios dentro.
FILTER_MATCH_MODES = ("equals", "contains")

# Signos con los que un valor mete varios dentro: "Negro/Rojo", "Negro (Black)".
COMPOSITE_SEPARATORS = r"[/,;()]"

# Las claves derivadas de D03, que la sección B de NB04 convirtió en obligatorias:
# el filtro nativo compara contra el valor almacenado, así que sin ellas pierde el
# 91 % del alcance (B.4). No son una opción de D13 y por eso están fuera de los
# esquemas: entran en todos, incluido el mínimo. Un esquema "mínimo" que no
# pudiera filtrar no sería mínimo, sería inservible.
FILTER_KEYS: tuple[str, ...] = ("brand_normalized", "color_normalized")

# Opciones de D13, declaradas aquí y no en una celda para que la comparación sea
# auditable y para que el notebook no pueda inventarse una cuarta a mitad de
# tabla. `record_id` no aparece en ninguna: es el ID del punto, no payload.
#
# Lo que D13 decide de verdad es solo el último escalón: `minimo` lo descartan
# los notebooks posteriores —NB05 necesita `title`, NB08 `catalog_version` y
# `active`, NB07 `title` y `color`—, así que la elección real es si el índice
# guarda además el texto de origen.
PAYLOAD_SCHEMAS: dict[str, tuple[str, ...]] = {
    # Lo estrictamente necesario: `product_id` es lo que piden los CSV de salida
    # y `brand` es sobre lo que filtra el enunciado.
    "minimo": ("product_id", "brand", *FILTER_KEYS),
    # Añade lo que consumen NB05 (mostrar el resultado), NB08 (comprobar la
    # versión y la baja) y NB07 (similitud de título, coincidencia de color).
    "completo": (
        "product_id", "title", "brand", "color", "catalog_version", "active", *FILTER_KEYS,
    ),
    # Igual que `completo` pero guardando además el texto de origen: hace el
    # índice autosuficiente para reconstruirse sin el CSV, y es el único campo
    # que ningún notebook posterior necesita.
    "completo_con_text": (
        "product_id", "title", "brand", "color", "catalog_version", "active", "text",
        *FILTER_KEYS,
    ),
}

# Opciones de D14. `omitir_campo` deja el punto sin esa clave; las otras dos la
# escriben siempre, con contenido distinto. La diferencia no es de estilo: en la
# mayoría de motores `brand ausente` y `brand == ""` responden distinto al mismo
# filtro, y eso decide si un producto sin marca es inalcanzable o alcanzable.
NULL_POLICIES = ("omitir_campo", "cadena_vacia", "centinela")


def _is_empty(value: object) -> bool:
    """Vacío = nulo, o cadena que solo tiene espacios."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return bool(pd.isna(value))


def _as_json_scalar(value: object) -> object:
    """Valor tal y como viaja en el payload: texto sin espacios sobrantes,
    números y booleanos como tipos nativos."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (bool, int, float)):
        return value
    return str(value)


def build_payload(
    row: Mapping[str, Any],
    *,
    fields: Sequence[str],
    null_policy: str,
    sentinel: str = NULL_PLACEHOLDER,
) -> dict[str, Any]:
    """Payload de un punto: los `fields` de la fila bajo la política D14.

    `null_policy` no tiene valor por defecto a propósito. Es la decisión que el
    plan marca como consecuencia directa sobre los filtros de NB05, y un
    argumento con defecto la convertiría en algo que se hereda sin pensarlo.
    """
    if null_policy not in NULL_POLICIES:
        raise ValueError(f"null_policy debe ser una de {NULL_POLICIES}")

    payload: dict[str, Any] = {}
    for field in fields:
        value = row.get(field)
        if not _is_empty(value):
            payload[field] = _as_json_scalar(value)
        elif null_policy == "cadena_vacia":
            payload[field] = ""
        elif null_policy == "centinela":
            payload[field] = sentinel
        # `omitir_campo`: la clave no se escribe.
    return payload


def add_normalized_key(
    frame: pd.DataFrame,
    *,
    field: str,
    mode: str,
    target: str | None = None,
) -> pd.DataFrame:
    """Copia del catálogo con la clave de filtro derivada que exige D03.

    D03 decidió dos cosas que solo son compatibles con una clave derivada: el
    valor **se guarda crudo** y **se normaliza al buscar**. Un filtro nativo
    compara valores almacenados, no consultas, así que la normalización tiene
    que estar materializada en el punto para que el motor pueda ejecutarla.

    Vale para cualquier campo filtrable, no solo para `brand`: la política es
    de la decisión, no del campo, y tenerla en una sola función es lo que
    impide que dos campos filtrables acaben normalizándose distinto.
    """
    if field not in frame.columns:
        raise ValueError(f"El catálogo no tiene la columna {field!r}.")
    result = frame.copy()
    result[target or f"{field}_normalized"] = [
        None if _is_empty(value) else normalize_brand(value, mode)
        for value in result[field]
    ]
    return result


def _value_bytes(value: object) -> int:
    """Bytes UTF-8 del valor serializado en el payload."""
    if isinstance(value, bool):
        return len("true" if value else "false")
    return len(str(value).encode("utf-8"))


def payload_bytes(payload: Mapping[str, Any]) -> int:
    """Bytes del payload contando claves y valores.

    Las claves cuentan porque los motores guardan el payload por punto, no un
    esquema compartido: con 15.000 puntos, un nombre de campo largo se paga
    15.000 veces.
    """
    return sum(len(key.encode("utf-8")) + _value_bytes(value) for key, value in payload.items())


def field_byte_profile(frame: pd.DataFrame, fields: Sequence[str]) -> pd.DataFrame:
    """Por campo: cuántos vacíos tiene y cuánto ocupa guardarlo en el payload.

    Los vacíos son la evidencia de D14 (a cuántos puntos afecta la política) y
    los bytes la de D13 (qué se paga por llevar el campo).
    """
    faltan = [field for field in fields if field not in frame.columns]
    if faltan:
        raise ValueError(f"El catálogo no tiene las columnas: {faltan}")

    n_rows = len(frame)
    rows = []
    for field in fields:
        values = list(frame[field])
        vacios = sum(_is_empty(value) for value in values)
        longitudes = pd.Series([
            0 if _is_empty(value) else _value_bytes(_as_json_scalar(value))
            for value in values
        ])
        rows.append({
            "campo": field,
            "n_vacios": int(vacios),
            "pct_vacios": round(100 * vacios / n_rows, 2) if n_rows else 0.0,
            "bytes_medios": round(float(longitudes.mean()), 1) if n_rows else 0.0,
            "bytes_p95": int(longitudes.quantile(0.95)) if n_rows else 0,
            "bytes_max": int(longitudes.max()) if n_rows else 0,
            "mb_total": round(float(longitudes.sum()) / MIB, 2),
        })
    return pd.DataFrame(rows)


def payload_budget(
    frame: pd.DataFrame,
    *,
    null_policy: str,
    schemas: Mapping[str, Sequence[str]] | None = None,
    sentinel: str = NULL_PLACEHOLDER,
) -> pd.DataFrame:
    """Coste de cada esquema de D13 sobre el catálogo real, bajo una política D14.

    Depende de las dos decisiones a la vez: `omitir_campo` no escribe la clave
    del campo vacío y `centinela` escribe una palabra donde no había nada, así
    que el mismo esquema no ocupa lo mismo con una política que con otra.
    """
    catalogo = schemas if schemas is not None else PAYLOAD_SCHEMAS
    filas = frame.to_dict("records")
    rows = []
    for nombre, fields in catalogo.items():
        tamanos = pd.Series([
            payload_bytes(build_payload(
                fila, fields=fields, null_policy=null_policy, sentinel=sentinel,
            ))
            for fila in filas
        ])
        rows.append({
            "esquema": nombre,
            "n_campos": len(fields),
            "bytes_medios": round(float(tamanos.mean()), 1),
            "bytes_p95": int(tamanos.quantile(0.95)),
            "bytes_max": int(tamanos.max()),
            "mb_total": round(float(tamanos.sum()) / MIB, 2),
        })
    return pd.DataFrame(rows)


def index_footprint(
    n_points: int,
    dim: int,
    *,
    payload_bytes_medios: float = 0.0,
    bytes_por_escalar: int = BYTES_PER_FLOAT32,
) -> dict[str, float]:
    """Lo que ocupa el índice en crudo: vectores + payload, sin grafo ANN.

    Es un suelo, no una estimación del contenedor: HNSW añade sus enlaces
    (≈ `m × 2 × 4` bytes por punto) y el motor sus estructuras. Sirve para
    descartar lo inviable, no para prometer una cifra de RAM.
    """
    if n_points < 0 or dim < 1:
        raise ValueError("n_points debe ser >= 0 y dim >= 1.")
    vectores = n_points * dim * bytes_por_escalar
    payload = n_points * payload_bytes_medios
    return {
        "n_points": n_points,
        "dim": dim,
        "bytes_por_vector": dim * bytes_por_escalar,
        "mb_vectores": round(vectores / MIB, 2),
        "mb_payload": round(payload / MIB, 2),
        "mb_total": round((vectores + payload) / MIB, 2),
    }


# Tamaño máximo por defecto de un mensaje gRPC. Es el techo real de un lote,
# porque los tres candidatos de D12 hablan gRPC: superarlo no degrada, corta.
GRPC_MAX_MESSAGE_MIB = 4


def batch_footprint(
    dim: int,
    *,
    n_points: int,
    payload_bytes_medios: float = 0.0,
    batch_sizes: Iterable[int] = (64, 128, 256),
    bytes_por_escalar: int = BYTES_PER_FLOAT32,
    limite_mensaje_mib: int = GRPC_MAX_MESSAGE_MIB,
) -> pd.DataFrame:
    """Lo que decide D15 para cada tamaño de lote candidato.

    El plan planteaba esta decisión como una de memoria —"con 8 GB y el modelo
    cargado, mide RAM antes de subir"—, pero en NB04 **el modelo no se carga**:
    los vectores vienen ya calculados de disco. Con 3.072 bytes por vector, el
    lote más grande de la lista ronda el megabyte y la RAM deja de decidir nada.

    Lo que sí decide, y es lo que devuelve esta tabla:

    - `mb_por_lote` y `pct_del_limite`: un lote que se acerque al máximo de
      mensaje de gRPC no se ralentiza, **falla**. Es el único techo duro, y sube
      con el esquema de payload que elija D13 — por eso las dos van juntas.
    - `n_lotes`: viajes de red. Menos lotes es menos ida y vuelta.
    - `puntos_reintentados_si_falla`: cuánto trabajo se pierde cuando un lote se
      cae. Un lote grande reintenta más. Es el precio del viaje ahorrado.

    Sigue siendo un suelo, no una promesa: el cliente serializa y el motor
    responde, así que el pico real del proceso es mayor.
    """
    rows = []
    limite_bytes = limite_mensaje_mib * MIB
    for size in batch_sizes:
        if size < 1:
            raise ValueError("Los tamaños de lote deben ser >= 1.")
        bytes_lote = size * (dim * bytes_por_escalar + payload_bytes_medios)
        rows.append({
            "lote": size,
            "mb_por_lote": round(bytes_lote / MIB, 2),
            "pct_del_limite": round(100 * bytes_lote / limite_bytes, 1),
            "n_lotes": -(-n_points // size),  # techo de la división
            "puntos_reintentados_si_falla": size,
            "cabe_en_un_mensaje": bytes_lote <= limite_bytes,
        })
    return pd.DataFrame(rows)


def writing_variants(value: str) -> dict[str, str]:
    """Las formas en que una misma consulta puede llegar escrita.

    Un filtro no lo escribe el catálogo, lo escribe una persona, y cada persona
    lo escribe distinto: todo en minúscula, con la inicial en mayúscula, a
    gritos, y con o sin la tilde que le corresponda. Suponer una sola forma
    —normalmente la minúscula— mete una hipótesis en el experimento sin
    declararla, y encima la más favorable a normalizar.

    Las variantes acentuadas solo se generan si el valor lleva tilde: no se
    puede adivinar dónde pondría una tilde quien escribe `azul`, pero sí
    quitársela a quien escribe `marrón`.
    """
    base = str(value).strip()
    if not base:
        raise ValueError("No hay valor del que generar variantes.")

    sin_tilde = strip_accents(base)
    variantes = {
        "minusculas": sin_tilde.lower(),
        "Capitalizada": sin_tilde.capitalize(),
        "MAYUSCULAS": sin_tilde.upper(),
    }
    if sin_tilde != base:
        variantes.update({
            "minusculas_con_tilde": base.lower(),
            "Capitalizada_con_tilde": base.capitalize(),
            "MAYUSCULAS_con_tilde": base.upper(),
        })
    return variantes


def filter_writing_robustness(
    frame: pd.DataFrame,
    value: str,
    *,
    field: str = "brand",
    match: str = "equals",
    modes: Sequence[str] = BRAND_NORMALIZATION_MODES,
) -> pd.DataFrame:
    """¿Cambia lo que encuentra el filtro según cómo lo escriba el usuario?

    Es la pregunta que decide si la clave normalizada hace falta, y se responde
    cruzando cada forma de escribir con cada modo de normalización. Una
    configuración robusta devuelve **lo mismo** escriba quien escriba; una
    frágil convierte el filtro en una lotería de mayúsculas.
    """
    rows = []
    for variante, escritura in writing_variants(value).items():
        alcance = filter_reach(
            frame, [escritura], field=field, match=match, modes=modes, ejemplos=0
        )
        for _, fila in alcance.iterrows():
            rows.append({
                "variante": variante,
                "escritura": escritura,
                "modo": fila["modo"],
                "n_productos": int(fila["n_productos"]),
            })
    return pd.DataFrame(rows)


def robustness_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Veredicto por modo, en dos ejes que es fácil confundir en uno.

    - **Consistente**: todas las formas de escribirlo encuentran lo mismo. Es lo
      mínimo exigible a un filtro que se le ofrece a una persona; lo contrario
      es una lotería de mayúsculas.
    - **Alcanza el máximo**: además encuentra tanto como el mejor modo de la
      tabla. Un modo puede ser perfectamente consistente y perfectamente
      incompleto —que todo el mundo encuentre la mitad también es consistente—,
      y ese caso se confundiría con el bueno si solo se mirara el primer eje.

    El techo es el mejor alcance observado entre los modos comparados, no un
    absoluto: con un solo modo en la tabla, el eje de completitud no dice nada.
    """
    resumen = (
        table.groupby("modo")["n_productos"]
        .agg(n_variantes="size", alcances_distintos="nunique", minimo="min", maximo="max")
        .reset_index()
    )
    resumen["consistente"] = resumen["alcances_distintos"] == 1
    techo = int(resumen["maximo"].max()) if len(resumen) else 0
    resumen["alcanza_el_maximo"] = resumen["maximo"] == techo
    resumen["pct_del_maximo"] = (
        (100 * resumen["maximo"] / techo).round(1) if techo else 0.0
    )
    return resumen


def filter_field_profile(frame: pd.DataFrame, fields: Sequence[str]) -> pd.DataFrame:
    """¿Este campo es una taxonomía o es texto libre que alguien rellenó?

    La pregunta no es cosmética: decide si un filtro de igualdad tiene sentido.
    Sobre una taxonomía —una lista corta de valores repetidos— la igualdad
    encuentra lo que hay. Sobre texto libre, la mayoría de los valores aparecen
    una sola vez, muchos son compuestos (`"Negro/Rojo"`) o de dos palabras
    (`"azul marino"`), y la igualdad deja fuera todo lo que no se escribió
    exactamente igual.

    - `pct_valores_unicos`: qué parte de los valores distintos aparece en un
      solo producto. Alto = cola larga = texto libre.
    - `pct_compuestos`: productos cuyo valor lleva un separador dentro.
    - `pct_multipalabra`: productos con dos o más palabras y sin separador.
    """
    faltan = [field for field in fields if field not in frame.columns]
    if faltan:
        raise ValueError(f"El catálogo no tiene las columnas: {faltan}")

    n_rows = len(frame)
    rows = []
    for field in fields:
        valores = frame[field].dropna().map(lambda v: str(v).strip())
        valores = valores[valores != ""]
        distintos = valores.value_counts()
        compuestos = valores.str.contains(COMPOSITE_SEPARATORS, regex=True)
        multipalabra = ~compuestos & valores.str.contains(r"\s", regex=True)
        rows.append({
            "campo": field,
            "n_con_valor": int(len(valores)),
            "pct_vacios": round(100 * (1 - len(valores) / n_rows), 2) if n_rows else 0.0,
            "n_valores_distintos": int(len(distintos)),
            "pct_valores_unicos": (
                round(100 * float((distintos == 1).mean()), 1) if len(distintos) else 0.0
            ),
            "pct_compuestos": round(100 * float(compuestos.mean()), 1) if len(valores) else 0.0,
            "pct_multipalabra": (
                round(100 * float(multipalabra.mean()), 1) if len(valores) else 0.0
            ),
        })
    return pd.DataFrame(rows)


def _matches(normalizadas: pd.Series, objetivo: str, match: str) -> pd.Series:
    if match == "equals":
        return normalizadas == objetivo
    return normalizadas.fillna("").str.contains(re.escape(objetivo), regex=True)


def _as_whole_word(valor: str, objetivo: str) -> bool:
    """¿El objetivo aparece como palabra suelta y no dentro de otra?"""
    patron = rf"(?:^|[^0-9a-z]){re.escape(objetivo)}(?:[^0-9a-z]|$)"
    return re.search(patron, valor) is not None


def filter_reach(
    frame: pd.DataFrame,
    values: Iterable[str],
    *,
    field: str = "brand",
    match: str = "equals",
    modes: Sequence[str] = BRAND_NORMALIZATION_MODES,
    ejemplos: int = 5,
) -> pd.DataFrame:
    """A cuántos productos llega un filtro, modo de normalización a modo.

    Ejecuta el filtro como lo ejecutaría el motor —sobre el valor almacenado—
    después de normalizar consulta y catálogo igual. La diferencia entre modos
    es lo que se gana o se pierde materializando la clave normalizada en el
    payload; la diferencia entre `equals` y `contains`, lo que se gana o se
    pierde exigiendo que el valor coincida entero.

    Con `contains` se informa además de `n_dentro_de_otra_palabra`: productos que
    entran porque el texto buscado aparece **dentro de otra palabra**, no como
    palabra suelta —`rosado` cuando se pidió `rosa`—. Es la superficie de falsos
    positivos de esa política: **cuanto más alto, peor**, y conviene mirarlo antes
    de adoptarla y no después de que alguien se queje.
    """
    if match not in FILTER_MATCH_MODES:
        raise ValueError(f"match debe ser uno de {FILTER_MATCH_MODES}")
    if field not in frame.columns:
        raise ValueError(f"El catálogo no tiene la columna {field!r}.")

    columna = frame[field]
    rows = []
    for mode in modes:
        normalizadas = pd.Series(
            [normalize_brand(value, mode) for value in columna], index=columna.index
        )
        for value in values:
            objetivo = normalize_brand(value, mode)
            casan = _matches(normalizadas, objetivo or "", match)
            crudos = sorted(set(columna[casan].dropna()))
            fila = {
                "filtro": value,
                "campo": field,
                "modo": mode,
                "match": match,
                "n_productos": int(casan.sum()),
                "n_valores_distintos": len(crudos),
                "valores": ", ".join(crudos[:ejemplos]),
            }
            if match == "contains":
                # Contado a mano y no con `.sum()` sobre una serie booleana: si el
                # filtro no encuentra nada, esa serie queda vacía y con tipo de
                # texto, y sumarla concatena en vez de sumar. El total sale como
                # cadena vacía y convertirlo a entero revienta — un fallo que solo
                # aparece en el caso en que no hay resultados.
                fila["n_dentro_de_otra_palabra"] = sum(
                    1
                    for encontrado in normalizadas[casan].fillna("")
                    if not _as_whole_word(str(encontrado), objetivo or "")
                )
            rows.append(fila)
    return pd.DataFrame(rows)


def combined_filter_selectivity(
    frame: pd.DataFrame,
    *,
    primary_values: Iterable[str],
    secondary_values: Iterable[str],
    primary_field: str = "brand",
    secondary_field: str = "color",
    mode: str = "unaccent",
    primary_match: str = "equals",
    secondary_match: str = "contains",
) -> pd.DataFrame:
    """Cuántos productos sobreviven a dos filtros a la vez, y por qué.

    Es el escenario que decide si vale filtrar *después* de recuperar: una marca
    sola ya es selectiva, y una marca **con** un color concreto puede no dejar
    ningún candidato entre los diez vecinos.

    El cero de ese cruce es ambiguo si se mira solo. Puede significar que la
    marca no tiene ningún producto de ese color —un hecho del catálogo— o que
    sus productos no tienen el color anotado —un hecho de la cobertura del dato,
    que cambia en cuanto alguien rellene la columna—. Por eso la tabla parte los
    productos de la marca en los que **tienen** el segundo campo y los que no, y
    marca explícitamente el caso en que el cero no dice nada del catálogo.

    Los dos campos se comparan normalizados con la misma política de D03, igual
    que `filter_reach`: filtrar el primero contra el valor crudo funcionaría solo
    mientras la consulta se escriba exactamente como está almacenada.
    """
    for field in (primary_field, secondary_field):
        if field not in frame.columns:
            raise ValueError(f"El catálogo no tiene la columna {field!r}.")
    for match in (primary_match, secondary_match):
        if match not in FILTER_MATCH_MODES:
            raise ValueError(f"match debe ser uno de {FILTER_MATCH_MODES}")

    col_primario = f"n_{primary_field}"
    col_con_dato = f"n_{primary_field}_con_{secondary_field}"
    col_sin_dato = f"n_{primary_field}_sin_{secondary_field}"
    col_ambos = f"n_{primary_field}_y_{secondary_field}"

    primarias = pd.Series(
        [normalize_brand(value, mode) for value in frame[primary_field]],
        index=frame.index, dtype=object,
    )
    secundarias = pd.Series(
        [normalize_brand(value, mode) for value in frame[secondary_field]],
        index=frame.index, dtype=object,
    )
    tiene_dato = pd.Series(
        [not _is_empty(value) for value in frame[secondary_field]],
        index=frame.index, dtype=bool,
    )

    n_total = len(frame)
    rows = []
    for primary in primary_values:
        del_primario = _matches(primarias, normalize_brand(primary, mode) or "", primary_match)
        n_primario = int(del_primario.sum())
        n_con_dato = int((del_primario & tiene_dato).sum())
        for secondary in secondary_values:
            casan = del_primario & _matches(
                secundarias, normalize_brand(secondary, mode) or "", secondary_match
            )
            n_ambos = int(casan.sum())
            rows.append({
                primary_field: primary,
                secondary_field: secondary,
                col_primario: n_primario,
                col_con_dato: n_con_dato,
                col_sin_dato: n_primario - n_con_dato,
                col_ambos: n_ambos,
                "pct_del_catalogo": round(100 * n_ambos / n_total, 3) if n_total else 0.0,
                # El cero que no es un hecho del catálogo: la marca tiene
                # productos, pero ninguno con el segundo campo relleno. Nada se
                # puede concluir de él, y mañana puede ser otro sin que cambie
                # ni el motor ni el filtro.
                "cero_por_falta_de_dato": n_ambos == 0 and n_con_dato == 0 and n_primario > 0,
            })
    return pd.DataFrame(rows)
