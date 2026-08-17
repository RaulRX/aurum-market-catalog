"""Plantillas de texto: qué se codifica de cada producto (familia A de NB03).

NB02 congeló la plantilla en **A0** —la columna `text` tal cual— para que el
único factor variable fuera el modelo (Regla 1). Aquí se invierte el montaje: el
modelo queda congelado en el ganador de NB02 y lo único que cambia es el texto.

La pregunta que responden estas plantillas no es de formato sino de contenido.
Un embedding resume el significado de todo lo que entra, así que 3.000
caracteres de prosa comercial —*"perfecto para regalo, calidad premium"*— acercan
el vector al lenguaje genérico de cualquier producto y lo alejan de lo que hace
distinto a *este*. Menos texto puede recuperar mejor, y eso se mide.

**Política de nulos (D02).** Un campo vacío no aporta sección: nada de
`"Color: ."` ni de `"Color: desconocido"`. Se decidió así porque el `text` de
origen ya lo hace —cuando `brand` falta, el texto nunca dice `"Marca: ."`, cero
casos sobre 1.500— y porque insertar un literal compartido en el 36,6 % del
catálogo crearía una señal común artificial: productos que se acercan por
compartir la palabra "desconocido", no por parecerse.

Ese razonamiento era sólido pero **no estaba medido**, que es justo lo que §3.1
del enunciado no da por bueno. Por eso existe `A3n`: misma receta que `A3` con
la política de nulos invertida. Comparar `A3` contra `A3n` pone un número a la
hipótesis de contaminación en lugar de dejarla en argumento.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

# Etiqueta con la que cada campo estructurado entra en las plantillas con
# etiquetas. Reproduce la nomenclatura del `text` de origen, para que A3 y A0
# hablen el mismo idioma y la comparación mida el contenido, no el vocabulario.
FIELD_LABELS = {"brand": "Marca", "color": "Color"}

# Relleno de A3n. Solo esa plantilla lo usa: es el literal cuya contaminación se
# quiere medir.
NULL_PLACEHOLDER = "desconocido"

REQUIRED_COLUMNS = ("title", "brand", "color", "text")


@dataclass(frozen=True, slots=True)
class CorpusContext:
    """Parámetros de plantilla que salen del propio corpus, no de una constante.

    Existe por A4. Un número escrito a mano —512, pongamos— sería una decisión
    de diseño disfrazada de detalle de implementación: nadie podría justificar
    por qué 512 y no 400, y el corte dejaría de tener sentido en cuanto cambiara
    el catálogo. Derivarlo de los datos hace la plantilla reproducible y
    defendible."""

    a4_chars: int


def corpus_context(frame: pd.DataFrame) -> CorpusContext:
    """Calcula los parámetros de plantilla que dependen del corpus.

    A4 recorta por la **mediana** de `text`, no por la media. La distribución
    está sesgada a la derecha y topada en 3.000 caracteres, así que la media se
    va por encima de lo típico y apenas tocaría a cuatro de cada diez fichas. La
    mediana parte el catálogo en dos mitades exactas y convierte A4 en una
    pregunta nítida: **¿sobra la mitad más larga de cada ficha?**"""
    if "text" not in frame.columns:
        raise ValueError("El catálogo necesita la columna 'text'.")
    lengths = frame["text"].fillna("").str.len()
    if lengths.empty:
        raise ValueError("El catálogo no tiene filas.")
    return CorpusContext(a4_chars=max(1, int(round(float(lengths.median())))))


def _clean(value: object) -> str:
    """Valor del campo como texto, o cadena vacía si es nulo o solo espacios."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value).strip()


def truncate_on_word_boundary(text: str, max_chars: int) -> str:
    """Recorta a `max_chars` sin partir la última palabra.

    Cortar a mitad de palabra fabrica subpalabras que no existen (`"resiste"` →
    `"resis"`), y el tokenizador las parte en piezas raras que no aparecían en el
    entrenamiento. El recorte limpio es la diferencia entre medir *"qué pasa si
    doy menos texto"* y medir *"qué pasa si doy texto roto"*."""
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError("max_chars debe ser un entero positivo.")
    if len(text) <= max_chars:
        return text
    corte = text[:max_chars]
    espacio = corte.rfind(" ")
    return (corte[:espacio] if espacio > 0 else corte).rstrip(" ,.;:-")


def compose_product_text(
    row: Mapping[str, Any],
    *,
    fields: Sequence[str] = ("brand", "color"),
    labels: bool = True,
    include_nulls: bool = False,
    null_placeholder: str = NULL_PLACEHOLDER,
) -> str:
    """Compone el texto de un producto a partir del título y los campos pedidos.

    `include_nulls=False` es **D02**: el campo vacío no aporta sección.
    Ponerlo a `True` es la variante `A3n`, que existe para medir si esa política
    era necesaria o solo prudente."""
    if "title" not in row:
        raise KeyError("La fila no tiene columna 'title'.")

    partes = [_clean(row["title"])]
    for field in fields:
        valor = _clean(row.get(field))
        if not valor:
            if not include_nulls:
                continue
            valor = null_placeholder
        partes.append(f"{FIELD_LABELS.get(field, field)}: {valor}" if labels else valor)

    separador = ". " if labels else " "
    return separador.join(parte for parte in partes if parte)


# Registro de plantillas. Cada una recibe una fila y el contexto del corpus, y
# devuelve el texto que se codificará. Están aquí y no en el notebook porque son
# el objeto de estudio de NB03: tienen que poder probarse sin red y sin modelo.
#
# Todas aceptan `ctx` aunque casi ninguna lo use: la firma uniforme deja escrito
# en el tipo que una plantilla *puede* depender del corpus, en vez de esconder
# esa dependencia dentro de A4.
TEMPLATES: dict[str, Callable[[Mapping[str, Any], CorpusContext], str]] = {
    # La columna del dataset tal cual. Es la que NB02 tuvo congelada.
    "A0": lambda row, ctx: _clean(row["text"]),
    # Solo el título: el extremo opuesto, y el único que nunca se trunca.
    "A1": lambda row, ctx: _clean(row["title"]),
    # Los campos estructurados sin etiquetas, para aislar si lo que aporta A3
    # es la información o la nomenclatura.
    "A2": lambda row, ctx: compose_product_text(row, labels=False),
    # Con etiquetas de campo, omitiendo los vacíos (D02).
    "A3": lambda row, ctx: compose_product_text(row),
    # Igual que A3 pero rellenando los vacíos: el control de D02.
    "A3n": lambda row, ctx: compose_product_text(row, include_nulls=True),
    # Recorte de `text` por la mediana del corpus: separa señal de relleno
    # comercial sin que el punto de corte lo elija nadie a dedo.
    "A4": lambda row, ctx: truncate_on_word_boundary(_clean(row["text"]), ctx.a4_chars),
    # A3 sin `color`, el campo con un 36,6 % de vacíos.
    "A5": lambda row, ctx: compose_product_text(row, fields=("brand",)),
}


# Plantillas que existen para poner a prueba una decisión ya tomada, no para
# competir por ser la elegida. Se codifican y se miden igual que el resto —sin
# eso no habría con qué comparar—, pero quedan fuera del conjunto de candidatas
# sobre el que se aplica la regla de elección.
#
# Declararlo aquí, y no con un filtro escrito a mano en una celda, es lo que
# permite que la exclusión sea auditable: el papel de A3n está fijado junto a su
# definición, no decidido al ver la tabla de resultados.
CONTROLES = frozenset({"A3n"})


def candidatas() -> list[str]:
    """Plantillas que compiten por ser la elegida: todas menos los controles."""
    return [nombre for nombre in TEMPLATES if nombre not in CONTROLES]


def render_template(
    frame: pd.DataFrame, template: str, *, context: CorpusContext | None = None
) -> list[str]:
    """Aplica una plantilla a todo el catálogo y devuelve los textos a codificar.

    El contexto se calcula del propio `frame` si no se pasa. Se admite
    inyectarlo para poder renderizar un subconjunto —una fila de ejemplo, un
    lote— con el corte derivado del catálogo entero, que es el que de verdad
    se va a codificar."""
    if template not in TEMPLATES:
        raise ValueError(f"Plantilla desconocida: {template!r}. Hay {sorted(TEMPLATES)}.")
    faltan = set(REQUIRED_COLUMNS) - set(frame.columns)
    if faltan:
        raise ValueError(f"Al catálogo le faltan columnas: {sorted(faltan)}")
    ctx = context if context is not None else corpus_context(frame)
    render = TEMPLATES[template]
    return [render(fila, ctx) for fila in frame.to_dict("records")]


def template_stats(
    frame: pd.DataFrame, templates: Iterable[str] | None = None
) -> pd.DataFrame:
    """Longitud en caracteres de cada plantilla sobre el mismo catálogo.

    Se mide **antes** de codificar: si dos plantillas producen textos casi
    idénticos, la diferencia de nDCG que salga después será ruido, no efecto, y
    conviene saberlo antes de interpretar la tabla."""
    nombres = list(templates) if templates is not None else list(TEMPLATES)
    ctx = corpus_context(frame)
    filas = []
    for nombre in nombres:
        textos = render_template(frame, nombre, context=ctx)
        longitudes = pd.Series([len(texto) for texto in textos])
        filas.append({
            "plantilla": nombre,
            "n_docs": len(textos),
            "chars_media": round(float(longitudes.mean()), 1),
            "chars_p50": int(longitudes.median()),
            "chars_p90": int(longitudes.quantile(0.9)),
            "chars_max": int(longitudes.max()),
            "n_vacios": int((longitudes == 0).sum()),
            "pct_vs_A0": None,
        })
    tabla = pd.DataFrame(filas)
    if "A0" in tabla["plantilla"].values:
        base = float(tabla.loc[tabla["plantilla"] == "A0", "chars_media"].iloc[0])
        tabla["pct_vs_A0"] = (100 * tabla["chars_media"] / base).round(1)
    return tabla
