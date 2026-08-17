"""Métricas de recuperación para Aurum Market.

Adaptado de `sesion_01/.../evaluation.py`, con dos cambios que impone este
proyecto y que no se pueden heredar de la sesión:

1. **El contrato de relevancia es `E=3, S=2, C=1, I=0`** (lo fija
   `manifest.json`, campo `selection.relevance_mapping`). La sesión usaba
   `E=1, S=0.1, C=0.01`, que aquí sería un error de métrica.
2. **D01** (`E+S` cuenta como relevante para Recall@10 y MRR@10) se
   materializa en `RELEVANCE_THRESHOLD = 2.0`; **D04** (un producto
   recuperado sin juicio puntúa 0) es el comportamiento por defecto al
   consultar los qrels con `.get(id, 0.0)`.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean

import pandas as pd

# manifest.json → selection.relevance_mapping
ESCI_RELEVANCE = {"E": 3.0, "S": 2.0, "C": 1.0, "I": 0.0}

# D01: relevante para Recall@10 / MRR@10 == Exact o Substitute == relevancia >= 2
RELEVANCE_THRESHOLD = 2.0

GAIN_MODES = ("exponential", "linear")

Qrels = Mapping[str, float]


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    """Métricas de una sola consulta."""

    query_id: str
    k: int
    precision: float
    recall: float
    mrr: float
    ndcg: float

    def as_row(self) -> dict[str, str | int | float]:
        """Fila plana, lista para una tabla o un JSON de artefacto."""
        return {
            "query_id": self.query_id,
            f"precision@{self.k}": round(self.precision, 4),
            f"recall@{self.k}": round(self.recall, 4),
            f"mrr@{self.k}": round(self.mrr, 4),
            f"ndcg@{self.k}": round(self.ndcg, 4),
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Métricas por consulta y su media macro."""

    k: int
    per_query: tuple[QueryMetrics, ...]
    mean_precision: float
    mean_recall: float
    mean_mrr: float
    mean_ndcg: float

    @property
    def summary(self) -> dict[str, float]:
        """Medias macro con las etiquetas `metrica_at_k` del `README_DATOS`."""
        return {
            f"precision_at_{self.k}": round(self.mean_precision, 4),
            f"recall_at_{self.k}": round(self.mean_recall, 4),
            f"mrr_at_{self.k}": round(self.mean_mrr, 4),
            f"ndcg_at_{self.k}": round(self.mean_ndcg, 4),
        }

    def per_query_frame(self) -> pd.DataFrame:
        """Tabla por consulta — el plan exige reportarla, no solo la media."""
        return pd.DataFrame([metrics.as_row() for metrics in self.per_query])


def qrels_from_judgements(
    relevancias: pd.DataFrame,
    *,
    query_id_col: str = "query_id",
    document_id_col: str = "product_id",
    label_col: str = "esci_label",
) -> dict[str, dict[str, float]]:
    """Convierte `relevancias_desarrollo.csv` en qrels graduados por consulta.

    Las etiquetas se traducen con `ESCI_RELEVANCE`, no con la columna
    `relevance` del CSV: así el contrato queda declarado en el código y una
    incoherencia en el fichero de entrada se detecta en vez de propagarse."""
    unknown = set(relevancias[label_col]) - set(ESCI_RELEVANCE)
    if unknown:
        raise ValueError(f"Etiquetas ESCI desconocidas: {sorted(unknown)}")
    qrels: dict[str, dict[str, float]] = {}
    for row in relevancias.itertuples(index=False):
        query_id = str(getattr(row, query_id_col))
        document_id = str(getattr(row, document_id_col))
        qrels.setdefault(query_id, {})[document_id] = ESCI_RELEVANCE[
            getattr(row, label_col)
        ]
    return qrels


def _top_k(ranking: Sequence[str], k: int) -> tuple[str, ...]:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k debe ser un entero positivo.")
    document_ids = tuple(str(item) for item in ranking[:k])
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("El ranking contiene document_ids duplicados en el top-k.")
    return document_ids


def precision_at_k(ranking: Sequence[str], qrels: Qrels, *, k: int) -> float:
    """Aciertos relevantes dividido por `k`.

    Se divide por el corte pedido y no por lo devuelto: así un sistema que
    entrega menos de `k` candidatos no sale beneficiado."""
    hits = sum(qrels.get(doc, 0.0) >= RELEVANCE_THRESHOLD for doc in _top_k(ranking, k))
    return hits / k


def recall_at_k(ranking: Sequence[str], qrels: Qrels, *, k: int) -> float:
    """Fracción de los relevantes juzgados (D01: E+S) recuperada en el top-k."""
    relevant = {doc for doc, rel in qrels.items() if rel >= RELEVANCE_THRESHOLD}
    if not relevant:
        return 0.0
    return len(relevant.intersection(_top_k(ranking, k))) / len(relevant)


def mrr_at_k(ranking: Sequence[str], qrels: Qrels, *, k: int) -> float:
    """Inverso de la posición del primer relevante; 0 si no hay ninguno."""
    for rank, document_id in enumerate(_top_k(ranking, k), start=1):
        if qrels.get(document_id, 0.0) >= RELEVANCE_THRESHOLD:
            return 1.0 / rank
    return 0.0


def _gain(relevance: float, mode: str) -> float:
    if mode == "exponential":
        return 2.0**relevance - 1.0
    return relevance


def _dcg(relevances: Sequence[float], mode: str) -> float:
    return sum(
        _gain(relevance, mode) / math.log2(rank + 1.0)
        for rank, relevance in enumerate(relevances, start=1)
    )


def ndcg_at_k(
    ranking: Sequence[str], qrels: Qrels, *, k: int, gain: str = "exponential"
) -> float:
    """nDCG con relevancia graduada `E=3, S=2, C=1, I=0`.

    `gain="exponential"` usa `2^rel - 1` (convención de las sesiones y de
    TREC); `gain="linear"` usa la relevancia tal cual. La elección rara vez
    cambia el orden de los sistemas comparados, pero sí el valor reportado:
    se declara una vez y no se cambia entre experimentos."""
    if gain not in GAIN_MODES:
        raise ValueError(f"gain debe ser uno de {GAIN_MODES}")
    observed = [qrels.get(doc, 0.0) for doc in _top_k(ranking, k)]
    ideal = sorted(qrels.values(), reverse=True)[:k]
    ideal_dcg = _dcg(ideal, gain)
    if ideal_dcg == 0:
        return 0.0
    return _dcg(observed, gain) / ideal_dcg


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    qrels_by_query: Mapping[str, Qrels],
    *,
    k: int = 10,
    gain: str = "exponential",
) -> EvaluationReport:
    """Evalúa todas las consultas con juicios y promedia en macro.

    Una consulta sin ranking se evalúa como ranking vacío (una ejecución
    incompleta no debe parecer mejor de lo que es), y un ranking para una
    consulta sin juicios es un error: casi siempre significa que los IDs no
    casan."""
    unknown = set(rankings) - set(qrels_by_query)
    if unknown:
        raise ValueError(f"Hay rankings sin qrels para: {sorted(unknown)}")

    per_query = tuple(
        QueryMetrics(
            query_id=query_id,
            k=k,
            precision=precision_at_k(rankings.get(query_id, ()), query_qrels, k=k),
            recall=recall_at_k(rankings.get(query_id, ()), query_qrels, k=k),
            mrr=mrr_at_k(rankings.get(query_id, ()), query_qrels, k=k),
            ndcg=ndcg_at_k(rankings.get(query_id, ()), query_qrels, k=k, gain=gain),
        )
        for query_id, query_qrels in sorted(qrels_by_query.items())
    )
    return EvaluationReport(
        k=k,
        per_query=per_query,
        mean_precision=fmean(item.precision for item in per_query),
        mean_recall=fmean(item.recall for item in per_query),
        mean_mrr=fmean(item.mrr for item in per_query),
        mean_ndcg=fmean(item.ndcg for item in per_query),
    )


def apply_tolerance_rule(
    tabla: pd.DataFrame,
    *,
    metrica: str = "ndcg_at_10",
    tolerancia: float = 0.02,
    coste: str = "dim",
    desempates: Sequence[str] = ("segundos",),
) -> pd.DataFrame:
    """Aplica la regla D09b: dentro de la tolerancia, gana la más barata.

    La regla se fijó en `config/config.yaml` **antes** de ver ninguna métrica,
    y ejecutarla como código en vez de a ojo es lo que hace verificable esa
    afirmación: el ganador sale de una función determinista sobre la tabla, no
    de una lectura interesada de ella.

    El procedimiento es literalmente el del fichero de configuración:

    1. `B` = mejor valor de `metrica` en toda la tabla.
    2. Son **admisibles** las filas con `metrica >= B - tolerancia`, porque con
       8 consultas una diferencia menor no distingue dos sistemas.
    3. Entre las admisibles gana la de menor `coste` (la dimensión: menos
       memoria en el motor y menos ancho de banda por consulta).
    4. A igualdad de coste, la de mayor `metrica`; después, los `desempates`
       en orden ascendente.

    Devuelve la tabla con `admisible` y `posicion_regla`, ordenada de forma que
    la primera fila es la ganadora."""
    faltan = {metrica, coste, *desempates} - set(tabla.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en la tabla: {sorted(faltan)}")
    if tabla.empty:
        raise ValueError("La tabla de candidatos está vacía.")

    resultado = tabla.copy()
    mejor = float(resultado[metrica].max())
    resultado["admisible"] = resultado[metrica] >= mejor - tolerancia
    # `admisible` primero (descendente: True antes que False), luego coste
    # ascendente, métrica descendente y los desempates ascendentes.
    columnas = ["admisible", coste, metrica, *desempates]
    ascendente = [False, True, False, *[True] * len(desempates)]
    resultado = resultado.sort_values(columnas, ascending=ascendente).reset_index(drop=True)
    resultado["posicion_regla"] = range(1, len(resultado) + 1)
    return resultado


def jaccard_at_k(
    ranking_a: Sequence[str], ranking_b: Sequence[str], *, k: int = 10
) -> float:
    """Solapamiento entre dos top-k: |A ∩ B| / |A ∪ B|.

    Métrica **sin etiquetas**: mide si dos formulaciones de la misma
    intención devuelven los mismos productos. Es lo único evaluable sobre las
    12 consultas de evaluación, cuyos juicios no están disponibles."""
    top_a = set(_top_k(ranking_a, k))
    top_b = set(_top_k(ranking_b, k))
    union = top_a | top_b
    if not union:
        return 0.0
    return len(top_a & top_b) / len(union)


def formulation_consistency(
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int = 10,
    separator: str = "-",
) -> pd.DataFrame:
    """Jaccard@k entre las formulaciones de cada intención de las ciegas.

    Espera IDs con la forma `EVAL-{intencion}-{formulacion}`. Devuelve una
    fila por intención con el solapamiento de cada par de formulaciones: un
    buscador semántico debería mantenerlo alto, porque las tres formas piden
    lo mismo."""
    por_intencion: dict[str, dict[str, Sequence[str]]] = {}
    for evaluation_id, ranking in rankings.items():
        partes = evaluation_id.split(separator)
        if len(partes) < 3:
            raise ValueError(
                f"{evaluation_id!r} no tiene la forma EVAL-intencion-formulacion."
            )
        por_intencion.setdefault(partes[1], {})[partes[2]] = ranking

    rows = []
    for intencion, formulaciones in sorted(por_intencion.items()):
        row: dict[str, str | float] = {"intencion": intencion}
        nombres = sorted(formulaciones)
        for i, primera in enumerate(nombres):
            for segunda in nombres[i + 1 :]:
                row[f"jaccard_{primera}_{segunda}"] = round(
                    jaccard_at_k(formulaciones[primera], formulaciones[segunda], k=k), 4
                )
        rows.append(row)
    return pd.DataFrame(rows)


def per_query_delta(
    frame: pd.DataFrame,
    *,
    sistema_a: str,
    sistema_b: str,
    columna_sistema: str = "plantilla",
    metrica: str = "ndcg@10",
) -> pd.DataFrame:
    """Diferencia consulta a consulta entre dos sistemas de una tabla larga.

    El agregado dice *cuánto* cambia; esta tabla dice *dónde*. Con 8 consultas
    la distinción es decisiva: una media puede moverse por encima del umbral
    porque una sola consulta cambió de sitio, y eso no es un efecto, es una
    consulta. `delta = a - b`, positivo cuando gana `sistema_a`."""
    faltan = {columna_sistema, "query_id", metrica} - set(frame.columns)
    if faltan:
        raise ValueError(f"A la tabla le faltan columnas: {sorted(faltan)}")

    tabla = frame.pivot_table(index="query_id", columns=columna_sistema, values=metrica)
    for sistema in (sistema_a, sistema_b):
        if sistema not in tabla.columns:
            raise ValueError(f"{sistema!r} no está en la columna {columna_sistema!r}.")

    salida = tabla[[sistema_a, sistema_b]].dropna().reset_index()
    salida["query_id"] = salida["query_id"].astype(str)
    salida["delta"] = (salida[sistema_a] - salida[sistema_b]).round(4)
    return salida.sort_values("delta", ascending=False).reset_index(drop=True)
