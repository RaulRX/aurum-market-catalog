"""NB06 · Fidelidad del ANN, latencia de consulta y selección bajo restricción.

El plan lo resume en una fórmula que no hay que confundir con la de
`evaluacion.py`:

    recall ANN@10  =  |IDs_ANN ∩ IDs_exactos| / 10   → ¿el ÍNDICE es fiel al espacio?
    Recall@10      =  |relevantes ∩ top10| / |rel|   → ¿la REPRESENTACIÓN es buena?

La primera no usa juicios de relevancia: compara el top-k que devuelve el
motor aproximado (`BuscadorVectorial` con `ef` fijado) contra el top-k que
devuelve el oráculo exacto (`busqueda.DenseRetriever`, ya construido y
probado en NB02). Por eso vive en un módulo aparte y no reutiliza
`evaluacion.recall_at_k`: mezclarlas sería aplanar justo la distinción que
más nota da en el enunciado.

El nDCG@10 sí se mide con `evaluacion.evaluate_rankings` -es la métrica de
negocio, y ya existe- comparando el ranking del oráculo con el del ANN
elegido; `comparar_ndcg_con_oraculo` solo empaqueta esa comparación.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_TOP_K = 10


# ═══════════════════════ fidelidad del índice (recall ANN) ═══════════════════


def ann_recall_per_query(
    oraculo: Mapping[str, Sequence[str]],
    ann: Mapping[str, Sequence[str]],
    *,
    k: int = DEFAULT_TOP_K,
) -> dict[str, float]:
    """`|top-k ANN ∩ top-k exacto| / k`, una consulta a la vez.

    Ambos argumentos son `{query_id: [document_id, ...]}`, la misma forma que
    devuelve `busqueda.rank_queries_dense`. El oráculo manda: toda consulta
    que aparece en `oraculo` tiene que tener su ranking ANN, o el resultado
    estaría comparando conjuntos distintos de consultas sin avisar.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k debe ser un entero positivo.")
    faltantes = set(oraculo) - set(ann)
    if faltantes:
        raise ValueError(
            f"Hay consultas con oráculo pero sin ranking ANN: {sorted(faltantes)}."
        )
    recalls: dict[str, float] = {}
    for query_id, exactos in oraculo.items():
        esperados = set(exactos[:k])
        if not esperados:
            raise ValueError(f"El oráculo de {query_id!r} no trae vecinos.")
        obtenidos = set(ann[query_id][:k])
        recalls[query_id] = len(esperados & obtenidos) / len(esperados)
    return recalls


def ann_recall_at_k(
    oraculo: Mapping[str, Sequence[str]],
    ann: Mapping[str, Sequence[str]],
    *,
    k: int = DEFAULT_TOP_K,
) -> float:
    """Media macro de `ann_recall_per_query`: la fidelidad que exige D16."""
    valores = ann_recall_per_query(oraculo, ann, k=k)
    return float(np.mean(list(valores.values())))


def resumen_recall(
    por_consulta: Mapping[str, float], *, k: int = DEFAULT_TOP_K
) -> dict[str, float]:
    """Media, mínimo y p5 del recall por consulta.

    La media sola puede esconder historias muy distintas -"pierde un poco
    siempre" y "perfecta salvo en dos consultas" dan la misma media-, así que
    el plan pide mirar también el mínimo y el p5, no solo la macro.
    """
    valores = np.asarray(list(por_consulta.values()), dtype=np.float64)
    if valores.size == 0:
        raise ValueError("No hay consultas con las que resumir el recall.")
    return {
        f"recall_ann_at_{k}": round(float(valores.mean()), 4),
        f"recall_ann_at_{k}_min": round(float(valores.min()), 4),
        f"recall_ann_at_{k}_p5": round(float(np.percentile(valores, 5)), 4),
    }


# ══════════════════════════ latencia de una consulta ═════════════════════════


def measure_search_latency(
    buscador: Any,
    consultas: Sequence[str],
    *,
    top_k: int = DEFAULT_TOP_K,
    repeticiones: int = 20,
    calentamiento: int = 3,
) -> dict[str, object]:
    """Latencia de `buscador.buscar()`, con calentamiento y repeticiones.

    Mide **una consulta cada vez** -es lo que ve un usuario real, no un lote-,
    recorriendo `consultas` en ciclo para que ni una consulta concreta ni una
    conexión recién abierta dominen la medición. El calentamiento no se
    contabiliza: paga el primer round-trip contra el motor, que no se repite
    en cada consulta real (mismo criterio que `embeddings.measure_encode_latency`).
    """
    materializadas = list(consultas)
    if not materializadas:
        raise ValueError("No hay consultas con las que medir.")
    if repeticiones < 1:
        raise ValueError("repeticiones debe ser >= 1.")

    for indice in range(max(0, calentamiento)):
        buscador.buscar(materializadas[indice % len(materializadas)], top_k=top_k)

    muestras: list[float] = []
    for indice in range(repeticiones):
        texto = materializadas[indice % len(materializadas)]
        inicio = time.perf_counter()
        buscador.buscar(texto, top_k=top_k)
        muestras.append((time.perf_counter() - inicio) * 1000)

    tiempos = np.asarray(muestras, dtype=np.float64)
    p50 = float(np.percentile(tiempos, 50))
    return {
        "ef": getattr(buscador, "ef", None),
        "n_llamadas": int(repeticiones),
        "ms_p50": round(p50, 2),
        "ms_p95": round(float(np.percentile(tiempos, 95)), 2),
        "ms_media": round(float(tiempos.mean()), 2),
        "ms_min": round(float(tiempos.min()), 2),
        "ms_max": round(float(tiempos.max()), 2),
        # QPS estimado en serie -una consulta detrás de otra-, sobre la
        # mediana: es la misma convención que usa el plan (`n_queries /
        # mediana_s`), no un throughput medido con lotes concurrentes.
        "qps_estimado": round(1000.0 / p50, 1) if p50 > 0 else float("inf"),
    }


# ═════════════════════════ el barrido y la restricción ═══════════════════════


def barrido_ef(
    construir_buscador: Callable[[int], Any],
    consultas: Mapping[str, str],
    oraculo: Mapping[str, Sequence[str]],
    *,
    valores_ef: Sequence[int],
    top_k: int = DEFAULT_TOP_K,
    repeticiones: int = 20,
    calentamiento: int = 3,
) -> pd.DataFrame:
    """Una fila por valor de `ef`: fidelidad frente al oráculo y latencia.

    `construir_buscador(ef)` crea el `BuscadorVectorial` de ese punto de la
    curva -normalmente la misma conexión y colección, cambiando solo `ef`-.
    El oráculo se recibe ya calculado: es el mismo top-k exacto para los
    cinco valores de `ef`, así que calcularlo una vez fuera del barrido evita
    pagarlo cinco veces sin cambiar el resultado.
    """
    if not valores_ef:
        raise ValueError("valores_ef no puede estar vacío.")
    if not consultas:
        raise ValueError("No hay consultas con las que barrer ef.")
    filas = []
    for ef in valores_ef:
        buscador = construir_buscador(ef)
        ann = {
            query_id: [r.document_id for r in buscador.buscar(texto, top_k=top_k)]
            for query_id, texto in consultas.items()
        }
        por_consulta = ann_recall_per_query(oraculo, ann, k=top_k)
        latencia = measure_search_latency(
            buscador,
            list(consultas.values()),
            top_k=top_k,
            repeticiones=repeticiones,
            calentamiento=calentamiento,
        )
        fila = {"ef": ef, **resumen_recall(por_consulta, k=top_k)}
        fila.update({clave: valor for clave, valor in latencia.items() if clave != "ef"})
        filas.append(fila)
    return pd.DataFrame(filas)


def aplicar_restriccion(
    tabla: pd.DataFrame,
    *,
    recall_minimo: float,
    p95_maximo_ms: float,
    columna_recall: str,
    columna_p95: str = "ms_p95",
) -> pd.DataFrame:
    """D16, aplicada tal cual: descarta lo que no cumple y, entre lo que
    sobra, gana el `ef` de menor p95 medido -el coste real, no un supuesto de
    que más `ef` siempre tarda más-. Empate a p95, gana el `ef` más bajo.

    No devuelve solo la fila ganadora: añade `cumple_d16` y `elegido_r04` a
    la tabla completa, para que la decisión se pueda contradecir mirando
    también las configuraciones que no ganaron -el mismo principio que las
    tablas de NB05, enseñar el resultado en vez de solo el veredicto-.
    """
    tabla = tabla.copy()
    tabla["cumple_d16"] = (tabla[columna_recall] >= recall_minimo) & (
        tabla[columna_p95] <= p95_maximo_ms
    )
    candidatas = tabla[tabla["cumple_d16"]].sort_values([columna_p95, "ef"])
    elegido_ef = candidatas.iloc[0]["ef"] if not candidatas.empty else None
    tabla["elegido_r04"] = (
        tabla["ef"] == elegido_ef if elegido_ef is not None else False
    )
    return tabla


# ═══════════════════ enseñar lo que sale, no solo el resumen ═════════════════


def tabla_recall_por_consulta(
    oraculo: Mapping[str, Sequence[str]],
    ann: Mapping[str, Sequence[str]],
    *,
    consultas: Mapping[str, str] | None = None,
    k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """Una fila por consulta: cuántos vecinos exactos trajo el ANN y cuáles no.

    El resumen (`resumen_recall`) puede esconder si el error se reparte
    -"pierde uno siempre"- o se concentra -"perfecta salvo en dos consultas"-;
    esta tabla lo enseña con los `product_id` perdidos delante, no con un
    número solo.
    """
    consultas = consultas or {}
    por_consulta = ann_recall_per_query(oraculo, ann, k=k)
    filas = []
    for query_id, recall in sorted(por_consulta.items()):
        esperados = set(oraculo[query_id][:k])
        obtenidos = set(ann[query_id][:k])
        perdidos = esperados - obtenidos
        filas.append({
            "query_id": query_id,
            "consulta": consultas.get(query_id, ""),
            "recall_ann": round(recall, 2),
            "vecinos_recuperados": f"{len(esperados & obtenidos)} de {len(esperados)}",
            "perdidos": ", ".join(sorted(perdidos)) if perdidos else "—",
        })
    return pd.DataFrame(filas)


def comparar_ndcg_con_oraculo(
    ranking_ann: Mapping[str, Sequence[str]],
    ranking_oraculo: Mapping[str, Sequence[str]],
    qrels_by_query: Mapping[str, Mapping[str, float]],
    *,
    k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """La métrica clave de NB06: si el nDCG no baja al pasar del oráculo al
    ANN elegido, la fidelidad perdida -si la hay- no le costó nada al negocio.

    Reutiliza `evaluacion.evaluate_rankings` -la misma función de NB01/NB09,
    no una reimplementación- sobre las mismas consultas con y sin aproximar.
    Solo entran las consultas con juicios (`qrels_by_query`): el recall ANN se
    mide sobre más consultas porque no necesita etiquetas, el nDCG solo puede
    medirse donde las hay.
    """
    from .evaluacion import evaluate_rankings

    exacto = evaluate_rankings(ranking_oraculo, qrels_by_query, k=k)
    aproximado = evaluate_rankings(ranking_ann, qrels_by_query, k=k)
    return pd.DataFrame([
        {"sistema": "oráculo exacto (DenseRetriever)", **exacto.summary},
        {"sistema": "ANN elegido (R04)", **aproximado.summary},
    ])


def comparar_ndcg_por_consulta(
    ranking_ann: Mapping[str, Sequence[str]],
    ranking_oraculo: Mapping[str, Sequence[str]],
    qrels_by_query: Mapping[str, Mapping[str, float]],
    *,
    consultas: Mapping[str, str] | None = None,
    k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """El desglose que `comparar_ndcg_con_oraculo` no puede dar: nDCG por
    consulta, oráculo frente a ANN, con la diferencia ya calculada.

    El agregado puede esconder dos historias muy distintas: que la pérdida se
    reparta un poco entre todas, o que se concentre en la misma consulta que
    ya delató un recall ANN bajo -y en ese caso, si `delta` sale ~0 pese a
    ello, es la prueba de que el sustituto que trajo el ANN era igual de
    relevante para el negocio aunque no fuera el vecino exacto.

    Ordenada por `delta` ascendente: la consulta que más pierde queda arriba.
    """
    from .evaluacion import evaluate_rankings

    consultas = consultas or {}
    exacto = evaluate_rankings(ranking_oraculo, qrels_by_query, k=k)
    aproximado = evaluate_rankings(ranking_ann, qrels_by_query, k=k)

    ndcg_exacto = {metricas.query_id: metricas.ndcg for metricas in exacto.per_query}
    ndcg_aproximado = {metricas.query_id: metricas.ndcg for metricas in aproximado.per_query}

    filas = [
        {
            "query_id": query_id,
            "consulta": consultas.get(query_id, ""),
            f"ndcg_at_{k}_oraculo": round(ndcg_oraculo, 4),
            f"ndcg_at_{k}_ann": round(ndcg_aproximado.get(query_id, 0.0), 4),
            "delta": round(ndcg_aproximado.get(query_id, 0.0) - ndcg_oraculo, 4),
        }
        for query_id, ndcg_oraculo in ndcg_exacto.items()
    ]
    return pd.DataFrame(filas).sort_values("delta")
