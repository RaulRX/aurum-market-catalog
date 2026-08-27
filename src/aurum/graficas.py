"""Figuras del proyecto: una sola identidad visual, sin depender de las sesiones.

La paleta y el layout vienen de `vector_search_session.plotting`, el módulo de
los notebooks del máster, pero **copiados y no importados**. La razón no es
estética sino de dependencias: `vector_search_session` es material de clase que
vive en este repositorio por conveniencia, con su propio modelo de dominio
(`SearchResult`, `LatencySummary`, su `EvaluationReport`). Acoplar los notebooks
del proyecto a ese paquete significa que borrarlo —o que evolucione— rompe el
entregable, y ata las figuras a unas clases que no son las nuestras.

El caso que lo demostró: `plot_metric_comparison` de la sesión comprueba
`isinstance(value, EvaluationReport)` contra **su** clase, no contra la de
`aurum.evaluacion`. Pasarle un informe nuestro no fallaba con un error claro:
caía por la rama `else`, lo trataba como `Mapping` y reventaba más adelante. Aquí
se resuelve por duck typing sobre `.summary`, que acepta las dos.

Los colores se mantienen idénticos a los de las sesiones a propósito: el
entregable enseña una sola identidad gráfica, y esa continuidad se conserva
aunque el código ya no se comparta.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PRIMARY_COLOR = "#2563EB"
SECONDARY_COLOR = "#14B8A6"
ACCENT_COLOR = "#F59E0B"
NEGATIVE_COLOR = "#DC2626"
TEXT_COLOR = "#172033"
MUTED_TEXT_COLOR = "#64748B"
GRID_COLOR = "#E2E8F0"
BAND_COLOR = "#94A3B8"
BACKGROUND_COLOR = "#FFFFFF"
COLOR_SEQUENCE = (
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    ACCENT_COLOR,
    "#8B5CF6",
    "#EC4899",
    "#0EA5E9",
    "#84CC16",
    "#F97316",
)


def apply_project_layout(
    figure: go.Figure,
    *,
    title: str,
    subtitle: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    height: int = 520,
) -> go.Figure:
    """Aplica el lenguaje visual del proyecto a una figura de Plotly.

    Centraliza paleta, tipografía, márgenes y leyenda para que dos figuras de
    dos notebooks distintos no parezcan de dos informes distintos."""
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title debe ser un string no vacío.")
    if height < 300:
        raise ValueError("height debe ser al menos 300 píxeles.")

    title_text = f"<b>{title}</b>"
    if subtitle:
        title_text += (
            f"<br><span style='font-size:13px;color:{MUTED_TEXT_COLOR}'>{subtitle}</span>"
        )

    figure.update_layout(
        template="plotly_white",
        title={"text": title_text, "x": 0.02, "xanchor": "left", "y": 0.97},
        height=height,
        margin={"l": 72, "r": 32, "t": 96, "b": 64},
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font={"family": "Arial, sans-serif", "size": 13, "color": TEXT_COLOR},
        colorway=list(COLOR_SEQUENCE),
        hoverlabel={"bgcolor": "white", "font_size": 13, "font_family": "Arial"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
        },
    )
    figure.update_xaxes(
        title=xaxis_title, showgrid=True, gridcolor=GRID_COLOR, zeroline=False, automargin=True
    )
    figure.update_yaxes(title=yaxis_title, showgrid=False, zeroline=False, automargin=True)
    return figure


def _summary_of(value: Any) -> dict[str, float]:
    """Resumen de métricas de un informe de evaluación o de un dict ya resumido.

    Se comprueba por **duck typing** sobre `.summary` y no con `isinstance`: así
    vale tanto el `EvaluationReport` de `aurum.evaluacion` como un diccionario
    suelto o cualquier informe equivalente, sin acoplar este módulo a una clase
    concreta ni crear un import circular con `evaluacion`."""
    summary = getattr(value, "summary", value)
    if not isinstance(summary, Mapping):
        raise TypeError(
            "Cada sistema debe ser un informe con `.summary` o un Mapping de "
            f"métrica -> valor; se recibió {type(value).__name__}."
        )
    return {str(name): float(metric) for name, metric in summary.items()}


def plot_metric_comparison(
    systems: Mapping[str, Any],
    *,
    title: str = "Calidad de recuperación por sistema",
    subtitle: str | None = None,
    height: int = 520,
) -> go.Figure:
    """Compara varios sistemas sobre las mismas métricas, en escala 0-1 común.

    Es la figura que exige §3.1 del enunciado: denso frente a baseline léxico.
    Exigir que todos los sistemas traigan **exactamente** las mismas métricas no
    es rigidez — comparar un sistema medido con nDCG contra otro medido con MRR
    produce un gráfico perfectamente presentable y sin ningún significado."""
    if not systems:
        raise ValueError("systems no puede estar vacío.")

    summaries = {name: _summary_of(value) for name, value in systems.items()}
    metric_names = list(next(iter(summaries.values())))
    if not metric_names:
        raise ValueError("Cada sistema debe aportar al menos una métrica.")
    expected = set(metric_names)

    figure = go.Figure()
    for system_name, summary in summaries.items():
        if set(summary) != expected:
            raise ValueError(
                f"'{system_name}' no trae las mismas métricas que el resto: "
                f"{sorted(set(summary) ^ expected)} de diferencia."
            )
        values = [summary[name] for name in metric_names]
        if not all(np.isfinite(value) for value in values):
            raise ValueError(f"'{system_name}' tiene métricas no finitas.")
        figure.add_trace(
            go.Bar(
                name=system_name,
                x=metric_names,
                y=values,
                text=[f"{value:.3f}" for value in values],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=f"<b>{system_name}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>",
            )
        )

    figure.update_layout(barmode="group")
    figure.update_yaxes(range=[0, 1.08], tickformat=".0%")
    return apply_project_layout(
        figure,
        title=title,
        subtitle=subtitle,
        xaxis_title="Métrica",
        yaxis_title="Resultado",
        height=height,
    )


def plot_dimension_curve(
    sweep: pd.DataFrame,
    *,
    metric: str = "ndcg_at_10",
    tolerance: float = 0.02,
    model_column: str = "modelo",
    dim_column: str = "dim",
    dash_column: str | None = None,
    hover_columns: tuple[str, ...] = ("bytes_por_vector",),
    title: str = "Calidad frente a dimensión (MRL)",
    subtitle: str | None = None,
    height: int = 540,
) -> go.Figure:
    """Curva calidad ↔ dimensión con la banda de admisibilidad de D09b.

    Lo que la tabla equivalente no puede enseñar es **dónde se cae la curva**:
    en forma de pivot hay que restar de cabeza para saber si 256 dimensiones
    pierden algo frente a 1.024.

    La banda sombreada es `[B - tolerance, B]`, donde `B` es el mejor valor de
    toda la tabla. Es la zona admisible de D09b: dentro de ella la regla se
    queda con la **menor dimensión**, así que el ganador es el punto admisible
    más a la izquierda. Dibujarla convierte el criterio en algo que se lee en el
    gráfico, en vez de un número que aparece sin contexto al aplicar la regla.

    El eje X va en escala logarítmica porque las dimensiones se barren
    dividiendo por dos; en lineal, los puntos pequeños se amontonan contra el
    margen y es justo ahí donde se decide el ahorro.

    `dash_column` separa una segunda variable por **tipo de trazo** en lugar de
    por color. Con dos ejes cruzados —modelo y contrato de entrada— meterlos
    ambos en el color da cinco tonos sin relación visible entre sí; con color
    por modelo y trazo por contrato, el ojo agrupa primero por modelo y compara
    las dos variantes dentro de cada uno, que es la comparación que interesa."""
    faltan = {model_column, dim_column, metric} - set(sweep.columns)
    if dash_column is not None:
        faltan |= {dash_column} - set(sweep.columns)
    if faltan:
        raise ValueError(f"Al barrido le faltan columnas: {sorted(faltan)}")
    if sweep.empty:
        raise ValueError("El barrido no tiene filas que dibujar.")
    if tolerance < 0:
        raise ValueError("tolerance no puede ser negativa.")

    # `px.line` une los puntos en el orden en que llegan: sin ordenar por
    # dimensión, la línea zigzaguea y sugiere una curva que no existe.
    orden = [dim_column] if dash_column is None else [dash_column, dim_column]
    datos = sweep.sort_values(orden)
    presentes = [column for column in hover_columns if column in datos.columns]
    extra = "".join(
        f"<br>{column}: %{{customdata[{i}]}}" for i, column in enumerate(presentes)
    )

    figure = px.line(
        datos,
        x=dim_column,
        y=metric,
        color=model_column,
        line_dash=dash_column,
        markers=True,
        log_x=True,
        custom_data=presentes or None,
    )
    figure.update_traces(
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            f"{dim_column}: %{{x}}<br>{metric}: %{{y:.4f}}{extra}<extra></extra>"
        )
    )

    mejor = float(datos[metric].max())
    if tolerance > 0:
        figure.add_hrect(
            y0=mejor - tolerance,
            y1=mejor,
            fillcolor=BAND_COLOR,
            opacity=0.18,
            line_width=0,
            layer="below",
        )
    figure.add_hline(
        y=mejor,
        line_dash="dot",
        line_color=MUTED_TEXT_COLOR,
        annotation_text=f"B = {mejor:.4f}",
        annotation_position="top left",
    )

    # Marcas solo en las dimensiones realmente barridas: los ticks automáticos
    # de una escala log caerían en 200, 300... que no corresponden a ninguna
    # medición y sugieren puntos intermedios que nadie ha evaluado.
    dims = sorted(datos[dim_column].unique())
    figure.update_xaxes(tickvals=dims, ticktext=[str(dim) for dim in dims])

    return apply_project_layout(
        figure,
        title=title,
        subtitle=subtitle,
        xaxis_title=f"{dim_column} del vector (escala logarítmica)",
        yaxis_title=metric,
        height=height,
    )


def plot_contract_delta(
    sweep: pd.DataFrame,
    *,
    metric: str = "ndcg_at_10",
    tolerance: float = 0.02,
    model_column: str = "modelo",
    dim_column: str = "dim",
    contract_column: str = "contrato",
    baseline_contract: str = "nativo",
    compared_contract: str = "sin_contrato",
    title: str = "Cuánto aporta el contrato de entrada, según la dimensión",
    subtitle: str | None = None,
    height: int = 540,
) -> go.Figure:
    """Δ de `metric` entre las dos ramas de contrato, frente a la dimensión.

    La tabla de la sección D mide esa diferencia **solo en la dimensión
    nativa**, y eso puede esconder que el signo cambie al truncar. Esta figura
    la calcula en todas las dimensiones barridas y la dibuja contra una línea
    en cero, de modo que un cruce sea imposible de pasar por alto.

    Lectura:

    - **Δ > 0** → aplicar el contrato **mejora**.
    - **Δ < 0** → aplicarlo **empeora**.
    - **Dentro de la banda `±tolerance`** → la diferencia no se distingue con
      el número de consultas disponible, así que no sostiene ninguna decisión.

    Los modelos que solo tengan una de las dos ramas se descartan en silencio:
    sin las dos no hay diferencia que calcular."""
    faltan = {model_column, dim_column, contract_column, metric} - set(sweep.columns)
    if faltan:
        raise ValueError(f"Al barrido le faltan columnas: {sorted(faltan)}")
    if tolerance < 0:
        raise ValueError("tolerance no puede ser negativa.")

    tabla = sweep.pivot_table(
        index=[model_column, dim_column], columns=contract_column, values=metric
    )
    if baseline_contract not in tabla.columns or compared_contract not in tabla.columns:
        raise ValueError(
            f"El barrido no contiene las dos ramas '{baseline_contract}' y "
            f"'{compared_contract}': sin ambas no hay diferencia que medir."
        )

    datos = tabla.dropna(subset=[baseline_contract, compared_contract]).reset_index()
    if datos.empty:
        raise ValueError(
            "Ningún modelo tiene las dos ramas de contrato en la misma dimensión."
        )
    datos["delta"] = datos[baseline_contract] - datos[compared_contract]
    datos = datos.sort_values(dim_column)

    figure = px.line(
        datos, x=dim_column, y="delta", color=model_column, markers=True, log_x=True
    )
    figure.update_traces(
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            f"{dim_column}: %{{x}}<br>Δ {metric}: %{{y:+.4f}}<extra></extra>"
        )
    )

    # La banda es ±tolerance alrededor de cero, no alrededor del mejor valor:
    # aquí no se busca al ganador, se busca si la diferencia existe.
    if tolerance > 0:
        figure.add_hrect(
            y0=-tolerance, y1=tolerance,
            fillcolor=BAND_COLOR, opacity=0.18, line_width=0, layer="below",
        )
    figure.add_hline(y=0, line_color=MUTED_TEXT_COLOR, line_width=1)

    dims = sorted(datos[dim_column].unique())
    figure.update_xaxes(tickvals=dims, ticktext=[str(dim) for dim in dims])

    return apply_project_layout(
        figure,
        title=title,
        subtitle=subtitle,
        xaxis_title=f"{dim_column} del vector (escala logarítmica)",
        yaxis_title=f"Δ {metric}  ({baseline_contract} − {compared_contract})",
        height=height,
    )


def plot_effect_vs_exposure(
    frame: pd.DataFrame,
    *,
    exposure: str,
    effect: str = "delta",
    label: str = "query_id",
    tolerance: float = 0.01,
    title: str = "¿El efecto guarda relación con cuánto le afecta el cambio?",
    subtitle: str | None = None,
    height: int = 540,
) -> go.Figure:
    """Un punto por consulta: cuánto la toca un cambio frente a cuánto se movió.

    Es la figura que separa **señal de perturbación**, y hace falta porque el
    agregado no puede distinguirlas. Si un cambio aporta información, las
    consultas más expuestas a él deberían ser las que más se mueven: los puntos
    dibujarían una tendencia ascendente. Una nube plana —o con el punto de
    exposición máxima pegado al cero— dice que el efecto no viene de la
    información que el cambio añade, sino de haber movido el espacio.

    La banda gris es la zona donde la diferencia no se distingue del ruido con
    las consultas disponibles. Cada punto lleva su identificador porque aquí lo
    interesante son **los casos concretos**, no la tendencia: la consulta que
    contradice la hipótesis vale más que el promedio de todas."""
    faltan = {exposure, effect, label} - set(frame.columns)
    if faltan:
        raise ValueError(f"A la tabla le faltan columnas: {sorted(faltan)}")
    if frame.empty:
        raise ValueError("No hay consultas que dibujar.")
    if tolerance < 0:
        raise ValueError("tolerance no puede ser negativa.")

    figure = px.scatter(
        frame, x=exposure, y=effect, text=frame[label].astype(str)
    )
    figure.update_traces(
        marker={"size": 13, "color": PRIMARY_COLOR, "line": {"width": 0}},
        textposition="top center",
        textfont={"size": 11, "color": MUTED_TEXT_COLOR},
        hovertemplate=(
            f"<b>%{{text}}</b><br>{exposure}: %{{x}}<br>{effect}: %{{y:+.4f}}<extra></extra>"
        ),
    )

    if tolerance > 0:
        figure.add_hrect(
            y0=-tolerance, y1=tolerance,
            fillcolor=BAND_COLOR, opacity=0.18, line_width=0, layer="below",
        )
    figure.add_hline(y=0, line_color=MUTED_TEXT_COLOR, line_width=1)

    return apply_project_layout(
        figure,
        title=title,
        subtitle=subtitle,
        xaxis_title=exposure,
        yaxis_title=effect,
        height=height,
    )


def plot_ann_pareto(
    barrido: pd.DataFrame,
    *,
    recall_minimo: float,
    p95_maximo_ms: float,
    recall_column: str,
    latency_column: str = "ms_p95",
    ef_column: str = "ef",
    elegido_column: str = "elegido_r04",
    title: str = "Fidelidad del ANN frente a latencia (NB06)",
    subtitle: str | None = None,
    height: int = 540,
) -> go.Figure:
    """La curva recall-latencia de NB06, con la región de D16 sombreada.

    Cada punto es un `ef` del barrido; la línea los une en el orden en que se
    barrieron para que se lea como una curva, no como una nube. La región
    sombreada es exactamente la de D16 -`latencia <= p95_maximo` **y**
    `recall >= recall_minimo`-, un solo rectángulo y no dos bandas cruzadas:
    dibujar la restricción como intersección es lo que permite ver de un
    vistazo si un punto la cumple sin leer la tabla.

    El punto marcado con `elegido_column` (R04, ya decidido por
    `ann.aplicar_restriccion`) sale en un color distinto: es la única forma de
    ver *dentro* de la región cuál es el barato, porque varios `ef` pueden
    cumplir D16 a la vez."""
    faltan = {recall_column, latency_column, ef_column} - set(barrido.columns)
    if faltan:
        raise ValueError(f"Al barrido le faltan columnas: {sorted(faltan)}")
    if barrido.empty:
        raise ValueError("El barrido no tiene filas que dibujar.")
    if recall_minimo < 0 or p95_maximo_ms < 0:
        raise ValueError("recall_minimo y p95_maximo_ms no pueden ser negativos.")

    datos = barrido.sort_values(ef_column).reset_index(drop=True)
    elegido = (
        datos[elegido_column].astype(bool)
        if elegido_column in datos.columns
        else pd.Series(False, index=datos.index)
    )
    colores = [ACCENT_COLOR if es_elegido else PRIMARY_COLOR for es_elegido in elegido]
    tamanos = [16 if es_elegido else 10 for es_elegido in elegido]

    x_max = float(max(datos[latency_column].max(), p95_maximo_ms) * 1.08)
    y_min = float(min(datos[recall_column].min(), recall_minimo) * 0.97)

    figure = go.Figure()
    figure.add_shape(
        type="rect",
        x0=0, x1=p95_maximo_ms, y0=recall_minimo, y1=1.02,
        fillcolor=BAND_COLOR, opacity=0.20, line_width=0, layer="below",
    )
    figure.add_trace(go.Scatter(
        x=datos[latency_column],
        y=datos[recall_column],
        mode="lines+markers+text",
        text=[str(valor) for valor in datos[ef_column]],
        textposition="top center",
        textfont={"size": 11, "color": MUTED_TEXT_COLOR},
        line={"color": PRIMARY_COLOR, "width": 1.5},
        marker={"size": tamanos, "color": colores, "line": {"width": 0}},
        name="ef",
        hovertemplate=(
            f"ef=%{{text}}<br>{latency_column}: %{{x:.2f}} ms<br>"
            f"{recall_column}: %{{y:.4f}}<extra></extra>"
        ),
    ))
    figure.add_vline(
        x=p95_maximo_ms, line_dash="dot", line_color=MUTED_TEXT_COLOR,
        annotation_text=f"p95 ≤ {p95_maximo_ms:g} ms", annotation_position="top right",
    )
    figure.add_hline(
        y=recall_minimo, line_dash="dot", line_color=MUTED_TEXT_COLOR,
        annotation_text=f"recall ≥ {recall_minimo:g}", annotation_position="bottom right",
    )
    figure.update_xaxes(range=[0, x_max])
    figure.update_yaxes(range=[y_min, 1.02])

    return apply_project_layout(
        figure,
        title=title,
        subtitle=subtitle,
        xaxis_title=f"{latency_column} (ms) — menor es mejor",
        yaxis_title=f"{recall_column} — mayor es mejor",
        height=height,
    )


def plot_duplicate_threshold_sweep(
    barrido: pd.DataFrame,
    *,
    max_fp: int,
    precision_column: str = "precision",
    recall_column: str = "recall",
    cumple_column: str = "cumple_d22",
    elegido_column: str = "elegido_r05",
    hover_columns: tuple[str, ...] = (
        "umbral_texto_solo", "margen_minimo", "umbral_texto_corroborado", "fp",
    ),
    title: str = "Precisión y recall del barrido de duplicados (NB07)",
    subtitle: str | None = None,
    height: int = 540,
) -> go.Figure:
    """El barrido de D22: cada punto es una combinación (umbral_texto_solo,
    margen_minimo, umbral_texto_corroborado).

    Mismo principio que `plot_ann_pareto`: enseñar el barrido completo, no
    solo el veredicto. Los puntos que no cumplen el suelo de `max_fp`
    falsos positivos se dibujan en rojo apagado en vez de descartarse -así
    se ve también lo que no ganó-; los que cumplen, en azul; el elegido
    (D22: mayor recall dentro del suelo) en ámbar y más grande."""
    faltan = {precision_column, recall_column, cumple_column} - set(barrido.columns)
    if faltan:
        raise ValueError(f"Al barrido le faltan columnas: {sorted(faltan)}")
    if barrido.empty:
        raise ValueError("El barrido no tiene filas que dibujar.")

    datos = barrido.reset_index(drop=True)
    cumple = datos[cumple_column].astype(bool)
    elegido = (
        datos[elegido_column].astype(bool)
        if elegido_column in datos.columns
        else pd.Series(False, index=datos.index)
    )
    colores = [
        ACCENT_COLOR if es_elegido else (PRIMARY_COLOR if es_cumple else NEGATIVE_COLOR)
        for es_elegido, es_cumple in zip(elegido, cumple)
    ]
    tamanos = [16 if es_elegido else 9 for es_elegido in elegido]

    presentes = [column for column in hover_columns if column in datos.columns]
    extra = "".join(
        f"<br>{column}: %{{customdata[{i}]}}" for i, column in enumerate(presentes)
    )

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=datos[precision_column],
        y=datos[recall_column],
        mode="markers",
        marker={"size": tamanos, "color": colores, "line": {"width": 0}},
        customdata=datos[presentes].to_numpy() if presentes else None,
        hovertemplate=(
            f"precisión: %{{x:.3f}}<br>recall: %{{y:.3f}}{extra}<extra></extra>"
        ),
        showlegend=False,
    ))
    figure.update_xaxes(range=[0, 1.02])
    figure.update_yaxes(range=[0, 1.02])

    return apply_project_layout(
        figure,
        title=title,
        subtitle=subtitle,
        xaxis_title=f"precisión — rojo: más de {max_fp} falsos positivos",
        yaxis_title="recall — ámbar: punto elegido (D22)",
        height=height,
    )
