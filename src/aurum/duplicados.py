"""NB07: deteccion de altas potencialmente duplicadas (D19-D22, config.yaml).

D19 fija que los candidatos salen siempre de consultar la base vectorial tal
como esta ahora -nunca de vectores cacheados en local-, y que NB07 se
calibra ANTES de que NB08 mute la coleccion. D20-D22 fijan que senales entran
en la regla, su forma exacta (dos caminos en OR) y el criterio del punto de
operacion. Este modulo implementa esas cuatro decisiones; no las repite en
prosa aqui -esa vive en `config.yaml` -> `nb07_duplicados`-.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from .datos import normalize_brand, tokenize

TOP_K_DUPLICADOS = 2


def normalizar_color(value: object) -> frozenset[str]:
    """Conjunto de palabras normalizado (D20): minusculas, sin acentos, con
    comas/"y"/"/" descartados por el propio tokenizador -no por una lista de
    separadores propia, que divergiria del resto del proyecto-."""
    return frozenset(tokenize(value, strip_accents=True))


def colores_coinciden(a: object, b: object) -> bool | None:
    """Igualdad de conjunto, no solapamiento parcial (D20): "Negro y Blanco"
    NO coincide con "Negro" -un color declarado de mas es informacion
    distinta, no la misma dicha de otra forma-. `None` cuando alguno de los
    dos esta vacio: no se puede verificar, y una cadena vacia no debe leerse
    como coincidencia."""
    conjunto_a, conjunto_b = normalizar_color(a), normalizar_color(b)
    if not conjunto_a or not conjunto_b:
        return None
    return conjunto_a == conjunto_b


def marcas_coinciden(a: object, b: object, *, modo: str = "unaccent") -> bool | None:
    """Igualdad exacta sobre el payload normalizado (D03/D20), nunca sobre el
    texto codificado -DEV-DUP-001 trae `brand="NIKE"` en el payload pero
    "Marca: ." en `text`-. `None` cuando alguna de las dos esta vacia (4,4%
    del catalogo no tiene `brand`)."""
    normal_a, normal_b = normalize_brand(a, modo), normalize_brand(b, modo)
    if not normal_a or not normal_b:
        return None
    return normal_a == normal_b


@dataclass(frozen=True, slots=True)
class SenalesDuplicado:
    """Las senales de D20 para una alta, ya calculadas contra su top-1."""

    incoming_id: str
    score_top1: float
    score_top2: float
    matched_product_id: str
    marca_coincide: bool | None
    color_coincide: bool | None
    is_duplicate: bool | None = None  # conocido en desarrollo; None en evaluacion

    @property
    def margen(self) -> float:
        """top1 - top2. `score_top2 = -inf` (sin segundo candidato) da un
        margen infinito a proposito: sin rival, el top1 no tiene con que
        empatar, y eso es la maxima confianza posible, no la minima."""
        return self.score_top1 - self.score_top2


def calcular_senales(
    buscador: object,
    altas: pd.DataFrame,
    *,
    etiquetas_col: str | None = "is_duplicate",
) -> tuple[SenalesDuplicado, ...]:
    """Recupera el top-2 de cada alta (D19: contra la base vectorial, en su
    estado actual) y calcula las senales de D20 frente a su top-1.

    `buscador` es cualquier objeto con `.buscar(texto, top_k=...)` que
    devuelva resultados con `.document_id`, `.score` y `.metadatos` -el
    contrato de `BuscadorVectorial.buscar`, no la clase en si: los tests
    pasan un doble en memoria, igual que el resto del proyecto-.
    """
    filas = []
    for _, alta in altas.iterrows():
        candidatos = buscador.buscar(str(alta["text"]), top_k=TOP_K_DUPLICADOS)
        if not candidatos:
            raise ValueError(
                f"{alta['incoming_id']}: la busqueda no devolvio ningun "
                f"candidato, no hay top-1 con el que comparar."
            )
        top1 = candidatos[0]
        top2 = candidatos[1] if len(candidatos) > 1 else None
        conocida = etiquetas_col is not None and etiquetas_col in alta.index
        filas.append(
            SenalesDuplicado(
                incoming_id=str(alta["incoming_id"]),
                score_top1=float(top1.score),
                score_top2=float(top2.score) if top2 is not None else float("-inf"),
                matched_product_id=top1.document_id,
                marca_coincide=marcas_coinciden(alta.get("brand"), top1.metadatos.get("brand")),
                color_coincide=colores_coinciden(alta.get("color"), top1.metadatos.get("color")),
                is_duplicate=bool(alta[etiquetas_col]) if conocida else None,
            )
        )
    return tuple(filas)


def regla_duplicado(
    senales: SenalesDuplicado,
    *,
    umbral_texto_solo: float,
    margen_minimo: float,
    umbral_texto_corroborado: float,
) -> bool:
    """D21: dos caminos en OR, ninguno bloqueado por el otro.

    Camino 1 (solo texto): la similitud semantica sola basta -por encima de
    `umbral_texto_solo`- si, ademas, es inequivocamente la mejor -eso mide
    el margen, exigido por encima de `margen_minimo`-. No mira marca ni
    color, a proposito: "mismo producto, otra talla o color" puede contar
    como duplicado sin que el color lo bloquee.

    Camino 2 (corroborado): una similitud mas baja -por encima de
    `umbral_texto_corroborado`, menor que `umbral_texto_solo`- se acepta si
    la marca O el color coinciden. El OR es deliberado, no un parche por
    datos faltantes: exigir las dos a la vez bloquearia el camino en un
    tramo no despreciable de candidatos (37,4% sin color, 4,4% sin marca),
    en contra de D22 (priorizar recall); ademas, misma marca con distinto
    color debe poder corroborar duplicado (D21), no descartarlo.
    """
    camino_1 = senales.score_top1 >= umbral_texto_solo and senales.margen >= margen_minimo
    camino_2 = senales.score_top1 >= umbral_texto_corroborado and (
        senales.marca_coincide is True or senales.color_coincide is True
    )
    return camino_1 or camino_2


def barrido_umbrales(
    senales: Sequence[SenalesDuplicado],
    *,
    valores_umbral_texto_solo: Sequence[float],
    valores_margen_minimo: Sequence[float],
    valores_umbral_texto_corroborado: Sequence[float],
) -> pd.DataFrame:
    """Una fila por combinacion (umbral_texto_solo, margen_minimo,
    umbral_texto_corroborado) con precision, recall, F1 y el recuento
    tp/fp/fn/tn sobre `senales` -D22 se aplica sobre esta tabla despues, no
    aqui-. Solo entra `umbral_texto_corroborado < umbral_texto_solo`, que es
    lo que D21 exige -el camino corroborado admite menos similitud porque
    tiene un respaldo que el otro camino no necesita-.

    Requiere `is_duplicate` conocido en todas las filas: es el barrido de
    calibracion sobre DESARROLLO, no la aplicacion sobre EVALUACION.
    """
    if not senales:
        raise ValueError("No hay senales con las que barrer umbrales.")
    if any(s.is_duplicate is None for s in senales):
        raise ValueError(
            "barrido_umbrales necesita is_duplicate en todas las filas -es "
            "el barrido de calibracion sobre DESARROLLO, no la aplicacion "
            "sobre EVALUACION-."
        )
    filas = []
    for umbral_texto_solo in valores_umbral_texto_solo:
        for margen_minimo in valores_margen_minimo:
            for umbral_texto_corroborado in valores_umbral_texto_corroborado:
                if umbral_texto_corroborado >= umbral_texto_solo:
                    continue
                tp = fp = fn = tn = 0
                for s in senales:
                    predicho = regla_duplicado(
                        s,
                        umbral_texto_solo=umbral_texto_solo,
                        margen_minimo=margen_minimo,
                        umbral_texto_corroborado=umbral_texto_corroborado,
                    )
                    real = bool(s.is_duplicate)
                    if predicho and real:
                        tp += 1
                    elif predicho and not real:
                        fp += 1
                    elif not predicho and real:
                        fn += 1
                    else:
                        tn += 1
                precision = tp / (tp + fp) if (tp + fp) else 0.0
                recall = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall)
                    else 0.0
                )
                filas.append(
                    {
                        "umbral_texto_solo": umbral_texto_solo,
                        "margen_minimo": margen_minimo,
                        "umbral_texto_corroborado": umbral_texto_corroborado,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "tn": tn,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    }
                )
    if not filas:
        raise ValueError(
            "Ninguna combinacion valida: revisa que haya algun "
            "umbral_texto_corroborado < umbral_texto_solo entre los valores dados."
        )
    return pd.DataFrame(filas)


def elegir_punto_operacion(barrido: pd.DataFrame, *, max_fp: int = 2) -> pd.DataFrame:
    """D22, aplicado tal cual: maximiza recall entre las combinaciones con
    como mucho `max_fp` falsos positivos sobre las 7 altas negativas de
    desarrollo -el suelo se expresa en cuenta de errores tolerados, no en un
    decimal de precision que 7 ejemplos no pueden sostener-. Empate a
    recall y fp, gana el F1 mas alto.

    Devuelve la tabla COMPLETA con `cumple_d22` y `elegido_r05`, no solo la
    fila ganadora -mismo patron que `aplicar_restriccion` en NB06-: para
    poder contradecir la eleccion mirando tambien lo que no gano.
    """
    tabla = barrido.copy()
    tabla["cumple_d22"] = tabla["fp"] <= max_fp
    candidatas = tabla[tabla["cumple_d22"]].sort_values(
        ["recall", "fp", "f1"], ascending=[False, True, False]
    )
    if candidatas.empty:
        tabla["elegido_r05"] = False
        return tabla
    ganadora = candidatas.iloc[0]
    tabla["elegido_r05"] = (
        (tabla["umbral_texto_solo"] == ganadora["umbral_texto_solo"])
        & (tabla["margen_minimo"] == ganadora["margen_minimo"])
        & (tabla["umbral_texto_corroborado"] == ganadora["umbral_texto_corroborado"])
    )
    return tabla


def resultados_duplicados(
    senales: Sequence[SenalesDuplicado],
    *,
    umbral_texto_solo: float,
    margen_minimo: float,
    umbral_texto_corroborado: float,
) -> pd.DataFrame:
    """La regla D21 ya congelada (R05), aplicada a `senales` en el formato
    exacto de `resultados_duplicados.csv` (README_DATOS): `incoming_id,
    predicted_duplicate, matched_product_id, score`. `matched_product_id`
    vacio si la prediccion es negativa."""
    filas = [
        {
            "incoming_id": s.incoming_id,
            "predicted_duplicate": (
                predicho := regla_duplicado(
                    s,
                    umbral_texto_solo=umbral_texto_solo,
                    margen_minimo=margen_minimo,
                    umbral_texto_corroborado=umbral_texto_corroborado,
                )
            ),
            "matched_product_id": s.matched_product_id if predicho else "",
            "score": s.score_top1,
        }
        for s in senales
    ]
    return pd.DataFrame(filas)
