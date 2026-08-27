"""Regenera `resultados/metricas_desarrollo.json` desde los artefactos existentes.

El enunciado (§8) pide que las métricas se regeneren **desde un único comando**, y
(§6) que el artefacto lleve "las métricas mínimas". Este script no recodifica ni
vuelve a buscar: consolida lo que los notebooks ya dejaron en `artifacts/`. Así
el comando cuesta segundos y puede ejecutarse antes de cada entrega sin depender
de una clave de API ni de tener un motor levantado.

Lo que aún no existe se escribe como `null` con el notebook que lo producirá al
lado. Un hueco declarado es información; un hueco ausente parece un olvido.

Uso: python scripts/consolidar_metricas.py [--salida RUTA]
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parent.parent
ARTEFACTOS = RAIZ / "artifacts"
SALIDA = RAIZ / "resultados" / "metricas_desarrollo.json"

# Las tres que la tabla de evidencia mínima del §5 exige sobre desarrollo, más
# precision, que ya se calcula y no cuesta nada arrastrar.
METRICAS_MINIMAS = ("ndcg_at_10", "recall_at_10", "mrr_at_10", "precision_at_10")

# Lo que falta y de dónde vendrá. Se declara aquí y no en un comentario para que
# aparezca en el propio artefacto.
PENDIENTES: dict[str, str] = {
    "fidelidad_ann": "NB06 · comparar IDs contra el oráculo exacto sobre una muestra",
    "latencia_busqueda": "NB06 · p50 y p95 con entorno, calentamiento y repeticiones",
    "filtros": "NB05 · las 4 consultas de consultas_filtradas.csv, marca cumplida",
    "duplicados": "NB07 · precision, recall y F1 sobre altas_desarrollo.csv",
    "mutaciones": "NB08 · recuento, lectura por ID y búsqueda tras los 24 eventos",
}


def _leer(nombre: str) -> dict[str, Any] | None:
    ruta = ARTEFACTOS / f"{nombre}.json"
    if not ruta.exists():
        return None
    return json.loads(ruta.read_text(encoding="utf-8"))


def _solo_minimas(metricas: dict[str, Any]) -> dict[str, Any]:
    return {clave: metricas.get(clave) for clave in METRICAS_MINIMAS}


def _ganadora(regla: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    """La fila que la regla dejó en primer lugar.

    Las reglas R01 y D09b se guardan como la tabla entera ordenada por
    `posicion_regla`, no como el ganador suelto: así el artefacto conserva por
    qué ganó y no solo que ganó. Aquí se coge la primera, comprobando que además
    sea admisible —una tabla donde ninguna lo es no tiene ganadora, y devolver la
    primera de todas formas sería inventarse una decisión.
    """
    if not regla:
        return None
    primera = min(regla, key=lambda fila: fila.get("posicion_regla", 10**6))
    return primera if primera.get("admisible") else None


def consolidar() -> dict[str, Any]:
    """Reúne las métricas de desarrollo de los tres artefactos que ya existen."""
    salida: dict[str, Any] = {
        "generado_en": datetime.now(UTC).isoformat(),
        "corpus_evaluado": "catalogo_productos",
        "consultas": "consultas_desarrollo.csv (8, con juicios ESCI)",
        "relevancia": None,
        "ranking": {},
        "pendiente": dict(PENDIENTES),
        "artefactos_ausentes": [],
    }

    baseline = _leer("baseline_lexico")
    if baseline:
        # La convención de relevancia sale del artefacto y no se escribe a mano:
        # §2 exige que no cambie en silencio entre experimentos, y copiarla
        # rompería justo esa garantía.
        salida["relevancia"] = baseline["configuracion"]["relevancia"]
        salida["umbral_relevante_recall_mrr"] = (
            baseline["configuracion"]["umbral_relevante_recall_mrr"]
        )
        for nombre, metricas in baseline["completo"]["metricas"].items():
            salida["ranking"][f"lexico_{nombre}"] = _solo_minimas(metricas)
    else:
        salida["artefactos_ausentes"].append("baseline_lexico.json (NB01)")

    representacion = _leer("comparativa_representacion")
    if representacion:
        # R01 se ratificó sobre el catálogo completo, no sobre la muestra: sobre
        # 1.500 el orden era otro. Se lee la tabla de `_completo` a propósito.
        ganadora = _ganadora(representacion.get("regla_r01_completo"))
        if ganadora:
            salida["ranking"]["denso_r01_ganadora"] = {
                "plantilla": ganadora.get("plantilla"),
                "decidida_sobre": "catalogo_productos",
                **_solo_minimas(ganadora),
            }
        else:
            salida["artefactos_ausentes"].append("R01 sin ganadora admisible en NB03")
    else:
        salida["artefactos_ausentes"].append("comparativa_representacion.json (NB03)")

    modelos = _leer("comparativa_modelos")
    if modelos:
        elegido = _ganadora(modelos.get("regla_d09b"))
        salida["modelo"] = (
            {
                "sistema": elegido.get("sistema"),
                "dim": elegido.get("dim"),
                "contrato": elegido.get("contrato"),
                "ndcg_at_10_muestra": elegido.get("ndcg_at_10"),
                "nota": "R02 se decidió sobre catalogo_muestra; P01 lo confirmó a escala completa",
            }
            if elegido
            else {"nota": "D09b no dejó ninguna configuración admisible; revisar NB02 G"}
        )
    else:
        salida["artefactos_ausentes"].append("comparativa_modelos.json (NB02)")

    return salida


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=Path, default=SALIDA)
    args = parser.parse_args()

    datos = consolidar()
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(
        json.dumps(datos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Escrito {args.salida.relative_to(RAIZ)}")
    print(f"  sistemas con métricas: {', '.join(datos['ranking']) or 'ninguno'}")
    if datos["artefactos_ausentes"]:
        print(f"  ⚠️ artefactos ausentes: {', '.join(datos['artefactos_ausentes'])}")
    print(f"  pendientes declarados : {', '.join(datos['pendiente'])}")
    # Falta trabajo por hacer, pero el comando ha hecho el suyo: no es un error.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
