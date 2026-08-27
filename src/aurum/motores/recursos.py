"""Paso 10 del guion de humo: cuánta máquina pide cada motor.

`docker stats` y `docker system df` se ejecutan en la terminal, fuera del
proceso de Python, así que lo que llega aquí es el texto
pegado tal cual, con la línea del comando y todo. Estas funciones lo parsean en
vez de dejar que los números se copien a mano a una tabla: el transcrito se
guarda en `artifacts/recursos/{motor}.txt` y la tabla se deriva de él, así que
cualquiera puede rehacer la cuenta y ver de dónde sale cada cifra.

Lo que **no** se puede leer de aquí está declarado en `MEDICION_ADVERTENCIAS`:
un `docker stats` es una foto en reposo, y el tamaño del volumen depende de la
edad del volumen tanto como del dato que guarda.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Una foto de `docker stats --no-stream`: id, nombre, CPU y la memoria con su
# límite. El resto de columnas (red, disco, PIDs) no entran en el paso 10.
_STATS = re.compile(
    r"^\s*(?P<id>[0-9a-f]{12})\s+(?P<nombre>\S+)\s+(?P<cpu>[\d.]+)%\s+"
    r"(?P<mem>[\d.]+\s*[A-Za-z]*B)\s*/\s*(?P<limite>[\d.]+\s*[A-Za-z]*B)\s+"
    r"(?P<pct>[\d.]+)%",
    re.MULTILINE,
)

# Una fila de volumen de `docker system df -v`: nombre, enlaces y tamaño. Las
# filas de contenedor de esa misma salida terminan en el nombre del contenedor y
# empiezan por un id hexadecimal, así que no casan con este patrón.
_VOLUMEN = re.compile(
    r"^(?P<nombre>(?![0-9a-f]{12}\s)[A-Za-z0-9][\w.-]*)\s+(?P<enlaces>\d+)\s+"
    r"(?P<tamano>[\d.]+\s*[A-Za-z]*B)\s*$",
    re.MULTILINE,
)

# El estado de cada contenedor en la misma salida. Interesa el "Up 6 minutes":
# es el tiempo en marcha, y sin él la comparación entre motores no dice en qué
# momento de su vida se midió cada uno.
_ESTADO = re.compile(
    r"^\s*[0-9a-f]{12}\s+.*?\s{2,}(?P<estado>Up\s.+?)\s{2,}(?P<nombre>\S+)\s*$",
    re.MULTILINE,
)

# Binarias las de `docker stats`, decimales las de `docker system df`. Mezclarlas
# es un error de hasta el 7 % en GB, que en una tabla comparativa se nota.
_BINARIOS = {"B": 1 / 1024**2, "KIB": 1 / 1024, "MIB": 1.0, "GIB": 1024.0, "TIB": 1024.0**2}
_DECIMALES = {"B": 1e-6, "KB": 1e-3, "MB": 1.0, "GB": 1e3, "TB": 1e6}

MEDICION_ADVERTENCIAS = (
    "La RAM es una foto en reposo tras ingerir, no un pico bajo carga: mide el "
    "coste base del proceso, que con 4,6 MB de vectores en la muestra es casi "
    "todo lo que hay.",
    "El tamaño del volumen NO es comparable entre motores: los volúmenes con "
    "nombre sobreviven a `down` y arrastran todo lo que cada motor escribió "
    "mientras se desarrollaba su adaptador, más la preasignación de WAL.",
    "Cada motor se midió con un tiempo en marcha distinto; la columna "
    "`en_marcha` lo declara en vez de dejarlo suponer.",
)


def _a_mib(texto: str) -> float:
    """`244.1MiB` → 244.1 · `3GiB` → 3072.0 (unidades binarias de `docker stats`)."""
    return _convertir(texto, _BINARIOS)


def _a_mb(texto: str) -> float:
    """`293.8MB` → 293.8 · `24.6kB` → 0.0246 (unidades decimales de `docker df`)."""
    return _convertir(texto, _DECIMALES)


def _convertir(texto: str, factores: dict[str, float]) -> float:
    casado = re.fullmatch(r"([\d.]+)\s*([A-Za-z]*B)", texto.strip())
    if casado is None:
        raise ValueError(f"No sé leer {texto!r} como un tamaño de Docker.")
    valor, unidad = casado.groups()
    clave = unidad.upper()
    if clave not in factores:
        raise ValueError(f"Unidad {unidad!r} desconocida en {texto!r}.")
    return float(valor) * factores[clave]


def parse_docker_stats(texto: str) -> pd.DataFrame:
    """La memoria por contenedor, a partir de la salida de `docker stats`.

    Ignora todo lo que no sea una fila de contenedor —la línea del comando, la
    cabecera, los separadores—, así que se le puede pasar el transcrito entero
    sin recortarlo. Recortarlo a mano sería el primer sitio donde se cuela un
    número que ya no corresponde a lo que se midió.
    """
    filas = [
        {
            "contenedor": casado["nombre"],
            "cpu_pct": float(casado["cpu"]),
            "ram_mib": round(_a_mib(casado["mem"]), 2),
            "limite_mib": round(_a_mib(casado["limite"]), 2),
            "pct_del_limite": float(casado["pct"]),
        }
        for casado in _STATS.finditer(texto)
    ]
    return pd.DataFrame(filas, columns=[
        "contenedor", "cpu_pct", "ram_mib", "limite_mib", "pct_del_limite",
    ])


def parse_volume_sizes(texto: str) -> pd.DataFrame:
    """El tamaño de cada volumen, a partir de `docker system df -v`."""
    filas = [
        {
            "volumen": casado["nombre"],
            "enlaces": int(casado["enlaces"]),
            "mb": round(_a_mb(casado["tamano"]), 2),
        }
        for casado in _VOLUMEN.finditer(texto)
    ]
    return pd.DataFrame(filas, columns=["volumen", "enlaces", "mb"])


def parse_uptime(texto: str) -> dict[str, str]:
    """Cuánto llevaba en marcha cada contenedor cuando se tomó la foto."""
    return {casado["nombre"]: casado["estado"].strip() for casado in _ESTADO.finditer(texto)}


def resource_row(
    texto: str,
    *,
    motor: str,
    exclude: tuple[str, ...] = (),
    volume_prefix: str | None = None,
) -> pd.DataFrame:
    """La fila del paso 10 para un motor, con una sola lectura del transcrito.

    `exclude` está para los contenedores que acompañan pero no son el motor. El
    caso real es `attu`, el panel de Milvus: sumar su memoria haría parecer a
    Milvus más caro frente a Qdrant —que sirve su panel desde el mismo proceso—
    y frente a Weaviate, que no trae ninguno. La regla ya está escrita en
    `docker/milvus/compose.yaml`; aquí solo se aplica, y se aplica nombrando al
    contenedor para que se vea qué se dejó fuera.
    """
    stats = parse_docker_stats(texto)
    en_marcha = parse_uptime(texto)
    contados = stats[~stats["contenedor"].isin(exclude)]

    # Los volúmenes SÍ hay que filtrarlos por motor. `docker system df -v` los
    # lista todos, y el `grep aurum-market` del Makefile deja pasar los de los
    # otros motores aunque estén parados: sus volúmenes con nombre sobreviven al
    # `down`. Sumarlos daba el total de los tres y no el del motor medido.
    prefijo = volume_prefix if volume_prefix is not None else f"aurum-market-{motor}"
    todos = parse_volume_sizes(texto)
    volumenes = todos[todos["volumen"].str.startswith(prefijo)]
    if contados.empty:
        raise ValueError(f"El transcrito de {motor} no trae ninguna fila de `docker stats`.")
    mas_apretado = contados.loc[contados["pct_del_limite"].idxmax()]
    return pd.DataFrame([{
        "motor": motor,
        "n_contenedores": len(contados),
        "ram_mib": round(contados["ram_mib"].sum(), 1),
        "volumen_mb": round(volumenes["mb"].sum(), 1),
        "mayor_consumidor": contados.loc[contados["ram_mib"].idxmax(), "contenedor"],
        # El único criterio de la fila 10 que estaba declarado ANTES de medir: el
        # `mem_limit` que cada compose fijó al escribirse. Inventar ahora un
        # umbral de MiB con los números delante sería ponerle la vara al ganador.
        "dentro_del_limite": bool((contados["pct_del_limite"] < 100).all()),
        "mas_apretado": f"{mas_apretado['contenedor']} al {mas_apretado['pct_del_limite']:.0f}%",
        "volumenes": ", ".join(volumenes["volumen"]),
        "en_marcha": " · ".join(
            sorted({en_marcha[c] for c in contados["contenedor"] if c in en_marcha})
        ),
        "excluidos": ", ".join(sorted(
            (set(exclude) & set(stats["contenedor"]))
            | set(todos.loc[~todos["volumen"].str.startswith(prefijo), "volumen"])
        )),
    }])


def resource_table(
    directory: Path | str,
    motores: tuple[str, ...],
    *,
    exclude: tuple[str, ...] = (),
) -> pd.DataFrame:
    """El paso 10 de los motores cuyo transcrito esté guardado en `directory`.

    Los que falten no se inventan ni rompen la tabla: simplemente no aparecen, y
    quien la lea ve qué motores tienen medido el paso y cuáles no.
    """
    origen = Path(directory)
    filas = [
        resource_row(
            (origen / f"{motor}.txt").read_text(encoding="utf-8"),
            motor=motor, exclude=exclude,
        )
        for motor in motores
        if (origen / f"{motor}.txt").is_file()
    ]
    if not filas:
        return pd.DataFrame(columns=["motor", "n_contenedores", "ram_mib", "volumen_mb"])
    return pd.concat(filas, ignore_index=True)


def resource_note(fila: pd.Series) -> str:
    """Lo que se anota en la fila 10 de la comparativa, desde la tabla."""
    contenedores = (
        "1 contenedor" if fila["n_contenedores"] == 1
        else f"{fila['n_contenedores']} contenedores"
    )
    mayor = (
        "" if fila["n_contenedores"] == 1
        else f", el mayor {fila['mayor_consumidor']}"
    )
    return (
        f"RAM {fila['ram_mib']:.1f} MiB en {contenedores}{mayor} · "
        f"volumen {fila['volumen_mb']:.1f} MB (no comparable entre motores) · "
        f"dentro del `mem_limit` declarado, el más apretado {fila['mas_apretado']} · "
        f"foto con el contenedor «{fila['en_marcha']}»"
    )
