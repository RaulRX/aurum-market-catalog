"""Los motores candidatos de D12, detrás de una sola interfaz.

El guion de la prueba de humo es el mismo para los tres (NB04),
así que se escribe **una vez** contra un puerto y cada motor aporta su adaptador.
Eso es lo que hace que la comparativa sea una comparativa: si cada motor tuviera
su propio guion, las diferencias de la tabla podrían venir del código y no del
motor.

La interfaz vive en `base`; el guion, en `humo`. Los adaptadores se importan
por separado y a propósito —`from aurum.motores.qdrant import QdrantStore`—
para que el SDK de un motor no haga falta si no se va a usar ese motor.
"""
from .aceptacion import (
    ACCEPTANCE_MANUAL_NUMBERS,
    ACCEPTANCE_MANUAL_STEPS,
    run_acceptance,
    self_retrieval_canaries,
    wait_until_indexed,
)
from .base import (
    CATALOG_PREFIX,
    COLLECTION_PREFIX,
    FILTER_OPERATORS,
    FilterCondition,
    Point,
    SearchHit,
    UnsupportedFilterError,
    VectorStore,
    catalog_collection_name,
    ensure_reset_allowed,
    guard_collection_name,
    reset_allowed,
)
from .humo import (
    MANUAL_STEP_NUMBERS,
    PROBE_ROLES,
    SMOKE_MANUAL_STEPS,
    FilterProbe,
    PersistenceCheck,
    Snapshot,
    SmokeResult,
    audited_key,
    contains_level,
    error_quality,
    load_smoke,
    persistence_check,
    probe_filters,
    read_snapshot,
    record_manual,
    run_smoke_test,
    save_smoke,
    smoke_differences,
    smoke_display,
    smoke_table,
)
from .recursos import (
    MEDICION_ADVERTENCIAS,
    parse_docker_stats,
    parse_uptime,
    parse_volume_sizes,
    resource_note,
    resource_row,
    resource_table,
)

__all__ = [
    "ACCEPTANCE_MANUAL_NUMBERS",
    "ACCEPTANCE_MANUAL_STEPS",
    "CATALOG_PREFIX",
    "COLLECTION_PREFIX",
    "FILTER_OPERATORS",
    "MANUAL_STEP_NUMBERS",
    "MEDICION_ADVERTENCIAS",
    "PROBE_ROLES",
    "SMOKE_MANUAL_STEPS",
    "FilterCondition",
    "FilterProbe",
    "PersistenceCheck",
    "Point",
    "SearchHit",
    "SmokeResult",
    "Snapshot",
    "UnsupportedFilterError",
    "VectorStore",
    "audited_key",
    "catalog_collection_name",
    "contains_level",
    "ensure_reset_allowed",
    "error_quality",
    "guard_collection_name",
    "load_smoke",
    "parse_docker_stats",
    "parse_uptime",
    "parse_volume_sizes",
    "persistence_check",
    "probe_filters",
    "read_snapshot",
    "record_manual",
    "reset_allowed",
    "resource_note",
    "resource_row",
    "resource_table",
    "run_acceptance",
    "run_smoke_test",
    "self_retrieval_canaries",
    "save_smoke",
    "smoke_differences",
    "smoke_display",
    "smoke_table",
    "wait_until_indexed",
]
