# Aurum Market — recorrido principal del repositorio.
#
# El enunciado (§6) pide que la entrega se ejecute en un entorno limpio siguiendo
# el README, y que el recorrido sea "claro y corto: preparar entorno, levantar el
# motor, ingerir, evaluar y limpiar". Estos objetivos son ese recorrido.
#
# `make help` los lista. Se van rellenando conforme cada notebook los habilita:
# un objetivo que aún no tiene con qué trabajar avisa y sale con error, en vez de
# terminar en silencio y aparentar que hizo algo.

.PHONY: help install test notebook notebooks metrics verify \
        motor-up motor-down motor-ps motor-stats all-down clean-pyc

PY      := .venv/Scripts/python.exe
MOTOR   ?= qdrant
NB      ?=
COMPOSE  = docker/$(MOTOR)/compose.yaml

help:
	@echo "Entorno"
	@echo "  make install                 instala las dependencias congeladas"
	@echo "  make test                    ejecuta la suite completa"
	@echo ""
	@echo "Notebooks  (fuente: scripts/notebook_cells.py)"
	@echo "  make notebooks               reconstruye TODOS los .ipynb"
	@echo "  make notebook NB=04_motor.ipynb   reconstruye uno"
	@echo ""
	@echo "Motores    (se levantan a mano; ningun notebook los arranca)"
	@echo "  make motor-up    MOTOR=qdrant|weaviate|milvus"
	@echo "  make motor-ps    MOTOR=..."
	@echo "  make motor-down  MOTOR=..."
	@echo "  make motor-stats             RAM y volumenes (paso 10 del guion)"
	@echo "  make all-down                apaga los tres, por si queda alguno"
	@echo ""
	@echo "Resultados"
	@echo "  make metrics                 regenera resultados/metricas_desarrollo.json"
	@echo "  make verify                  tests + metricas, el paso previo a entregar"

# ── entorno ──────────────────────────────────────────────────────────────────

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

# ── notebooks ────────────────────────────────────────────────────────────────
# ⚠️ Reconstruir BORRA las salidas. Si el notebook tiene resultados que quieres
# conservar, guárdalos antes: el builder parte de cero desde notebook_cells.py.

notebooks:
	$(PY) scripts/build_notebook.py

notebook:
	@test -n "$(NB)" || { echo "Falta NB=. Ejemplo: make notebook NB=04_motor.ipynb"; exit 1; }
	$(PY) scripts/build_notebook.py $(NB)

# ── motores ──────────────────────────────────────────────────────────────────
# Uno cada vez: los tres no caben en 7,9 GB. Ver docker/README.md.

motor-up:
	@test -f $(COMPOSE) || { echo "No existe $(COMPOSE). MOTOR debe ser qdrant, weaviate o milvus."; exit 1; }
	docker compose -f $(COMPOSE) up -d --wait

motor-ps:
	docker compose -f $(COMPOSE) ps

motor-down:
	docker compose -f $(COMPOSE) down

motor-stats:
	docker stats --no-stream
	@echo "--- volúmenes ---"
	docker system df -v | grep aurum-market || true

all-down:
	-docker compose -f docker/qdrant/compose.yaml down
	-docker compose -f docker/weaviate/compose.yaml down
	-docker compose -f docker/milvus/compose.yaml down

# ── resultados ───────────────────────────────────────────────────────────────
# §8: "las métricas pueden regenerarse desde un único comando". Consolida los
# artefactos que ya han producido los notebooks; no re-codifica nada.

metrics:
	$(PY) scripts/consolidar_metricas.py

verify: test metrics
	@echo "OK - tests en verde y metricas regeneradas."

# ── limpieza ─────────────────────────────────────────────────────────────────

clean-pyc:
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
