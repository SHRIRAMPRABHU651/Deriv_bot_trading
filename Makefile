.PHONY: install test lint typecheck check run demo clean security

PY ?= python
VENV ?= .venv
ifeq ($(OS),Windows_NT)
  BIN := $(VENV)/Scripts
else
  BIN := $(VENV)/bin
endif

install:
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt

test:
	$(BIN)/python -m pytest -q

lint:
	$(BIN)/ruff check .

typecheck:
	$(BIN)/mypy app tests research scripts

security:
	$(BIN)/python scripts/check.py

# lint + typecheck + all tests (+ secret scan). Must exit 0.
check: lint typecheck security test

run:
	$(BIN)/python -m app.main

# DEMO only, autostart. Refuses to run without DEMO credentials in .env.
demo:
	$(BIN)/python scripts/run_demo.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
