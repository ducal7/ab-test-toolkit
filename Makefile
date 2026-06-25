.PHONY: data report test lint all

# Use the project virtualenv if present, else fall back to `python`.
PY ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python)

data:
	$(PY) -m ab.data

report:
	$(PY) -m ab.report

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

all: lint test data report
