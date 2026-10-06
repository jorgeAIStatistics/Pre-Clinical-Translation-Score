# Preclinical Translational Risk Scorer — developer entry points.
#
# Every target here is a thin, readable wrapper around a command you could type
# yourself. Nothing is hidden: if a target breaks, the underlying command is on
# the line above it. Run `make` with no argument for the list.
#
# The backend deliberately has no activated-virtualenv requirement. Every recipe
# calls the interpreter in ./.venv by absolute path, so `make` behaves the same
# from a shell, an editor task runner or CI.

SHELL     := /bin/bash
ROOT      := $(CURDIR)
VENV      := $(ROOT)/.venv
PY        := $(VENV)/bin/python
PIP       := $(VENV)/bin/pip
ALEMBIC   := $(VENV)/bin/alembic
UVICORN   := $(VENV)/bin/uvicorn
RUFF      := $(VENV)/bin/ruff
BACKEND   := $(ROOT)/backend
FRONTEND  := $(ROOT)/frontend

# Overridable on the command line, e.g. `make api API_PORT=8001`.
API_HOST  ?= 127.0.0.1
API_PORT  ?= 8000
WEB_PORT  ?= 3000
PROGRAMS  ?= 600
ENDPOINT  ?= ph2_efficacy_failure
COHORT    ?= oncology_ph2_efficacy

.DEFAULT_GOAL := help
.PHONY: help install db seed train api web test lint clean demo facts

help:  ## Show this help.
	@echo "Preclinical Translational Risk Scorer"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-10s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Demonstration data only. See README.md before interpreting any number."

# Fallback dependency set, transcribed from backend/pyproject.toml. It exists only
# because the editable install is currently broken: pyproject declares
# `readme = "../README.md"`, and setuptools refuses to read a readme from outside
# the project directory, so metadata generation fails with
#   DistutilsOptionError: Cannot access '.../backend/../README.md'
# Delete FALLBACK_DEPS and the `||` branch below once that line is fixed — the
# package is importable from backend/ without installation, so only the
# dependencies are actually required.
FALLBACK_DEPS := \
  "numpy>=1.26" "pandas>=2.2" "scipy>=1.12" "scikit-learn>=1.4" "statsmodels>=0.14" \
  "SQLAlchemy>=2.0" "alembic>=1.13" \
  "fastapi>=0.110" "uvicorn[standard]>=0.29" "pydantic>=2.6" "pydantic-settings>=2.2" \
  "httpx>=0.27" "python-multipart>=0.0.9" \
  "passlib[bcrypt]>=1.7.4" "python-jose[cryptography]>=3.3" \
  "pytest>=8.0" "pytest-cov>=5.0" "ruff>=0.4"

install:  ## Create the venv if absent, install the backend (editable, with dev extras) and the frontend.
	@test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	@$(PIP) install -e "$(BACKEND)[dev]" || { \
	  echo ""; \
	  echo ">> Editable install of backend/ failed."; \
	  echo ">> Known cause: backend/pyproject.toml sets readme = \"../README.md\","; \
	  echo ">> which setuptools refuses to read from outside the project directory."; \
	  echo ">> Installing dependencies directly instead. The package is importable"; \
	  echo ">> from backend/ without installation, so every documented command works."; \
	  echo ""; \
	  $(PIP) install $(FALLBACK_DEPS); \
	}
	cd $(FRONTEND) && npm install --no-audit --no-fund

db:  ## Apply Alembic migrations to TRS_DATABASE_URL (SQLite by default).
	cd $(BACKEND) && $(ALEMBIC) upgrade head

seed:  ## Generate the synthetic demonstration dataset. DESTRUCTIVE-ADJACENT: appends programs.
	cd $(BACKEND) && $(PY) -m trs.cli seed --programs $(PROGRAMS)

train:  ## Build the cohort and train, validate, calibrate and register a model.
	cd $(BACKEND) && $(PY) -m trs.cli build-cohort --name $(COHORT) --endpoint $(ENDPOINT)
	cd $(BACKEND) && $(PY) -m trs.cli train --endpoint $(ENDPOINT)

api:  ## Run the FastAPI server. OpenAPI at /api/docs.
	cd $(BACKEND) && $(UVICORN) trs.api.main:app --reload --host $(API_HOST) --port $(API_PORT)

web:  ## Run the Next.js development server.
	cd $(FRONTEND) && npm run dev -- --port $(WEB_PORT)

test:  ## Run the backend test suite (offline; no network, in-memory database).
	cd $(BACKEND) && $(PY) -m pytest tests

facts:  ## Print every checkable number, for verifying documentation against the live system.
	cd $(BACKEND) && $(PY) -m trs.cli facts

lint:  ## Ruff on the backend, TypeScript on the frontend.
	@if [ -x "$(RUFF)" ]; then \
	  cd $(BACKEND) && $(RUFF) check trs tests; \
	else \
	  echo "ruff is not installed in $(VENV); run 'make install' to add the dev extras."; \
	fi
	cd $(FRONTEND) && npm run typecheck

clean:  ## Remove build and test caches. Leaves the database, model artifacts and connector cache alone.
	find $(BACKEND) -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf $(BACKEND)/.pytest_cache $(BACKEND)/.ruff_cache $(BACKEND)/.coverage
	rm -rf $(FRONTEND)/.next $(FRONTEND)/tsconfig.tsbuildinfo

demo:  ## Full bootstrap from an empty database: migrate, seed, build cohort, train.
	cd $(BACKEND) && $(ALEMBIC) upgrade head
	cd $(BACKEND) && $(PY) -m trs.cli pipeline --programs $(PROGRAMS) --endpoint $(ENDPOINT) --name $(COHORT)
