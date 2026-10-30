# feels — common tasks. `make help` lists them.
# Prefers the repo venv (.venv) when present, else the current python.

PY := $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)
PORT ?= 8080

.PHONY: help setup dashboard mock teach teach-dry list lint test

help: ## show this list
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /  —  /'

setup: ## one-command dev setup (venv, deps, .env, sanity check)
	bash setup.sh

dashboard: ## run the live dashboard (needs robot + Atlas in .env)
	$(PY) realtime_plot.py --port $(PORT)

mock: ## run the full UI on recorded data, no hardware (purple MOCK badge)
	FEELS_MOCK=1 $(PY) realtime_plot.py --port $(PORT)

teach: ## batch-teach swirl recordings into Atlas (preview with `make teach-dry` first)
	$(PY) teach_csv.py recordings/ --require-swirl --session fri

teach-dry: ## preview what teach_csv.py would teach
	$(PY) teach_csv.py recordings/ --require-swirl --dry-run

list: ## show what's currently in the Atlas signature library
	$(PY) teach_csv.py --list

lint: ## pyflakes over the active scripts (repeatable_grasp.py is locked — excluded)
	$(PY) -m pyflakes atlas.py realtime_plot.py feature_extractor.py teach_csv.py predict_mystery.py

test: ## offline smoke suite: compile + CLI help + mock dashboard endpoints
	$(PY) -m py_compile atlas.py realtime_plot.py feature_extractor.py teach_csv.py predict_mystery.py
	@for f in teach_csv predict_mystery feature_extractor atlas realtime_plot; do $(PY) $$f.py --help >/dev/null || exit 1; done
	@FEELS_MOCK=1 $(PY) realtime_plot.py --port 18099 > /tmp/feels-test.log 2>&1 & echo $$! > /tmp/feels-test.pid; \
	sleep 8; \
	(curl -sf http://localhost:18099/ -o /dev/null && \
	 curl -sf -X POST http://localhost:18099/api/cycle/predict -o /dev/null && \
	 curl -sf http://localhost:18099/docs -o /dev/null && \
	 curl -sf http://localhost:18099/docs/problem -o /dev/null && \
	 curl -sf http://localhost:18099/docs/approach -o /dev/null && \
	 curl -sf http://localhost:18099/docs/prediction -o /dev/null && \
	 curl -sf http://localhost:18099/docs/force -o /dev/null && \
	 curl -sf http://localhost:18099/docs/api -o /dev/null && \
	 echo ALL_SMOKE_PASS); \
	rc=$$?; kill $$(cat /tmp/feels-test.pid) 2>/dev/null || true; \
	rm -f /tmp/feels-test.pid /tmp/feels-test.log; exit $$rc
