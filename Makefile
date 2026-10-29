.PHONY: setup lint format test audit demo clean

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -e ".[dev,fairness]"
	. .venv/bin/activate && pre-commit install

lint:
	ruff check .
	ruff format --check .

format:
	ruff check . --fix
	ruff format .

test:
	pytest --cov=cfa --cov-report=term-missing

audit:
	bandit -r src -q
	pip-audit

demo:
	cfa demo --out outputs

clean:
	rm -rf outputs .pytest_cache .ruff_cache .coverage htmlcov
