install:
	pip install -r requirements.txt

run:
	python -m src.run_mvp

test:
	pytest tests -q.PHONY: install test

install:
	pip install -r requirements.txt

test:
	pytest -q
